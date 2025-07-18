import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from real_world_src.models.tom_graph_encoder import ToMGraphEncoder, CampusDataLoader
from real_world_src.models.tom_latent_mind import HierarchicalMindStateVAE
from real_world_src.models.tom_goal_predictor import GoalPredictorHead

torch.backends.cudnn.benchmark = True

class StepwiseMemmapDataset(Dataset):
    def __init__(self, npz_path):
        print(f"Loading stepwise data from {npz_path} with memory mapping...")
        self.data = np.load(npz_path, mmap_mode='r')
        self.trajs = self.data['trajs']
        self.targets = self.data['targets']
        self.true_goals = self.data['true_goals']
        self.ts = self.data['ts']
    
    def __len__(self):
        return self.trajs.shape[0]
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.trajs[idx]),
            int(self.targets[idx]),
            int(self.true_goals[idx]),
            int(self.ts[idx])
        )

def train_stepwise_goal_inference(
    epochs=10, batch_size=512, log_wandb=True, max_seq_len=100, num_workers=0, top_k=5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if log_wandb:
        wandb.init(project="tom-graph-causalnet-distributional", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len
        })
    
    # Load preprocessed data
    data_path = 'stepwise_data.npz'
    if not os.path.exists(data_path):
        print("stepwise_data.npz not found. Running preprocessing script...")
        os.system(f"python real_world_src/models/preprocess_stepwise_data.py")
    dataset = StepwiseMemmapDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # Model setup
    data_loader = CampusDataLoader()
    node_id_mapping = data_loader.node_id_mapping
    num_nodes = len(node_id_mapping)
    graph_data = data_loader.prepare_graph_data()
    node_feat_dim = graph_data['x'].shape[1]
    
    model = nn.Module()
    model.encoder = ToMGraphEncoder(
        node_feat_dim=node_feat_dim,
        time_emb_dim=16,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        use_gat=True
    ).to(device)
    
    model.latent_vae = HierarchicalMindStateVAE(
        input_dim=128,
        latent_dim=32,
        hidden_dim=128
    ).to(device)
    
    model.goal_head = GoalPredictorHead(
        latent_dim=32,
        output_type='node',
        num_nodes=num_nodes
    ).to(device)
    
    model.encoder.trajectory_encoder.node_embedding = nn.Embedding(num_nodes, node_feat_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    graph_data_on_device = {k: v.to(device, non_blocking=True) for k, v in graph_data.items()}
    
    model.train()
    for epoch in range(epochs):
        total_loss, total_brier, total_acc, total_topk, count = 0, 0, 0, 0, 0
        for batch_idx, (trajs, target_goals, true_goals, ts) in enumerate(dataloader):
            batch_size, seq_len = trajs.shape
            timestamps = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(0).repeat(batch_size, 1)
            mask = (trajs != 0).float()
            batch_trajectory_data = {
                'node_ids': trajs.to(device, non_blocking=True),
                'timestamps': timestamps,
                'mask': mask.to(device, non_blocking=True)
            }
            target_goals = target_goals.to(device, non_blocking=True)
            optimizer.zero_grad()
            fused = model.encoder(batch_trajectory_data, graph_data_on_device)
            latents = model.latent_vae(fused)
            logits = model.goal_head(latents['z_belief'], latents['z_desire'], latents['z_intention'])
            loss = nn.CrossEntropyLoss()(logits, target_goals)
            loss.backward()
            optimizer.step()
    
            with torch.no_grad():
                probs = model.goal_head.get_probabilities(logits)
                true_idx = target_goals
                true_onehot = torch.zeros_like(probs)
                true_onehot[torch.arange(probs.size(0)), true_idx] = 1.0
                brier = ((probs - true_onehot) ** 2).sum(dim=1).mean().item()
    
                pred = torch.argmax(probs, dim=-1)
                acc = (pred == target_goals).float().mean().item()
    
                topk = torch.topk(probs, k=top_k, dim=-1).indices
                topk_acc = (topk == target_goals.unsqueeze(1)).any(dim=1).float().mean().item()
    
                total_loss += loss.item() * target_goals.size(0)
                total_brier += brier * target_goals.size(0)
                total_acc += acc * target_goals.size(0)
                total_topk += topk_acc * target_goals.size(0)
    
                count += target_goals.size(0)
    
            if log_wandb:
                wandb.log({
                    "loss_step": loss.item(),
                    "brier_score_step": brier,
                    "accuracy_step": acc,
                    f"top{top_k}_accuracy_step": topk_acc
                }, commit=False)
                if batch_idx == np.random.randint(0, len(dataloader)):
                    gt = torch.zeros(num_nodes)
                    gt[target_goals[0].item()] = 1.0
                    pred_dist = probs[0].detach().cpu().numpy()
                    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                    ax.bar(np.arange(len(gt)), gt.numpy(), color='g', alpha=0.5, label='Ground Truth')
                    ax.bar(np.arange(len(pred_dist)), pred_dist, color='b', alpha=0.3, label='Predicted')
                    ax.set_title(f'Goal Distribution (Step {ts[0].item()})')
                    ax.set_xlabel('Node Index')
                    ax.set_ylabel('Probability')
                    ax.legend()
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plt.close(fig)
                    img = Image.open(buf)
                    img_np = np.array(img)
                    wandb.log({"goal_distribution_plot_step": wandb.Image(img_np)}, commit=False)
    
        avg_loss = total_loss / count
        avg_brier = total_brier / count
        avg_acc = total_acc / count
        avg_topk = total_topk / count
    
        if log_wandb:
            wandb.log({
                "epoch": epoch+1,
                "loss_epoch": avg_loss,
                "brier_score_epoch": avg_brier,
                "accuracy_epoch": avg_acc,
                f"top{top_k}_accuracy_epoch": avg_topk
            })
    
    if log_wandb:
        wandb.finish()
    
    torch.save(model.state_dict(), "tom_stepwise_goal_inference.pth")

if __name__ == "__main__":
    train_stepwise_goal_inference(epochs=10, batch_size=512, log_wandb=True, max_seq_len=100, num_workers=0, top_k=5) 