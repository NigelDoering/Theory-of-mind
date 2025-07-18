import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from real_world_src.models.tom_graph_encoder import ToMGraphEncoder, CampusDataLoader
from real_world_src.models.tom_latent_mind import HierarchicalMindStateVAE
from real_world_src.models.tom_goal_predictor import GoalPredictorHead

torch.backends.cudnn.benchmark = True

class StepwiseInMemoryDataset(Dataset):
    def __init__(self, path_data, goal_data, node_id_mapping, max_seq_len=100):
        self.samples = []
        valid_node_ids = set(node_id_mapping.keys())
        for episode, agent_dict in goal_data.items():
            for agent_id, goal_node in agent_dict.items():
                if goal_node not in valid_node_ids:
                    continue
                if agent_id in path_data.get(episode, {}):
                    traj = path_data[episode][agent_id]
                    traj = [n for n in traj if n in valid_node_ids]
                    if len(traj) < 2:
                        continue
                    traj_idx = [node_id_mapping[n] for n in traj]
                    goal_idx = node_id_mapping[goal_node]
                    for t in range(2, min(len(traj_idx)+1, max_seq_len+1)):
                        partial = traj_idx[:t]
                        pad_len = max_seq_len - len(partial)
                        if pad_len > 0:
                            partial = partial + [0]*pad_len
                        if t == len(traj_idx):
                            target_goal = goal_idx
                        else:
                            target_goal = partial[t-1]
                        self.samples.append((partial, target_goal, goal_idx, t))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        traj, target_goal, true_goal, t = self.samples[idx]
        return torch.tensor(traj, dtype=torch.long), target_goal, true_goal, t

class TomNetCausal(nn.Module):
    def __init__(self, node_feat_dim, num_nodes, time_emb_dim=16, hidden_dim=128, latent_dim=32, n_layers=2, n_heads=4, dropout=0.1, use_gat=True):
        super().__init__()
        self.encoder = ToMGraphEncoder(
            node_feat_dim=node_feat_dim,
            time_emb_dim=time_emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            use_gat=use_gat
        )
        self.latent_vae = HierarchicalMindStateVAE(
            input_dim=hidden_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        self.goal_head = GoalPredictorHead(
            latent_dim=latent_dim,
            output_type='node',
            num_nodes=num_nodes
        )
    def forward(self, batch_trajectory_data, graph_data):
        fused = self.encoder(batch_trajectory_data, graph_data)
        latents = self.latent_vae(fused)
        logits = self.goal_head(latents['z_belief'], latents['z_desire'], latents['z_intention'])
        return latents, fused, logits

def train_pipeline(
    epochs=10, batch_size=64, log_wandb=True, max_seq_len=100, top_k=5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if log_wandb:
        wandb.init(project="tom-graph-causalnet-distributional", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len
        })
    # Load data
    data_loader = CampusDataLoader()
    print("Done campus data loader")
    path_data = data_loader.path_data
    goal_data = data_loader.goal_data
    node_id_mapping = data_loader.node_id_mapping
    num_nodes = len(node_id_mapping)
    graph_data = data_loader.prepare_graph_data()
    node_feat_dim = graph_data['x'].shape[1]
    print("Done graph data, starting with pt dataset")
    dataset = StepwiseInMemoryDataset(path_data, goal_data, node_id_mapping, max_seq_len=max_seq_len)
    print("Done pt dataset, starting with dataloader")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=24)
    print("Done dataloader, starting with model")

    # Model
    model = TomNetCausal(
        node_feat_dim=node_feat_dim,
        num_nodes=num_nodes,
        time_emb_dim=16,
        hidden_dim=128,
        latent_dim=32,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        use_gat=True
    ).to(device)
    
    model.encoder.trajectory_encoder.node_embedding = nn.Embedding(num_nodes, node_feat_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    graph_data_on_device = {k: v.to(device, non_blocking=True) for k, v in graph_data.items()}
    
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        total_loss, total_brier, total_acc, total_topk, count = 0, 0, 0, 0, 0
        for batch_idx, (trajs, target_goals, true_goals, ts) in enumerate(dataloader):
            batch_size_, seq_len = trajs.shape
            timestamps = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(0).repeat(batch_size_, 1)
            mask = (trajs != 0).float()
            batch_trajectory_data = {
                'node_ids': trajs.to(device, non_blocking=True),
                'timestamps': timestamps,
                'mask': mask.to(device, non_blocking=True)
            }
            target_goals = target_goals.to(device, non_blocking=True)
            optimizer.zero_grad()
            latents, fused, logits = model(batch_trajectory_data, graph_data_on_device)
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
    torch.save(model.state_dict(), "tomnet_pipeline_model.pth")
    print("Model saved as tomnet_pipeline_model.pth")

def main():
    train_pipeline(epochs=10, batch_size=2048, log_wandb=True, max_seq_len=100, top_k=5)

if __name__ == "__main__":
    main() 