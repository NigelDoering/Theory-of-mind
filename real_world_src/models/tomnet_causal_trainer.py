import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import argparse
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from real_world_src.models.tomnet_causal_dataloader import CampusDataLoader
from real_world_src.models.tom_graph_encoder import ToMGraphEncoder
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
    epochs=10, batch_size=64, log_wandb=True, max_seq_len=100, top_k=5, gpu=0, data_dir="./data/1k/"
):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if log_wandb:
        wandb.init(project="tom-graph-causalnet-distributional", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len
        })
    # Load data
    data_loader = CampusDataLoader(data_dir=data_dir)
    # print("Done campus data loader")
    path_data = data_loader.path_data
    goal_data = data_loader.goal_data
    node_id_mapping = data_loader.node_id_mapping
    num_nodes = len(node_id_mapping)

    # Prepare graph data (only x and edge_index, with pin_memory)
    graph_cuda_path = 'graph_data_cuda.pt'
    if os.path.exists(graph_cuda_path):
        print(f"Loading graph data from {graph_cuda_path}...")
        import time
        t0 = time.time()
        graph_data_on_device = torch.load(graph_cuda_path)
        print(f"Loaded graph_data_cuda.pt in {time.time() - t0:.2f} seconds")
        # Do not reference graph_data here
    else:
        print("Done graph data, starting with pt dataset")
        import time
        t0 = time.time()
        graph_data = data_loader.prepare_graph_data()
        # Only move x and edge_index
        graph_data_on_device = {k: v.to(device, non_blocking=True) for k, v in graph_data.items() if k in ['x', 'edge_index']}
        print(f"Moved graph_data to device in {time.time() - t0:.2f} seconds")
        # Save for future runs
        torch.save(graph_data_on_device, graph_cuda_path)
        print(f"Saved graph_data_cuda.pt for future runs.")
    node_feat_dim = graph_data_on_device['x'].shape[1]
    # print("Done graph data, starting with pt dataset")
    dataset = StepwiseInMemoryDataset(path_data, goal_data, node_id_mapping, max_seq_len=max_seq_len)
    # print("Done pt dataset, starting with dataloader")
    # Train/validation split
    val_split = 0.2
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Train/val split: {n_train} train, {n_val} val")
    # print("Done dataloader, starting with model")

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
    # graph_data_on_device = {k: v.to(device, non_blocking=True) for k, v in graph_data.items()} # This line is removed as per the edit hint
    
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        total_loss, total_brier, total_acc, total_topk, count = 0, 0, 0, 0, 0
        for batch_idx, (trajs, target_goals, true_goals, ts) in enumerate(train_loader):
            # print(f"Batch {batch_idx+1} of {len(train_loader)}")
            batch_size_, seq_len = trajs.shape
            timestamps = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(0).repeat(batch_size_, 1)
            mask = (trajs != 0).float()
            # print("obtaining batch trajectory data")
            batch_trajectory_data = {
                'node_ids': trajs.to(device, non_blocking=True),
                'timestamps': timestamps,
                'mask': mask.to(device, non_blocking=True)
            }
            # print("obtaining target goals")
            target_goals = target_goals.to(device, non_blocking=True)
            optimizer.zero_grad()
            # print("obtaining latents, fused, logits")
            latents, fused, logits = model(batch_trajectory_data, graph_data_on_device)
            # print("obtaining loss")
            loss = nn.CrossEntropyLoss()(logits, target_goals)
            loss.backward()
            optimizer.step()
            # print("obtaining probs")
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
            # print("logging")
            if log_wandb:
                wandb.log({
                    "training/loss_step": loss.item(),
                    "training/brier_score_step": brier,
                    "training/accuracy_step": acc,
                    f"training/top{top_k}_accuracy_step": topk_acc
                }, commit=False)
        avg_loss = total_loss / count
        avg_brier = total_brier / count
        avg_acc = total_acc / count
        avg_topk = total_topk / count
        if log_wandb:
            wandb.log({
                "training/epoch": epoch+1,
                "training/loss_epoch": avg_loss,
                "training/brier_score_epoch": avg_brier,
                "training/accuracy_epoch": avg_acc,
                f"top{top_k}_accuracy_epoch": avg_topk
            })
        # Validation every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss, val_brier, val_acc, val_topk, val_count = 0, 0, 0, 0, 0
            with torch.no_grad():
                for val_trajs, val_target_goals, val_true_goals, val_ts in val_loader:
                    batch_size_, seq_len = val_trajs.shape
                    timestamps = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(0).repeat(batch_size_, 1)
                    mask = (val_trajs != 0).float()
                    batch_trajectory_data = {
                        'node_ids': val_trajs.to(device, non_blocking=True),
                        'timestamps': timestamps,
                        'mask': mask.to(device, non_blocking=True)
                    }
                    val_target_goals = val_target_goals.to(device, non_blocking=True)
                    latents, fused, logits = model(batch_trajectory_data, graph_data_on_device)
                    loss = nn.CrossEntropyLoss()(logits, val_target_goals)
                    probs = model.goal_head.get_probabilities(logits)
                    true_idx = val_target_goals
                    true_onehot = torch.zeros_like(probs)
                    true_onehot[torch.arange(probs.size(0)), true_idx] = 1.0
                    brier = ((probs - true_onehot) ** 2).sum(dim=1).mean().item()
                    pred = torch.argmax(probs, dim=-1)
                    acc = (pred == val_target_goals).float().mean().item()
                    topk = torch.topk(probs, k=top_k, dim=-1).indices
                    topk_acc = (topk == val_target_goals.unsqueeze(1)).any(dim=1).float().mean().item()
                    val_loss += loss.item() * val_target_goals.size(0)
                    val_brier += brier * val_target_goals.size(0)
                    val_acc += acc * val_target_goals.size(0)
                    val_topk += topk_acc * val_target_goals.size(0)
                    val_count += val_target_goals.size(0)
            val_loss /= val_count
            val_brier /= val_count
            val_acc /= val_count
            val_topk /= val_count
            print(f"Validation after epoch {epoch+1}: loss={val_loss:.4f}, brier={val_brier:.4f}, acc={val_acc:.4f}, top{top_k}_acc={val_topk:.4f}")
            if log_wandb:
                wandb.log({
                    "validation/val_loss": val_loss,
                    "validation/val_brier_score": val_brier,
                    "validation/val_accuracy": val_acc,
                    f"validation/val_top{top_k}_accuracy": val_topk,
                    "validation/val_epoch": epoch+1
                })
            model.train()
    if log_wandb:
        wandb.finish()
    model_save_path = f"./trained_models/tomnet_causal_model_{os.path.basename(data_dir).replace('/', '_')}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train TomNet Causal Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--log_wandb", action='store_true', help="Log metrics to Weights & Biases")
    parser.add_argument("--max_seq_len", type=int, default=100, help="Maximum sequence length for trajectories")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k accuracy to compute")
    parser.add_argument("--gpu", type=int, default=0, help="Specify GPU to use, e.g., '0' for cuda:0")
    parser.add_argument("--data_dir", type=str, default="./data/1k/", help="Directory for the dataset")

    args = parser.parse_args()
    train_pipeline(epochs=args.epochs, batch_size=args.batch_size, log_wandb=args.log_wandb, max_seq_len=args.max_seq_len, top_k=args.top_k, gpu=args.gpu)

if __name__ == "__main__":
    main() 