import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from real_world_src.models.tom_graph_encoder import ToMGraphEncoder, CampusDataLoader
from real_world_src.models.tom_latent_mind import HierarchicalMindStateVAE
from real_world_src.models.tom_goal_predictor import GoalPredictorHead
# -----------------------------
# ToM-GraphCausalNet Trainer
# -----------------------------
class ToMGraphCausalNet(nn.Module):
    """
    End-to-end model: Encoder + Hierarchical Latent Mind-State + Goal Predictor
    """
    def __init__(self, node_feat_dim=4, time_emb_dim=16, hidden_dim=128, latent_dim=32, n_layers=2, n_heads=4, dropout=0.1, use_gat=True, goal_output_type='coord', num_nodes=None, mdn_components=0):
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
            output_type=goal_output_type,
            num_nodes=num_nodes,
            mdn_components=mdn_components
        )

    def forward(self, trajectory_data, graph_data):
        fused = self.encoder(trajectory_data, graph_data)  # (batch_size, hidden_dim)
        latents = self.latent_vae(fused)
        goal_pred = self.goal_head(latents['z_belief'], latents['z_desire'], latents['z_intention'])
        return latents, fused, goal_pred

# -----------------------------
# PyTorch Dataset for Per-Episode Supervision
# -----------------------------
class EpisodeGoalDataset(Dataset):
    def __init__(self, path_data, goal_data, node_id_mapping, max_seq_len=100):
        self.samples = []
        valid_node_ids = set(node_id_mapping.keys())
        for episode, agent_dict in goal_data.items():
            for agent_id, goal_node in agent_dict.items():
                # Only keep if goal_node is valid
                if goal_node not in valid_node_ids:
                    continue
                # Get trajectory for this agent in this episode
                if agent_id in path_data.get(episode, {}):
                    traj = path_data[episode][agent_id]
                    # Filter out invalid nodes
                    traj = [n for n in traj if n in valid_node_ids]
                    if len(traj) == 0:
                        continue
                    # Map to indices
                    traj = [node_id_mapping[n] for n in traj]
                    # Pad or truncate
                    if len(traj) < max_seq_len:
                        traj = traj + [0] * (max_seq_len - len(traj))
                    else:
                        traj = traj[:max_seq_len]
                    goal_idx = node_id_mapping[goal_node]
                    self.samples.append((traj, goal_idx))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        traj, goal_idx = self.samples[idx]
        return torch.tensor(traj, dtype=torch.long), goal_idx

# Collate function for batching
def episode_goal_collate_fn(batch):
    trajs, goal_idxs = zip(*batch)
    trajs = torch.stack(trajs)  # (batch, seq_len)
    goal_idxs = torch.tensor(goal_idxs, dtype=torch.long)
    batch_size, seq_len = trajs.shape
    timestamps = torch.arange(seq_len, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    mask = (trajs != 0).float()
    batch_trajectory_data = {
        'node_ids': trajs,
        'timestamps': timestamps,
        'mask': mask
    }
    return batch_trajectory_data, goal_idxs

# Brier score computation
def brier_score(pred_probs, true_idx):
    # pred_probs: (batch, num_nodes), true_idx: (batch,)
    true_onehot = torch.zeros_like(pred_probs)
    true_onehot[torch.arange(pred_probs.size(0)), true_idx] = 1.0
    return ((pred_probs - true_onehot) ** 2).sum(dim=1).mean().item()

# -----------------------------
# Full Training Loop with wandb Logging and Brier Score
# -----------------------------
def train_with_real_data_distributional(
    epochs=25, batch_size=16, log_wandb=True, top_k=5, max_seq_len=100, wandb_log_interval=1, num_workers=2
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if log_wandb:
        wandb.init(project="tom-graph-causalnet-distributional", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "top_k": top_k,
            "max_seq_len": max_seq_len
        })

    # Load data
    data_loader = CampusDataLoader()
    path_data = data_loader.path_data
    goal_data = data_loader.goal_data
    node_id_mapping = data_loader.node_id_mapping
    num_nodes = len(node_id_mapping)
    print(f"Number of unique nodes in the framework: {num_nodes}")
    graph_data = data_loader.prepare_graph_data()
    node_feat_dim = graph_data['x'].shape[1]

    # Dataset and DataLoader
    dataset = EpisodeGoalDataset(path_data, goal_data, node_id_mapping, max_seq_len=max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=episode_goal_collate_fn
    )
    print(f"Loaded {len(dataset)} episode-agent samples.")

    # Model setup
    model = ToMGraphCausalNet(
        node_feat_dim=node_feat_dim,
        time_emb_dim=16,
        hidden_dim=128,
        latent_dim=32,
        goal_output_type='node',
        num_nodes=num_nodes,
        mdn_components=0
    ).to(device)
    model.encoder.trajectory_encoder.node_embedding = nn.Embedding(num_nodes, node_feat_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("Starting training loop...")
    model.train()
    total_kl, total_recon, total_goal, total_loss, total_brier = 0, 0, 0, 0, 0
    correct, total, topk_correct = 0, 0, 0
    num_batches = 0
    # Move graph_data to device ONCE
    graph_data_on_device = {k: v.to(device, non_blocking=True) for k, v in graph_data.items()}

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} start")
        for batch_idx, (batch_trajectory_data, goal_idxs) in enumerate(dataloader):
            # Remove or reduce print statements here
            batch_trajectory_data = {k: v.to(device, non_blocking=True) for k, v in batch_trajectory_data.items()}
            goal_idxs = goal_idxs.to(device, non_blocking=True)
            # Use graph_data_on_device, not graph_data
            latents, fused, goal_logits = model(batch_trajectory_data, graph_data_on_device)
            # Forward pass
            # optimizer.zero_grad() # This line is moved up
            kl_loss = model.latent_vae.total_kl_loss(latents)
            recon_loss = model.latent_vae.recon_loss(fused, latents)
            # Cross-entropy loss for one-hot ground truth
            goal_loss = nn.CrossEntropyLoss()(goal_logits, goal_idxs)
            loss = kl_loss + recon_loss + goal_loss
            loss.backward()
            optimizer.step()
            total_kl += kl_loss.item()
            total_recon += recon_loss.item()
            total_goal += goal_loss.item()
            total_loss += loss.item()
            num_batches += 1
            # Metrics
            probs = model.goal_head.get_probabilities(goal_logits)
            top1_pred = torch.argmax(probs, dim=-1)
            correct += (top1_pred == goal_idxs).sum().item()
            topk = torch.topk(probs, k=top_k, dim=-1).indices
            for b in range(goal_idxs.size(0)):
                if goal_idxs[b].item() in topk[b].tolist():
                    topk_correct += 1
            total += goal_idxs.size(0)
            # Brier score
            brier = brier_score(probs, goal_idxs)
            total_brier += brier

            # --- wandb logging for every batch ---
            if log_wandb:
                batch_acc = (top1_pred == goal_idxs).float().mean().item()
                batch_topk_acc = sum([goal_idxs[b].item() in topk[b].tolist() for b in range(goal_idxs.size(0))]) / goal_idxs.size(0)
                wandb.log({
                    "brier_score_batch": brier,
                    "accuracy_batch": batch_acc,
                    f"top{top_k}_accuracy_batch": batch_topk_acc,
                    "loss_batch": loss.item()
                }, commit=False)
            if log_wandb and (num_batches % wandb_log_interval == 0):
                import matplotlib.pyplot as plt
                import io
                import numpy as np
                from PIL import Image
                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                gt = torch.zeros(num_nodes)
                gt[goal_idxs[0].item()] = 1.0
                pred = probs[0].detach().cpu().numpy()
                ax.bar(np.arange(len(gt)), gt.numpy(), color='g', alpha=0.5, label='Ground Truth')
                ax.bar(np.arange(len(pred)), pred, color='b', alpha=0.3, label='Predicted')
                ax.set_title('Goal Distribution (First Sample in Batch)')
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
                wandb.log({"goal_distribution_plot": wandb.Image(img_np)}, commit=False)
                entropy = -np.sum(pred * np.log(pred + 1e-8))
                wandb.log({"predicted_goal_entropy": entropy}, commit=False)
                wandb.log({
                    "latent_belief": wandb.Histogram(latents['z_belief'].detach().cpu().numpy()),
                    "latent_desire": wandb.Histogram(latents['z_desire'].detach().cpu().numpy()),
                    "latent_intention": wandb.Histogram(latents['z_intention'].detach().cpu().numpy()),
                    "brier_score": brier
                }, commit=False)
        acc = correct / total if total > 0 else 0.0
        topk_acc = topk_correct / total if total > 0 else 0.0
        avg_brier = total_brier / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} | KL: {total_kl/num_batches:.4f} | Recon: {total_recon/num_batches:.4f} | Goal: {total_goal/num_batches:.4f} | Total: {total_loss/num_batches:.4f} | Acc: {acc:.4f} | Top-{top_k} Acc: {topk_acc:.4f} | Brier: {avg_brier:.4f}")
        if log_wandb:
            wandb.log({
                "epoch": epoch+1,
                "kl_loss": total_kl/num_batches,
                "recon_loss": total_recon/num_batches,
                "goal_loss": total_goal/num_batches,
                "total_loss": total_loss/num_batches,
                "accuracy": acc,
                f"top{top_k}_accuracy": topk_acc,
                "brier_score": avg_brier
            })
    print("✅ Distributional training completed!")
    if log_wandb:
        wandb.finish()
    torch.save(model.state_dict(), "tom_graph_causalnet_distributional.pth")
    print("✅ Model saved!")

if __name__ == "__main__":
    # Set up for distributional goal prediction and hybrid evaluation
    train_with_real_data_distributional(epochs=25, batch_size=16, log_wandb=True, top_k=5, max_seq_len=100)