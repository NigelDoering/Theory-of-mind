import torch
import torch.nn as nn
import numpy as np
import argparse
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from real_world_src.models.tomnet_causal_dataloader import CampusDataLoader
from real_world_src.models.tomnet_causal_trainer import TomNetCausal

def brier_score(probs, true_idx):
    true_onehot = torch.zeros_like(probs)
    true_onehot[torch.arange(probs.size(0)), true_idx] = 1.0
    return ((probs - true_onehot) ** 2).sum(dim=1).mean().item()

def eval_traversal_lengths(model, data_loader, graph_data_on_device, device, n_agents=10, top_k=5, log_wandb=True):
    results = []
    agent_ids = list(data_loader.path_data[list(data_loader.path_data.keys())[0]].keys())
    np.random.shuffle(agent_ids)
    agent_ids = agent_ids[:n_agents]
    for agent_id in agent_ids:
        # For each episode for this agent
        for episode in data_loader.path_data:
            if agent_id not in data_loader.path_data[episode]:
                continue
            traj = data_loader.path_data[episode][agent_id]
            goal = data_loader.goal_data[episode][agent_id]
            node_id_mapping = data_loader.node_id_mapping
            traj_idx = [node_id_mapping[n] for n in traj if n in node_id_mapping]
            goal_idx = node_id_mapping[goal]
            if len(traj_idx) < 2:
                continue
            # Evaluate at every 5% of the trajectory
            steps = np.linspace(2, len(traj_idx), num=21, dtype=int)
            for t in steps:
                partial = traj_idx[:t]
                pad_len = len(traj_idx) - len(partial)
                padded = partial + [0]*pad_len
                seq_len = len(padded)
                node_ids = torch.tensor([padded], dtype=torch.long, device=device)
                timestamps = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(0)
                mask = (node_ids != 0).float()
                batch_trajectory_data = {
                    'node_ids': node_ids,
                    'timestamps': timestamps,
                    'mask': mask
                }
                with torch.no_grad():
                    latents, fused, logits = model(batch_trajectory_data, graph_data_on_device)
                    probs = model.goal_head.get_probabilities(logits)
                    pred = torch.argmax(probs, dim=-1).item()
                    brier = brier_score(probs, torch.tensor([goal_idx], device=device))
                    topk = torch.topk(probs, k=top_k, dim=-1).indices
                    topk_acc = int(goal_idx in topk[0].tolist())
                    acc = int(pred == goal_idx)
                results.append({
                    'agent_id': agent_id,
                    'episode': episode,
                    't': t,
                    'frac': t/len(traj_idx),
                    'brier': brier,
                    'topk_acc': topk_acc,
                    'acc': acc
                })
                if log_wandb:
                    wandb.log({
                        'brier': brier,
                        'topk_acc': topk_acc,
                        'acc': acc,
                        'traversal_frac': t/len(traj_idx),
                        'agent_id': agent_id
                    })
    return results

def plot_results(results, top_k, save_dir):
    import pandas as pd
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='frac', y='brier', data=df, ci='sd')
    plt.title('Brier Score vs. Fraction of Trajectory')
    plt.xlabel('Fraction of Trajectory')
    plt.ylabel('Brier Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'brier_vs_frac.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='frac', y='acc', data=df, ci='sd', label='Top-1 Acc')
    sns.lineplot(x='frac', y='topk_acc', data=df, ci='sd', label=f'Top-{top_k} Acc')
    plt.title('Accuracy vs. Fraction of Trajectory')
    plt.xlabel('Fraction of Trajectory')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acc_vs_frac.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ToMNet Causal Model on Traversal Lengths")
    parser.add_argument('--n_agents', type=int, default=10, help='Number of agents to evaluate')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k accuracy to compute')
    parser.add_argument('--log_wandb', action='store_true', help='Log metrics to Weights & Biases')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    parser.add_argument('--data_dir', type=str, default='./data/1k/', help='Directory for the dataset')
    parser.add_argument('--model_path', type=str, default='./trained_models/tomnet_causal_model_1k.pth', help='Path to trained model')
    parser.add_argument('--save_dir', type=str, default='./eval_results', help='Directory to save plots')
    parser.add_argument('--node_mapping_path', type=str, default='./data/1k/node_mapping.pkl', help='Path to node mapping file (must match training)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.log_wandb:
        wandb.init(project="tomnet-causal-eval", config=vars(args))

    # Load data with fixed mapping
    data_loader = CampusDataLoader(data_dir=args.data_dir, node_mapping_path=args.node_mapping_path, mode='eval')
    path_data = data_loader.path_data
    goal_data = data_loader.goal_data
    node_id_mapping = data_loader.node_id_mapping
    num_nodes = len(node_id_mapping)
    # Create data_utils directory if it doesn't exist
    data_utils_dir = os.path.join(os.path.dirname(args.data_dir), 'data_utils')
    os.makedirs(data_utils_dir, exist_ok=True)
    graph_cuda_path = os.path.join(data_utils_dir, 'graph_data_cuda.pt')
    if os.path.exists(graph_cuda_path):
        graph_data_on_device = torch.load(graph_cuda_path)
    else:
        graph_data = data_loader.prepare_graph_data()
        graph_data_on_device = {k: v.to(device, non_blocking=True) for k, v in graph_data.items() if k in ['x', 'edge_index']}
        torch.save(graph_data_on_device, graph_cuda_path)
    node_feat_dim = graph_data_on_device['x'].shape[1]

    # Load model
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
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    def safe_eval_traversal_lengths(*a, **kw):
        # Wraps eval_traversal_lengths to check node index range before model call
        results = []
        for res in eval_traversal_lengths(*a, **kw):
            # Check node index range
            node_ids = res.get('node_ids', None)
            if node_ids is not None:
                max_idx = np.max(node_ids)
                if max_idx >= num_nodes:
                    print(f"[ERROR] Max node index {max_idx} >= embedding size {num_nodes}")
                    print(f"Node IDs: {node_ids}")
                    raise ValueError("Node index out of range for embedding!")
            results.append(res)
        return results

    results = safe_eval_traversal_lengths(model, data_loader, graph_data_on_device, device, n_agents=args.n_agents, top_k=args.top_k, log_wandb=args.log_wandb)

    os.makedirs(args.save_dir, exist_ok=True)
    plot_results(results, args.top_k, args.save_dir)
    print(f"Plots saved to {args.save_dir}")
    if args.log_wandb:
        wandb.save(os.path.join(args.save_dir, '*.png'))
        wandb.finish()

if __name__ == "__main__":
    main() 