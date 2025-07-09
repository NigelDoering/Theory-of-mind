import numpy as np
import pymc
import networkx as nx
import matplotlib.pyplot as plt
import random
import json
import wandb
import seaborn as sns

import sys
import os
import osmnx as ox

# Adjust this path as needed to point to your project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("PYTHONPATH:", sys.path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import multiprocessing as mp
mp.set_start_method('fork', force=True)

from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.agents.agent_factory import AgentFactory
from real_world_src.agents.agent_species import ShortestPathAgent
from real_world_src.simulation.simulator import Simulator

from real_world_src.utils.run_manager import RunManager
from real_world_src.utils.config import VISUAL_CONFIG
from real_world_src.utils.config import get_agent_color

import pickle
import wandb
import random

# ## Step 1: Loading the Data
# Initialize campus environment
campus = CampusEnvironment()

# Need to establish the set of common goals (just choose the landmark nodes)
goals = [469084068, 49150691, 768264666, 1926666015, 1926673385, 49309735,
         273627682, 445989107, 445992528, 446128310, 1772230346, 1926673336, 
         2872424923, 3139419286, 4037576308]

with open('./data/base/100_agents.pkl', 'rb') as f:
    agents = pickle.load(f)

with open("./data/base/100_path_data.json", 'r') as file:
    path_data = json.load(file)

with open("./data/base/100_goal_data.json", 'r') as file:
    goal_data = json.load(file)

def convert_keys_to_int(data):
    if isinstance(data, dict):
        return {int(k) if isinstance(k, str) and k.isdigit() else k: convert_keys_to_int(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_int(item) for item in data]
    else:
        return data

goal_data = convert_keys_to_int(goal_data)
path_data = convert_keys_to_int(path_data)

def check_indices(tensor, num_embeddings, name):
    if not torch.all((tensor >= 0) & (tensor < num_embeddings)):
        bad = tensor[(tensor < 0) | (tensor >= num_embeddings)]
        print(f"[ERROR] {name}: Found out-of-bounds indices: {bad}")
        raise ValueError(f"{name}: Found out-of-bounds indices: {bad}")


# ## Step 2: Defining ToMnet
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:1")
print("Using device:", device)

class CharacterNet(nn.Module):
    """
    Variational CharacterNet: outputs z_char, mu_char, logvar_char.
    """
    def __init__(self, 
                 node_embeddings: np.ndarray, 
                 h_lstm: int = 64, 
                 T_sup: int = 50, 
                 K: int = 10, 
                 z_dim: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        num_nodes, d_emb = node_embeddings.shape
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(node_embeddings), freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(d_emb, h_lstm, batch_first=True)
        self.layer_norm = nn.LayerNorm(h_lstm)
        self.dropout = nn.Dropout(dropout)
        
        self.K = K
        self.T_sup = T_sup
        self.z_dim = z_dim
        
        self.fc_mu = nn.Linear(h_lstm, z_dim)
        self.fc_logvar = nn.Linear(h_lstm, z_dim)

    def forward(self, support_trajs):
        
        B, K, T = support_trajs.size()
        assert K == self.K and T == self.T_sup
        
        flat = support_trajs.view(B * K, T)
        check_indices(flat, self.embedding.num_embeddings, "CharacterNet.support_trajs")
        emb = self.embedding(flat)
        _, (h_n, _) = self.lstm(emb)
        h_n = h_n.squeeze(0)
        h_n = self.layer_norm(self.dropout(h_n))
        
        chars = h_n.view(B, K, -1).mean(dim=1)  # [B, h_lstm]
        mu = self.fc_mu(chars)
        logvar = self.fc_logvar(chars)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar

class MentalNet(nn.Module):
    """
    Variational MentalNet: outputs z_mental, mu_mental, logvar_mental.
    """
    def __init__(self, 
                 node_embeddings: np.ndarray, 
                 h_lstm: int = 64, 
                 T_q: int = 20, 
                 dropout: float = 0.5, 
                 use_attention: bool = True, 
                 z_dim: int = 64):
        super().__init__()
        num_nodes, d_emb = node_embeddings.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(node_embeddings), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(d_emb, h_lstm, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(2 * h_lstm)
        self.dropout = nn.Dropout(dropout)
        
        self.T_q = T_q
        self.z_dim = z_dim
        
        self.use_attention = use_attention
        
        if use_attention:
            self.attn = nn.Linear(h_lstm * 2, 1)
        
        self.fc_mu = nn.Linear(2 * h_lstm, z_dim)
        self.fc_logvar = nn.Linear(2 * h_lstm, z_dim)

    def forward(self, prefix: torch.LongTensor, prefix_len: torch.LongTensor):
        B, T = prefix.size()
        assert T == self.T_q, f"Expected T_q={self.T_q}, got {T}"
        
        check_indices(prefix, self.embedding.num_embeddings, "MentalNet.prefix")
        emb = self.embedding(prefix)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths=prefix_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=self.T_q)
        out = self.layer_norm(self.dropout(out))  # [B, T_q, 2*h_lstm]
        
        if self.use_attention:
            mask = torch.arange(self.T_q, device=prefix.device)[None, :] < prefix_len[:, None]
            attn_scores = self.attn(out).squeeze(-1)
            attn_scores[~mask] = float('-inf')
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
            feat = (out * attn_weights).sum(dim=1)
        else:
            idx = (prefix_len - 1).clamp(min=0)
            feat = out[torch.arange(B), idx]
        
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar

class ToMNet(nn.Module):
    """
    ToMNet with DVIB: CharacterNet + MentalNet + fusion MLP + prediction heads.
    """
    def __init__(self, 
                 node_embeddings: np.ndarray, 
                 num_nodes: int, 
                 num_goals: int, 
                 K: int = 10, 
                 T_sup: int = 50, 
                 T_q: int = 20, 
                 h_char: int = 64, 
                 h_ment: int = 64, 
                 z_dim: int = 32, 
                 dvib_z_dim: int = 64):
        super().__init__()
        num_nodes, d_emb = node_embeddings.shape
        
        self.char_net = CharacterNet(node_embeddings, h_char, T_sup, K, z_dim=dvib_z_dim)
        self.mental_net = MentalNet(node_embeddings, h_ment, T_q, z_dim=dvib_z_dim)
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(node_embeddings), freeze=False, padding_idx=0)
        
        fusion_dim = dvib_z_dim + dvib_z_dim + d_emb
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, z_dim),
            nn.LayerNorm(z_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.goal_head = nn.Linear(z_dim, num_goals)
        self.next_head = nn.Linear(z_dim, num_nodes)

    def forward(self, sup, prefix, prefix_len):
        B, K, T_sup = sup.shape
        _, T_q = prefix.shape
        
        z_char, mu_char, logvar_char = self.char_net(sup)
        z_mental, mu_mental, logvar_mental = self.mental_net(prefix, prefix_len)
        
        last_indices = (prefix_len - 1).clamp(min=0)
        last_nodes = prefix[torch.arange(B), last_indices]
        last_emb = self.embedding(last_nodes)
        
        fusion_input = torch.cat([z_char, z_mental, last_emb], dim=1)
        z = self.fusion(fusion_input)
        
        next_logits = self.next_head(z)
        goal_logits = self.goal_head(z)
        
        return next_logits, goal_logits, mu_char, logvar_char, mu_mental, logvar_mental

# ## Step 3: Prepare the Datasets
all_nodes = set()
for episode in path_data.values():
    for path in episode.values():
        if isinstance(path, (list, tuple, set)):
            all_nodes.update(path)
        else:
            all_nodes.add(path)
all_nodes.update(campus.G_undirected.nodes())
all_nodes = list(all_nodes)

node2idx = {n: i for i, n in enumerate(all_nodes)}
print(f"Number of nodes in node2idx: {len(node2idx)}")

V = len(all_nodes)

# build goal2idx likewise for your goals list
goal2idx = {g:i for i,g in enumerate(goals)}
G = len(goals)

node_embeddings = np.load("data/node2vec_embeddings.npy")

train_agent_ids = list(range(0, 70))
test_agent_ids = list(range(70, 100))

# hyper‐params
K     = 10    # number of support trajectories per agent
T_sup = 75    # max length (pad/truncate) of each support trajectory
T_q   = 20    # prefix length for query trajectories

all_episodes    = list(path_data.keys())
examples_train  = []
examples_test   = []

for agent in agents:
    a_id = agent.id

    # choose which list to append into
    if a_id in train_agent_ids:
        target = examples_train
    elif a_id in test_agent_ids:
        target = examples_test
    else:
        # silently skip any id outside 0–99
        continue

    for ep in all_episodes:
        # ——— 1) build the K‐shot “support set” for this (agent, ep) ———
        other_eps   = [e for e in all_episodes if e != ep]
        support_eps = random.sample(other_eps, K)

        sup_tensor = torch.zeros(K, T_sup, dtype=torch.long)
        for k, se in enumerate(support_eps):
            raw_sup  = path_data[se][a_id]           # e.g. [n0, n1, n2, …]
            idxs_sup = [node2idx[n] for n in raw_sup]
            L        = min(len(idxs_sup), T_sup)
            sup_tensor[k, :L] = torch.tensor(idxs_sup[:L], dtype=torch.long)

        # ——— 2) unroll *this* episode’s path into (prefix→next) queries ———
        raw_q        = path_data[ep][a_id]
        idxs_q       = [node2idx[n] for n in raw_q]
        true_goal_idx = goal2idx[goal_data[ep][a_id]]

        for t in range(1, len(idxs_q)):
            prefix_idxs = idxs_q[:t]     # length t (we’ll pad later)
            next_idx    = idxs_q[t]      # ground‐truth “next node”

            target.append((
                sup_tensor.clone(),      # [K×T_sup] LongTensor
                prefix_idxs,             # Python list of length t
                next_idx,                # int
                true_goal_idx            # int
            ))

print(f"# train examples: {len(examples_train)}")
print(f"# test  examples: {len(examples_test)}")

def augment_trajectory(traj, drop_prob=0.1, swap_prob=0.05):
    # Randomly drop nodes
    traj = [n for n in traj if random.random() > drop_prob or len(traj) <= 2]
    # Randomly swap adjacent nodes
    if len(traj) > 2 and random.random() < swap_prob:
        i = random.randint(0, len(traj)-2)
        traj[i], traj[i+1] = traj[i+1], traj[i]
    return traj


from torch.utils.data import Dataset, DataLoader

class ToMNetDataset(Dataset):
    def __init__(self, examples, T_q, pad_value=0):
        """
        examples: list of tuples
            (sup_tensor, prefix_idxs, next_idx, true_goal_idx)
        T_q: int
            length that we will pad/truncate every prefix to
        pad_value: int
            index to use for padding prefixes
        """
        self.examples = examples
        self.T_q       = T_q
        self.pad_value = pad_value

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sup_tensor, prefix_list, next_idx, true_goal_idx = self.examples[idx]
        # Augment prefix_list (query) and support trajectories
        prefix_list = augment_trajectory(prefix_list)
        # Optionally, augment support trajectories as well
        for k in range(sup_tensor.shape[0]):
            traj = sup_tensor[k].tolist()
            traj = augment_trajectory(traj)
            sup_tensor[k] = torch.tensor(traj + [0]*(sup_tensor.shape[1]-len(traj)), dtype=torch.long)[:sup_tensor.shape[1]]
        return sup_tensor, torch.tensor(prefix_list, dtype=torch.long), next_idx, true_goal_idx

    # def __getitem__(self, idx):
    #     sup_tensor, prefix_list, next_idx, true_goal_idx = self.examples[idx]
    #     # sup_tensor: Tensor[K, T_sup]
    #     # prefix_list: Python list, length <= T_q (un‐padded)
    #     # next_idx: int
    #     # true_goal_idx: int
    #     return sup_tensor, torch.tensor(prefix_list, dtype=torch.long), next_idx, true_goal_idx

def tomnet_collate(batch, T_q, pad_value=0):
    """
    batch: list of tuples from __getitem__()
      sup_tensor:  K×T_sup
      prefix:     [t] (list of ints)
      next_idx:   scalar int
      goal_idx:   scalar int

    Returns:
      sup_batch:  (B, K, T_sup)
      prefix_batch: (B, T_q)
      next_batch:   (B,)
      goal_batch:   (B,)
      prefix_lens:  (B,)  # optional if you need to mask
    """
    sup_list, prefix_list, next_list, goal_list = zip(*batch)
    B = len(batch)

    sup_batch = torch.stack(sup_list, dim=0)    # (B, K, T_sup)

    prefix_batch = torch.full((B, T_q), pad_value, dtype=torch.long)
    prefix_lens  = torch.zeros(B, dtype=torch.long)
    for i, p in enumerate(prefix_list):
        L = min(len(p), T_q)
        if L == 0:
            # Ensure at least one token (e.g., pad with 0)
            prefix_batch[i, 0] = pad_value
            prefix_lens[i] = 1
        else:
            prefix_batch[i, :L] = p[:L]
            prefix_lens[i]      = L

    next_batch = torch.tensor(next_list, dtype=torch.long)     # (B,)
    goal_batch = torch.tensor(goal_list, dtype=torch.long)     # (B,)

    return sup_batch, prefix_batch, next_batch, goal_batch, prefix_lens

def tomnet_collate_fn(batch):
    # use your existing tomnet_collate, but wrap it
    return tomnet_collate(batch, T_q=T_q, pad_value=0)

batch_size = 256

train_ds = ToMNetDataset(examples_train, T_q=T_q, pad_value=0)
test_ds  = ToMNetDataset(examples_test,  T_q=T_q, pad_value=0)

test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=6, collate_fn=tomnet_collate_fn)

# 3) (optional) split into train/val
n_total = len(train_ds)
n_val   = int(0.1 * n_total)
n_train = n_total - n_val
# train_ds, val_ds = random_split((train_ds), [n_train, n_val])
train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True,
                                           collate_fn=tomnet_collate_fn,
                                           num_workers=16)
val_loader   = torch.utils.data.DataLoader(test_ds,   batch_size, shuffle=False,
                                           collate_fn=tomnet_collate_fn,

                                           num_workers=16)

# Building the required compute metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

def compute_metrics(model, loader, device):
    model.eval()
    all_goal_true, all_goal_pred = [], []
    all_next_true, all_next_pred = [], []
    with torch.no_grad():
        for sup, prefix, next_idx, goal_idx, pre_len in loader:
            sup = sup.to(device)
            prefix = prefix.to(device)
            pre_len = pre_len.to(device)
            next_idx = next_idx.to(device)
            goal_idx = goal_idx.to(device)
            next_logits, goal_logits, *_ = model(sup, prefix, pre_len)
            # Next-node prediction
            next_pred = next_logits.argmax(dim=1)
            all_next_true.extend(next_idx.cpu().numpy())
            all_next_pred.extend(next_pred.cpu().numpy())
            # Goal prediction
            goal_pred = goal_logits.argmax(dim=1)
            all_goal_true.extend(goal_idx.cpu().numpy())
            all_goal_pred.extend(goal_pred.cpu().numpy())
    # Accuracy
    goal_acc = accuracy_score(all_goal_true, all_goal_pred)
    next_acc = accuracy_score(all_next_true, all_next_pred)
    # Confusion matrices
    goal_cm = confusion_matrix(all_goal_true, all_goal_pred)
    next_cm = confusion_matrix(all_next_true, all_next_pred)
    return goal_acc, next_acc, goal_cm, next_cm

def plot_and_log_confusion_matrix(cm, labels, title, wandb_key):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="viridis", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    # Log to wandb
    wandb.log({wandb_key: wandb.Image(plt.gcf())})
    # plt.show()
    # plt.close()

# ## Step 4: Model Training
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
device

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# TODO: Need to play with the hyperparameters
# 1) hyper‐parameters 
lr         = 1e-3
weight_decay = 5e-4
num_epochs = 30

# 2) model, losses, optimizer
model = ToMNet(
    node_embeddings = node_embeddings,
    num_nodes       = len(node2idx),
    num_goals       = len(goal2idx),
    T_sup           = 75
).to(device)

loss_next = nn.CrossEntropyLoss()
loss_goal = nn.CrossEntropyLoss()
opt       = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = ReduceLROnPlateau(
    optimizer=opt, mode='min', factor=0.5, patience=5)

beta = 1e-3  # TODO: Test out different values of annealing

best_val_loss = float('inf')
best_state    = None
patience          = 5
epochs_no_improve = 0

# Loss Weights
goal_weight = 2.0

# Initialize wandb at the start of your run (do this only once)
wandb.init(project="tomnet", name="New_TomNet_run", config={
    "epochs": num_epochs,
    "batch_size": batch_size,
    "lr": lr,
    "weight_decay": weight_decay,
    "goal_weight": goal_weight,
})

for epoch in range(1, num_epochs+1):
    
    model.train()
    total_train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
    
    for sup, prefix, next_idx, goal_idx, pre_len in train_bar:
        sup = sup.to(device)
        prefix = prefix.to(device)
        pre_len = pre_len.to(device)
        next_idx = next_idx.to(device)
        goal_idx = goal_idx.to(device)

        opt.zero_grad()
        pred_next_logits, pred_goal_logits, mu_char, logvar_char, mu_mental, logvar_mental = model(sup, prefix, pre_len)

        # KL divergence for both character and mental
        kl_char = -0.5 * torch.sum(1 + logvar_char - mu_char.pow(2) - logvar_char.exp(), dim=1).mean()
        kl_mental = -0.5 * torch.sum(1 + logvar_mental - mu_mental.pow(2) - logvar_mental.exp(), dim=1).mean()
        loss_dvib = beta * (kl_char + kl_mental)

        L_next = loss_next(pred_next_logits, next_idx)
        L_goal = loss_goal(pred_goal_logits, goal_idx)
        loss = L_next + goal_weight*L_goal + loss_dvib

        loss.backward()
        opt.step()

        total_train_loss += loss.item() * prefix.size(0)
        train_bar.set_postfix(train_loss=loss.item())
        
    avg_train_loss = total_train_loss / n_train

    # —————— Validation ——————
    model.eval()
    total_val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]  ", leave=False)
    with torch.no_grad():
        for sup, prefix, next_idx, goal_idx, pre_len in val_bar:
            sup     = sup.to(device)       # [B,K,T_sup]
            prefix  = prefix.to(device)    # [B,T_q]
            pre_len = pre_len.to(device)   # [B]
            next_idx= next_idx.to(device)
            goal_idx= goal_idx.to(device)
    
            # forward
            next_logits, goal_logits, mu_char, logvar_char, mu_mental, logvar_mental = model(sup, prefix, pre_len)
            L_next        = loss_next(next_logits, next_idx)
            L_goal        = loss_goal(goal_logits,   goal_idx)
            batch_loss    = (L_next + L_goal).item()
    
            total_val_loss += batch_loss * prefix.size(0)
            val_bar.set_postfix(val_loss=batch_loss)
    
    avg_val_loss = total_val_loss / n_val

    goal_acc, next_acc, goal_cm, next_cm = compute_metrics(model, val_loader, device)
    per_goal_acc = goal_cm.diagonal() / goal_cm.sum(axis=1)
    print(f"Goal Acc: {goal_acc:.3f} | Next-node Acc: {next_acc:.3f}")
    # print("Goal Confusion Matrix:\n", goal_cm)
    # print("Next-node Confusion Matrix:\n", next_cm)
    # print("Per-goal accuracy:", per_goal_acc)

    # Log confusion matrices to wandb
    goal_labels = [str(g) for g in range(len(goal2idx))]
    next_labels = [str(n) for n in range(len(node2idx))]
    plot_and_log_confusion_matrix(goal_cm, goal_labels, "Goal Confusion Matrix", "goal_confusion_matrix")
    plot_and_log_confusion_matrix(next_cm, next_labels, "Next-node Confusion Matrix", "next_confusion_matrix")

    # Log scalar metrics to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "goal_accuracy": goal_acc,
        "next_node_accuracy": next_acc,
        "per_goal_accuracy": wandb.Histogram(per_goal_acc)
    })

    scheduler.step(avg_val_loss)

    # print a summary line
    # print(f"Epoch {epoch}/{num_epochs}  "
        #   f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

    # save best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state    = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# finally, load best
model.load_state_dict(best_state)
print("Training complete. Best val loss:", best_val_loss)

torch.save(model.state_dict(), "./models/All_en_tomnet_cuda.pth", _use_new_zipfile_serialization=False)

# ## Step 5: Testing and Evaluation with ToMnet
from real_world_src.utils.metrics import brier_along_path, accuracy_along_path

import torch.nn.functional as F

def make_support_tensor(agent_id, episode_id, path_data, node2idx, K, T_sup):
    # all eps for this agent
    all_eps = [ep for ep in path_data.keys() if ep != episode_id]
    # pick K random others:
    support_eps = random.sample(all_eps, K)
    sup_tensor = torch.zeros(K, T_sup, dtype=torch.long)
    for k, ep in enumerate(support_eps):
        raw = path_data[ep][agent_id]            # list of node‐ids
        idxs = [node2idx[n] for n in raw]
        L = min(len(idxs), T_sup)
        sup_tensor[k, :L] = torch.tensor(idxs[:L], dtype=torch.long)
    return sup_tensor  # (K×T_sup)

def infer_goal_dists(
    model, agent_id, test_ep,
    path_data, node2idx, goal2idx,
    K, T_sup, T_q,
    device='cuda:1'
):
    model.eval()
    # 1) build support once
    sup     = make_support_tensor(agent_id, test_ep, path_data, node2idx, K, T_sup)
    sup     = sup.to(device).unsqueeze(0)     # add batch‐dim → [1,K,T_sup]

    raw_seq = path_data[test_ep][agent_id]
    idxs    = [node2idx[n] for n in raw_seq]
    N       = len(idxs)

    goal_dists = []   # will be list of length N each [num_goals]
    with torch.no_grad():
        for t in range(1, N):
            # build prefix up to t (we treat t=0 as “no steps seen”)
            prefix_len = min(t, T_q)
            # pad prefix to T_q
            prefix = torch.zeros(T_q, dtype=torch.long)
            if prefix_len>0:
                prefix[:prefix_len] = torch.tensor(idxs[:prefix_len], dtype=torch.long)
            # move to device and batch‐dim
            prefix     = prefix.to(device).unsqueeze(0)       # [1,T_q]
            prefix_len = torch.tensor([prefix_len], dtype=torch.long, device=device)

            # forward through ToMNet
            _, goal_logits = model(sup, prefix, prefix_len)   # [1, num_goals]
            p_goal = F.softmax(goal_logits, dim=-1)[0]        # remove batch‐dim → [num_goals]

            goal_dists.append(p_goal.cpu().numpy())

    return goal_dists   # shape (N × num_goals) array

agent_id=2
test_ep=2

dists = infer_goal_dists(
    model, agent_id, test_ep,
    path_data, node2idx, goal2idx,
    K=10, T_sup=75, T_q=20,
    device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
)

dists[0]

len(path_data[test_ep][agent_id])

# goal2idx: { goal_node_id → index }
idx2goal = { idx: goal for goal, idx in goal2idx.items() }

goal_posteriors = [
    { idx2goal[i]: float(p) for i, p in enumerate(prob_row) }
    for prob_row in dists
]

scores = brier_along_path(path_data[test_ep][agent_id], 
                                  goal_data[test_ep][agent_id], 
                                  goal_posteriors, 
                                  goals)

scores

goal_posteriors