import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
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

# FUNCTIONS

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

def augment_trajectory(traj, drop_prob=0.1, swap_prob=0.05):
    # Randomly drop nodes
    traj = [n for n in traj if random.random() > drop_prob or len(traj) <= 2]
    # Randomly swap adjacent nodes
    if len(traj) > 2 and random.random() < swap_prob:
        i = random.randint(0, len(traj)-2)
        traj[i], traj[i+1] = traj[i+1], traj[i]
    return traj

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