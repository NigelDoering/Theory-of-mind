import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.utils.data import random_split


import random
import json
import pickle
from tqdm import tqdm

import sys
import os

# Adjust this path as needed to point to your project root
sys.path.append(os.path.abspath("../.."))



from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.utils.run_manager import RunManager

# import multiprocessing as mp
# mp.set_start_method('fork', force=True)

class CharacterNet(nn.Module):
    """
    Encode K full trajectories per agent into a single 'character' vector c ∈ R^h.
    Input shape: (B, K, T_sup) of node indices (long).
    """
    def __init__(self,
                 num_nodes:int,
                 d_emb:int   = 16,
                 h_lstm:int  = 64,
                 T_sup:int   = 50,
                 K:int       = 10):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, d_emb, padding_idx=0)
        self.lstm = nn.LSTM(d_emb, h_lstm, batch_first=True)
        self.K = K
        self.T_sup = T_sup

    def forward(self, support_trajs):
        # support_trajs: LongTensor[B, K, T_sup]
        B, K, T = support_trajs.size()
        assert K==self.K and T==self.T_sup

        # (B*K, T)
        flat = support_trajs.view(B*K, T)
        # (B*K, T, d_emb)
        emb = self.embedding(flat)
        # run LSTM
        _, (h_n, _) = self.lstm(emb)  # h_n: (1, B*K, h_lstm)
        h_n = h_n.squeeze(0)          # (B*K, h_lstm)
        # reshape to (B, K, h_lstm) and mean-pool over K
        chars = h_n.view(B, K, -1).mean(dim=1)  # (B, h_lstm)
        return chars                            # → c
    
class MentalNet(nn.Module):
    """
    Encode the query prefix into a 'mental' vector m ∈ R^h'.
    Inputs:
    - prefix     : LongTensor of shape [B, T_q] (node indices, padded with 0)
    - prefix_len : LongTensor of shape [B]   (true lengths in 1..T_q)
    Outputs:
    - m          : FloatTensor of shape [B, h_lstm]
    """
    def __init__(self,
                num_nodes:int,
                d_emb:int  = 16,
                h_lstm:int = 64,
                T_q:int    = 20):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, d_emb, padding_idx=0)
        self.lstm      = nn.LSTM(d_emb, h_lstm, batch_first=True)
        self.T_q       = T_q

    def forward(self, prefix: torch.LongTensor, prefix_len: torch.LongTensor):
        B, T = prefix.size()
        assert T == self.T_q, f"Expected T_q={self.T_q}, got {T}"

        # embed all time-steps
        emb = self.embedding(prefix)  # [B, T_q, d_emb]

        # pack by actual lengths
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths=prefix_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # run through LSTM
        _, (h_n, _) = self.lstm(packed)
        # h_n: [1, B, h_lstm]
        m = h_n.squeeze(0)            # [B, h_lstm]

        return m
    
class ToMNet(nn.Module):
    """
    Full ToMNet: CharacterNet + MentalNet + fusion MLP + prediction heads.
    """
    def __init__(self,
                 num_nodes:int,
                 num_goals:int,
                 K:int=10,
                 T_sup:int=50,
                 T_q:int=20,
                 d_emb:int=16,
                 h_char:int=64,
                 h_ment:int=64,
                 z_dim:int=64):
        super().__init__()
        # submodules
        self.char_net   = CharacterNet(num_nodes, d_emb, h_char, T_sup, K)
        self.mental_net = MentalNet(num_nodes, d_emb, h_ment, T_q)
        # embedding to get last‐step token embedding
        self.embedding  = nn.Embedding(num_nodes, d_emb, padding_idx=0)

        # a small MLP to fuse [h_char + h_ment + d_emb] → z_dim
        fusion_dim = h_char + h_ment + d_emb
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
            nn.ReLU()
        )
        # final prediction heads
        self.goal_head = nn.Linear(z_dim, num_goals)
        self.next_head = nn.Linear(z_dim, num_nodes)

    def forward(self,
                sup: torch.LongTensor,       # [B, K, T_sup]
                prefix: torch.LongTensor,    # [B, T_q]
                prefix_len: torch.LongTensor # [B]
               ):
        B, K, T_sup = sup.shape
        _, T_q     = prefix.shape

        # 1) character‐level features from the K support trajectories
        #    --> sup_feat: [B, h_char]
        sup_feat = self.char_net(sup)

        # 2) mental‐net encoding of the current prefix
        #    --> ment_feat: [B, h_ment]
        ment_feat = self.mental_net(prefix, prefix_len)

        # 3) take the *last non‐padded* token in each prefix, embed it
        #    prefix_len is in [1..T_q], so subtract 1 for zero‐based index
        last_indices = (prefix_len - 1).clamp(min=0)          # [B]
        # gather the node index at that last step
        last_nodes   = prefix[torch.arange(B), last_indices] # [B]
        # embed it
        last_emb     = self.embedding(last_nodes)            # [B, d_emb]

        # 4) fuse all three representations
        #    concat → [B, h_char + h_ment + d_emb]
        fusion_input = torch.cat([sup_feat, ment_feat, last_emb], dim=1)
        z            = self.fusion(fusion_input)             # [B, z_dim]

        # 5) heads
        next_logits = self.next_head(z)  # [B, num_nodes]
        goal_logits = self.goal_head(z)  # [B, num_goals]

        return next_logits, goal_logits
    
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
        # sup_tensor: Tensor[K, T_sup]
        # prefix_list: Python list, length <= T_q (un‐padded)
        # next_idx: int
        # true_goal_idx: int
        return sup_tensor, torch.tensor(prefix_list, dtype=torch.long), next_idx, true_goal_idx
    
def main():
    # Initialize campus environment
    campus = CampusEnvironment()

    # Need to establish the set of common goals (just choose the landmark nodes)
    goals = [469084068, 49150691, 768264666, 1926666015, 1926673385, 49309735,
            273627682, 445989107, 445992528, 446128310, 1772230346, 1926673336, 
            2872424923, 3139419286, 4037576308]

    with open('../../notebooks/agents.pkl', 'rb') as f:
        agents = pickle.load(f)

    with open("../../notebooks/data/path_data.json", 'r') as file:
        path_data = json.load(file)

    with open("../../notebooks/data/goal_data.json", 'r') as file:
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

    # build node2idx so that every node in campus.G_undirected maps to 0…V−1
    all_nodes = list(campus.G_undirected.nodes())
    node2idx  = {n:i for i,n in enumerate(all_nodes)}
    V = len(all_nodes)

    # build goal2idx likewise for your goals list
    goal2idx = {g:i for i,g in enumerate(goals)}
    G = len(goals)

    train_agent_ids = list(range(0, 70))
    test_agent_ids = list(range(70, 100))

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
            true_goal_idx = goal2idx[ goal_data[ep][a_id] ]

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

    batch_size = 128

    train_ds = ToMNetDataset(examples_train, T_q=T_q, pad_value=0)
    
    # Save for testing later 
    with open("model_runs/test_examples.pkl", "wb") as f:
        pickle.dump(examples_test, f) 

    # 3) (optional) split into train/val
    n_total = len(train_ds)
    n_val   = int(0.1 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split((train_ds), [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True,
                                            collate_fn=tomnet_collate_fn,
                                            num_workers=6)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size, shuffle=False,
                                            collate_fn=tomnet_collate_fn,
                                            num_workers=6)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 1) hyper‐parameters
    lr         = 1e-3
    weight_decay = 1e-5
    num_epochs = 30



    # 2) model, losses, optimizer
    model = ToMNet(
        num_nodes   = len(node2idx),
        num_goals   = len(goal2idx),
        T_sup=75
        # … etc …
    ).to(device)

    loss_next = nn.CrossEntropyLoss()
    loss_goal = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, num_epochs+1):
        # —————— Training ——————
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
        for sup, prefix, next_idx, goal_idx, pre_len in train_bar:
            sup      = sup.to(device)       # [B, K, T_sup]
            prefix   = prefix.to(device)    # [B, T_q]
            pre_len  = pre_len.to(device)   # [B]
            next_idx = next_idx.to(device)  # [B]
            goal_idx = goal_idx.to(device)  # [B]

            opt.zero_grad()
            # forward
            pred_next_logits, pred_goal_logits = model(sup, prefix, pre_len)

            # compute losses
            L_next = loss_next(pred_next_logits, next_idx)
            L_goal = loss_goal(pred_goal_logits,   goal_idx)
            loss   = L_next + L_goal

            # backward + step
            loss.backward()
            opt.step()

            total_train_loss += loss.item() * prefix.size(0)

            # update tqdm bar with current batch loss
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
                p_next, p_goal = model(sup, prefix, pre_len)
                L_next        = loss_next(p_next, next_idx)
                L_goal        = loss_goal(p_goal,   goal_idx)
                batch_loss    = (L_next + L_goal).item()
        
                total_val_loss += batch_loss * prefix.size(0)
                val_bar.set_postfix(val_loss=batch_loss)
        
        avg_val_loss = total_val_loss / n_val

        # print a summary line
        print(f"Epoch {epoch}/{num_epochs}  "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = model.state_dict()

    # finally, load best
    model.load_state_dict(best_state)
    print("Training complete. Best val loss:", best_val_loss)

    torch.save(model.state_dict(), "model_runs/tomnet_cpu.pth", _use_new_zipfile_serialization=False)

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

    # stack support tensors
    sup_batch = torch.stack(sup_list, dim=0)    # (B, K, T_sup)

    # pad prefixes to length T_q
    prefix_batch = torch.full((B, T_q), pad_value, dtype=torch.long)
    prefix_lens  = torch.zeros(B, dtype=torch.long)
    for i, p in enumerate(prefix_list):
        L = min(len(p), T_q)
        prefix_batch[i, :L] = p[:L]
        prefix_lens[i]      = L

    next_batch = torch.tensor(next_list, dtype=torch.long)     # (B,)
    goal_batch = torch.tensor(goal_list, dtype=torch.long)     # (B,)

    return sup_batch, prefix_batch, next_batch, goal_batch, prefix_lens

def tomnet_collate_fn(batch):
    # use your existing tomnet_collate, but wrap it
    return tomnet_collate(batch, T_q=T_q, pad_value=0)

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
    device='cuda'
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

if __name__ == "__main__":
    main()


    