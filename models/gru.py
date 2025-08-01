import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SimpleGRU(nn.Module):
    """
    A simple GRU-based model for goal inference from agent paths.
    """
    def __init__(self, num_nodes, num_goals, embedding_dim=16, hidden_dim=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_goals)

    def forward(self, path_idxs):
        """
        path_idxs: (batch_size, seq_len)
        """
        emb = self.embedding(path_idxs)  # (batch_size, seq_len, embedding_dim)
        _, h_n = self.gru(emb)           # h_n: (num_layers, batch_size, hidden_dim)
        h_last = h_n[-1]                 # (batch_size, hidden_dim)
        logits = self.fc(h_last)         # (batch_size, num_goals)
        return logits

    def predict_goal_distribution(self, path_idxs):
        logits = self.forward(path_idxs)
        probs = torch.softmax(logits, dim=-1)
        return probs

def train_gru_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for paths, goals in dataloader:
            paths = paths.to(device)
            goals = goals.to(device)
            optimizer.zero_grad()
            logits = model(paths)
            loss = criterion(logits, goals)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def evaluate_gru_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for paths, goals in dataloader:
            paths = paths.to(device)
            goals = goals.to(device)
            logits = model(paths)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == goals).sum().item()
            total += goals.size(0)
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation accuracy: {accuracy:.4f}")
    return accuracy

class GRUDataset(Dataset):
    """
    Dataset for GRU model: yields (path_idxs, goal_idx)
    """
    def __init__(self, path_data, goal_data, node2idx, goal2idx):
        self.samples = []
        for episode in path_data:
            for agent_id in path_data[episode]:
                path = path_data[episode][agent_id]
                goal = goal_data[episode][agent_id]
                path_idx = [node2idx[n] for n in path]
                goal_idx = goal2idx[goal]
                self.samples.append((path_idx, goal_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_idx, goal_idx = self.samples[idx]
        path_idx = torch.tensor(path_idx, dtype=torch.long)
        goal_idx = torch.tensor(goal_idx, dtype=torch.long)
        return path_idx, goal_idx

def gru_collate_fn(batch):
    """
    Pads sequences in the batch to the same length.
    """
    paths, goals = zip(*batch)
    paths_padded = torch.nn.utils.rnn.pad_sequence(paths, batch_first=True, padding_value=0)
    goals = torch.stack(goals)
    return paths_padded, goals

def predict_goal_posterior_gru(model, path, node2idx, goal2idx, device):
    """
    Given a path (list of node IDs), returns a dict mapping goal IDs to probabilities.
    """
    model.eval()
    path_idx = torch.tensor([[node2idx[n] for n in path]], dtype=torch.long).to(device)  # shape (1, seq_len)
    with torch.no_grad():
        probs = model.predict_goal_distribution(path_idx).squeeze(0).cpu().numpy()  # (num_goals,)
    idx2goal = {v: k for k, v in goal2idx.items()}
    return {idx2goal[i]: float(probs[i]) for i in range(len(probs))}

def predict_goal_posteriors_along_path(model, path, node2idx, goal2idx, device):
    """
    For a given path, returns a list of posterior distributions over goals for each prefix.
    Each element is a dict mapping goal IDs to probabilities.
    """
    model.eval()
    idx2goal = {v: k for k, v in goal2idx.items()}
    posteriors = []
    with torch.no_grad():
        for t in range(1, len(path)+1):
            prefix = path[:t]
            path_idx = torch.tensor([[node2idx[n] for n in prefix]], dtype=torch.long).to(device)
            probs = model.predict_goal_distribution(path_idx).squeeze(0).cpu().numpy()
            posteriors.append({idx2goal[i]: float(probs[i]) for i in range(len(probs))})
    return posteriors