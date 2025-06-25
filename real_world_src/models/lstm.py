import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    """
    A simple LSTM-based model for goal inference from agent paths.
    """
    def __init__(self, num_nodes, num_goals, embedding_dim=16, hidden_dim=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_goals)

    def forward(self, path_idxs):
        """
        path_idxs: (batch_size, seq_len) tensor of node indices
        Returns: (batch_size, num_goals) logits
        """
        x = self.embedding(path_idxs)  # (batch, seq, embedding_dim)
        _, (h_n, _) = self.lstm(x)     # h_n: (num_layers, batch, hidden_dim)
        h_last = h_n[-1]               # (batch, hidden_dim)
        logits = self.fc(h_last)       # (batch, num_goals)
        return logits

    def predict_goal_distribution(self, path_idxs):
        """
        Returns softmax probabilities over goals.
        """
        logits = self.forward(path_idxs)
        return F.softmax(logits, dim=-1)

def train_lstm_model(model, dataloader, optimizer, device, num_epochs=10):
    """
    Train the LSTM model.
    dataloader: yields (path_idxs, goal_idx) pairs
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for path_idxs, goal_idx in dataloader:
            path_idxs = path_idxs.to(device)
            goal_idx = goal_idx.to(device)
            optimizer.zero_grad()
            logits = model(path_idxs)
            loss = criterion(logits, goal_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}")

def evaluate_lstm_model(model, dataloader, device):
    """
    Evaluate the LSTM model.
    Returns average accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for path_idxs, goal_idx in dataloader:
            path_idxs = path_idxs.to(device)
            goal_idx = goal_idx.to(device)
            logits = model(path_idxs)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == goal_idx).sum().item()
            total += goal_idx.size(0)
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation accuracy: {accuracy:.4f}")
    return accuracy

class LSTMDataset(torch.utils.data.Dataset):
    """
    Dataset for LSTM model: yields (path_idxs, goal_idx)
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

def lstm_collate_fn(batch):
    """
    Pads sequences in the batch to the same length.
    """
    paths, goals = zip(*batch)
    paths_padded = torch.nn.utils.rnn.pad_sequence(paths, batch_first=True, padding_value=0)
    goals = torch.stack(goals)
    return paths_padded, goals

def predict_goal_posterior(model, path, node2idx, goal2idx, device):
    """
    Given a path (list of node IDs), returns a dict mapping goal IDs to probabilities.
    """
    model.eval()
    path_idx = torch.tensor([[node2idx[n] for n in path]], dtype=torch.long).to(device)  # shape (1, seq_len)
    with torch.no_grad():
        probs = model.predict_goal_distribution(path_idx).squeeze(0).cpu().numpy()  # (num_goals,)
    idx2goal = {v: k for k, v in goal2idx.items()}
    return {idx2goal[i]: float(probs[i]) for i in range(len(probs))}