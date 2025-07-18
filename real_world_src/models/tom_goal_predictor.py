import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Goal Predictor Head
# -----------------------------
class GoalPredictorHead(nn.Module):
    """
    Predicts the agent's ultimate goal from latent mind-states.
    Supports both coordinate regression and node classification.
    Optionally, can be extended to Mixture Density Network (MDN).
    """
    def __init__(self, latent_dim=32, output_type='coord', num_nodes=None, mdn_components=0):
        """
        Args:
            latent_dim: Dim of each latent (belief, desire, intention)
            output_type: 'coord' for regression, 'node' for classification
            num_nodes: Number of graph nodes (for classification)
            mdn_components: If >0, use MDN with this many components
        """
        super().__init__()
        self.output_type = output_type
        self.mdn_components = mdn_components
        input_dim = latent_dim * 3
        hidden_dim = 128
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if mdn_components > 0:
            # Mixture Density Network for 2D coordinates
            self.mdn_pi = nn.Linear(hidden_dim, mdn_components)
            self.mdn_mu = nn.Linear(hidden_dim, mdn_components * 2)  # 2D mean per component
            self.mdn_sigma = nn.Linear(hidden_dim, mdn_components * 2)  # 2D std per component
        elif output_type == 'coord':
            self.head = nn.Linear(hidden_dim, 2)  # Predict (y, x) coordinate
        elif output_type == 'node':
            assert num_nodes is not None, 'num_nodes required for node classification'
            self.head = nn.Linear(hidden_dim, num_nodes)
        else:
            raise ValueError('Invalid output_type')

    def forward(self, z_belief, z_desire, z_intention):
        # Concatenate all latents
        z = torch.cat([z_belief, z_desire, z_intention], dim=-1)
        h = self.base(z)
        if self.mdn_components > 0:
            pi = F.log_softmax(self.mdn_pi(h), dim=-1)  # (batch, K)
            mu = self.mdn_mu(h).view(-1, self.mdn_components, 2)  # (batch, K, 2)
            sigma = torch.exp(self.mdn_sigma(h)).view(-1, self.mdn_components, 2)  # (batch, K, 2)
            return {'pi': pi, 'mu': mu, 'sigma': sigma}
        elif self.output_type == 'coord':
            coord = self.head(h)
            return coord  # (batch, 2)
        elif self.output_type == 'node':
            logits = self.head(h)
            return logits  # (batch, num_nodes)

    # -----------------------------
    # Loss Functions
    # -----------------------------
    def regression_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def classification_loss(self, logits, target):
        return F.cross_entropy(logits, target)

    def mdn_loss(self, mdn_out, target):
        # Negative log-likelihood for MDN (target: (batch, 2))
        pi, mu, sigma = mdn_out['pi'], mdn_out['mu'], mdn_out['sigma']
        target = target.unsqueeze(1)  # (batch, 1, 2)
        norm = torch.distributions.Normal(mu, sigma)
        log_prob = norm.log_prob(target).sum(-1)  # (batch, K)
        weighted = pi + log_prob  # log(pi) + log_prob
        nll = -torch.logsumexp(weighted, dim=-1).mean()
        return nll

    def distribution_loss(self, logits, target_dist):
        """
        KL-divergence loss between predicted logits (after softmax) and ground-truth distribution.
        Args:
            logits: (batch, num_nodes) - raw model outputs
            target_dist: (batch, num_nodes) - ground-truth distribution (should sum to 1)
        Returns:
            Mean KL-divergence loss
        """
        pred_log_prob = F.log_softmax(logits, dim=-1)
        # Add small epsilon for numerical stability
        target_dist = target_dist + 1e-8
        kl = F.kl_div(pred_log_prob, target_dist, reduction='batchmean')
        return kl

    def get_probabilities(self, logits):
        """
        Utility to get softmaxed probabilities from logits (for node classification output).
        """
        return F.softmax(logits, dim=-1)

# -----------------------------
# Main Function for Testing
# -----------------------------
def main():
    """Test the GoalPredictorHead with dummy data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    latent_dim = 32
    z_belief = torch.randn(batch_size, latent_dim).to(device)
    z_desire = torch.randn(batch_size, latent_dim).to(device)
    z_intention = torch.randn(batch_size, latent_dim).to(device)
    # Coordinate regression
    coord_head = GoalPredictorHead(latent_dim, output_type='coord').to(device)
    coord_pred = coord_head(z_belief, z_desire, z_intention)
    print('Coord prediction:', coord_pred.shape)
    # Node classification
    node_head = GoalPredictorHead(latent_dim, output_type='node', num_nodes=100).to(device)
    node_logits = node_head(z_belief, z_desire, z_intention)
    print('Node logits:', node_logits.shape)
    # MDN
    mdn_head = GoalPredictorHead(latent_dim, output_type='coord', mdn_components=5).to(device)
    mdn_out = mdn_head(z_belief, z_desire, z_intention)
    print('MDN pi:', mdn_out['pi'].shape, 'mu:', mdn_out['mu'].shape, 'sigma:', mdn_out['sigma'].shape)
    # Loss test
    target_coord = torch.randn(batch_size, 2).to(device)
    print('Regression loss:', coord_head.regression_loss(coord_pred, target_coord).item())
    target_node = torch.randint(0, 100, (batch_size,)).to(device)
    print('Classification loss:', node_head.classification_loss(node_logits, target_node).item())
    print('MDN loss:', mdn_head.mdn_loss(mdn_out, target_coord).item())
    print('✅ GoalPredictorHead test completed!')

if __name__ == "__main__":
    main()