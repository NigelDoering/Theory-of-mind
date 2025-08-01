import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# VAE Block (Encoder, Reparam, Decoder)
# -----------------------------
class VAEBlock(nn.Module):
    """
    A basic VAE block: encoder -> latent (mu, logvar) -> reparam -> decoder
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return z, mu, logvar, recon

# -----------------------------
# Hierarchical Mind-State VAE
# -----------------------------
class HierarchicalMindStateVAE(nn.Module):
    """
    Hierarchical VAE for Belief, Desire, Intention latent mind-states.
    Each layer is conditioned on the previous (e.g., desire on belief).
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # VAE for Belief
        self.belief_vae = VAEBlock(input_dim, latent_dim, hidden_dim)
        # VAE for Desire (conditioned on Belief)
        self.desire_vae = VAEBlock(input_dim + latent_dim, latent_dim, hidden_dim)
        # VAE for Intention (conditioned on Desire)
        self.intention_vae = VAEBlock(input_dim + latent_dim, latent_dim, hidden_dim)

    def forward(self, x):
        # x: (batch_size, input_dim) - fused encoding from ToMGraphEncoder
        # 1. Belief
        z_belief, mu_belief, logvar_belief, recon_belief = self.belief_vae(x)
        # 2. Desire (conditioned on belief)
        desire_input = torch.cat([x, z_belief], dim=-1)
        z_desire, mu_desire, logvar_desire, recon_desire = self.desire_vae(desire_input)
        # 3. Intention (conditioned on desire)
        intention_input = torch.cat([x, z_desire], dim=-1)
        z_intention, mu_intention, logvar_intention, recon_intention = self.intention_vae(intention_input)
        # Return all latents and reconstructions
        return {
            'z_belief': z_belief, 'mu_belief': mu_belief, 'logvar_belief': logvar_belief, 'recon_belief': recon_belief,
            'z_desire': z_desire, 'mu_desire': mu_desire, 'logvar_desire': logvar_desire, 'recon_desire': recon_desire,
            'z_intention': z_intention, 'mu_intention': mu_intention, 'logvar_intention': logvar_intention, 'recon_intention': recon_intention
        }

    def kl_loss(self, mu, logvar):
        # Standard VAE KL divergence
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    def total_kl_loss(self, outs):
        # Sum KL for all levels
        return self.kl_loss(outs['mu_belief'], outs['logvar_belief']) + \
               self.kl_loss(outs['mu_desire'], outs['logvar_desire']) + \
               self.kl_loss(outs['mu_intention'], outs['logvar_intention'])

    def recon_loss(self, x, outs):
        # Sum recon losses for all levels
        return F.mse_loss(outs['recon_belief'], x) + \
               F.mse_loss(outs['recon_desire'], torch.cat([x, outs['z_belief']], dim=-1)) + \
               F.mse_loss(outs['recon_intention'], torch.cat([x, outs['z_desire']], dim=-1))

# -----------------------------
# Main Function for Testing
# -----------------------------
def main():
    """Test the HierarchicalMindStateVAE with dummy data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    batch_size = 8
    input_dim = 128  # Should match ToMGraphEncoder output
    latent_dim = 32
    x = torch.randn(batch_size, input_dim).to(device)
    model = HierarchicalMindStateVAE(input_dim, latent_dim).to(device)
    outs = model(x)
    print("Latent shapes:")
    print("z_belief:", outs['z_belief'].shape)
    print("z_desire:", outs['z_desire'].shape)
    print("z_intention:", outs['z_intention'].shape)
    kl = model.total_kl_loss(outs)
    recon = model.recon_loss(x, outs)
    print(f"KL loss: {kl.item():.4f}, Recon loss: {recon.item():.4f}")
    print("✅ HierarchicalMindStateVAE test completed!")

if __name__ == "__main__":
    main()