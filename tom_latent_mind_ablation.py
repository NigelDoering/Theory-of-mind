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
    
    Supports ablation modes:
    - 'none': No VAE components, returns zero tensors
    - 'belief': Only belief VAE active
    - 'belief_desire': Belief and desire VAEs active
    - 'full': All three VAEs active (default)
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128, ablation_mode='full'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.ablation_mode = ablation_mode

        # VAE for Belief
        if ablation_mode in ['belief', 'belief_desire', 'full']:
            self.belief_vae = VAEBlock(input_dim, latent_dim, hidden_dim)
        else:
            self.belief_vae = None
            
        # VAE for Desire (conditioned on Belief)
        if ablation_mode in ['belief_desire', 'full']:
            self.desire_vae = VAEBlock(input_dim + latent_dim, latent_dim, hidden_dim)
        else:
            self.desire_vae = None
            
        # VAE for Intention (conditioned on Desire)
        if ablation_mode == 'full':
            self.intention_vae = VAEBlock(input_dim + latent_dim, latent_dim, hidden_dim)
        else:
            self.intention_vae = None

    def forward(self, x):
        # x: (batch_size, input_dim) - fused encoding from ToMGraphEncoder
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize outputs with zeros
        z_belief = torch.zeros(batch_size, self.latent_dim, device=device)
        mu_belief = torch.zeros(batch_size, self.latent_dim, device=device)
        logvar_belief = torch.zeros(batch_size, self.latent_dim, device=device)
        recon_belief = torch.zeros_like(x)
        
        z_desire = torch.zeros(batch_size, self.latent_dim, device=device)
        mu_desire = torch.zeros(batch_size, self.latent_dim, device=device)
        logvar_desire = torch.zeros(batch_size, self.latent_dim, device=device)
        recon_desire = torch.zeros(batch_size, x.shape[1] + self.latent_dim, device=device)
        
        z_intention = torch.zeros(batch_size, self.latent_dim, device=device)
        mu_intention = torch.zeros(batch_size, self.latent_dim, device=device)
        logvar_intention = torch.zeros(batch_size, self.latent_dim, device=device)
        recon_intention = torch.zeros(batch_size, x.shape[1] + self.latent_dim, device=device)
        
        # 1. Belief
        if self.belief_vae is not None:
            z_belief, mu_belief, logvar_belief, recon_belief = self.belief_vae(x)
            
        # 2. Desire (conditioned on belief)
        if self.desire_vae is not None:
            desire_input = torch.cat([x, z_belief], dim=-1)
            z_desire, mu_desire, logvar_desire, recon_desire = self.desire_vae(desire_input)
            
        # 3. Intention (conditioned on desire)
        if self.intention_vae is not None:
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
        # Sum KL for active levels only
        total_kl = 0.0
        if self.belief_vae is not None:
            total_kl += self.kl_loss(outs['mu_belief'], outs['logvar_belief'])
        if self.desire_vae is not None:
            total_kl += self.kl_loss(outs['mu_desire'], outs['logvar_desire'])
        if self.intention_vae is not None:
            total_kl += self.kl_loss(outs['mu_intention'], outs['logvar_intention'])
        return total_kl

    def recon_loss(self, x, outs):
        # Sum recon losses for active levels only
        total_recon = 0.0
        if self.belief_vae is not None:
            total_recon += F.mse_loss(outs['recon_belief'], x)
        if self.desire_vae is not None:
            total_recon += F.mse_loss(outs['recon_desire'], torch.cat([x, outs['z_belief']], dim=-1))
        if self.intention_vae is not None:
            total_recon += F.mse_loss(outs['recon_intention'], torch.cat([x, outs['z_desire']], dim=-1))
        return total_recon

# -----------------------------
# Main Function for Testing
# -----------------------------
def main():
    """Test the HierarchicalMindStateVAE with dummy data for all ablation modes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 8
    input_dim = 128  # Should match ToMGraphEncoder output
    latent_dim = 32
    x = torch.randn(batch_size, input_dim).to(device)
    
    # Test all ablation modes
    modes = ['none', 'belief', 'belief_desire', 'full']
    
    for mode in modes:
        print(f"\n--- Testing ablation mode: {mode} ---")
        model = HierarchicalMindStateVAE(input_dim, latent_dim, ablation_mode=mode).to(device)
        outs = model(x)
        
        print("Latent shapes:")
        print("z_belief:", outs['z_belief'].shape)
        print("z_desire:", outs['z_desire'].shape)
        print("z_intention:", outs['z_intention'].shape)
        
        kl = model.total_kl_loss(outs)
        recon = model.recon_loss(x, outs)
        print(f"KL loss: {kl:.4f}, Recon loss: {recon:.4f}")
        
        # Check if latents are zero for inactive components
        if mode == 'none':
            assert torch.allclose(outs['z_belief'], torch.zeros_like(outs['z_belief'])), "Belief should be zero in 'none' mode"
            assert torch.allclose(outs['z_desire'], torch.zeros_like(outs['z_desire'])), "Desire should be zero in 'none' mode"
            assert torch.allclose(outs['z_intention'], torch.zeros_like(outs['z_intention'])), "Intention should be zero in 'none' mode"
        elif mode == 'belief':
            assert not torch.allclose(outs['z_belief'], torch.zeros_like(outs['z_belief'])), "Belief should be non-zero in 'belief' mode"
            assert torch.allclose(outs['z_desire'], torch.zeros_like(outs['z_desire'])), "Desire should be zero in 'belief' mode"
            assert torch.allclose(outs['z_intention'], torch.zeros_like(outs['z_intention'])), "Intention should be zero in 'belief' mode"
        elif mode == 'belief_desire':
            assert not torch.allclose(outs['z_belief'], torch.zeros_like(outs['z_belief'])), "Belief should be non-zero in 'belief_desire' mode"
            assert not torch.allclose(outs['z_desire'], torch.zeros_like(outs['z_desire'])), "Desire should be non-zero in 'belief_desire' mode"
            assert torch.allclose(outs['z_intention'], torch.zeros_like(outs['z_intention'])), "Intention should be zero in 'belief_desire' mode"
        elif mode == 'full':
            assert not torch.allclose(outs['z_belief'], torch.zeros_like(outs['z_belief'])), "Belief should be non-zero in 'full' mode"
            assert not torch.allclose(outs['z_desire'], torch.zeros_like(outs['z_desire'])), "Desire should be non-zero in 'full' mode"
            assert not torch.allclose(outs['z_intention'], torch.zeros_like(outs['z_intention'])), "Intention should be non-zero in 'full' mode"
    
    print("\n✅ All HierarchicalMindStateVAE ablation tests completed!")

if __name__ == "__main__":
    main()