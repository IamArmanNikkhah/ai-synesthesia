# utils/utils.py
import math
import torch
import torch.autograd.functional

def collect_grad_norms(parameters, eps=1e-8):
    """Collect the overall L2 norm of gradients from given parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm) + eps

def compute_jacobian_norm(model, input_seq, num_iter=4):
    """
    Approximate the spectral norm (largest singular value) of the Jacobian of `model`
    with respect to `input_seq` using power iteration.
    For simplicity, we work on the first sample of the batch.
    """
    x = input_seq[0:1].detach().requires_grad_(True)  # shape: (1, seq_length, input_size)
    output = model(x)  # shape: (1, output_size)
    # Initialize a random vector with the same shape as output
    v = torch.randn_like(output)
    v = v / (v.norm() + 1e-8)
    for _ in range(num_iter):
        # Compute the Jacobian-vector product (jvp)
        jvp = torch.autograd.functional.jvp(model, (x,), (v,), create_graph=True)[1]
        # Compute the vector-Jacobian product to obtain J^T(J*v)
        v_new = torch.autograd.functional.vjp(model, x, jvp, create_graph=True)[1]
        v_new = v_new.view(-1)
        norm_v_new = v_new.norm() + 1e-8
        v = (v_new / norm_v_new).view_as(output)
    jvp_final = torch.autograd.functional.jvp(model, (x,), (v,), create_graph=False)[1]
    spectral_norm = jvp_final.norm().item()
    return spectral_norm

def compute_effective_alpha(user_alpha, safe_alpha, pred_grad_norm, jacobian_norm, epoch, total_epochs):
    """
    Compute the effective α using cosine decay scheduling and the tension protocol.
    
    The effective α is given by a cosine-decayed version of the user-specified α,
    but it is capped by safe_alpha and only allowed to increase if the prediction gradient
    norm is sufficiently small.
    """
    # Cosine decay scheduling
    decayed_alpha = user_alpha * 0.5 * (1 + math.cos(math.pi * (epoch / total_epochs)))
    if epoch == 0:  # Kickoff criteria on first batch/epoch
        decayed_alpha = min(user_alpha, safe_alpha)
    # Tension protocol: only allow increase if the prediction gradient norm is sufficiently low.
    if pred_grad_norm < 0.8 * safe_alpha * jacobian_norm:
        effective_alpha = decayed_alpha
    else:
        effective_alpha = safe_alpha
    # Enforce the Chaplygin bound.
    effective_alpha = min(effective_alpha, safe_alpha)
    # Convergence validation.
    assert effective_alpha <= safe_alpha + 1e-6, "Effective α exceeds safe bound!"
    return effective_alpha

def hybrid_elbo_loss(
    recon_mel: torch.Tensor,
    recon_energy: torch.Tensor,
    mel: torch.Tensor,
    energy: torch.Tensor,
    predicted_latent: torch.Tensor,
    actual_latent: torch.Tensor,
    z_mean: torch.Tensor,
    z_log_var: torch.Tensor,
    prediction_weight: float
):
    """
    Compute the hybrid ELBO loss as the sum of reconstruction loss, prediction loss,
    and the Kullback-Leibler divergence (KLD) from the CVAE.
    """
    import torch.nn.functional as F
    recon_loss_mel = F.mse_loss(recon_mel, mel)
    recon_loss_energy = F.mse_loss(recon_energy, energy)
    recon_loss = recon_loss_mel + recon_loss_energy
    pred_loss = F.mse_loss(predicted_latent, actual_latent)
    # Compute KLD (assuming diagonal covariance)
    kld = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / z_mean.size(0)
    return recon_loss, pred_loss, kld
