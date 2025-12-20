import mlx.core as mx
import mlx.nn as nn

def property_guided_loss(recon_x, x, pred_y, y, mu, logvar, alpha=10.0, beta=1.0):
    """
    Computes the Joint VAE loss: Reconstruction + Property Prediction + KL Divergence.
    
    Args:
        recon_x: Reconstructed alloy composition (Batch, 30)
        x:       Original alloy composition (Batch, 30)
        pred_y:  Predicted Property (e.g., Yield Strength) (Batch, 1)
        y:       True Property (Batch, 1)
        mu:      Latent mean
        logvar:  Latent log variance
        alpha:   Weight for property loss. Controls how much the model prioritizes 
                 optimizing the property (Strength) vs. learning chemistry.
        beta:    Weight for KLD (Regularization).
    """
    
    # 1. Chemistry Loss (Reconstruction)
    # Measures how well the output alloy matches the input alloy.
    loss_chem = nn.losses.mse_loss(recon_x, x, reduction='mean')
    
    # 2. Physics Loss (Property Regression)
    # Measures how accurate the strength prediction is.
    loss_prop = nn.losses.mse_loss(pred_y, y, reduction='mean')
    
    # 3. Regularization (KL Divergence)
    # Forces the latent space to be continuous and smooth.
    loss_kld = -0.5 * mx.sum(1 + logvar - mx.power(mu, 2) - mx.exp(logvar))
    
    # TOTAL LOSS
    # We multiply property loss by alpha to balance the scales.
    return loss_chem + (alpha * loss_prop) + (beta * loss_kld)