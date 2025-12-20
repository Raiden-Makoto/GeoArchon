import mlx.nn as nn
import mlx.core as mx
import mlx.utils as utils
import numpy as np
from tqdm import tqdm

from .hea_vae import HEA_VAE
from utils import load_hea_data
from utils.loss_funcs import property_guided_loss
from mlx.optimizers import AdamW

class Trainer():
    def __init__(self, model=None, opt=None, alpha: float=4.0, beta: float=1.0):
        self.model = HEA_VAE() if model is None else model
        self.opt = AdamW(learning_rate=1e-4) if opt is None else opt
        self.alpha = alpha
        self.beta = beta
        
        # Create loss and gradient function using MLX's value_and_grad
        def loss_fn(model, x, y):
            recon_x, pred_y, mu, logvar = model(x)
            loss = property_guided_loss(
                recon_x,
                x,
                pred_y,
                y,
                mu,
                logvar, 
                alpha=self.alpha,
                beta=self.beta
            )
            return loss
        
        self.loss_fn = loss_fn
        self.loss_and_grad_fn = mx.value_and_grad(loss_fn)

    def train(self, epochs: int=1, csv_path='data/MPEA_cleaned.csv', batch_size=64, log_file=None):
        """
        Train the HEA VAE model using MLX.
        
        Args:
            epochs: Number of training epochs
            csv_path: Path to the CSV data file
            batch_size: Batch size for training
            log_file: Optional path to log file for writing training summaries
        """
        # Load data - returns generator function, not dataloader
        data_gen, y_mean, y_std, input_dim, n_samples = load_hea_data(
            csv_path=csv_path, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Calculate number of batches per epoch
        n_batches_per_epoch = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(1, 1+epochs):
            total_loss = 0.0
            chem_loss_acc = 0.0
            prop_loss_acc = 0.0
            kld_loss_acc = 0.0
            
            # Create progress bar for this epoch
            pbar = tqdm(data_gen(), total=n_batches_per_epoch, 
                       desc=f"Epoch {epoch}/{epochs}", 
                       unit="batch")
            
            # Iterate through batches
            for x_batch, y_batch in pbar:
                # Forward pass and compute gradients
                loss, grads = self.loss_and_grad_fn(self.model, x_batch, y_batch)
                
                # Update model parameters (MLX optimizer updates model in place)
                self.opt.update(self.model, grads)
                
                # Evaluate the loss to get actual values (MLX is lazy)
                # Convert MLX array to numpy for logging
                loss_val = float(loss)
                
                # Compute individual loss components for logging
                recon_x, pred_y, mu, logvar = self.model(x_batch)
                loss_chem = float(nn.losses.mse_loss(recon_x, x_batch, reduction='mean'))
                loss_prop = float(nn.losses.mse_loss(pred_y, y_batch, reduction='mean'))
                # KL divergence term (normalized by batch size)
                batch_size = mu.shape[0]
                loss_kld = float(-0.5 * mx.sum(1 + logvar - mx.power(mu, 2) - mx.exp(logvar)) / batch_size)
                
                # Diagnostic: check if posterior is collapsing
                mu_mean = float(mx.mean(mx.abs(mu)))
                logvar_mean = float(mx.mean(logvar))
                
                # Track metrics
                total_loss += float(loss_val)
                chem_loss_acc += float(loss_chem)
                prop_loss_acc += float(loss_prop)
                kld_loss_acc += float(loss_kld)
                
                # Update progress bar with current metrics
                pbar.set_postfix({
                    'Loss': f'{loss_val:.4f}',
                    'Chem': f'{loss_chem:.4f}',
                    'Prop': f'{loss_prop:.4f}',
                    'KLD': f'{loss_kld:.4f}',
                    #'|mu|': f'{mu_mean:.3f}',
                    #'logvar': f'{logvar_mean:.2f}'
                })
            
            # Close progress bar and print epoch summary
            pbar.close()
            avg_loss = total_loss / n_batches_per_epoch if n_batches_per_epoch > 0 else 0.0
            avg_chem = chem_loss_acc / n_batches_per_epoch if n_batches_per_epoch > 0 else 0.0
            avg_prop = prop_loss_acc / n_batches_per_epoch if n_batches_per_epoch > 0 else 0.0
            avg_kld = kld_loss_acc / n_batches_per_epoch if n_batches_per_epoch > 0 else 0.0
            
            # Calculate weighted components for analysis
            weighted_prop = avg_prop * self.alpha
            weighted_kld = avg_kld * self.beta
            
            # Log epoch summary
            summary = f"Epoch {epoch}/{epochs} Summary: Total Loss={avg_loss:.4f} | Chem_Err={avg_chem:.4f} | Prop_Err={avg_prop:.4f} | KLD={avg_kld:.4f} | Components: Chem({avg_chem:.4f}) + Prop×α({weighted_prop:.4f}) + KLD×β({weighted_kld:.4f})\n"
            
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(summary)

        print("Training Complete.")
