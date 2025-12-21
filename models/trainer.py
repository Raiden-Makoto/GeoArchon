import mlx.nn as nn
import mlx.core as mx
import mlx.utils as utils
import numpy as np
import os
from tqdm import tqdm

from .hea_vae import HEA_VAE
from utils import load_hea_data
from utils.loss_funcs import property_guided_loss
from mlx.optimizers import AdamW

class Trainer():
    def __init__(self, model=None, opt=None, alpha: float=25.0, beta: float=1.0):
        self.model = HEA_VAE() if model is None else model
        self.opt = AdamW(learning_rate=1e-4) if opt is None else opt
        self.alpha = alpha
        self.beta = beta

        # Create loss and gradient function using MLX's value_and_grad
        # Note: beta will be dynamically set during training for KL annealing
        def loss_fn(model, x, y, beta_val):
            recon_x, pred_y, mu, logvar = model(x)
            loss = property_guided_loss(
                recon_x,
                x,
                pred_y,
                y,
                mu,
                logvar, 
                alpha=self.alpha,
                beta=beta_val
            )
            return loss
        
        self.loss_fn = loss_fn
        # Create a function that returns value_and_grad with current beta
        def get_loss_and_grad_fn(beta_val):
            def loss_fn_with_beta(model, x, y):
                return loss_fn(model, x, y, beta_val)
            return mx.value_and_grad(loss_fn_with_beta)
        
        self.get_loss_and_grad_fn = get_loss_and_grad_fn

    def train(self, epochs: int=1, csv_path='data/MPEA_cleaned.csv', batch_size=64, 
              log_file=None, val_split=0.2, early_stopping_patience=10, early_stopping_min_delta=0.0,
              save_dir='models', kl_annealing_epochs=0, kl_annealing_start=0.0, model_name=None):
        """
        Train the HEA VAE model using MLX.
        
        Args:
            epochs: Number of training epochs
            csv_path: Path to the CSV data file
            batch_size: Batch size for training
            log_file: Optional path to log file for writing training summaries
            val_split: Fraction of data to use for validation (0.0 to disable)
            early_stopping_patience: Number of epochs to wait before stopping if no improvement
            early_stopping_min_delta: Minimum change to qualify as an improvement
            save_dir: Directory to save model checkpoints (default: 'models')
            kl_annealing_epochs: Number of epochs to anneal KL weight from start to target (0 = no annealing)
            kl_annealing_start: Starting value for beta (KL weight) during annealing (default: 0.0)
        """
        # Load data - returns generator functions for train and validation
        train_gen, val_gen, y_mean, y_std, input_dim, n_train, n_val = load_hea_data(
            csv_path=csv_path, 
            batch_size=batch_size, 
            shuffle=True,
            val_split=val_split
        )
        
        # Calculate number of batches per epoch
        n_batches_per_epoch = (n_train + batch_size - 1) // batch_size
        n_val_batches = (n_val + batch_size - 1) // batch_size if n_val > 0 else 0
        
        # Early stopping variables
        # Monitor property error (not total loss) to avoid false triggers during KL annealing
        best_val_prop_err = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        # KL Annealing setup
        use_kl_annealing = kl_annealing_epochs > 0
        
        # Learning rate scheduling: decrease LR during last 20% of annealing epochs
        initial_lr = self.opt.learning_rate
        final_lr = initial_lr * 0.01  # 1e-3 -> 1e-5 (reduced for stability)
        lr_decay_start_epoch = int(kl_annealing_epochs * 0.8) if use_kl_annealing else epochs + 1
        lr_decay_epochs = kl_annealing_epochs - lr_decay_start_epoch if use_kl_annealing else 0
        
        if use_kl_annealing:
            print(f"KL Annealing enabled: beta will increase from {kl_annealing_start:.4f} to {self.beta:.4f} over {kl_annealing_epochs} epochs")
            if lr_decay_epochs > 0:
                print(f"Learning rate decay: {initial_lr:.2e} -> {final_lr:.2e} over epochs {lr_decay_start_epoch+1}-{kl_annealing_epochs} (last {lr_decay_epochs} epochs of annealing)")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"KL Annealing: beta from {kl_annealing_start:.4f} to {self.beta:.4f} over {kl_annealing_epochs} epochs\n")
                    if lr_decay_epochs > 0:
                        f.write(f"Learning rate decay: {initial_lr:.2e} -> {final_lr:.2e} over epochs {lr_decay_start_epoch+1}-{kl_annealing_epochs}\n")
        else:
            # No annealing - early stopping is active from the start
            print(f"\n{'='*60}")
            print(f"Early Stopping Active (No KL Annealing)")
            print(f"  Beta (KL weight): {self.beta:.4f}")
            print(f"  Early stopping patience: {early_stopping_patience} epochs")
            print(f"{'='*60}\n")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"\nEarly Stopping Active (No KL Annealing) - patience={early_stopping_patience}\n")
        
        # Create models directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Use custom model name if provided, otherwise use default
            if model_name:
                model_path = os.path.join(save_dir, f"{model_name}.npz")
            else:
                model_path = os.path.join(save_dir, "hea_vae_best.npz")
        else:
            model_path = None

        for epoch in range(1, 1+epochs):
            total_loss = 0.0
            chem_loss_acc = 0.0
            prop_loss_acc = 0.0
            kld_loss_acc = 0.0
            
            # Compute annealed beta for this epoch
            if use_kl_annealing:
                # Linear annealing: beta = start + (target - start) * min(1.0, epoch / annealing_epochs)
                annealing_progress = min(1.0, epoch / kl_annealing_epochs)
                current_beta = kl_annealing_start + (self.beta - kl_annealing_start) * annealing_progress
            else:
                current_beta = self.beta
            
            # Check if annealing just completed (transition from annealing to non-annealing)
            is_annealing = use_kl_annealing and epoch <= kl_annealing_epochs
            was_annealing = use_kl_annealing and (epoch - 1) <= kl_annealing_epochs and (epoch - 1) > 0
            annealing_just_completed = was_annealing and not is_annealing
            
            # Learning rate decay during last 20% of annealing epochs
            if use_kl_annealing and lr_decay_epochs > 0 and epoch > lr_decay_start_epoch and epoch <= kl_annealing_epochs:
                # Linear decay over the last 20% of annealing epochs
                lr_decay_progress = (epoch - lr_decay_start_epoch) / lr_decay_epochs
                current_lr = initial_lr + (final_lr - initial_lr) * lr_decay_progress
                self.opt.learning_rate = current_lr
                if epoch == lr_decay_start_epoch + 1:
                    # Log when LR decay starts
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(f"Epoch {epoch}: Learning rate decay started: {initial_lr:.2e} -> {final_lr:.2e}\n")
            elif not is_annealing and use_kl_annealing and lr_decay_epochs > 0:
                # After annealing, keep LR at final value (only set once)
                if self.opt.learning_rate != final_lr:
                    self.opt.learning_rate = final_lr
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(f"Epoch {epoch}: Learning rate set to final value: {final_lr:.2e}\n")
            
            # Reset patience counter and best validation property error when annealing completes
            # This is important because the loss scale changes significantly when beta goes from 0 to target
            if annealing_just_completed:
                patience_counter = 0
                best_val_prop_err = float('inf')  # Reset to allow model to establish new baseline
                best_epoch = 0
                print(f"\n{'='*60}")
                print(f"KL Annealing Complete: Early Stopping Now Active")
                print(f"  Beta has reached target value: {self.beta:.4f}")
                print(f"  Early stopping patience: {early_stopping_patience} epochs")
                print(f"{'='*60}\n")
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"\nKL Annealing Complete: Early Stopping Now Active (patience={early_stopping_patience})\n")
            
            # Get loss and grad function with current beta
            loss_and_grad_fn = self.get_loss_and_grad_fn(current_beta)
            
            # Create progress bar for this epoch
            pbar = tqdm(train_gen(), total=n_batches_per_epoch, 
                       desc=f"Epoch {epoch}/{epochs}", 
                       unit="batch")
            
            # Iterate through training batches
            for x_batch, y_batch in pbar:
                # Forward pass and compute gradients with current beta
                loss, grads = loss_and_grad_fn(self.model, x_batch, y_batch)
                
                # Gradient clipping to stabilize training and reduce loss fluctuations
                grads = utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
                
                # Update model parameters (MLX optimizer updates model in place)
                self.opt.update(self.model, grads)
                
                # Evaluate the loss to get actual values (MLX is lazy)
                # Convert MLX array to numpy for logging
                loss_val = float(loss)
                
                # Compute individual loss components for logging
                recon_x, pred_y, mu, logvar = self.model(x_batch)
                loss_chem = float(nn.losses.mse_loss(recon_x, x_batch, reduction='sum'))
                loss_prop = float(nn.losses.mse_loss(pred_y, y_batch, reduction='sum'))
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
                
                # Update progress bar with essential metrics only
                pbar.set_postfix({'Loss': f'{loss_val:.4f}'})
            
            # Close progress bar
            pbar.close()
            
            # Calculate training metrics
            avg_loss = total_loss / n_batches_per_epoch if n_batches_per_epoch > 0 else 0.0
            avg_chem = chem_loss_acc / n_batches_per_epoch if n_batches_per_epoch > 0 else 0.0
            avg_prop = prop_loss_acc / n_batches_per_epoch if n_batches_per_epoch > 0 else 0.0
            avg_kld = kld_loss_acc / n_batches_per_epoch if n_batches_per_epoch > 0 else 0.0
            
            # Evaluate on validation set if available
            val_loss = None
            val_prop_err = None
            if val_gen is not None and n_val_batches > 0:
                val_loss_total = 0.0
                val_prop_err_total = 0.0
                for x_val, y_val in val_gen():
                    recon_x, pred_y, mu, logvar = self.model(x_val)
                    val_loss_batch = self.loss_fn(self.model, x_val, y_val, current_beta)
                    val_loss_total += float(val_loss_batch)
                    # Compute property error separately for early stopping
                    val_prop_err_batch = float(nn.losses.mse_loss(pred_y, y_val, reduction='sum'))
                    val_prop_err_total += val_prop_err_batch
                val_loss = val_loss_total / n_val_batches
                val_prop_err = val_prop_err_total / n_val_batches
                
                # Early stopping check (ONLY active after annealing completes)
                # Monitor PROPERTY ERROR, not total loss, to avoid false triggers during KL annealing
                is_annealing = use_kl_annealing and epoch <= kl_annealing_epochs
                
                # Only track best model and update patience AFTER annealing completes
                if not is_annealing:
                    if val_prop_err < best_val_prop_err - early_stopping_min_delta:
                        best_val_prop_err = val_prop_err
                        best_epoch = epoch
                        patience_counter = 0
                        
                        # Save best model
                        if model_path:
                            self.model.save_weights(model_path)
                            print(f"  -> Saved best model to {model_path}")
                    else:
                        # Increment patience counter only after annealing
                        patience_counter += 1
                else:
                    # During annealing: save model if it's the best so far (for recovery), but don't track for early stopping
                    # This allows us to keep the best model even during annealing
                    if val_prop_err < best_val_prop_err:
                        best_val_prop_err = val_prop_err
                        best_epoch = epoch
                        if model_path:
                            self.model.save_weights(model_path)
                            print(f"  -> Saved best model to {model_path}")
                    # Do NOT increment patience counter during annealing
                
                # Log epoch summary with validation
                weighted_prop = avg_prop * self.alpha
                weighted_kld = avg_kld * current_beta
                summary = f"Epoch {epoch}/{epochs} Summary: Train Loss={avg_loss:.4f} | Val Loss={val_loss:.4f} | Chem={avg_chem:.4f} | Prop={avg_prop:.4f} | KLD={avg_kld:.4f} | Beta={current_beta:.4f}"
                if not is_annealing and patience_counter > 0 and early_stopping_patience != float('inf'):
                    summary += f" | Patience: {patience_counter}/{early_stopping_patience}"
                if is_annealing:
                    summary += " | (Annealing - early stopping disabled)"
                elif early_stopping_patience == float('inf'):
                    summary += " | (Early stopping disabled)"
                summary += "\n"
                
                # Print epoch summary to console (matches log file format)
                print(summary.strip())  # Use the same summary string that's written to log file
                
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(summary)
                
                # Early stopping (ONLY after annealing completes - CRITICAL: must check is_annealing)
                # Monitor PROPERTY ERROR, not total loss, to avoid false triggers during KL annealing
                # Early stopping can ONLY trigger if:
                #   1. Annealing is complete (not is_annealing)
                #   2. We've waited at least 'patience' epochs after annealing started monitoring
                if not is_annealing and patience_counter >= early_stopping_patience:
                    print(f"\n{'='*60}")
                    print(f"Early Stopping Triggered!")
                    print(f"  No improvement in property error for {early_stopping_patience} epochs")
                    print(f"  Best validation property error: {best_val_prop_err:.4f} at epoch {best_epoch}")
                    print(f"  (Total validation loss: {val_loss:.4f})")
                    if model_path:
                        print(f"  Best model saved at: {model_path}")
                    print(f"{'='*60}\n")
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(f"\nEarly stopping at epoch {epoch}. Best val prop error: {best_val_prop_err:.4f} at epoch {best_epoch}\n")
                            if model_path:
                                f.write(f"Best model saved at: {model_path}\n")
                    break
            else:
                # No validation set - just log training metrics
                weighted_prop = avg_prop * self.alpha
                weighted_kld = avg_kld * self.beta
                summary = f"Epoch {epoch}/{epochs} Summary: Total Loss={avg_loss:.4f} | Chem_Err={avg_chem:.4f} | Prop_Err={avg_prop:.4f} | KLD={avg_kld:.4f} | Components: Chem({avg_chem:.4f}) + Prop×α({weighted_prop:.4f}) + KLD×β({weighted_kld:.4f})\n"
                
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(summary)

        # Final summary
        print("Training Complete.")
        if val_gen is not None and best_epoch > 0:
            print(f"Best validation property error: {best_val_prop_err:.4f} at epoch {best_epoch}")
            if model_path:
                print(f"Best model saved at: {model_path}")