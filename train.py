#!/usr/bin/env python3
"""
Training script for HEA VAE model.
Usage: python train.py [--epochs EPOCHS] [--alpha ALPHA] [--lr LR] [--log] [--no-early-stopping]
"""

import argparse
import datetime
import os
from models.hea_vae import HEA_VAE
from models.trainer import Trainer
from mlx.optimizers import AdamW


def main():
    parser = argparse.ArgumentParser(description='Train HEA VAE model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--alpha', type=float, default=25.0,
                        help='Weight for property loss (default: 25.0)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--log', action='store_true', default=False,
                        help='Enable logging to file (default: False, logging disabled)')
    parser.add_argument('--no-early-stopping', action='store_true', default=False,
                        help='Disable early stopping (default: False, early stopping enabled)')
    
    args = parser.parse_args()
    
    # Fixed/default values for other parameters
    batch_size = 100
    beta = 0.01  # Final beta value after annealing (increased from 0.005 for better regularization)
    csv_path = 'data/MPEA_cleaned.csv'
    val_split = 0.2
    # Set early stopping patience to a very large number if disabled
    early_stopping_patience = float('inf') if args.no_early_stopping else 10
    early_stopping_min_delta = 0.0
    # Revised beta annealing schedule:
    # - Cosine schedule (smooth S-curve) instead of linear
    # - 40 epochs (reduced from 50 for faster convergence)
    # - Starts at 0.0, reaches 0.01 (increased from 0.005 for better latent space regularization)
    kl_annealing_epochs = 40
    kl_annealing_start = 0.0  # Beta starts at 0 and increases to target via cosine schedule
    
    # Create log file only if logging is enabled
    log_file = None
    if args.log:
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create log file with timestamp in logs folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"training_{timestamp}.log")
    
    print("=" * 60)
    print("HEA VAE Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Alpha (property loss weight): {args.alpha}")
    print(f"Learning Rate: {args.lr}")
    if args.no_early_stopping:
        print("Early Stopping: Disabled")
    else:
        print(f"Early Stopping: Enabled (patience: {early_stopping_patience})")
    if log_file:
        print(f"Log File: {log_file}")
    else:
        print("Logging: Disabled")
    print("=" * 60)
    print()
    
    # Write header to log file only if logging is enabled
    if log_file:
        with open(log_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("HEA VAE Training Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Alpha (property loss weight): {args.alpha}\n")
            f.write(f"Beta (KL divergence weight): {beta}\n")
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Data Path: {csv_path}\n")
            f.write(f"Validation Split: {val_split}\n")
            if val_split > 0:
                if args.no_early_stopping:
                    f.write(f"Early Stopping: Disabled\n")
                else:
                    f.write(f"Early Stopping Patience: {early_stopping_patience}\n")
                    f.write(f"Early Stopping Min Delta: {early_stopping_min_delta}\n")
            if kl_annealing_epochs > 0:
                f.write(f"KL Annealing: {kl_annealing_start:.4f} -> {beta:.4f} over {kl_annealing_epochs} epochs\n")
            f.write("=" * 60 + "\n\n")
    
    # Initialize optimizer with specified learning rate
    optimizer = AdamW(learning_rate=args.lr)
    
    # Initialize trainer - ensure model matches dataset (30 element columns)
    trainer = Trainer(model=HEA_VAE(input_dim=30, latent_dim=4, hidden_dim=512, dropout_rate=0.1), 
                      opt=optimizer, alpha=args.alpha, beta=beta)
    
    # Train the model (saves to models/hea_vae_best.npz by default)
    trainer.train(
        epochs=args.epochs,
        csv_path=csv_path,
        batch_size=batch_size,
        log_file=log_file,
        val_split=val_split,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        save_dir='models',
        kl_annealing_epochs=kl_annealing_epochs,
        kl_annealing_start=kl_annealing_start
    )
    
    # Write completion to log file if logging is enabled
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"\nTraining completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print()
    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
