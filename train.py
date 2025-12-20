#!/usr/bin/env python3
"""
Training script for HEA VAE model.
Usage: python train.py [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--alpha ALPHA] [--beta BETA] [--lr LR] [--csv-path CSV_PATH]
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
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--alpha', type=float, default=4.0,
                        help='Weight for property loss (default: 4.0)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for KL divergence loss (default: 1.0)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--csv-path', type=str, default='data/MPEA_cleaned.csv',
                        help='Path to CSV data file (default: data/MPEA_cleaned.csv)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of data for validation (default: 0.2, set to 0 to disable)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping (default: 10)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                        help='Minimum change to qualify as improvement (default: 0.0)')
    parser.add_argument('--kl-annealing-epochs', type=int, default=50,
                        help='Number of epochs to anneal KL weight from start to target (0 = no annealing, default: 50)')
    parser.add_argument('--kl-annealing-start', type=float, default=0.0,
                        help='Starting value for beta (KL weight) during annealing (default: 0.0)')
    parser.add_argument('--log', action='store_true', default=False,
                        help='Enable logging to file (default: False, logging disabled)')
    
    args = parser.parse_args()
    
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
    print(f"Batch Size: {args.batch_size}")
    print(f"Alpha (property loss weight): {args.alpha}")
    print(f"Beta (KL divergence weight): {args.beta}")
    print(f"Learning Rate: {args.lr}")
    print(f"Data Path: {args.csv_path}")
    print(f"Validation Split: {args.val_split}")
    if args.val_split > 0:
        print(f"Early Stopping Patience: {args.early_stopping_patience}")
        print(f"Early Stopping Min Delta: {args.early_stopping_min_delta}")
    if args.kl_annealing_epochs > 0:
        print(f"KL Annealing: {args.kl_annealing_start:.4f} -> {args.beta:.4f} over {args.kl_annealing_epochs} epochs")
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
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Alpha (property loss weight): {args.alpha}\n")
            f.write(f"Beta (KL divergence weight): {args.beta}\n")
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Data Path: {args.csv_path}\n")
            f.write(f"Validation Split: {args.val_split}\n")
            if args.val_split > 0:
                f.write(f"Early Stopping Patience: {args.early_stopping_patience}\n")
                f.write(f"Early Stopping Min Delta: {args.early_stopping_min_delta}\n")
            if args.kl_annealing_epochs > 0:
                f.write(f"KL Annealing: {args.kl_annealing_start:.4f} -> {args.beta:.4f} over {args.kl_annealing_epochs} epochs\n")
            f.write("=" * 60 + "\n\n")
    
    # Initialize optimizer with specified learning rate
    optimizer = AdamW(learning_rate=args.lr)
    
    # Initialize trainer
    trainer = Trainer(model=HEA_VAE(latent_dim=4), opt=optimizer, alpha=args.alpha, beta=args.beta)
    
    # Train the model
    trainer.train(
        epochs=args.epochs,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        log_file=log_file,
        val_split=args.val_split,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        save_dir='models',
        kl_annealing_epochs=args.kl_annealing_epochs,
        kl_annealing_start=args.kl_annealing_start
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

