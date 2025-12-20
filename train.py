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
                        help='Learning rate (default: 1e-)')
    parser.add_argument('--csv-path', type=str, default='data/MPEA_cleaned.csv',
                        help='Path to CSV data file (default: data/MPEA_cleaned.csv)')
    
    args = parser.parse_args()
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_{timestamp}.log"
    
    print("=" * 60)
    print("HEA VAE Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Alpha (property loss weight): {args.alpha}")
    print(f"Beta (KL divergence weight): {args.beta}")
    print(f"Learning Rate: {args.lr}")
    print(f"Data Path: {args.csv_path}")
    print(f"Log File: {log_file}")
    print("=" * 60)
    print()
    
    # Write header to log file
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
        log_file=log_file
    )
    
    # Write completion to log file
    with open(log_file, 'a') as f:
        f.write(f"\nTraining completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print()
    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

