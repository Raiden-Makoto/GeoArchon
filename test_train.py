#!/usr/bin/env python3
"""Test script to run training for one epoch."""

from models.trainer import Trainer

if __name__ == "__main__":
    print("Initializing trainer...")
    trainer = Trainer(alpha=10.0, beta=1.0)
    
    print("\nStarting training for 1 epoch...")
    trainer.train(epochs=1, csv_path='data/MPEA_cleaned.csv', batch_size=64)
    
    print("\nTest completed successfully!")

