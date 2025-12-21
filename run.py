#!/usr/bin/env python3
"""
Complete pipeline script for HEA VAE: Preprocess → Train → Evaluate

This script runs the full pipeline:
1. Preprocess dataset (add physics descriptors)
2. Train the VAE model
3. Evaluate the trained model

Usage:
    python run_pipeline.py [--epochs EPOCHS] [--alpha ALPHA] [--lr LR] [--log] [--early-stopping] [--skip-preprocess] [--skip-train] [--skip-eval]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_preprocess():
    """Run data preprocessing to add physics descriptors."""
    print("=" * 60)
    print("STEP 1: Preprocessing Dataset")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "data/preprocess.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print("✓ Preprocessing completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Preprocessing failed:")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("✗ Error: data/preprocess.py not found")
        return False


def run_train(epochs, alpha, lr, log, early_stopping):
    """Run model training."""
    print("\n" + "=" * 60)
    print("STEP 2: Training Model")
    print("=" * 60)
    
    # Check if cleaned dataset exists
    if not os.path.exists("data/MPEA_cleaned.csv"):
        print("✗ Error: data/MPEA_cleaned.csv not found")
        print("  Run preprocessing first or use --skip-preprocess to skip it")
        return False
    
    cmd = [
        sys.executable, "utils/train.py",
        "--epochs", str(epochs),
        "--alpha", str(alpha),
        "--lr", str(lr)
    ]
    
    if log:
        cmd.append("--log")
    
    if early_stopping:
        cmd.append("--early-stopping")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print("\n✓ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        return False


def run_evaluate(model_path="models/hea_vae_best.npz"):
    """Run model evaluation."""
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating Model")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"✗ Error: Model file not found: {model_path}")
        print("  Train the model first or use --skip-train to skip training")
        return False
    
    cmd = [
        sys.executable, "utils/evaluate.py",
        "--model", model_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print("\n✓ Evaluation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Evaluation failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n✗ Evaluation interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete HEA VAE pipeline: Preprocess → Train → Evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python run_pipeline.py

  # Run with custom parameters
  python run_pipeline.py --epochs 200 --alpha 30 --lr 1e-4

  # Skip preprocessing (use existing cleaned dataset)
  python run_pipeline.py --skip-preprocess

  # Only evaluate existing model
  python run_pipeline.py --skip-preprocess --skip-train
        """
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--alpha', type=float, default=25.0,
                        help='Weight for property loss (default: 25.0)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--log', action='store_true', default=False,
                        help='Enable logging to file (default: False)')
    parser.add_argument('--early-stopping', action='store_true', default=False,
                        help='Enable early stopping (default: False, disabled)')
    
    # Skip options
    parser.add_argument('--skip-preprocess', action='store_true', default=False,
                        help='Skip preprocessing step (default: False)')
    parser.add_argument('--skip-train', action='store_true', default=False,
                        help='Skip training step (default: False)')
    parser.add_argument('--skip-eval', action='store_true', default=False,
                        help='Skip evaluation step (default: False)')
    
    # Model path for evaluation
    parser.add_argument('--model', type=str, default='models/hea_vae_best.npz',
                        help='Path to model for evaluation (default: models/hea_vae_best.npz)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("HEA VAE Complete Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Logging: {'Enabled' if args.log else 'Disabled'}")
    print(f"  Early Stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
    print(f"  Skip Preprocess: {args.skip_preprocess}")
    print(f"  Skip Train: {args.skip_train}")
    print(f"  Skip Eval: {args.skip_eval}")
    print("=" * 60)
    print()
    
    success = True
    
    # Step 1: Preprocessing
    if not args.skip_preprocess:
        if not run_preprocess():
            success = False
            print("\n✗ Pipeline failed at preprocessing step")
            sys.exit(1)
    else:
        print("Skipping preprocessing step...")
    
    # Step 2: Training
    if not args.skip_train:
        if not run_train(args.epochs, args.alpha, args.lr, args.log, args.early_stopping):
            success = False
            print("\n✗ Pipeline failed at training step")
            sys.exit(1)
    else:
        print("Skipping training step...")
    
    # Step 3: Evaluation
    if not args.skip_eval:
        if not run_evaluate(args.model):
            success = False
            print("\n✗ Pipeline failed at evaluation step")
            sys.exit(1)
    else:
        print("Skipping evaluation step...")
    
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nOutput files:")
        if not args.skip_preprocess:
            print("  - data/MPEA_cleaned.csv (preprocessed dataset)")
        if not args.skip_train:
            print(f"  - {args.model} (trained model)")
        if not args.skip_eval:
            print("  - figures/eval_property_parity.png (evaluation plot)")
            print("  - figures/eval_latent_space.png (latent space visualization)")
    else:
        print("✗ PIPELINE FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

