"""
Evaluation script for HEA VAE model.

Generates:
- Property prediction parity plot
- Latent space visualization
- Generative capability check
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.hea_vae import HEA_VAE
from utils.dataloader import load_hea_data


def evaluate(model_path='models/hea_vae_best.npz', figures_dir='figures'):
    """
    Evaluate a trained HEA VAE model.
    
    Args:
        model_path: Path to saved model weights (.npz file)
        figures_dir: Directory to save evaluation plots
    """
    print("=" * 60)
    print("HEA VAE Model Evaluation")
    print("=" * 60)
    
    # Ensure figures directory exists
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load Data
    print("\n[1/4] Loading data...")
    train_gen, val_gen, y_mean, y_std, input_dim, n_train, n_val = load_hea_data(
        batch_size=128, shuffle=False, val_split=0.0
    )
    data_gen = train_gen  # Use all data for evaluation
    
    print(f"  Input dimension: {input_dim}")
    print(f"  Total samples: {n_train}")
    print(f"  Target normalization: mean={y_mean:.2f}, std={y_std:.2f}")
    
    # 2. Load Model
    print(f"\n[2/4] Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = HEA_VAE(latent_dim=4, hidden_dim=512, dropout_rate=0.0)
    model.load_weights(model_path)
    print("  Model loaded successfully")
    
    # 3. Run Inference
    print("\n[3/4] Running inference...")
    y_true_all = []
    y_pred_all = []
    z_all = []
    x_recon_all = []
    
    for x_batch, y_batch in data_gen():
        # Forward pass
        recon_x, pred_y, mu, logvar = model(x_batch)
        
        # Store results
        y_true_all.append(np.array(y_batch))
        y_pred_all.append(np.array(pred_y))
        z_all.append(np.array(mu))  # Use mean of latent distribution
        x_recon_all.append(np.array(recon_x))
    
    # Concatenate all batches
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    z_latent = np.concatenate(z_all, axis=0)
    x_recon = np.concatenate(x_recon_all, axis=0)
    
    # Denormalize predictions
    y_true_mpa = (y_true * y_std) + y_mean
    y_pred_mpa = (y_pred * y_std) + y_mean
    
    # 4. Calculate Metrics
    print("\n[4/4] Calculating metrics...")
    r2 = r2_score(y_true_mpa, y_pred_mpa)
    mse = mean_squared_error(y_true_mpa, y_pred_mpa)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_mpa - y_pred_mpa))
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"R² Score:        {r2:.4f}")
    print(f"MSE (MPa²):      {mse:.2f}")
    print(f"RMSE (MPa):      {rmse:.2f}")
    print(f"MAE (MPa):       {mae:.2f}")
    print("=" * 60)
    
    # 5. Generate Plots
    
    # 5a. Property Prediction Parity Plot
    print("\nGenerating plots...")
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_mpa, y_pred_mpa, alpha=0.5, s=20, c='blue', edgecolors='black', linewidth=0.5)
    
    min_val = min(y_true_mpa.min(), y_pred_mpa.min())
    max_val = max(y_true_mpa.max(), y_pred_mpa.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title(f'Property Prediction Parity\n$R^2 = {r2:.3f}$, RMSE = {rmse:.1f} MPa', fontsize=14)
    plt.xlabel('Actual Yield Strength (MPa)', fontsize=12)
    plt.ylabel('Predicted Yield Strength (MPa)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/eval_property_parity.png', dpi=150)
    print(f"  Saved: {figures_dir}/eval_property_parity.png")
    plt.close()
    
    # 5b. Latent Space Visualization (2D projection using first 2 dimensions)
    if z_latent.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            z_latent[:, 0], z_latent[:, 1], 
            c=y_true_mpa.flatten(), 
            cmap='viridis', 
            alpha=0.6, 
            s=30,
            edgecolors='black',
            linewidth=0.3
        )
        plt.colorbar(scatter, label='Yield Strength (MPa)')
        plt.title('Latent Space Visualization\n(First 2 Dimensions)', fontsize=14)
        plt.xlabel('Latent Dimension 1', fontsize=12)
        plt.ylabel('Latent Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/eval_latent_space.png', dpi=150)
        print(f"  Saved: {figures_dir}/eval_latent_space.png")
        plt.close()
    
    # 5c. Generative Capability Check
    print("\nChecking generative capability...")
    # Sample random points from latent space
    n_samples = 5
    z_sample = mx.random.normal(shape=(n_samples, 4))  # Sample from standard normal
    
    # Decode to get generated compositions
    generated_comps = []
    for i in range(n_samples):
        z_i = z_sample[i:i+1]  # Shape: (1, 4)
        recon_i = model.decode(z_i)  # Decode to composition
        pred_i = model.predict(z_i)  # Predict property
        generated_comps.append({
            'composition': np.array(recon_i),
            'predicted_ys': float((np.array(pred_i) * y_std) + y_mean)
        })
    
    print(f"  Generated {n_samples} sample compositions:")
    for i, comp in enumerate(generated_comps):
        comp_sum = comp['composition'].sum()
        print(f"    Sample {i+1}: Sum={comp_sum:.4f}, Predicted YS={comp['predicted_ys']:.1f} MPa")
    
    # Check reconstruction quality
    x_original = np.concatenate([batch[0] for batch in data_gen()], axis=0)[:100]  # First 100 samples
    x_recon_sample = x_recon[:100]
    recon_error = np.mean((x_original - x_recon_sample) ** 2, axis=1)
    print(f"\n  Reconstruction MSE (first 100 samples): {np.mean(recon_error):.6f}")
    print(f"  Composition sums (original): {x_original.sum(axis=1)[:5]}")
    print(f"  Composition sums (reconstructed): {x_recon_sample.sum(axis=1)[:5]}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    return {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'y_true': y_true_mpa,
        'y_pred': y_pred_mpa,
        'z_latent': z_latent
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate HEA VAE model')
    parser.add_argument(
        '--model',
        type=str,
        default='models/hea_vae_best.npz',
        help='Path to model weights file (default: models/hea_vae_best.npz)'
    )
    parser.add_argument(
        '--figures-dir',
        type=str,
        default='figures',
        help='Directory to save plots (default: figures)'
    )
    
    args = parser.parse_args()
    
    evaluate(model_path=args.model, figures_dir=args.figures_dir)
