import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your local modules
from models.hea_vae import HEA_VAE
from utils.dataloader import load_hea_data

def evaluate_stability(model_path='models/hea_vae_best.npz', data_path='data/HEA_stability_train.csv', figures_dir='figures'):
    print(f"Evaluating model from: {model_path}")
    
    # Ensure figures directory exists
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load Data to get Input Dims and Normalization Stats
    print("Loading data...")
    # shuffle=False to keep the order for plotting
    # load_hea_data returns: (data_gen, val_gen, y_mean, y_std, input_dim, n_train, n_val)
    # When val_split=0.0, val_gen is None and n_val is 0
    data_gen, _, y_mean, y_std, input_dim, n_samples, _ = load_hea_data(csv_path=data_path, batch_size=256, shuffle=False, val_split=0.0)

    # 2. Initialize Model
    # Must match training settings (latent=4, hidden=512)
    model = HEA_VAE(input_dim=input_dim, latent_dim=4, hidden_dim=512, dropout_rate=0.0)
    try:
        model.load_weights(model_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 3. Run Inference
    print(f"Running inference on {n_samples} alloys...")
    y_true_list = []
    y_pred_list = []
    mu_list = []

    for x_batch, y_batch in data_gen():
        # Forward pass returns: recon_x, pred_y, mu, logvar
        _, pred_y, mu, _ = model(x_batch)

        y_true_list.append(np.array(y_batch))
        y_pred_list.append(np.array(pred_y))
        mu_list.append(np.array(mu))

    # Concatenate all batches
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    mu = np.concatenate(mu_list, axis=0)

    # 4. Denormalize Targets
    # y = y_norm * std + mean
    y_true_ev = (y_true * y_std) + y_mean
    y_pred_ev = (y_pred * y_std) + y_mean

    # 5. Calculate Metrics
    r2 = r2_score(y_true_ev, y_pred_ev)
    rmse = np.sqrt(mean_squared_error(y_true_ev, y_pred_ev))

    # Classification Metric: Did we correctly guess "Stable" (< 0.05 eV)?
    actual_stable = y_true_ev <= 0.05
    pred_stable = y_pred_ev <= 0.05
    accuracy = np.mean(actual_stable == pred_stable)

    print("\n" + "="*40)
    print("STABILITY PREDICTION RESULTS")
    print("="*40)
    print(f"R² Score:     {r2:.4f}")
    print(f"RMSE:         {rmse:.4f} eV/atom")
    print(f"Stability Acc: {accuracy*100:.2f}% (Threshold: 0.05 eV)")
    print("="*40)

    # 6. Plot 1: Parity Plot (Truth vs Prediction)
    plt.figure(figsize=(7, 6))
    # Downsample for plotting if dataset is huge (>10k points)
    if len(y_true_ev) > 10000:
        idx = np.random.choice(len(y_true_ev), 10000, replace=False)
        plt.scatter(y_true_ev[idx], y_pred_ev[idx], alpha=0.3, s=5, c='teal')
    else:
        plt.scatter(y_true_ev, y_pred_ev, alpha=0.3, s=5, c='teal')
    
    # Reference Line
    min_val = min(y_true_ev.min(), y_pred_ev.min())
    max_val = max(y_true_ev.max(), y_pred_ev.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    
    # Highlight the "Stable Region"
    plt.axvspan(-10, 0.05, color='green', alpha=0.1, label='Stable Region (<0.05 eV)')
    plt.axhspan(-10, 0.05, color='green', alpha=0.1)

    plt.xlabel('Actual Stability (eV/atom)')
    plt.ylabel('Predicted Stability (eV/atom)')
    plt.title(f'Stability Prediction\n$R^2 = {r2:.3f}$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 0.5) # Focus on the relevant energy range
    plt.ylim(-0.5, 0.5)
    plt.tight_layout()
    output_path = os.path.join(figures_dir, 'eval_stability_parity.png')
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")

    # 7. Plot 2: Latent Space Map
    # Use PCA to squash 4D latent space -> 2D
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(mu)

    plt.figure(figsize=(8, 6))
    if len(z_2d) > 10000:
        idx = np.random.choice(len(z_2d), 10000, replace=False)
        z_plot = z_2d[idx]
        c_plot = y_true_ev[idx]
    else:
        z_plot = z_2d
        c_plot = y_true_ev
        
    # Color by Stability (Blue=Stable, Red=Unstable)
    # We clip the color range to make the stable region visible
    sc = plt.scatter(z_plot[:, 0], z_plot[:, 1], c=c_plot, cmap='coolwarm_r', s=5, alpha=0.6, vmin=-0.2, vmax=0.2)
    plt.colorbar(sc, label='Stability (eV/atom)')
    plt.title('Latent Space Map\n(Blue = Stable Alloys, Red = Unstable)')
    plt.xlabel('Latent PC1')
    plt.ylabel('Latent PC2')
    plt.tight_layout()
    output_path = os.path.join(figures_dir, 'eval_latent_space.png')
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")

if __name__ == "__main__":
    evaluate_stability()