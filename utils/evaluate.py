import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.hea_vae import HEA_VAE
from utils.dataloader import load_hea_data

def evaluate(model_path='models/hea_vae_best.npz', figures_dir='figures'):
    # Create figures directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    # 1. Load Data & Stats
    # We need y_mean and y_std to un-normalize the predictions for plotting
    print("Loading data...")
    train_gen, val_gen, y_mean, y_std, input_dim, n_train, n_val = load_hea_data(
        batch_size=128, shuffle=False, val_split=0.0  # No validation split for evaluation
    )
    data_gen = train_gen  # Use training generator for full dataset

    # 2. Load Model
    # Ensure latent_dim matches what you trained with (train.py uses 4)
    print(f"Loading model from {model_path}...")
    model = HEA_VAE(latent_dim=4)
    model.load_weights(model_path)

    # 3. Inference Loop
    y_true_all = []
    y_pred_all = []
    mu_all = []
    
    print("Running inference...")
    for x_batch, y_batch in data_gen():
        # Forward pass
        # returns: recon_x, pred_y, mu, logvar
        recon_x, pred_y, mu, logvar = model(x_batch)
        
        # Store results (convert to numpy)
        y_true_all.append(np.array(y_batch))
        y_pred_all.append(np.array(pred_y))
        mu_all.append(np.array(mu))

    # Concatenate all batches
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    mu = np.concatenate(mu_all, axis=0)

    # 4. Denormalize Properties (back to MPa)
    y_true_mpa = (y_true * y_std) + y_mean
    y_pred_mpa = (y_pred * y_std) + y_mean

    # ---------------------------------------------------------
    # PLOT 1: Property Prediction (Parity Plot)
    # ---------------------------------------------------------
    r2 = r2_score(y_true_mpa, y_pred_mpa)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_mpa, y_pred_mpa, alpha=0.5, s=10)
    
    # Plot perfect diagonal line
    min_val = min(y_true_mpa.min(), y_pred_mpa.min())
    max_val = max(y_true_mpa.max(), y_pred_mpa.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title(f'Property Prediction (Yield Strength)\n$R^2 = {r2:.3f}$')
    plt.xlabel('Actual YS (MPa)')
    plt.ylabel('Predicted YS (MPa)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(figures_dir, 'eval_property_parity.png')
    plt.savefig(plot_path)
    print(f"Saved {plot_path} (R2={r2:.3f})")

    # ---------------------------------------------------------
    # PLOT 2: Latent Space Visualization
    # ---------------------------------------------------------
    # If latent_dim > 2, use PCA to project to 2D
    if mu.shape[1] > 2:
        pca = PCA(n_components=2)
        mu_2d = pca.fit_transform(mu)
        xlabel, ylabel = 'PC1', 'PC2'
    else:
        mu_2d = mu
        xlabel, ylabel = 'Latent Dim 1', 'Latent Dim 2'

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=y_true_mpa.flatten(), 
                     cmap='viridis', alpha=0.6, s=15)
    plt.colorbar(sc, label='Yield Strength (MPa)')
    plt.title('Latent Space Organization')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(figures_dir, 'eval_latent_space.png')
    plt.savefig(plot_path)
    print(f"Saved {plot_path}")

    # ---------------------------------------------------------
    # CHECK 3: Generative Capability
    # ---------------------------------------------------------
    print("\n--- Generative Check ---")
    # Sample 3 random points from normal distribution
    z_sample = mx.random.normal(shape=(3, 4)) # shape=(3, latent_dim)
    
    # Decode
    generated_alloys = model.decode(z_sample)
    predicted_props = model.predict(z_sample)
    
    # Convert to numpy for printing
    gen_np = np.array(generated_alloys)
    prop_np = np.array(predicted_props)
    
    # Element names (approximate list based on your data columns)
    # You might want to grab the actual columns from your dataframe if possible
    # For now, we print the top 3 elements for each alloy
    for i in range(3):
        print(f"\nGenerated Alloy {i+1}:")
        print(f"  Predicted Strength: {(float(prop_np[i]) * y_std) + y_mean:.0f} MPa")
        
        # Find indices of elements with > 5% concentration
        indices = np.where(gen_np[i] > 0.05)[0]
        elements_str = []
        for idx in indices:
            elements_str.append(f"Elem_{idx}: {gen_np[i][idx]*100:.1f}%")
        print("  Composition:", ", ".join(elements_str))

if __name__ == "__main__":
    evaluate()