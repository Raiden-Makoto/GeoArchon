import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your local modules
from models.hea_vae import HEA_VAE
from utils.dataloader import load_hea_data

def train_latent_booster(model_path='models/hea_vae_best.npz'):
    print("="*60)
    print("LATENT BOOSTING STAGE")
    print("="*60)

    # 1. Load Data
    # We use shuffle=False to ensure the order of X and y matches perfectly
    print("[1/5] Loading Data...")
    train_gen, val_gen, y_mean, y_std, input_dim, n_train, n_val = load_hea_data(batch_size=100, shuffle=False, val_split=0.0)
    # Use train generator (val_split=0 means all data in train_gen)
    data_gen = train_gen
    
    # 2. Load the Pre-Trained VAE
    # IMPORTANT: Ensure these match your training architecture!
    # Based on our conversation: latent=4, hidden=512
    print(f"[2/5] Loading VAE from {model_path}...")
    vae = HEA_VAE(latent_dim=4, hidden_dim=512, dropout_rate=0.0)
    try:
        vae.load_weights(model_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Did you ensure hidden_dim matches your training run?")
        return

    # Note: MLX doesn't have eval() mode - dropout is controlled by dropout_rate parameter (set to 0.0 above)

    # 3. Extract Latent Features (The "Translation" Step)
    print("[3/5] Extracting latent features (mu) for all alloys...")
    X_latent_list = []
    y_true_list = []
    
    # Iterate through the entire dataset
    for x_batch, y_batch in data_gen():
        # Encode only: x -> encoder -> mu
        # We ignore logvar because we want the "best guess" map location
        mu, _ = vae.encode(x_batch)
        
        X_latent_list.append(np.array(mu))
        y_true_list.append(np.array(y_batch))
    
    # Concatenate all batches into one large feature matrix
    X_latent = np.concatenate(X_latent_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0).ravel() # Flatten to 1D
    
    print(f"      Extracted dataset shape: X={X_latent.shape}, y={y_true.shape}")

    # 4. Train Gradient Boosting Regressor (The "Specialist")
    print("[4/5] Training Gradient Boosting Regressor...")
    
    # Split data to verify generalization
    X_train, X_test, y_train, y_test = train_test_split(X_latent, y_true, test_size=0.2, random_state=42)
    
    # XGBoost/GradientBoosting settings
    # n_estimators=2000: High number of trees (booster learns slowly but deeply)
    # learning_rate=0.02: Slow learning prevents overfitting
    # max_depth=5: Deep enough to capture non-linear cliffs in strength
    gbr = GradientBoostingRegressor(
        n_estimators=2000, 
        learning_rate=0.02, 
        max_depth=5, 
        subsample=0.8, # Train on random 80% of data per tree (adds robustness)
        validation_fraction=0.1,
        n_iter_no_change=50, # Early stopping for the booster
        random_state=42,
        verbose=1
    )
    
    gbr.fit(X_train, y_train)
    
    # 5. Evaluate
    print("[5/5] Evaluating...")
    y_pred_norm = gbr.predict(X_test)
    
    # Denormalize predictions back to MPa
    y_test_mpa = (y_test * y_std) + y_mean
    y_pred_mpa = (y_pred_norm * y_std) + y_mean
    
    r2 = r2_score(y_test_mpa, y_pred_mpa)
    mse = mean_squared_error(y_test_mpa, y_pred_mpa)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"R² Score:  {r2:.4f}")
    print(f"RMSE:      {rmse:.2f} MPa")
    print("="*30)
    
    # 6. Plotting
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test_mpa, y_pred_mpa, alpha=0.6, s=20, c='darkorange', edgecolors='k', linewidth=0.2)
    
    # Perfect line
    min_val = min(y_test_mpa.min(), y_pred_mpa.min())
    max_val = max(y_test_mpa.max(), y_pred_mpa.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title(f'Latent Boosting Prediction\n$R^2 = {r2:.3f}$')
    plt.xlabel('Actual Yield Strength (MPa)')
    plt.ylabel('Predicted Yield Strength (MPa)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Ensure figures directory exists
    import os
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/eval_booster_parity.png')
    print("Plot saved to figures/eval_booster_parity.png")

if __name__ == "__main__":
    train_latent_booster()