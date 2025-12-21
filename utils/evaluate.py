import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.hea_vae import HEA_VAE
from utils.dataloader import load_hea_data

def ensemble_evaluate(model_paths=[
    'models/hea_vae_alpha_25.npz', 
    'models/hea_vae_alpha_41.npz', 
    'models/hea_vae_alpha_50.npz',
]):
    print("Loading Data...")
    train_gen, val_gen, y_mean, y_std, input_dim, n_train, n_val = load_hea_data(batch_size=100, shuffle=False, val_split=0.0)
    # Use train generator (val_split=0 means all data in train_gen)
    data_gen = train_gen

    # 1. Load All Models
    models = []
    for path in model_paths:
        print(f"Loading {path}...")
        # Ensure these params match your training (latent=4, hidden=512)
        model = HEA_VAE(latent_dim=4, hidden_dim=512, dropout_rate=0.0) 
        model.load_weights(path)
        # Note: MLX doesn't have eval() mode - dropout is controlled by dropout_rate parameter
        models.append(model)

    # 2. Run Inference
    y_true_all = []
    y_pred_ensemble = []

    for x_batch, y_batch in data_gen():
        # Get predictions from all 3 models
        batch_preds = []
        for model in models:
            _, pred_y, _, _ = model(x_batch)
            batch_preds.append(pred_y)
        
        # AVERAGE THE PREDICTIONS (The "Ensemble" Magic)
        # Stack them to shape (3, batch_size, 1) then mean over axis 0
        stacked_preds = mx.stack(batch_preds)
        avg_pred = mx.mean(stacked_preds, axis=0)
        
        y_true_all.append(np.array(y_batch))
        y_pred_ensemble.append(np.array(avg_pred))

    # 3. Process Results
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_ensemble, axis=0)

    # Denormalize
    y_true_mpa = (y_true * y_std) + y_mean
    y_pred_mpa = (y_pred * y_std) + y_mean

    # 4. Calculate Score
    r2 = r2_score(y_true_mpa, y_pred_mpa)
    mse = np.mean((y_true_mpa - y_pred_mpa) ** 2)
    
    print("\n=== ENSEMBLE RESULTS ===")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE (MPa): {mse:.2f}")

    # 5. Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_mpa, y_pred_mpa, alpha=0.5, s=10, c='purple')
    
    min_val = min(y_true_mpa.min(), y_pred_mpa.min())
    max_val = max(y_true_mpa.max(), y_pred_mpa.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title(f'Ensemble Prediction (3 Models)\n$R^2 = {r2:.3f}$')
    plt.xlabel('Actual YS (MPa)')
    plt.ylabel('Predicted YS (MPa)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Ensure figures directory exists
    import os
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/eval_ensemble_parity.png')
    print("Saved figures/eval_ensemble_parity.png")

if __name__ == "__main__":
    ensemble_evaluate()