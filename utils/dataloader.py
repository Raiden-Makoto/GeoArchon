import mlx.core as mx
import pandas as pd
import numpy as np
from pathlib import Path

def load_hea_data(csv_path='data/MPEA_cleaned.csv', batch_size=64, shuffle=True):
    """
    Load HEA (High Entropy Alloy) data using MLX.
    
    Args:
        csv_path: Path to the CSV file (default: 'data/MPEA_cleaned.csv')
        batch_size: Batch size for data loading (default: 64)
        shuffle: Whether to shuffle the data (default: True)
    
    Returns:
        data_generator: Generator function that yields batches
        y_mean: Mean of the target property (for denormalization)
        y_std: Standard deviation of the target property (for denormalization)
        input_dim: Number of input features (30)
        n_samples: Total number of valid samples
    """
    # 1. Load Data
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 2. Filter: Keep only rows with valid Yield Strength
    target_col = 'PROPERTY: YS (MPa)'
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # 3. Extract Features (X)
    # The first 30 columns are the element fractions
    element_cols = df.columns[:30]
    X = df_clean[element_cols].values.astype(np.float32)
    
    # 4. Extract Targets (y) and Normalize
    y = df_clean[target_col].values.astype(np.float32).reshape(-1, 1)
    
    # Save statistics for later un-normalization
    y_mean = float(y.mean())
    y_std = float(y.std())
    
    # Standardize (Z-score normalization)
    y_norm = (y - y_mean) / y_std
    
    # 5. Convert to MLX arrays
    X_mlx = mx.array(X)
    y_mlx = mx.array(y_norm)
    
    n_samples = len(X_mlx)
    
    print(f"Loaded {n_samples} valid alloys.")
    print(f"Target Property Normalized (Mean: {y_mean:.2f}, Std: {y_std:.2f})")
    
    # 6. Create indices for batching
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    def data_generator():
        """Generator function that yields batches of data."""
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            # Convert indices to MLX array for indexing
            batch_indices_mlx = mx.array(batch_indices)
            X_batch = X_mlx[batch_indices_mlx]
            y_batch = y_mlx[batch_indices_mlx]
            yield X_batch, y_batch
    
    return data_generator, y_mean, y_std, X.shape[1], n_samples

# Usage example
#if __name__ == "__main__":
#    data_gen, y_mean, y_std, input_dim, n_samples = load_hea_data() 
#    # Example: iterate through batches
#   for batch_idx, (X_batch, y_batch) in enumerate(data_gen()):
#        print(f"Batch {batch_idx}: X shape = {X_batch.shape}, y shape = {y_batch.shape}")
#        if batch_idx >= 2:  # Just show first few batches
#            break