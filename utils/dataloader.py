import mlx.core as mx
import pandas as pd
import numpy as np
from pathlib import Path

# We'll identify element columns dynamically from the dataset
# Elements are columns that are NOT properties, metadata, or physics descriptors

def load_hea_data(csv_path='data/MPEA_cleaned.csv', batch_size=64, shuffle=True, val_split=0.2):
    """
    Load HEA (High Entropy Alloy) data using MLX.
    
    CRITICAL: Proper normalization is essential:
    - Element fractions (X): NOT normalized (preserve compositional constraint: sum to 1.0)
    - Target property (y): Z-score normalized (mean=0, std=1) for stable training
    
    Args:
        csv_path: Path to the CSV file (default: 'data/MPEA_cleaned.csv')
        batch_size: Batch size for data loading (default: 64)
        shuffle: Whether to shuffle the data (default: True)
        val_split: Fraction of data to use for validation (default: 0.2)
    
    Returns:
        train_generator: Generator function that yields training batches
        val_generator: Generator function that yields validation batches (or None if val_split=0)
        y_mean: Mean of the target property (for denormalization)
        y_std: Standard deviation of the target property (for denormalization)
        input_dim: Number of input features (number of element columns found)
        n_train_samples: Number of training samples
        n_val_samples: Number of validation samples
    """
    # 1. Load Data
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 2. Filter: Keep only rows with valid Yield Strength
    target_col = 'PROPERTY: YS (MPa)'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # 3. Extract Features (X) - Element fractions by column name
    # Identify element columns: all columns that are NOT properties, metadata, or physics descriptors
    all_cols = df_clean.columns.tolist()
    excluded_prefixes = ['PROPERTY', 'PHYS:', 'IDENTIFIER', 'REFERENCE']
    element_cols = [
        col for col in all_cols 
        if not any(col.startswith(prefix) for prefix in excluded_prefixes)
    ]
    
    if len(element_cols) == 0:
        raise ValueError("No element columns found in dataset. Expected columns like Al, Co, Fe, etc.")
    
    # Extract element fractions (preserve compositional constraint)
    X = df_clean[element_cols].values.astype(np.float32)
    
    # Verify compositional constraint (should sum to ~1.0 per sample)
    X_sums = X.sum(axis=1)
    if not np.allclose(X_sums, 1.0, atol=1e-4):
        print(f"Warning: Some samples don't sum to 1.0 (min={X_sums.min():.6f}, max={X_sums.max():.6f})")
        print(f"  Mean sum: {X_sums.mean():.6f}, Std: {X_sums.std():.6f}")
        # Normalize to ensure sum = 1.0 (CRITICAL for compositional constraint)
        X = X / X_sums[:, np.newaxis]
        print("  Normalized element fractions to sum to 1.0")
    
    # CRITICAL: Do NOT normalize X further - preserve compositional constraint
    # Element fractions are already in [0,1] range and sum to 1.0
    
    # 4. Extract Targets (y) and Normalize - CRITICAL STEP
    y = df_clean[target_col].values.astype(np.float32).reshape(-1, 1)
    
    # Save statistics for later denormalization
    y_mean = float(y.mean())
    y_std = float(y.std())
    
    if y_std == 0:
        raise ValueError("Target property has zero standard deviation - cannot normalize")
    
    # CRITICAL: Z-score normalization (mean=0, std=1)
    # This is essential for stable training and proper loss scaling
    y_norm = (y - y_mean) / y_std
    
    # Verify normalization
    y_norm_mean = float(y_norm.mean())
    y_norm_std = float(y_norm.std())
    if not np.allclose([y_norm_mean, y_norm_std], [0.0, 1.0], atol=1e-5):
        print(f"Warning: Normalization may be incorrect (mean={y_norm_mean:.6f}, std={y_norm_std:.6f})")
    
    # 5. Convert to MLX arrays
    # X is kept as raw element fractions (preserves compositional constraint: sum to 1)
    X_mlx = mx.array(X)
    y_mlx = mx.array(y_norm)
    
    n_samples = len(X_mlx)
    
    # 6. Split into train and validation sets
    if val_split > 0:
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        # Create indices for batching
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        print(f"Loaded {n_samples} valid alloys.")
        print(f"  Training samples: {n_train}")
        print(f"  Validation samples: {n_val}")
        print(f"Input Features: {len(element_cols)} element fractions (range: [{X.min():.4f}, {X.max():.4f}], sum to 1.0)")
        print(f"Target Property: Normalized (original mean: {y_mean:.2f}, std: {y_std:.2f})")
        print(f"  Normalized mean: {y_norm_mean:.6f}, std: {y_norm_std:.6f}")
        
        def train_generator():
            """Generator function that yields training batches."""
            # Shuffle training indices each epoch
            epoch_indices = train_indices.copy()
            if shuffle:
                np.random.shuffle(epoch_indices)
            
            for i in range(0, n_train, batch_size):
                batch_indices = epoch_indices[i:i + batch_size]
                batch_indices_mlx = mx.array(batch_indices)
                X_batch = X_mlx[batch_indices_mlx]
                y_batch = y_mlx[batch_indices_mlx]
                yield X_batch, y_batch
        
        def val_generator():
            """Generator function that yields validation batches."""
            for i in range(0, n_val, batch_size):
                batch_indices = val_indices[i:i + batch_size]
                batch_indices_mlx = mx.array(batch_indices)
                X_batch = X_mlx[batch_indices_mlx]
                y_batch = y_mlx[batch_indices_mlx]
                yield X_batch, y_batch
        
        return train_generator, val_generator, y_mean, y_std, X.shape[1], n_train, n_val
    else:
        # No validation split - return all data as training
        print(f"Loaded {n_samples} valid alloys.")
        print(f"Input Features: {len(element_cols)} element fractions (range: [{X.min():.4f}, {X.max():.4f}], sum to 1.0)")
        print(f"Target Property: Normalized (original mean: {y_mean:.2f}, std: {y_std:.2f})")
        print(f"  Normalized mean: {y_norm_mean:.6f}, std: {y_norm_std:.6f}")
        
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        def data_generator():
            """Generator function that yields batches of data."""
            epoch_indices = indices.copy()
            if shuffle:
                np.random.shuffle(epoch_indices)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = epoch_indices[i:i + batch_size]
                batch_indices_mlx = mx.array(batch_indices)
                X_batch = X_mlx[batch_indices_mlx]
                y_batch = y_mlx[batch_indices_mlx]
                yield X_batch, y_batch
        
        return data_generator, None, y_mean, y_std, X.shape[1], n_samples, 0

# Usage example
#if __name__ == "__main__":
#    data_gen, y_mean, y_std, input_dim, n_samples = load_hea_data() 
#    # Example: iterate through batches
#   for batch_idx, (X_batch, y_batch) in enumerate(data_gen()):
#        print(f"Batch {batch_idx}: X shape = {X_batch.shape}, y shape = {y_batch.shape}")
#        if batch_idx >= 2:  # Just show first few batches
#            break