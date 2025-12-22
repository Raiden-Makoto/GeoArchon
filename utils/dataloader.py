import pandas as pd
import numpy as np
import mlx.core as mx

def load_hea_data(csv_path='data/HEA_stability_train.csv', batch_size=64, shuffle=True, val_split=0.0):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Identify Target
    # We look for the stability column we created
    target_col = 'Stability (eV/atom)'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV!")
        
    # 2. Identify Input Features (Elements)
    # We explicitly exclude metadata columns to find the element columns
    exclude_cols = [target_col, 'reduced_formula', 'fid', 'chemical_system', 'formula', 'nelements']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Found {len(feature_cols)} input features (elements).")
    
    # 3. Extract and Normalize Data
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)
    
    # Split into train/val if requested
    n_samples = X.shape[0]
    if val_split > 0.0:
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        # Shuffle indices before splitting
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        # Normalize using training statistics only
        y_mean = y_train.mean()
        y_std = y_train.std()
        y_train_norm = (y_train - y_mean) / (y_std + 1e-6)
        y_val_norm = (y_val - y_mean) / (y_std + 1e-6)
        
        # Convert to MLX Arrays
        X_train_mx = mx.array(X_train)
        y_train_mx = mx.array(y_train_norm)
        X_val_mx = mx.array(X_val)
        y_val_mx = mx.array(y_val_norm)
        
        # Create generators
        def train_generator():
            train_idx = np.arange(n_train)
            if shuffle:
                np.random.shuffle(train_idx)
            for start_idx in range(0, n_train, batch_size):
                end_idx = min(start_idx + batch_size, n_train)
                batch_indices = train_idx[start_idx:end_idx]
                batch_idx_mx = mx.array(batch_indices)
                yield X_train_mx[batch_idx_mx], y_train_mx[batch_idx_mx]
        
        def val_generator():
            val_idx = np.arange(n_val)
            for start_idx in range(0, n_val, batch_size):
                end_idx = min(start_idx + batch_size, n_val)
                batch_indices = val_idx[start_idx:end_idx]
                batch_idx_mx = mx.array(batch_indices)
                yield X_val_mx[batch_idx_mx], y_val_mx[batch_idx_mx]
        
        print(f"Split: {n_train} train, {n_val} validation")
        return train_generator, val_generator, y_mean, y_std, X.shape[1], n_train, n_val
    else:
        # No validation split
        y_mean = y.mean()
        y_std = y.std()
        y_norm = (y - y_mean) / (y_std + 1e-6)
        
        input_dim = X.shape[1]
        
        # Convert to MLX Arrays
        X_mx = mx.array(X)
        y_mx = mx.array(y_norm)
        
        # Create Generator
        def data_generator():
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)
                
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Convert numpy indices to mlx array for indexing
                batch_idx_mx = mx.array(batch_indices)
                
                yield X_mx[batch_idx_mx], y_mx[batch_idx_mx]

        return data_generator, None, y_mean, y_std, input_dim, n_samples, 0