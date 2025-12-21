import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
from pathlib import Path

def train_xgb():
    print("=" * 60)
    print("XGBoost Training on HEA Dataset")
    print("=" * 60)
    
    # Load cleaned dataset (includes physics descriptors)
    print("Loading data from data/MPEA_cleaned.csv...")
    df = pd.read_csv('data/MPEA_cleaned.csv')
    
    # Target: Yield Strength
    target_col = 'PROPERTY: YS (MPa)'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Filter out rows with missing target values
    df_clean = df.dropna(subset=[target_col]).copy()
    y = df_clean[target_col].values
    
    # Inputs: Element fractions + Physics descriptors
    # Exclude property columns, metadata, and target
    exclude_cols = [col for col in df_clean.columns 
                    if col.startswith('PROPERTY') or 
                       'IDENTIFIER' in col or 
                       'REFERENCE' in col]
    
    X = df_clean.drop(columns=exclude_cols).values
    feature_names = df_clean.drop(columns=exclude_cols).columns.tolist()
    
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]} (elements + physics descriptors)")
    print(f"  Target: {target_col}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # XGBoost Regressor
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror',
        early_stopping_rounds=50  # New API: set in constructor
    )
    
    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print("\n" + "=" * 60)
    print("XGBOOST RESULTS")
    print("=" * 60)
    print(f"R² Score:        {r2:.4f}")
    print(f"MSE (MPa²):      {mse:.2f}")
    print(f"RMSE (MPa):      {rmse:.2f}")
    print(f"MAE (MPa):       {mae:.2f}")
    print("=" * 60)
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Plot: Parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, c='green', s=20, edgecolors='black', linewidth=0.5)
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title(f'XGBoost Prediction Parity\n$R^2 = {r2:.3f}$, RMSE = {rmse:.1f} MPa', fontsize=14)
    plt.xlabel('Actual Yield Strength (MPa)', fontsize=12)
    plt.ylabel('Predicted Yield Strength (MPa)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/eval_xgb_parity.png', dpi=150)
    print(f"\nSaved: figures/eval_xgb_parity.png")
    plt.close()
    
    # Feature Importance
    print("\nGenerating feature importance plot...")
    sorted_idx = model.feature_importances_.argsort()
    top_n = min(15, len(feature_names))
    top_features = [feature_names[i] for i in sorted_idx[-top_n:]]
    top_importances = model.feature_importances_[sorted_idx][-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances, color='steelblue', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features (XGBoost)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('figures/xgb_importance.png', dpi=150)
    print(f"Saved: figures/xgb_importance.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("XGBoost training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    train_xgb()