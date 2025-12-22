import mlx.core as mx
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.hea_vae import HEA_VAE
from utils.dataloader import load_hea_data

def generate_stable_alloys():
    print("="*60)
    print("GENERATING STABLE HEA CANDIDATES (Energy Minimization)")
    print("="*60)

    # 1. Load Data Config
    # We load the data just to get the correct 'input_dim' and element names
    # (Must match what the model was trained on)
    print("Loading data configuration...")
    # load_hea_data returns: (data_gen, val_gen, y_mean, y_std, input_dim, n_train, n_val)
    _, _, y_mean, y_std, input_dim, _, _ = load_hea_data('data/HEA_stability_train.csv', batch_size=1, shuffle=False, val_split=0.0)
    
    # Extract element names from the CSV header to decode the output
    df_ref = pd.read_csv('data/HEA_stability_train.csv')
    # Exclude non-element columns
    exclude_cols = ['Stability (eV/atom)', 'reduced_formula', 'fid', 'chemical_system', 'formula', 'nelements']
    element_names = [c for c in df_ref.columns if c not in exclude_cols]
    
    print(f"Model expects {input_dim} elements: {element_names[:5]}...")

    # 2. Load Model
    # Ensure hidden_dim matches your training (default 512)
    print("Loading VAE model...")
    vae = HEA_VAE(input_dim=input_dim, latent_dim=4, hidden_dim=512, dropout_rate=0.0) 
    try:
        vae.load_weights('models/hea_vae_best.npz')
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # 3. Scan Latent Space
    print("Scanning latent space for stability minima...")
    # We sample 50,000 random points. Since Beta was low (0.05), the space is distinct.
    # We use sigma=2.0 to search slightly outside the average to find new mixes.
    z_search = mx.random.normal(shape=(50000, 4)) * 2.0
    
    # Predict Stability (Energy)
    pred_y_norm = vae.predict(z_search)
    
    # Denormalize to get real eV/atom
    pred_y_ev = (np.array(pred_y_norm).flatten() * y_std) + y_mean
    
    # 4. Filter for STABLE Alloys (Energy < 0.05 eV/atom)
    # < 0.00 is theoretically stable.
    # < 0.05 is "Metastable" (Synthesizable).
    cutoff = 0.05
    stable_indices = np.where(pred_y_ev < cutoff)[0]
    
    print(f"Found {len(stable_indices)} candidates with E < {cutoff} eV/atom.")
    
    if len(stable_indices) == 0:
        print("No stable candidates found. Try relaxing the cutoff or searching more points.")
        return

    # Decode the stable vectors into compositions
    best_z = z_search[mx.array(stable_indices)]
    compositions = np.array(vae.decode(best_z))
    energies = pred_y_ev[stable_indices]
    
    # 5. Format & HEA Filter
    results = []
    print("Filtering for HEA complexity (>= 4 elements)...")
    
    for i in range(len(stable_indices)):
        comp = compositions[i]
        energy = energies[i]
        
        # HEA Filter: At least 4 elements with > 5% concentration
        # This prevents the model from just giving us "Pure Tungsten"
        major_elements = np.sum(comp > 0.05)
        
        if major_elements >= 4:
            # Create formula string
            formula_parts = []
            alloy_dict = {'Predicted_Stability_eV': round(energy, 4)}
            
            # Sort elements by percentage
            sorted_idx = np.argsort(comp)[::-1]
            for idx in sorted_idx:
                pct = comp[idx]
                if pct > 0.01: # 1% display cutoff
                    name = element_names[idx]
                    alloy_dict[name] = round(pct * 100, 1)
                    formula_parts.append(f"{name}{int(pct*100)}")
            
            alloy_dict['Formula'] = "".join(formula_parts)
            results.append(alloy_dict)

    # 6. Save Results
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        # Sort by Stability (Lower is Better)
        df_results = df_results.sort_values(by='Predicted_Stability_eV', ascending=True).head(50)
        
        # Reorder columns
        cols = ['Formula', 'Predicted_Stability_eV'] + [c for c in df_results.columns if c not in ['Formula', 'Predicted_Stability_eV']]
        df_results = df_results[cols].fillna(0)
        
        print("\nTop 10 Stable Candidates:")
        print(df_results.head(10))
        
        output_filename = 'generated_stable_heas.csv'
        df_results.to_csv(output_filename, index=False)
        print(f"\nSuccess! Saved top 50 candidates to '{output_filename}'")
    else:
        print("Candidates were stable, but none met the HEA criteria (>= 4 elements).")

if __name__ == "__main__":
    generate_stable_alloys()