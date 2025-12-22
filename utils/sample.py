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

# ==========================================
# PHYSICAL CONSTANTS (The "Textbook" Data)
# ==========================================
# Sources: Takeuchi & Inoue (2005), Guo et al (2011)
# Format: {Element: (Atomic Radius in Angstrom, VEC)}
element_props = {
    # Transition Metals (3d)
    'Ti': (1.462, 4), 'V':  (1.316, 5), 'Cr': (1.249, 6), 
    'Mn': (1.350, 7), 'Fe': (1.241, 8), 'Co': (1.252, 9), 
    'Ni': (1.246, 10),'Cu': (1.278, 11),
    # Refractory (4d/5d)
    'Zr': (1.602, 4), 'Nb': (1.429, 5), 'Mo': (1.363, 6), 
    'Hf': (1.578, 4), 'Ta': (1.430, 5), 'W':  (1.371, 6),
    # Others
    'Al': (1.432, 3), 'Si': (1.176, 4), 'C':  (0.770, 4), 'B': (0.820, 3)
}

def get_phase_prediction(delta, vec):
    """
    Classifies alloy based on Hume-Rothery Rules for HEAs.
    Rules from: Guo et al., J. Appl. Phys (2011).
    """
    phase = "Uncertain"
    
    # 1. Solid Solution Check (Geometric)
    if delta <= 6.6:
        # It's likely a simple solid solution (not brittle intermetallic)
        
        # 2. Phase Selection (Electronic)
        if vec >= 8.0:
            phase = "FCC (Stable/Ductile)"
        elif vec <= 6.87:
            phase = "BCC (Strong/Brittle)"
        else:
            phase = "FCC + BCC Mix"
            
    else:
        phase = "Multi-Phase / Intermetallic (Risk of Brittleness)"
        
    return phase

def calculate_physics(comps_dict, element_names):
    """
    Calculate physics-based properties for a composition.
    
    Args:
        comps_dict: Dictionary {element_name: fraction (0-1)}
        element_names: List of element names in order
    
    Returns:
        dict with 'Delta_r (%)', 'VEC', 'Predicted_Phase'
    """
    # Calculate average properties
    avg_r = 0.0
    avg_vec = 0.0
    
    for el, frac in comps_dict.items():
        if el in element_props:
            r, vec = element_props[el]
            avg_r += frac * r
            avg_vec += frac * vec
    
    # Calculate lattice distortion (Delta)
    delta_sq_sum = 0.0
    for el, frac in comps_dict.items():
        if el in element_props:
            r, _ = element_props[el]
            delta_sq_sum += frac * (1 - r/avg_r)**2
    
    delta = 100 * np.sqrt(delta_sq_sum)
    
    # Predict phase
    prediction = get_phase_prediction(delta, avg_vec)
    
    return {
        'Delta_r (%)': round(delta, 2),
        'VEC': round(avg_vec, 2),
        'Predicted_Phase': prediction
    }

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
    
    # 5. Format & HEA Filter + Verification
    results = []
    print("Filtering for HEA complexity (>= 4 elements) and verifying...")
    
    for i in range(len(stable_indices)):
        comp = compositions[i]
        energy = energies[i]
        
        # HEA Filter: At least 4 elements with > 5% concentration
        # This prevents the model from just giving us "Pure Tungsten"
        major_elements = np.sum(comp > 0.05)
        
        if major_elements >= 4:
            # Create formula string and composition dict
            formula_parts = []
            alloy_dict = {'Predicted_Stability_eV': round(energy, 4)}
            comps_dict = {}  # For physics calculation
            
            # Sort elements by percentage
            sorted_idx = np.argsort(comp)[::-1]
            for idx in sorted_idx:
                pct = comp[idx]
                if pct > 0.01: # 1% display cutoff
                    name = element_names[idx]
                    alloy_dict[name] = round(pct * 100, 1)
                    comps_dict[name] = pct  # Store as fraction for physics
                    formula_parts.append(f"{name}{int(pct*100)}")
            
            alloy_dict['Formula'] = "".join(formula_parts)
            
            # Calculate physics-based verification
            physics = calculate_physics(comps_dict, element_names)
            alloy_dict.update(physics)
            
            results.append(alloy_dict)

    # 6. Save Results with Verification
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        # Sort by Stability (Lower is Better)
        df_results = df_results.sort_values(by='Predicted_Stability_eV', ascending=True).head(50)
        
        # Reorder columns: Formula, Stability, Phase info, then element percentages
        priority_cols = ['Formula', 'Predicted_Stability_eV', 'Predicted_Phase', 'Delta_r (%)', 'VEC']
        other_cols = [c for c in df_results.columns if c not in priority_cols]
        cols = priority_cols + other_cols
        # Only include columns that exist
        cols = [c for c in cols if c in df_results.columns]
        df_results = df_results[cols].fillna(0)
        
        print("\nTop 10 Verified Stable Candidates:")
        display_cols = ['Formula', 'Predicted_Stability_eV', 'Predicted_Phase', 'Delta_r (%)', 'VEC']
        display_cols = [c for c in display_cols if c in df_results.columns]
        print(df_results[display_cols].head(10))
        
        output_filename = 'verified_stable_heas.csv'
        df_results.to_csv(output_filename, index=False)
        print(f"\nSuccess! Saved top 50 verified candidates to '{output_filename}'")
    else:
        print("Candidates were stable, but none met the HEA criteria (>= 4 elements).")

if __name__ == "__main__":
    generate_stable_alloys()