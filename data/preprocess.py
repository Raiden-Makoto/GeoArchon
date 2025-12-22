import pandas as pd
import numpy as np
import re

def parse_formula(formula):
    """Parses chemical formula into a dictionary of elements and fractions."""
    # Matches Element (e.g., 'Fe') and optional Number (e.g., '0.5' or '2')
    pattern = re.compile(r'([A-Z][a-z]*)(\d*\.?\d*)')
    matches = pattern.findall(formula)
    
    composition = {}
    for element, amount in matches:
        if amount == '':
            amount = 1.0
        else:
            amount = float(amount)
        composition[element] = amount
    
    # Normalize to sum = 1.0 (atomic fraction)
    total_atoms = sum(composition.values())
    for k in composition:
        composition[k] /= total_atoms
        
    return composition

def process_dft_data():
    print("Loading full dataset...")
    # Replace with your actual large filename
    df = pd.read_csv('HEA_dataset.csv') 
    
    # 1. Filter for HEAs (4 or more elements)
    print(f"Total entries: {len(df)}")
    df_hea = df[df['nelements'] >= 4].copy()
    print(f"Found {len(df_hea)} HEAs (nelements >= 4).")
    
    if len(df_hea) == 0:
        print("Error: No HEAs found. Check your input file.")
        return

    # 2. Parse Compositions
    print("Parsing formulas...")
    comps_list = []
    
    # We need to find ALL unique elements present in these HEAs to build our vector
    all_elements = set()
    
    for formula in df_hea['reduced_formula']:
        comp = parse_formula(formula)
        comps_list.append(comp)
        all_elements.update(comp.keys())
        
    sorted_elements = sorted(list(all_elements))
    print(f"Identified {len(sorted_elements)} unique elements in the dataset.")
    print(f"Elements: {', '.join(sorted_elements)}")
    
    # 3. Create the Vector Table
    # Turn list of dicts into a DataFrame with columns for each element
    df_comps = pd.DataFrame(comps_list).fillna(0.0)
    
    # Ensure all columns are present and sorted
    # (This aligns the input for the Neural Network)
    df_comps = df_comps.reindex(columns=sorted_elements, fill_value=0.0)
    
    # 4. Merge with Target (Stability)
    # Target: e_above_hull (Energy Above Hull)
    # Lower is better. 0.0 is stable. >0.1 is usually unstable.
    df_final = pd.concat([
        df_comps, 
        df_hea[['e_above_hull', 'reduced_formula']].reset_index(drop=True)
    ], axis=1)
    
    # Rename target for compatibility with our training code
    df_final.rename(columns={'e_above_hull': 'Stability (eV/atom)'}, inplace=True)
    
    # 5. Save
    output_file = 'HEA_stability_train.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")
    print(df_final.head())

if __name__ == "__main__":
    process_dft_data()