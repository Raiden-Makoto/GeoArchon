"""
Data preprocessing utilities for HEA dataset.

This module provides functions to:
1. Calculate physics-based descriptors from element compositions
2. Parse chemical formulas into element fractions
3. Clean and normalize the dataset
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


# ============================================================================
# Element Properties Database
# ============================================================================

# Element: (Atomic Radius (pm), VEC, Electronegativity, Melting Point (K))
# Data source: Mendeleev / WebElements
ELEMENT_PROPERTIES = {
    'Al': (143, 3, 1.61, 933),
    'Co': (125, 9, 1.88, 1768),
    'Cr': (128, 6, 1.66, 2180),
    'Fe': (126, 8, 1.83, 1811),
    'Ni': (124, 10, 1.91, 1728),
    'Cu': (128, 11, 1.90, 1357),
    'Mn': (127, 7, 1.55, 1519),
    'Ti': (147, 4, 1.54, 1941),
    'V':  (134, 5, 1.63, 2183),
    'Nb': (146, 5, 1.60, 2750),
    'Mo': (139, 6, 2.16, 2896),
    'Zr': (160, 4, 1.33, 2128),
    'Hf': (159, 4, 1.30, 2506),
    'Ta': (146, 5, 1.50, 3290),
    'W':  (139, 6, 2.36, 3695),
    'C':  (77,  4, 2.55, 3823),
    'B':  (82,  3, 2.04, 2349),
    'Si': (111, 4, 1.90, 1687)
}


# ============================================================================
# Physics Descriptor Calculation
# ============================================================================

def calculate_physics_descriptors(row, element_properties=ELEMENT_PROPERTIES):
    """
    Calculate weighted physical properties for a single alloy composition.
    
    Computes:
    - Atomic Size Mismatch (Delta): sqrt(sum(c_i * (1 - r_i/r_avg)^2))
    - Average Valence Electron Concentration (VEC)
    - Average Electronegativity
    - Average Melting Point
    
    Args:
        row: pandas Series with element fractions as values
        element_properties: dict mapping element symbols to (radius, vec, electroneg, melting_point)
    
    Returns:
        pandas Series with ['PHYS: Delta', 'PHYS: VEC', 'PHYS: Electronegativity', 'PHYS: Tm']
    """
    avg_r = 0.0
    avg_vec = 0.0
    avg_chi = 0.0
    avg_tm = 0.0
    
    # Track concentrations and radii for Delta calculation
    concs = []
    radii = []
    
    for element, props in element_properties.items():
        if element in row.index:
            conc = row[element]
            if conc > 0:
                r, vec, chi, tm = props
                avg_r += conc * r
                avg_vec += conc * vec
                avg_chi += conc * chi
                avg_tm += conc * tm
                
                concs.append(conc)
                radii.append(r)
    
    # Calculate Atomic Size Mismatch (Delta)
    # Delta = sqrt(sum(c_i * (1 - r_i/r_avg)^2)) * 100
    if avg_r > 0 and len(concs) > 0:
        delta_sq = sum([c * (1 - r/avg_r)**2 for c, r in zip(concs, radii)])
        delta = 100 * np.sqrt(delta_sq)
    else:
        delta = 0.0
        
    return pd.Series(
        [delta, avg_vec, avg_chi, avg_tm],
        index=['PHYS: Delta', 'PHYS: VEC', 'PHYS: Electronegativity', 'PHYS: Tm']
    )


def add_physics_descriptors(df, element_properties=ELEMENT_PROPERTIES, output_path=None):
    """
    Add physics-based descriptors to a dataframe with element compositions.
    
    Args:
        df: DataFrame with element fraction columns
        element_properties: dict mapping element symbols to properties
        output_path: Optional path to save enriched dataset
    
    Returns:
        DataFrame with added physics descriptor columns
    """
    # Identify element columns
    element_cols = [c for c in df.columns if c in element_properties.keys()]
    
    if len(element_cols) == 0:
        print("Warning: No recognized element columns found.")
        return df
    
    print(f"Found {len(element_cols)} recognized elements: {element_cols}")
    print("Calculating physics descriptors...")
    
    # Calculate physics descriptors for each row
    physics_df = df[element_cols].apply(
        lambda row: calculate_physics_descriptors(row, element_properties),
        axis=1
    )
    
    # Merge with original dataframe
    df_enriched = pd.concat([df, physics_df], axis=1)
    
    # Save if output path provided
    if output_path:
        df_enriched.to_csv(output_path, index=False)
        print(f"Saved enriched dataset to {output_path}")
    
    return df_enriched


# ============================================================================
# Formula Parsing
# ============================================================================

def parse_formula(formula):
    """
    Parse a chemical formula string into element composition dictionary.
    
    Handles formats like:
    - "Al0.25 Co1" (with spaces)
    - "Al0.25Co1" (without spaces)
    - "Al1" (implicit coefficient)
    
    Args:
        formula: String containing chemical formula
    
    Returns:
        dict mapping element symbols to amounts
    """
    # Pattern: [A-Z][a-z]* (Element) followed by [\d\.]* (Number)
    pattern = re.compile(r"([A-Z][a-z]*)([\d\.]*)")
    
    elements = {}
    matches = pattern.findall(formula)
    
    for element, amount_str in matches:
        # Handle implicit coefficient of 1
        if amount_str == '':
            amount = 1.0
        else:
            try:
                amount = float(amount_str)
            except ValueError:
                amount = 1.0
        
        # Sum amounts if element appears multiple times
        if element in elements:
            elements[element] += amount
        else:
            elements[element] = amount
    
    return elements


def parse_formulas_to_dataframe(df, formula_col='FORMULA'):
    """
    Parse chemical formulas in a dataframe and convert to element fraction matrix.
    
    Args:
        df: DataFrame with formula column
        formula_col: Name of column containing formulas
    
    Returns:
        DataFrame with element fractions (normalized to sum to 1 per row)
    """
    if formula_col not in df.columns:
        raise ValueError(f"Column '{formula_col}' not found in dataframe")
    
    print(f"Parsing formulas from column '{formula_col}'...")
    
    # Parse all formulas
    composition_list = df[formula_col].apply(parse_formula).tolist()
    comp_df = pd.DataFrame(composition_list).fillna(0.0)
    
    # Normalize to fractions (sum to 1 per row)
    row_sums = comp_df.sum(axis=1)
    comp_df_normalized = comp_df.div(row_sums, axis=0)
    
    print(f"Shape of composition matrix: {comp_df_normalized.shape}")
    print(f"Elements found: {comp_df_normalized.columns.tolist()}")
    
    return comp_df_normalized


# ============================================================================
# Dataset Cleaning
# ============================================================================

def clean_property_columns(df):
    """
    Clean and convert property columns to numeric types.
    
    Args:
        df: DataFrame with property columns
    
    Returns:
        DataFrame with cleaned property columns
    """
    property_cols = [c for c in df.columns if 'PROPERTY' in c]
    
    if len(property_cols) == 0:
        return df
    
    # Define numeric property columns
    numeric_prop_cols = [
        'PROPERTY: HV',
        'PROPERTY: YS (MPa)',
        'PROPERTY: UTS (MPa)', 
        'PROPERTY: Elongation (%)',
        'PROPERTY: Elongation plastic (%)',
        'PROPERTY: Exp. Young modulus (GPa)',
        'PROPERTY: Calculated Young modulus (GPa)'
    ]
    
    cleaned_props = df[property_cols].copy()
    
    # Convert to numeric, coercing errors to NaN
    for col in numeric_prop_cols:
        if col in cleaned_props.columns:
            cleaned_props[col] = pd.to_numeric(cleaned_props[col], errors='coerce')
    
    return cleaned_props


def create_cleaned_dataset(
    input_path='data/MPEA_dataset.csv',
    output_path='data/MPEA_cleaned.csv',
    formula_col='FORMULA'
):
    """
    Create cleaned dataset from raw formula-based dataset.
    
    Args:
        input_path: Path to input CSV with formulas
        output_path: Path to save cleaned CSV
        formula_col: Name of column containing formulas
    
    Returns:
        DataFrame with cleaned data
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Parse formulas to element fractions
    comp_df = parse_formulas_to_dataframe(df, formula_col=formula_col)
    
    # Clean property columns
    cleaned_props = clean_property_columns(df)
    
    # Get metadata columns
    meta_cols = [c for c in df.columns if 'IDENTIFIER' in c or 'REFERENCE' in c]
    
    # Combine all parts
    final_df = pd.concat([comp_df, cleaned_props, df[meta_cols]], axis=1)
    
    # Save
    final_df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")
    
    return final_df


# ============================================================================
# Main Processing Functions
# ============================================================================

def process_full_dataset(
    input_path='data/MPEA_dataset.csv',
    output_path='data/MPEA_cleaned.csv'
):
    """
    Process the full dataset: parse formulas, clean properties, and add physics descriptors.
    
    Args:
        input_path: Path to original dataset with formulas
        output_path: Path to save fully processed dataset
    """
    print(f"Loading original dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Step 1: Parse formulas to element fractions
    print("\nStep 1: Parsing chemical formulas...")
    comp_df = parse_formulas_to_dataframe(df, formula_col='FORMULA')
    
    # Step 2: Clean property columns
    print("\nStep 2: Cleaning property columns...")
    cleaned_props = clean_property_columns(df)
    
    # Step 3: Get metadata columns
    meta_cols = [c for c in df.columns if 'IDENTIFIER' in c or 'REFERENCE' in c]
    
    # Step 4: Combine composition, properties, and metadata
    print("\nStep 3: Combining composition, properties, and metadata...")
    df_combined = pd.concat([comp_df, cleaned_props, df[meta_cols]], axis=1)
    
    # Step 5: Add physics descriptors
    print("\nStep 4: Adding physics descriptors...")
    df_final = add_physics_descriptors(df_combined)
    
    # Step 6: Save final dataset
    print(f"\nStep 5: Saving fully processed dataset to {output_path}...")
    df_final.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("Processing Summary:")
    print("=" * 60)
    print(f"  Original samples: {len(df)}")
    print(f"  Element columns: {len(comp_df.columns)}")
    print(f"  Property columns: {len(cleaned_props.columns)}")
    print(f"  Physics descriptors: 4 (Delta, VEC, Electronegativity, Tm)")
    print(f"  Total columns: {len(df_final.columns)}")
    print(f"\nSaved to: {output_path}")
    
    return df_final


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Processing Full Dataset")
    print("=" * 60)
    print("  Input:  data/MPEA_dataset.csv (original)")
    print("  Output: data/MPEA_cleaned.csv (fully processed)")
    print("=" * 60)
    
    process_full_dataset(
        input_path='data/MPEA_dataset.csv',
        output_path='data/MPEA_cleaned.csv'
    )
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)