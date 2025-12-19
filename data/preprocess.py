import pandas as pd #type: ignore
import re

import re
import numpy as np

def parse_formula(formula):
    # This regex looks for an Element (Capital followed by optional lowercase)
    # followed by an optional number.
    # It handles "Al0.25 Co1" or "Al0.25Co1" (spaces or no spaces)
    # Remove any whitespace to handle both "Al 1" and "Al1" cases uniformly, 
    
    # Pattern: [A-Z][a-z]* (Element) followed by [\d\.]* (Number)
    pattern = re.compile(r"([A-Z][a-z]*)([\d\.]*)")
    
    elements = {}
    matches = pattern.findall(formula)
    for element, amount_str in matches:
        if amount_str == '': amount = 1.0
        else:
            try: amount = float(amount_str)
            except ValueError: amount = 1.0 
        
        # safety check in case we have repeats
        if element in elements: elements[element] += amount
        else: elements[element] = amount
    return elements

df = pd.read_csv('MPEA_dataset.csv')

# Apply to all
composition_list = df['FORMULA'].apply(parse_formula).tolist()
comp_df = pd.DataFrame(composition_list).fillna(0.0)
row_sums = comp_df.sum(axis=1)
comp_df_normalized = comp_df.div(row_sums, axis=0)

print("\nShape of composition matrix:", comp_df_normalized.shape)
print("Elements found:", comp_df_normalized.columns.tolist())
print(comp_df_normalized.head())

property_cols = [c for c in df.columns if 'PROPERTY' in c]
meta_cols = [c for c in df.columns if 'IDENTIFIER' in c or 'REFERENCE' in c]

# Clean up property columns (force numeric)
cleaned_props = df[property_cols].copy()
numeric_prop_cols = [
    'PROPERTY: HV',
    'PROPERTY: YS (MPa)',
    'PROPERTY: UTS (MPa)', 
    'PROPERTY: Elongation (%)',
    'PROPERTY: Elongation plastic (%)',
    'PROPERTY: Exp. Young modulus (GPa)',
    'PROPERTY: Calculated Young modulus (GPa)']

for col in numeric_prop_cols:
    if col in cleaned_props.columns:
        cleaned_props[col] = pd.to_numeric(cleaned_props[col], errors='coerce')

final_df = pd.concat([comp_df_normalized, cleaned_props, df[meta_cols]], axis=1)
final_df.to_csv('MPEA_cleaned.csv')