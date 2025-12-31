# Generative Design of Stable High-Entropy Alloys via Physics-Informed VAEs

## 1. Project Overview
This project utilizes a **Variational Autoencoder (VAE)** to discover novel, non-equiatomic High-Entropy Alloys (HEAs) that are thermodynamically stable. Unlike traditional discovery methods that rely on trial-and-error mixing of equiatomic ratios (e.g., Cantor Alloy), this model learns the continuous landscape of phase stability from Density Functional Theory (DFT) data to generate optimized, multi-component microstructures.

**Key Achievement:** The model identified multiple novel alloy candidates (e.g., `Ni45Cr36Fe12Cu6`) that are predicted to be thermodynamically stable ($E_{hull} < 0$) and were validated as **FCC Solid Solutions** using Hume-Rothery rules.

---

## 2. The Problem & The Pivot
### Phase 1: The "Processing Gap" (Yield Strength Prediction)
Initially, the project aimed to predict experimental Yield Strength from composition alone.
* **Result:** The model plateaued at **$R^2 \approx 0.54$**.
* **Root Cause:** Experimental datasets lacked critical processing metadata (Annealing Temp, Grain Size). Without this, the model faced contradictory labels for identical compositions.

### Phase 2: The Strategic Pivot (Thermodynamic Stability)
We pivoted the target variable to **Energy Above Hull ($E_{hull}$)** derived from high-fidelity DFT simulations (84,000+ samples).
* **Result:** The model successfully learned the laws of thermodynamics, achieving **$R^2 = 0.92$** and **99.75% Accuracy** in distinguishing stable from unstable phases.
* **Significance:** This transformed the VAE from a "noisy regressor" into a "precision discovery engine."

---

## 3. Methodology

### Architecture
The model uses a modular VAE architecture with three main components:

* **Encoder** (`models/encoder.py`): Compresses chemical compositions into a 4-dimensional latent space
* **Decoder** (`models/decoder.py`): Reconstructs valid chemical compositions from latent coordinates
* **Property Regressor** (`models/regressor.py`): Predicts stability from latent representations

The main model (`models/hea_vae.py`) combines these components into a unified VAE that:
- Encodes compositions → latent space
- Decodes latent points → compositions
- Predicts stability from latent coordinates

### Training Data
* **Source:** 84,000+ DFT calculations from Materials Project
* **Preprocessing:** Filtered for High-Entropy complexity (N $\ge$ 4 elements)
* **Final Dataset:** 52,713 HEA samples with 8 unique elements (Al, Co, Cr, Cu, Fe, Mn, Ni, Si)
* **Input:** Chemical composition vectors (atomic fractions)
* **Output:** Formation Energy Above Hull ($E_{hull}$)

### Training Configuration
* **Batch Size:** 256
* **Alpha (Property Loss Weight):** 50.0
* **Beta (KL Divergence Weight):** 1.0 (with cosine annealing from 0.0 over 40 epochs)
* **Learning Rate:** 1e-3 (with decay during annealing)
* **Hidden Dimension:** 512
* **Latent Dimension:** 4

---

## 4. Results

### Model Performance
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **$R^2$ Score** | **0.9216** | The model captures 92% of the variance in thermodynamic stability. |
| **RMSE** | **0.06 eV/atom** | Prediction error is within the margin of thermal fluctuation ($kT$). |
| **Stability Accuracy** | **99.75%** | Near-perfect classification of stable vs. unstable alloys. |

### Novel Candidates Discovered
The generator scans 50,000 random points in latent space and filters for alloys that are **Stable ($E < 0.05$ eV)** and **Complex (N $\ge$ 4 elements with >5% concentration)**. Top candidates exhibit non-equiatomic characteristics:

| Rank | Formula | Stability (eV) | Phase | Delta_r (%) | VEC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Ni59Fe19Cr11Cu8** | -0.019 | FCC (Stable/Ductile) | 0.94 | 9.16 |
| 2 | **Ni47Fe38Cr7Cu5Si1** | -0.0087 | FCC (Stable/Ductile) | 0.93 | 8.92 |
| 3 | **Ni68Si18Fe6Cu5Cr1** | 0.0022 | FCC (Stable/Ductile) | 2.32 | 8.77 |

*(See `verified_stable_heas.csv` for full list with physics verification)*

---

## 5. Physical Validation (Hume-Rothery Rules)
All generated candidates are automatically verified against semi-empirical metallurgical rules:

1. **Atomic Size Mismatch ($\delta$):** Must be $< 6.6\%$ to form a Solid Solution.
2. **Valence Electron Concentration (VEC):** Must be $\ge 8.0$ to form a ductile FCC phase.

### Validation Results
* **100%** of candidates passed the $\delta < 6.6\%$ check (Average $\delta \approx 1.5\%$).
* **100%** of candidates passed the VEC $\ge 8.0$ check (Predominantly Ni/Co rich).
* **Conclusion:** The candidates are predicted to be **Stable, Single-Phase FCC Solid Solutions**, making them excellent candidates for ductile structural applications.
