# Adapting the Self-Optimized Spectral Distance (SSD) Framework for Broadband Coherent Anti-Stokes Raman Scattering (BCARS): Mathematical Foundations and Required Modifications

---

## 1. The Core Problem: Why SSD Cannot Be Applied to BCARS "As-Is"

The SSD algorithm was designed for spontaneous Raman hyperspectral imaging, where the physics is governed by a **linear additive observation model**:

$$Y = X + S \quad \quad (1)$$

where **Y** ∈ ℝ^{HW×B} is the raw noisy measurement, **X** is the desired clean Raman hyperspectral image, and **S** is composite additive noise (Poisson-Gaussian + cosmic rays). The critical assumption here is that the noise **S** is *additive* and *independent* of the signal **X**. The clean signal **X** is directly proportional to the imaginary part of the linear susceptibility, which produces positive Lorentzian peaks — the standard Raman fingerprint.

In BCARS, the physics is fundamentally different. The measured CARS intensity at each pixel and each spectral channel is:

$$I_{CARS}(\omega) \propto |\chi^{(3)}_{NR}(\omega) + \chi^{(3)}_{R}(\omega)|^2 \cdot I_p^2 \cdot I_S \quad \quad (2)$$

where χ^{(3)}\_{NR} is the nonresonant third-order susceptibility (real-valued, slowly varying) and χ^{(3)}\_{R} is the resonant susceptibility (complex-valued, containing the Raman information):

$$\chi^{(3)}_R(\omega) = \sum_m \frac{A_m}{\Omega_m - \omega - i\Gamma_m} \quad \quad (3)$$

Expanding the modulus-squared in Eq. (2):

$$I_{CARS}(\omega) \propto |\chi^{(3)}_R|^2 + |\chi^{(3)}_{NR}|^2 + 2\,\text{Re}\{\chi^{(3)}_R \cdot \chi^{(3)*}_{NR}\} \quad \quad (4)$$

This is a **nonlinear coherent mixing** — the NRB does not simply add to the Raman signal; it *multiplies* and *interferes* with it. The measured CARS spectrum therefore contains dispersive line shapes, peak shifts, and intensity distortions that bear no direct resemblance to the spontaneous Raman spectrum. The target quantity for chemical analysis is Im{χ^{(3)}\_{R}(ω)}, which mirrors the spontaneous Raman spectrum.

**The fundamental incompatibility**: SSD's observation model (Eq. 1) assumes Y = X + S. In BCARS, even a noise-free measurement (S = 0) is **not** the target Raman spectrum. Instead, Y\_CARS = f(χ\_R, χ\_{NR}) where f is a nonlinear function involving coherent mixing and modulus-squared detection. Simply denoising the BCARS hyperspectral cube with the original SSD yields a clean CARS image — but not a Raman image.

---

## 2. The Original SSD Mathematical Framework (Recap)

### 2.1 Optimization Problem

SSD solves:

$$(\hat{X}, \hat{S}) = \arg\min_{X,S} \frac{1}{2}\|Y - X - S\|_F^2 + \lambda R(X) + \lambda_S \|S\|_1 \quad \quad (5)$$

via ADMM with auxiliary variable **Z**:

$$(\hat{X}, \hat{Z}, \hat{S}) = \arg\min_{X,Z,S} \frac{1}{2}\|Y - X - S\|_F^2 + \lambda R(Z) + \lambda_S \|S\|_1, \quad \text{s.t. } X = Z \quad \quad (6)$$

The scaled augmented Lagrangian is:

$$\mathcal{L}_\rho(X, Z, S, U) = \frac{1}{2}\|Y - X - S\|_F^2 + \frac{\rho}{2}\|X - Z + U\|_F^2 + \lambda R(Z) + \lambda_S \|S\|_1 \quad \quad (7)$$

ADMM iterates over K outer iterations:

- **X-update**: X^k = (I + ρI)^{-1}[Y − S^{k-1} + ρ(Z^{k-1} − U^{k-1})]
- **Z-update**: Z^k via the SSD prior (the core innovation — see below)
- **S-update**: S^k = S\_{λ\_S}(Y − X^k) (soft-thresholding for sparse noise)
- **U-update**: U^k = U^{k-1} + (X^k − Z^k)

### 2.2 The SSD Prior (Z-Update)

**Definition (SSD Prior)**: The Raman hyperspectral image Z can be decomposed as:

$$Z = PI + OSD \quad \quad (8)$$

where:

- **PI** (Prior Image) is the band-averaged spatial image:

$$PI_t = \frac{1}{B}\sum_{b=1}^{B} Z_t(:, b) \quad \quad (9)$$

- **ISD** (Initial Spectral Distance) is the deviation of each band from PI:

$$ISD_t(:, b) = Z_t(:, b) - PI_t \quad \quad (10)$$

- **OSD** (Optimized Spectral Distance) is the UNN-refined version:

$$OSD_t = UNN(ISD_t) \quad \quad (11)$$

The UNN parameters W are optimized self-supervisedly:

$$W_t = \arg\min_W \frac{\rho}{2\lambda}\|(PI_t + UNN(ISD_t)) - X^k - U^{k-1}\|_F^2 + R(PI_t + UNN(ISD_t)) \quad \quad (12)$$

with the regularization:

$$R(Z_{t+1}) = \|Y - Z_{t+1}\|_1 + \lambda_R \|Z_{t+1}\|_{SSTV} \quad \quad (13)$$

### 2.3 Key Assumptions in the Original SSD That Break for BCARS

| SSD Assumption | Spontaneous Raman | BCARS |
|---|---|---|
| Y = X + S (additive noise model) | ✅ Valid | ❌ Fails — coherent mixing |
| PI (band mean) ≈ smooth spatial structure | ✅ Mean of Raman bands = useful morphology | ⚠️ Mean of CARS bands ≈ NRB-dominated profile |
| Spectral distance captures band-to-band variation | ✅ Positive Raman peaks vary linearly | ❌ Dispersive line shapes, phase information lost |
| ‖S‖₁ promotes sparse noise | ✅ Cosmic rays, shot noise | ⚠️ Still applicable for additive detector noise |
| SSTV enforces smooth spatial-spectral structure | ✅ Raman features are spatially smooth | ⚠️ CARS spectral structure is not simply smooth — dispersive features have sign changes |

---

## 3. Strategy for Adaptation: The BCARS-SSD Framework

I propose a two-stage architecture, each requiring specific mathematical modifications to the SSD framework.

### Strategy Overview

**Stage A** — "BCARS-SSD-Denoise": Use a minimally-modified SSD to denoise the raw BCARS hyperspectral cube (replace SVD in the standard BCARS pipeline). This requires adapting the PI and regularization terms.

**Stage B** — "BCARS-SSD-Retrieve": Replace the observation model entirely to perform joint denoising + NRB removal (phase retrieval) within a single spatially-aware optimization. This is the transformative contribution.

---

## 4. Stage A: SSD for BCARS Denoising (Minimal Modifications)

### 4.1 Modified Observation Model

The observation model remains additive but now explicitly describes CARS intensity:

$$Y_{CARS} = I_{CARS} + S \quad \quad (14)$$

where I\_{CARS} ∈ ℝ^{HW×B} is the clean (noise-free) CARS intensity image and S is detector noise. The target is X = I\_{CARS} (a clean CARS image, not yet a Raman image).

The optimization problem (Eq. 5) applies directly with X → I\_{CARS}.

### 4.2 Modified Prior Image

**Problem**: In spontaneous Raman, the band-mean PI captures useful spatial morphology because all Raman bands are positive and structurally correlated. In BCARS, the band-mean is heavily dominated by the NRB envelope, which varies smoothly across the spectral axis but can have very different spatial contrast than the resonant features.

**Modification 1 — Weighted Prior Image**: Instead of a uniform band average, use a spectrally-weighted average that emphasizes bands with high resonant content:

$$PI_t^{(w)} = \frac{1}{\sum_b w_b}\sum_{b=1}^{B} w_b \cdot Z_t(:, b) \quad \quad (15)$$

where the weights w\_b can be:
- Estimated from the spectral variance across pixels (bands with high inter-pixel variance likely contain resonant information)
- Set by a preliminary NRB estimate (downweighting bands dominated by NRB)
- Learned as additional parameters within the optimization

**Modification 2 — Median Prior Image**: Replace the mean with a spectral median to reduce the influence of the NRB-dominated baseline:

$$PI_t^{(med)}(:) = \text{median}_b\{Z_t(:, b)\} \quad \quad (16)$$

### 4.3 Modified Regularization

**Problem**: The SSTV regularization ‖Z‖\_{SSTV} penalizes spectral gradients. In CARS spectra, dispersive line shapes inherently have sharp spectral gradients (sign changes from positive to negative lobes). Over-smoothing these would destroy the phase information needed for subsequent NRB removal.

**Modification 3 — Anisotropic SSTV with Spectral Awareness**:

$$R(Z) = \|Y - Z\|_1 + \lambda_{TV}\|Z\|_{TV} + \lambda_{STV}\|Z\|_{STV}^{(adapt)} \quad \quad (17)$$

where the spectral TV term uses an adaptive weight that reduces penalization near spectral features:

$$\|Z\|_{STV}^{(adapt)} = \sum_{b=1}^{B-1} \alpha_b \|Z(:, b+1) - Z(:, b)\| \quad \quad (18)$$

and α\_b is small near known Raman peak positions (where spectral gradients are expected) and large in featureless regions.

### 4.4 Complete Stage-A Algorithm

The ADMM iterations proceed identically to the original SSD, with:
1. PI computed via Eq. (15) or (16) instead of Eq. (9)
2. Regularization via Eq. (17-18) instead of Eq. (13)
3. UNN architecture unchanged — it learns to map ISD → OSD in the CARS intensity domain
4. Output: A denoised CARS hyperspectral cube, which then requires a **secondary** NRB removal step (TDKK, MEM, or a DL model like VECTOR/GAN)

---

## 5. Stage B: Joint Denoising and NRB Removal — The Full BCARS-SSD Framework

This is the core theoretical contribution. The goal is to embed the CARS physics directly into the SSD optimization, so that the algorithm simultaneously denoises and performs phase retrieval in a spatially-aware, self-supervised manner.

### 5.1 The New Observation Model

We must replace Y = X + S with the actual BCARS forward model. Define:

- **χ\_R** ∈ ℂ^{HW×B} — the complex resonant susceptibility at each pixel and spectral channel (the target)
- **χ\_{NR}** ∈ ℝ^{HW×B} — the nonresonant susceptibility (real-valued, to be estimated or provided)
- **E(ω)** ∈ ℝ^{1×B} — the excitation spectral profile (known from calibration)

The noise-free CARS measurement is:

$$I_{CARS}(\omega) = E(\omega) \cdot |\chi_{NR}(\omega) + \chi_R(\omega)|^2 \quad \quad (19)$$

With additive detector noise:

$$Y_{CARS} = I_{CARS} + S = E \odot |\chi_{NR} + \chi_R|^2 + S \quad \quad (20)$$

where ⊙ denotes element-wise multiplication. The target is to recover Im{χ\_R}, which is the Raman-equivalent spectrum.

### 5.2 Reformulated Optimization Problem

Replace the original SSD optimization (Eq. 5) with:

$$(\hat{\chi}_R, \hat{\chi}_{NR}, \hat{S}) = \arg\min_{\chi_R, \chi_{NR}, S} \frac{1}{2}\|Y_{CARS} - E \odot |\chi_{NR} + \chi_R|^2 - S\|_F^2 + \lambda R(\text{Im}\{\chi_R\}) + \lambda_S \|S\|_1 + \lambda_{NR} R_{NR}(\chi_{NR}) \quad \quad (21)$$

The key differences from the original Eq. (5) are:

1. **The data fidelity term** is now nonlinear: the forward model involves a modulus-squared operation
2. **The regularization R(·)** is applied to Im{χ\_R}, which should look like a clean Raman hyperspectral image
3. **A new NRB regularization** R\_{NR}(χ\_{NR}) enforces physically-motivated constraints on the nonresonant background

### 5.3 Physics-Informed NRB Regularization

The NRB has well-known physical properties that can be encoded as priors:

$$R_{NR}(\chi_{NR}) = \lambda_{smooth}\|\chi_{NR}\|_{STV} + \lambda_{real}\|\text{Im}\{\chi_{NR}\}\|_F^2 \quad \quad (22)$$

The first term enforces spectral smoothness of the NRB (it varies slowly across the spectral axis). The second term penalizes any imaginary component, since χ\_{NR} is real-valued away from electronic resonances. In practice, if the NRB spectral profile is measured from a reference sample (glass, water), χ\_{NR} can be further constrained:

$$R_{NR}(\chi_{NR}) = \lambda_{ref}\|\chi_{NR} - \chi_{NR}^{(ref)}\|_F^2 \quad \quad (23)$$

where χ\_{NR}^{(ref)} is the measured reference, allowing pixel-to-pixel variation.

### 5.4 Modified ADMM Splitting for the Nonlinear Forward Model

The nonlinear data fidelity term makes the original ADMM splitting non-trivial. We introduce auxiliary variables to linearize the problem.

**Step 1 — Variable Splitting**: Introduce auxiliary variables:
- **P** = χ\_{NR} + χ\_R (total susceptibility)
- **Q** = |P|² (intensity before excitation profile)

$$\min_{\chi_R, \chi_{NR}, P, Q, S} \frac{1}{2}\|Y_{CARS} - E \odot Q - S\|_F^2 + \lambda R(\text{Im}\{\chi_R\}) + \lambda_S\|S\|_1 + \lambda_{NR}R_{NR}(\chi_{NR}) \quad \quad (24)$$
$$\text{s.t.} \quad P = \chi_{NR} + \chi_R, \quad Q = |P|^2$$

**Step 2 — Augmented Lagrangian**:

$$\mathcal{L} = \frac{1}{2}\|Y_{CARS} - E \odot Q - S\|_F^2 + \frac{\rho_1}{2}\|P - \chi_{NR} - \chi_R + U_1\|_F^2 + \frac{\rho_2}{2}\|Q - |P|^2 + U_2\|_F^2 + \lambda R(\text{Im}\{\chi_R\}) + \lambda_S\|S\|_1 + \lambda_{NR}R_{NR}(\chi_{NR}) \quad \quad (25)$$

**Step 3 — ADMM Sub-problems**:

**(a) Q-update** (linear in Q):

$$Q^{k+1} = \arg\min_Q \frac{1}{2}\|Y_{CARS} - E \odot Q - S^k\|_F^2 + \frac{\rho_2}{2}\|Q - |P^k|^2 + U_2^k\|_F^2 \quad \quad (26)$$

This has a closed-form, element-wise solution:

$$Q^{k+1} = \frac{E \odot (Y_{CARS} - S^k) + \rho_2(|P^k|^2 - U_2^k)}{E \odot E + \rho_2} \quad \quad (27)$$

**(b) P-update** (involves modulus-squared constraint):

$$P^{k+1} = \arg\min_P \frac{\rho_1}{2}\|P - \chi_{NR}^k - \chi_R^k + U_1^k\|_F^2 + \frac{\rho_2}{2}\|Q^{k+1} - |P|^2 + U_2^k\|_F^2 \quad \quad (28)$$

This sub-problem involves a quartic term due to |P|². It can be solved iteratively via gradient descent or via a proximal linearization where |P|² is linearized around the current estimate:

$$|P|^2 \approx |P^k|^2 + 2\,\text{Re}\{(P^k)^* \cdot (P - P^k)\} \quad \quad (29)$$

yielding a quadratic sub-problem with closed-form solution.

**(c) χ\_R-update** — This is where the SSD prior is applied:

$$\chi_R^{k+1} = \arg\min_{\chi_R} \frac{\rho_1}{2}\|P^{k+1} - \chi_{NR}^k - \chi_R + U_1^k\|_F^2 + \lambda R(\text{Im}\{\chi_R\}) \quad \quad (30)$$

### 5.5 The Adapted SSD Prior for χ\_R

This is the crucial adaptation. The SSD prior is now applied to the *imaginary part* of χ\_R, which should resemble a clean Raman hyperspectral image.

**Definition (BCARS-SSD Prior)**: The resonant susceptibility image can be decomposed as:

$$\text{Im}\{\chi_R\} = PI_R + OSD_R \quad \quad (31)$$

where:

$$PI_R^t = \frac{1}{B}\sum_{b=1}^{B} \text{Im}\{\chi_R^t(:, b)\} \quad \quad (32)$$

$$ISD_R^t(:, b) = \text{Im}\{\chi_R^t(:, b)\} - PI_R^t \quad \quad (33)$$

$$OSD_R^t = UNN_R(ISD_R^t) \quad \quad (34)$$

The real part of χ\_R is not independently parameterized — it is determined by the Kramers-Kronig relation from the imaginary part. This is a powerful physics-informed constraint:

$$\text{Re}\{\chi_R(\omega)\} = \frac{1}{\pi}\mathcal{P}\int_{-\infty}^{\infty}\frac{\text{Im}\{\chi_R(\omega')\}}{\omega' - \omega}d\omega' \quad \quad (35)$$

which can be implemented numerically as a Hilbert transform:

$$\text{Re}\{\chi_R\} = \mathcal{H}[\text{Im}\{\chi_R\}] \quad \quad (36)$$

This means the UNN only needs to generate OSD\_R for the imaginary part, and the real part is computed analytically. The complete χ\_R is then:

$$\chi_R = \mathcal{H}[\text{Im}\{\chi_R\}] + i \cdot \text{Im}\{\chi_R\} = \mathcal{H}[PI_R + OSD_R] + i(PI_R + OSD_R) \quad \quad (37)$$

### 5.6 Modified UNN Objective Function

The UNN parameters are updated self-supervisedly within each ADMM iteration:

$$W_t = \arg\min_W \frac{\rho_1}{2\lambda}\|P^{k+1} - \chi_{NR}^k - \chi_R^{(t+1)} + U_1^k\|_F^2 + R_{Raman}(\text{Im}\{\chi_R^{(t+1)}\}) \quad \quad (38)$$

where χ\_R^{(t+1)} is constructed from UNN output via Eq. (37), and:

$$R_{Raman}(\text{Im}\{\chi_R\}) = \|Y_{CARS} - E \odot |\chi_{NR}^k + \chi_R^{(t+1)}|^2\|_1 + \lambda_R\|\text{Im}\{\chi_R^{(t+1)}\}\|_{SSTV} \quad \quad (39)$$

The first term is a data-driven prior ensuring the reconstructed susceptibility is consistent with the measured CARS data. The second term is the spectral-spatial total variation on the *Raman-like* output.

**Critical difference from original SSD Eq. (12)**: The data fidelity term now passes through the nonlinear CARS forward model rather than being a simple ‖Y − Z‖₁.

**(d) χ\_{NR}-update**:

$$\chi_{NR}^{k+1} = \arg\min_{\chi_{NR}} \frac{\rho_1}{2}\|P^{k+1} - \chi_{NR} - \chi_R^{k+1} + U_1^k\|_F^2 + \lambda_{NR}R_{NR}(\chi_{NR}) \quad \quad (40)$$

With the smoothness prior on χ\_{NR}, this is a standard TV-regularized least squares problem.

**(e) S-update**: Identical to original SSD:

$$S^{k+1} = \mathcal{S}_{\lambda_S}(Y_{CARS} - E \odot |P^{k+1}|^2) \quad \quad (41)$$

**(f) Dual variable updates**:

$$U_1^{k+1} = U_1^k + (P^{k+1} - \chi_{NR}^{k+1} - \chi_R^{k+1}) \quad \quad (42)$$
$$U_2^{k+1} = U_2^k + (Q^{k+1} - |P^{k+1}|^2) \quad \quad (43)$$

---

## 6. Summary of All Required Modifications

### 6.1 Observation Model (Most Critical)

| Component | Original SSD | BCARS-SSD Stage A | BCARS-SSD Stage B |
|---|---|---|---|
| Forward model | Y = X + S | Y = I\_CARS + S (unchanged form) | Y = E⊙\|χ\_NR + χ\_R\|² + S |
| Target variable | X (clean Raman image) | X (clean CARS image) | Im{χ\_R} (Raman-equivalent) |
| Data fidelity | ‖Y − X − S‖²\_F | ‖Y − X − S‖²\_F | ‖Y − E⊙\|χ\_NR + χ\_R\|² − S‖²\_F |

### 6.2 Prior Image Construction

| | Original SSD | BCARS-SSD |
|---|---|---|
| Definition | PI = (1/B)Σ Z(:,b) | PI\_R = (1/B)Σ Im{χ\_R(:,b)} or weighted variant |
| Domain | Raman intensity (positive) | Raman-equivalent (positive Lorentzian peaks) |
| Physical meaning | Average morphology | Average Raman-like morphology |

### 6.3 Spectral Distance & UNN

| | Original SSD | BCARS-SSD |
|---|---|---|
| ISD input | Z(:,b) − PI | Im{χ\_R(:,b)} − PI\_R |
| UNN output | OSD (intensity domain) | OSD\_R (Raman imaginary domain) |
| Reconstruction | Z = PI + OSD | χ\_R = H[PI\_R + OSD\_R] + i(PI\_R + OSD\_R) |

### 6.4 Regularization

| | Original SSD | BCARS-SSD |
|---|---|---|
| Data prior | ‖Y − Z‖₁ | ‖Y − E⊙\|χ\_NR + χ\_R\|²‖₁ |
| Spatial-spectral TV | ‖Z‖\_SSTV | ‖Im{χ\_R}‖\_SSTV |
| Additional | — | R\_NR(χ\_NR): smoothness + real-valuedness |

### 6.5 ADMM Variables

| | Original SSD | BCARS-SSD |
|---|---|---|
| Primal variables | X, Z, S | χ\_R, χ\_NR, P, Q, S |
| Dual variables | U | U₁, U₂ |
| Penalty parameters | ρ | ρ₁, ρ₂ |

---

## 7. UNN Architecture Modifications

### 7.1 Input/Output Channels

The original UNN takes ISD ∈ ℝ^{HW×B} and outputs OSD ∈ ℝ^{HW×B}. For BCARS-SSD Stage B:

- **Input**: ISD\_R ∈ ℝ^{HW×B} (spectral distance in the Raman imaginary domain)
- **Output**: OSD\_R ∈ ℝ^{HW×B} (optimized spectral distance in Raman imaginary domain)
- **Post-processing**: Hilbert transform to construct full complex χ\_R

The network dimensions are unchanged (same spatial and spectral dimensions), but the *domain* of the data is different.

### 7.2 SDTrans Module Adaptation

The Spectral Distance Transformer (SDTrans) captures long-range spectral and spatial correlations. In BCARS:

- Spectral correlations are **stronger** because the Kramers-Kronig relation imposes a mathematical relationship between all spectral channels
- The SDTrans attention mechanism should naturally capture these correlations, but its effectiveness can be enhanced by:
  - Initializing attention weights with the Hilbert transform kernel structure
  - Adding a "KK-consistency" loss term that penalizes deviations between Re{χ\_R} and H[Im{χ\_R}]

### 7.3 Spatial-Aware Attention (SAA) Adaptation

The SAA mechanism remains largely unchanged because spatial correlations in BCARS images are governed by the same physical principles (adjacent pixels have similar chemistry). However, in BCARS microscopy the spatial sampling is often coarser and faster than in spontaneous Raman, making the SAA's ability to exploit spatial correlations even more valuable.

---

## 8. Initialization Strategy

### 8.1 For Stage A (BCARS Denoising)

Initialize Z⁰ = Y\_CARS (identical to original SSD).

### 8.2 For Stage B (Joint Denoising + NRB Removal)

Initialization is critical for convergence of the nonlinear problem:

1. **χ\_{NR}⁰**: Initialize from a reference NRB measurement (glass/water), or from a low-pass filtered version of the mean CARS spectrum across all pixels
2. **χ\_R⁰**: Initialize from a preliminary TDKK or MEM phase retrieval on the raw (noisy) data — this gives a rough starting point even if noisy
3. **P⁰**: χ\_{NR}⁰ + χ\_R⁰
4. **Q⁰**: |P⁰|²
5. **S⁰**: 0 (no noise estimate initially)
6. **U₁⁰, U₂⁰**: 0

---

## 9. Computational Considerations

### 9.1 Hilbert Transform in the Loop

Each UNN forward pass requires computing the Hilbert transform (Eq. 36) to construct Re{χ\_R} from the UNN's prediction of Im{χ\_R}. The Hilbert transform is computed via FFT:

$$\mathcal{H}[f] = \mathcal{F}^{-1}[-i \cdot \text{sgn}(\omega) \cdot \mathcal{F}[f]] \quad \quad (44)$$

This is O(B log B) per pixel and is fully parallelizable across all HW pixels. In PyTorch/JAX, this is efficiently implemented using `torch.fft.fft` and is differentiable, so gradients can flow through it during backpropagation.

### 9.2 Complexity Comparison

| | Original SSD | BCARS-SSD Stage B |
|---|---|---|
| ADMM variables | 4 (X, Z, S, U) | 7 (χ\_R, χ\_NR, P, Q, S, U₁, U₂) |
| Closed-form sub-problems | X, S, U | Q, S, U₁, U₂ |
| Iterative sub-problems | Z (UNN optimization) | P (linearized), χ\_R (UNN + Hilbert), χ\_NR (TV-regularized) |
| Per-UNN-epoch cost | O(HW·B) | O(HW·B·log B) due to Hilbert transform |

The additional cost is modest — dominated by the Hilbert transform which is an FFT operation.

---

## 10. Expected Advantages Over Current BCARS Methods

### 10.1 Versus Pixel-wise Deep Learning (SpecNet, VECTOR, LSTM, GAN)

Current BCARS deep learning methods process spectra **pixel-by-pixel** as isolated 1D sequences. They ignore spatial correlations entirely. BCARS-SSD would be the first method to perform **spatially-aware NRB removal**, exploiting the fact that neighboring pixels share similar chemistry and therefore similar χ\_R and χ\_{NR}.

This is analogous to the demonstrated superiority of SSD over per-spectrum denoising in spontaneous Raman: removing spatial TV constraints severely degrades morphological accuracy.

### 10.2 Versus KK/MEM

Classical phase retrieval methods (TDKK, MEM) are also pixel-wise and require a separately measured NRB reference. BCARS-SSD would jointly estimate χ\_{NR} from the data while exploiting spatial smoothness of the NRB — eliminating the need for a reference measurement.

### 10.3 Self-Supervised Advantage

Like the original SSD, BCARS-SSD requires **no training data**. It optimizes directly on the measured hyperspectral cube. This eliminates the training-testing domain gap that plagues supervised methods (which are trained on synthetic data and may fail on experimental data with different NRB profiles or noise characteristics).

---

## 11. Open Questions and Challenges

1. **Convergence of the nonlinear ADMM**: The modulus-squared nonlinearity in the forward model makes the overall problem non-convex. Convergence guarantees from convex ADMM do not apply. Empirical convergence analysis and careful selection of penalty parameters (ρ₁, ρ₂) will be essential.

2. **Identifiability**: Can χ\_R and χ\_NR be simultaneously recovered from I\_CARS without additional constraints? The answer depends on the spectral bandwidth and the relative strength of the NRB. The Kramers-Kronig constraint on χ\_R and the smoothness constraint on χ\_NR provide the necessary regularization, but this should be verified on simulated data.

3. **Excitation profile E(ω)**: In practice, the excitation spectral profile varies across the field of view and drifts over time. It may need to be treated as another variable to be estimated.

4. **Complex-valued UNN**: While the UNN only outputs Im{χ\_R} (real-valued), the loss function involves the complex χ\_R and the nonlinear forward model. The gradient computation through the Hilbert transform must be carefully implemented.

5. **Hyperparameter selection**: The BCARS-SSD Stage B has more hyperparameters than the original SSD (ρ₁, ρ₂, λ, λ\_S, λ\_{NR}, λ\_R, λ\_{smooth}, λ\_{real}). Automated hyperparameter selection strategies will be important.

---

## 12. Recommended Implementation Roadmap

**Phase 1 — Validation on synthetic data**: Generate BCARS hyperspectral cubes using the known forward model (Eq. 2-4) with ground-truth χ\_R and χ\_{NR}. Test Stage A (denoising only) and compare with SVD. Test Stage B (joint retrieval) and compare with pixel-wise TDKK/MEM and deep learning methods (VECTOR, GAN).

**Phase 2 — Ablation studies**: Systematically evaluate the contribution of each modification: (a) weighted PI vs. standard PI, (b) adaptive SSTV vs. standard SSTV, (c) Hilbert transform constraint vs. unconstrained real/imaginary parts, (d) joint χ\_{NR} estimation vs. fixed reference.

**Phase 3 — Experimental validation**: Apply to real BCARS microscopy data (tissue, cells) and benchmark against the standard pipeline (SVD → TDKK → unmixing) and deep learning pipelines.

---

*This analysis identifies the complete set of mathematical modifications required to adapt SSD for BCARS, from the observation model through the optimization algorithm to the network architecture. The most transformative element is embedding the CARS forward model and the Kramers-Kronig constraint into the SSD optimization framework, enabling spatially-aware, self-supervised phase retrieval — a capability that does not exist in any current BCARS processing method.*
