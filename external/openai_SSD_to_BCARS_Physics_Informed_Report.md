# Adapting Self‑Optimized Spectral Distance (SSD) to Broadband CARS (BCARS)

**Purpose of this report**  
You asked how SSD (self‑optimized spectral distance) can be applied to BCARS, *mathematically*, and what must be modified. This report provides:

- A **physics-grounded forward model** for BCARS that replaces SSD’s additive-noise observation model.
- A **concrete set of modifications** to SSD’s prior-image + spectral-distance framework so it can work with BCARS coherent mixing.
- A **fully specified optimization objective** (including variables, constraints, regularizers, and an SSD-style untrained neural network parameterization).
- A practical optimization plan and alternative “SSD as denoiser” pipeline.

---

## 1) Key difference: Raman vs BCARS observation models

### 1.1 SSD (spontaneous Raman) assumes an additive measurement model
SSD (as proposed for low-light Raman hyperspectral imaging) starts from an additive decomposition:

\[
Y = X + S,
\]

where:

- \(Y\) = measured hyperspectral cube (noisy Raman)
- \(X\) = unknown clean hyperspectral cube
- \(S\) = composite noise term (shot + detector + other artefacts)

The SSD prior is expressed via a **prior image** (PI) and a **spectral distance** (SD) for each band relative to the PI. The SD is predicted/refined by an **untrained neural network (UNN)** with spatial–spectral attention.

> Why this matters: this model is linear in the unknown \(X\).  

### 1.2 BCARS measures *coherent intensity*, not an additive Raman signal
In BCARS (and CARS generally), what you measure is intensity proportional to the **squared modulus** of a complex third-order susceptibility:

\[
I_{\mathrm{CARS}}(\omega,\mathbf{r}) \propto \left|\chi^{(3)}_{\mathrm{NR}}(\omega,\mathbf{r}) + \chi^{(3)}_{\mathrm{R}}(\omega,\mathbf{r})\right|^2,
\]

with:

- \(\chi^{(3)}_{\mathrm{NR}}\): non-resonant background (NRB), often treated as **real** and slowly varying
- \(\chi^{(3)}_{\mathrm{R}}\): resonant susceptibility (complex), containing the vibrational information

Expanding (assuming \(\chi_{\mathrm{NR}}\) is real):

\[
I_{\mathrm{CARS}} \propto 
\underbrace{|\chi_{\mathrm{NR}}|^2}_{\text{large smooth term}}
+
\underbrace{|\chi_{\mathrm{R}}|^2}_{\text{small term}}
+
\underbrace{2 \chi_{\mathrm{NR}}\Re[\chi_{\mathrm{R}}]}_{\text{heterodyne interference term}}.
\]

> Why this matters: the “background” is not simply added; it **mixes coherently** and changes line shapes.

---

## 2) What breaks if you run “plain SSD” directly on raw BCARS

SSD’s SD uses a **prior image**, typically the mean across spectral bands:

\[
\mathrm{PI}(\mathbf{r}) = \frac{1}{B}\sum_{b=1}^B Y(\omega_b,\mathbf{r}),
\qquad
\mathrm{SD}(\omega_b,\mathbf{r}) = Y(\omega_b,\mathbf{r}) - \mathrm{PI}(\mathbf{r}) \;\; (\text{or a variant}).
\]

In BCARS:

- The mean/PI is dominated by \(|\chi_{\mathrm{NR}}|^2\) and instrument response.
- The informative term is mainly the **interference** \(2\chi_{\mathrm{NR}}\Re[\chi_{\mathrm{R}}]\), which can be positive or negative *relative to* the NRB-dominated baseline.
- Therefore, a mean-based PI is **not a neutral reference** like a baseline in spontaneous Raman; it is part of the coherent mixing physics.

**Conclusion:** “vanilla SSD” can be excellent for denoising *intensity* BCARS data, but **cannot, by itself, yield Raman-like spectra** (e.g., \(\Im[\chi_{\mathrm{R}}]\)) without changing the forward model.

---

## 3) Two ways to apply SSD to BCARS

### 3.1 Option A — SSD strictly as a denoiser (recommended baseline pipeline)
1. Treat raw BCARS cube \(Y\) as “noisy intensity cube”.
2. Use SSD to reconstruct \(X\approx\) denoised intensity cube.
3. Apply a separate NRB removal / phase retrieval:
   - KK / time-domain KK (TDKK)
   - MEM
   - or a trained model (SpecNet, VECTOR, LSTM/BiLSTM, GAN variants)

**Mathematics:** SSD stays unchanged; it solves only \(Y \approx X + S\).  
**What you gain:** strong spatially consistent denoising (often better than SVD).

### 3.2 Option B — Physics-informed SSD for *joint* NRB removal + Raman retrieval
You **redefine** the SSD objective so that the unknowns are not “clean intensity \(X\)” but instead the **latent physical quantities** \(\chi_{\mathrm{NR}}\) and \(\chi_{\mathrm{R}}\) (or equivalent parameterizations).

This is the core of “SSD adapted to BCARS”.

---

## 4) BCARS physics-informed parameterization for inversion

### 4.1 Choose the latent variables you want
A convenient representation:

- \(n(\omega,\mathbf{r}) := \chi^{(3)}_{\mathrm{NR}}(\omega,\mathbf{r}) \in \mathbb{R}_{\ge 0}\) (NRB amplitude; real)
- \(r(\omega,\mathbf{r}) := \Im[\chi^{(3)}_{\mathrm{R}}(\omega,\mathbf{r})]\) (Raman-like absorptive spectrum)

Then enforce causality/analyticity (KK relation) by linking the real part:

\[
\Re[\chi^{(3)}_{\mathrm{R}}](\omega,\mathbf{r}) = \mathcal{H}_\omega\{ r(\omega,\mathbf{r}) \} + c(\mathbf{r}),
\]

where:

- \(\mathcal{H}_\omega\{\cdot\}\) is a Hilbert transform along the spectral axis (in wavenumber/frequency domain),
- \(c(\mathbf{r})\) is an (optional) constant/slowly varying detrending term per pixel.

Define:

\[
h(\omega,\mathbf{r}) := \mathcal{H}_\omega\{ r(\omega,\mathbf{r}) \}.
\]

Then a simplified forward model (up to known excitation scaling) becomes:

\[
\widehat{I}(\omega,\mathbf{r})
=
\alpha(\omega)\left( \left[n(\omega,\mathbf{r}) + h(\omega,\mathbf{r})\right]^2 + r(\omega,\mathbf{r})^2 \right)
+ b(\omega),
\]

with:

- \(\alpha(\omega)\ge 0\): instrument stimulation / detection scaling (known if calibrated; otherwise estimate)
- \(b(\omega)\): additive offset / dark current baseline (optional; can be a scalar or per-band)

> This is already far more BCARS-faithful than \(Y=X+S\).

---

## 5) How to modify SSD’s PI/SD concept for BCARS

SSD’s strength is the *representation + self-supervised optimization*:

- decompose a cube into PI + SD,
- predict SD with a spatial–spectral UNN,
- iterate with an ADMM-like procedure.

For BCARS you can keep the **mechanism**, but you must move it to the **latent fields** \(n(\omega,\mathbf{r})\) and \(r(\omega,\mathbf{r})\), not the measured intensity \(I\).

### 5.1 Define prior images (PI) that make physical sense

**NRB prior image**
Because \(n(\omega,\mathbf{r})\) should be smooth and slowly varying, define:

\[
\mathrm{PI}_n(\mathbf{r}) := \operatorname{mean}_\omega \left( n(\omega,\mathbf{r}) \right)
\quad \text{or} \quad
\mathrm{PI}_n(\mathbf{r}) := \operatorname{median}_\omega \left( n(\omega,\mathbf{r}) \right).
\]

**Raman prior image**
For \(r(\omega,\mathbf{r})\), a better prior is often simply:

\[
\mathrm{PI}_r(\mathbf{r}) := 0,
\]

because the Raman imaginary part is peak-like; its mean across a wide range is close to zero compared to NRB.

### 5.2 Define spectral distances on the latent fields
\[
\mathrm{SD}_n(\omega,\mathbf{r}) := n(\omega,\mathbf{r}) - \mathrm{PI}_n(\mathbf{r}),
\qquad
\mathrm{SD}_r(\omega,\mathbf{r}) := r(\omega,\mathbf{r}) - \mathrm{PI}_r(\mathbf{r}) = r(\omega,\mathbf{r}).
\]

SSD’s UNN then predicts **optimized** SDs \(\mathrm{OSD}_n, \mathrm{OSD}_r\) that are spatially and spectrally coherent.

---

## 6) Fully specified objective: Physics-informed BCARS‑SSD

Below is one complete, implementable objective that:

- respects coherent mixing (BCARS physics),
- uses KK/Hilbert structure for \(\Re[\chi_R]\),
- keeps SSD’s “PI + SD + UNN” idea,
- adds explicit priors tailored to \(n\) (smooth) and \(r\) (peaky).

### 6.1 Data, dimensions, and operators
- Observed cube: \(Y \in \mathbb{R}^{B\times H\times W}\) (BCARS intensity)
- Spectral index: \(b \in \{1,\dots,B\}\), spatial pixel: \(\mathbf{r}\in\Omega\) (2D grid)
- Unknown NRB field: \(n \in \mathbb{R}^{B\times H\times W}\)
- Unknown Raman imag field: \(r \in \mathbb{R}^{B\times H\times W}\)
- Hilbert transform along spectral axis: \(h = \mathcal{H}_\omega\{r\}\)
- Spatial gradient: \(\nabla_{xy}\) and TV functional:
  \[
  \mathrm{TV}_{xy}(u)=\sum_{b}\sum_{\mathbf{r}}\sqrt{\|\nabla_{xy}u(b,\mathbf{r})\|_2^2+\varepsilon}.
  \]
- Spectral second derivative operator \(D^2_\omega\) (finite differences).

### 6.2 UNN (SSD-style) parameterization
Let \(G_n(\cdot;W_n)\) and \(G_r(\cdot;W_r)\) be two SSD-style UNNs (U-Net-like + SD transformer + spatial-aware attention), each taking an **initial spectral distance** (ISD) tensor and outputting an **optimized spectral distance** (OSD).

Construct latent fields as:

\[
n = \mathrm{PI}_n + G_n(\mathrm{ISD}_n; W_n),
\qquad
r = \mathrm{PI}_r + G_r(\mathrm{ISD}_r; W_r).
\]

Practical ISD choices (from raw \(Y\), no external NRB measurement required):

- A robust “NRB-dominated” amplitude surrogate:
  \[
  a(\omega,\mathbf{r}) := \sqrt{\max(Y(\omega,\mathbf{r})-b_0,0)}.
  \]
  Then \(\mathrm{ISD}_n := a - \operatorname{median}_\omega(a)\).

- A “high-pass / interference” surrogate for Raman:
  \[
  \mathrm{ISD}_r := \frac{Y - \operatorname{LP}_\omega(Y)}{\operatorname{LP}_\omega(Y)+\delta},
  \]
  where \(\operatorname{LP}_\omega\) is a low-pass filter along \(\omega\).

(These are initializations/features; the objective below is what “forces” physical consistency.)

### 6.3 Forward model
Predict intensity:

\[
\widehat{Y}(n,r;\alpha,b) = 
\alpha(\omega)\left(\left[n + \mathcal{H}_\omega\{r\}\right]^2 + r^2\right) + b(\omega).
\]

We allow \(\alpha(\omega)\) and \(b(\omega)\) to be either:
- fixed (from calibration), or
- estimated jointly with weak smoothness priors.

### 6.4 Noise / data fidelity term
A simple, robust choice:

\[
\mathcal{D}(Y,\widehat{Y})=
\frac{1}{2}\left\| W \odot \left(Y-\widehat{Y}\right)\right\|_F^2,
\]

where \(W\) is a per-band or per-pixel weight (e.g., inverse-variance estimate).  
(If you want a Poisson–Gaussian model, replace this with an appropriate negative log-likelihood or an Anscombe-transformed least squares.)

### 6.5 Regularizers (priors)
**NRB prior (smooth in ω and spatially):**
\[
\mathcal{R}_n(n)=
\lambda_{n,\omega}\|D^2_\omega n\|_2^2 + \lambda_{n,xy}\mathrm{TV}_{xy}(n).
\]

**Raman prior (peak-like in ω, spatially consistent):**
\[
\mathcal{R}_r(r)=
\lambda_{r,\omega}\|D^2_\omega r\|_1 + \lambda_{r,xy}\mathrm{TV}_{xy}(r).
\]

(Using an \(\ell_1\) penalty on spectral curvature encourages a sparse set of sharp features—peaks—while allowing spatial coherence through TV.)

**Optional detrending prior** (for KK constant term):
\[
\mathcal{R}_c(c)=\lambda_c \mathrm{TV}_{xy}(c).
\]

### 6.6 Constraints
\[
n \ge 0,\quad \alpha(\omega)\ge 0,\quad r \ge 0 \;\; (\text{optional but often valid}).
\]

If your preprocessing yields baseline shifts or phase-error artefacts that can make \(r\) locally negative, drop \(r\ge 0\) and instead penalize negative parts softly.

### 6.7 Complete objective (the “full specified objective”)
**Decision variables**: \(W_n, W_r, \alpha(\omega), b(\omega)\) (and optional \(c(\mathbf{r})\)).  
**Latents**: \(n(W_n), r(W_r)\) via UNN outputs; \(h=\mathcal{H}_\omega\{r\}\).

\[
\boxed{
\begin{aligned}
\min_{W_n,W_r,\alpha,b,c}\;\;
&\underbrace{
\frac{1}{2}\left\|W\odot\left(Y-\widehat{Y}(n,r;\alpha,b)\right)\right\|_F^2
}_{\text{data fidelity}}\\
&+\underbrace{\lambda_{n,\omega}\|D^2_\omega n\|_2^2 + \lambda_{n,xy}\mathrm{TV}_{xy}(n)}_{\mathcal{R}_n(n)}\\
&+\underbrace{\lambda_{r,\omega}\|D^2_\omega r\|_1 + \lambda_{r,xy}\mathrm{TV}_{xy}(r)}_{\mathcal{R}_r(r)}\\
&+\underbrace{\lambda_c \mathrm{TV}_{xy}(c)}_{\mathcal{R}_c(c)\;\text{(optional)}}\\[3pt]
\text{s.t.}\;\;
&n=\mathrm{PI}_n + G_n(\mathrm{ISD}_n;W_n),\\
&r=\mathrm{PI}_r + G_r(\mathrm{ISD}_r;W_r),\\
&h = \mathcal{H}_\omega\{r\} + c,\\
&\widehat{Y}(n,r;\alpha,b)=\alpha(\omega)\big((n+h)^2+r^2\big)+b(\omega),\\
&n\ge 0,\;\alpha(\omega)\ge 0 \;\; (\text{and optionally } r\ge 0).
\end{aligned}}
\]

This objective is *explicit*, physics-informed, and preserves SSD’s key idea: **use an untrained, scene-specific network as an implicit prior**, but now on **physically meaningful latent quantities** rather than directly on the measured intensity.

---

## 7) Optimization strategy (practical)

The objective is nonconvex (because of UNNs and quadratic forward model), but is optimized reliably in practice via alternating minimization.

### 7.1 Alternating minimization template
At iteration \(k\):

1. **Update UNN weights \(W_n\)** to reduce objective with \(W_r,\alpha,b,c\) fixed  
   - backprop through \(\widehat{Y}\) and Hilbert transform
2. **Update UNN weights \(W_r\)** similarly
3. **Update \(\alpha(\omega)\)** (closed form if \(b\) fixed):
   \[
   \alpha(\omega_b) \leftarrow 
   \frac{\sum_{\mathbf{r}} w^2\,(Y-b)\,Q}{\sum_{\mathbf{r}} w^2\,Q^2 + \eta},
   \quad Q=((n+h)^2+r^2)
   \]
4. **Update \(b(\omega)\)** (closed form if \(\alpha\) fixed):
   \[
   b(\omega_b) \leftarrow \operatorname{mean}_{\mathbf{r}}\left( Y - \alpha Q\right)
   \]
5. Apply constraints (projection) \(n\leftarrow\max(n,0)\), \(\alpha\leftarrow\max(\alpha,0)\), etc.

### 7.2 Why this is “SSD-like”
- The UNNs are **untrained priors** tuned to the current cube (like DIP/SSD).
- The SD representation reduces the learning burden and encourages consistency across bands.
- You can still use an ADMM splitting (like SSD) if you want separate variables \(Z_n,Z_r\) with quadratic penalties.

---

## 8) What you should validate experimentally

To show that “BCARS-SSD” works (not just denoises), evaluate:

1. **Spectral fidelity**
   - compare recovered \(r=\Im[\chi_R]\) to:
     - KK/MEM outputs (with reference NRB when available)
     - spontaneous Raman spectra of the same sample if feasible
2. **Peak accuracy**
   - peak positions, widths, and relative intensities
3. **Spatial consistency**
   - sharper morphology and reduced pixel-wise flicker versus 1D-only deep models
4. **Runtime**
   - per-frame inference time (for microscopy) vs methods like MEM/TDKK and trained DL

---

## 9) Practical notes and risks

- **Hilbert transform implementation:** discrete Hilbert along spectral axis can be done via FFT; but band-limited edges cause artefacts → consider padding/extrapolation and a phase-detrending term \(c(\mathbf{r})\).
- **Non-uniqueness:** without any constraint, \(n\) and \(r\) can trade off. The smoothness prior on \(n\) and sparsity-like prior on \(r\) (plus KK) are crucial.
- **Instrument response:** if \(\alpha(\omega)\) is unknown and varies strongly, estimating it jointly can be unstable. Calibrating \(\alpha(\omega)\) from a standard sample is helpful.
- **Three-color vs two-color BCARS:** the effective stimulation profile changes. If \(\alpha(\omega)\) captures the excitation envelope, the same objective still applies (but you may need a better prior on \(\alpha\)).

---

## 10) References (starting points)

- **SSD for low-light Raman HSI**: Chen et al., *Nature Computational Science* (accepted 2026), DOI: 10.1038/s43588-026-00957-3.  
- **SpecNet (CNN) for NRB removal**: Valensise et al., *APL Photonics* 5, 061305 (2020).  
- **Deep learning phase retrieval with LSTM**: Houhou et al., *Optics Express* 28(14), 21002–21024 (2020), DOI: 10.1364/OE.390413.  
- **Review + benchmarking of BCARS deep models**: Vernuccio et al., *Scientific Reports* 14:23903 (2024), DOI: 10.1038/s41598-024-74912-5.  
- **Time-domain KK for BCARS phase retrieval (reference approach)**: (NIST-hosted) “Broadband CARS spectral phase retrieval using a time-domain Kramers–Kronig transform” (2009).

---

## 11) Summary: what to modify, in one checklist

To adapt SSD from Raman to BCARS you must:

1. **Replace the observation model**  
   from \(Y=X+S\) → to a BCARS forward model \(Y \approx \alpha|n+\chi_R|^2 + b\).

2. **Move SSD’s PI/SD concept to latent physics fields**  
   define PI/SD on \(n\) (NRB) and \(r=\Im[\chi_R]\), not on raw intensity.

3. **Add physics constraints**  
   KK/Hilbert linking \(\Re[\chi_R]\) and \(\Im[\chi_R]\); smooth NRB; optional nonnegativity.

4. **Optimize scene-specifically**  
   UNN weights are optimized on the measured cube itself (SSD philosophy), but under the BCARS forward model.

If you implement the objective in Section 6, you have a fully specified, end‑to‑end “BCARS‑SSD” formulation.

