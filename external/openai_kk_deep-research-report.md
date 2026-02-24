# Adapting Self-Optimized Spectral Distance Denoising to BCARS With KK-Based Raman Retrieval

[Download this report as Markdown](sandbox:/mnt/data/ssd_bcars_report.md)

## Executive overview

Broadband coherent anti-Stokes Raman scattering (BCARS) produces hyperspectral data cubes with high speed and strong signal through coherent nonlinear optics, but its raw spectra are **not** “Raman peaks sitting on a baseline.” The measured spectrum is an **intensity** proportional to the squared magnitude of a complex third-order susceptibility, where the *non-resonant background* (NRB) mixes coherently with the resonant vibrational response. The coherent cross-term is what both amplifies and distorts the Raman-like information you ultimately want. citeturn0search2turn16search4turn16search7

A self-optimized spectral distance (SSD) method (as used in low-light Raman hyperspectral imaging) is therefore best viewed as a **powerful 3D (spatial–spectral) denoiser / regularizer**. It fits naturally into BCARS *before* phase retrieval, where the measurement can be approximated as “true intensity + noise.” It does **not** remove NRB by itself unless you explicitly change the forward model in the SSD objective to match BCARS coherent mixing. citeturn16search4turn15search2

The most practical, physics-consistent, and low-risk pathway—especially when you **already measure NRB**—is:

- Use SSD to denoise the **raw BCARS cube** (and optionally the NRB reference cube).
- Run **KK/TDKK** using the **measured NRB**.
- Optionally apply SSD (or simpler spatial regularization) to stabilize spatial maps of KK error-correction parameters and/or the retrieved Raman-like cube.

This keeps the validated KK step intact while transplanting SSD’s main advantage (3D spatial awareness) into the BCARS pipeline. citeturn0search6turn0search7turn16search4turn16search7

![Conceptual pipeline](sandbox:/mnt/data/bcars_ssd_pipeline.png)

## BCARS mathematical foundations relevant to denoising and NRB removal

BCARS is commonly described via the third-order nonlinear susceptibility. The detected CARS intensity at Raman shift (or equivalently frequency) \(\omega\) is proportional to the squared magnitude of the sum of non-resonant and resonant susceptibilities:
\[
I_{\mathrm{CARS}}(\omega)\ \propto\ \left|\chi^{(3)}_{\mathrm{NR}}(\omega) + \chi^{(3)}_{\mathrm{R}}(\omega)\right|^2.
\]
Expanding:
\[
I_{\mathrm{CARS}}(\omega)\ \propto\ |\chi_{\mathrm{NR}}|^2\ +\ |\chi_{\mathrm{R}}|^2\ +\ 2\,\Re\{\chi_{\mathrm{NR}}\chi^*_{\mathrm{R}}\}.
\]
The third term is an interference (heterodyne) term that makes NRB removal fundamentally different from subtracting a baseline in spontaneous Raman: the NRB is *inside* the modulus-square and cannot be removed by a purely additive model. citeturn0search2turn16search4turn16search7

Because typical BCARS measurements record **intensity only** (magnitude, not phase), extracting Raman-like absorptive spectra requires **phase retrieval**. Two widely used families are:

- Kramers–Kronig (KK), including **time-domain KK (TDKK)**, which retrieves phase from the modulus under analytic/causality-type assumptions. citeturn0search6turn16search4  
- Maximum entropy method (MEM), which is commonly used and has been discussed as functionally equivalent to KK under holomorphic/minimum-phase-type conditions often met in practice. citeturn16search4turn0search7  

A further practical reality is that KK/TDKK and MEM depend on the NRB estimate/profile; researchers often measure NRB on a reference sample such as glass or water, but that surrogate can differ from the true in-sample NRB and creates amplitude/phase errors. Camp *et al.* develop a framework for correcting such phase-retrieval errors (detrending/scaling) to enable quantitative comparability. citeturn0search7turn16search4

## What SSD contributes and what its standard assumptions miss for BCARS

A “self-optimized spectral distance (SSD)” Raman hyperspectral denoising approach is publicly referenced as a code release for low-light Raman hyperspectral imaging via a Code Ocean capsule citation. citeturn15search2turn10search0  

Accessible public sources do **not** (from what could be retrieved here) provide full, inspectable methodological details for that specific SSD implementation because the capsule page itself returns access errors. As a result, this report treats “SSD” at the level of the properties you described and that are broadly consistent with known self-supervised / untrained-prior hyperspectral denoisers: (i) it uses **spatial–spectral correlation** across the full cube and (ii) it can be implemented using **spatio-spectral total variation** plus an **untrained neural-network prior** (DIP-like). citeturn17search0turn17search10  

Those two components are well-grounded:

- **SSTV / spatio-spectral TV** regularizers are widely used for hyperspectral denoising because they model joint smoothness in (x, y, ω) and reduce “spectral jaggedness” that can occur if bands are denoised independently. citeturn17search10turn17search5turn17search27  
- **Deep Image Prior (DIP)** shows that a *randomly initialized* convolutional generator can act as an implicit prior for denoising and other inverse problems without external training data. citeturn17search0turn17search1  

Where the default SSD framing can break for BCARS is the observation model. If SSD assumes something like
\[
Y = X + \varepsilon
\]
and treats \(X\) as “the clean spectrum,” then it can denoise BCARS intensities but it cannot, by itself, attack NRB mixing—because the NRB is not an additive baseline. citeturn16search4turn0search2

## Using SSD for BCARS as pure denoising before KK

BCARS pipelines often include an early denoising step, and published work explicitly discusses using statistical dimensionality reduction / denoising (e.g., SVD) before applying KK-based phase retrieval. citeturn16search7  

This makes “SSD-for-denoising” a direct drop-in conceptually: you apply SSD to the *raw intensity cube* to reduce random noise while preserving spatial morphology and spectral continuity, then you perform KK/TDKK with your measured NRB.

**Key modification: denoise in an NRB-normalized domain to prevent the NRB from dominating SSD’s spectral distance**

In BCARS, the NRB envelope can dominate mean intensity across bands. If SSD computes a “prior image” as the mean across bands (as in your description), that prior image risks being a proxy for “NRB morphology” rather than “vibrational contrast.”

When you have a measured NRB reference \(I_{\mathrm{NRB}}(\omega)\), a robust fix is to apply SSD in an NRB-normalized domain where the NRB envelope cancels to first order:

\[
A(x,y,\omega)= \sqrt{\frac{I_{\mathrm{CARS}}(x,y,\omega)}{I_{\mathrm{NRB}}(\omega)}}
\quad \text{and/or}\quad
R(x,y,\omega)= \log\left(\frac{I_{\mathrm{CARS}}(x,y,\omega)}{I_{\mathrm{NRB}}(\omega)}\right).
\]

Both are aligned with KK/TDKK-style phase retrieval, which uses modulus/log-modulus information to recover phase. After SSD denoising in \(R\)-space, reconstruct a denoised intensity cube via:
\[
\widehat{I}_{\mathrm{CARS}}(x,y,\omega)= I_{\mathrm{NRB}}(\omega)\,\exp\!\left(\widehat{R}(x,y,\omega)\right),
\]
then run KK/TDKK using \(\widehat{I}_{\mathrm{CARS}}\) and the measured \(I_{\mathrm{NRB}}\). citeturn0search6turn16search4turn0search7

**Modify SSD’s data fidelity for BCARS noise (recommended)**

BCARS noise is typically a mixture of photon shot noise and detector/electronics noise; treating it as i.i.d. Gaussian everywhere is usually suboptimal. A practical internal modification is to replace a plain MSE data term with either:

- **Weighted least squares (WLS)** (heteroscedastic Gaussian approximation), or  
- A **Poisson(-Gaussian)** negative log-likelihood (when counts/shot noise dominate),  

while keeping the same SSTV/untrained-prior regularization backbone. This is standard practice in multispectral/hyperspectral denoising under photon noise. citeturn17search24turn17search10turn17search0

## Making SSD NRB-aware while preserving KK with measured NRB

There are two levels of “SSD for NRB removal” that complement your “preferably still the KK step” requirement.

**NRB-aware SSD as a KK stabilizer**

This option preserves the KK step operationally and uses SSD only where its assumptions are valid.

1) Apply SSD denoising to \(R=\log(I_{\mathrm{CARS}}/I_{\mathrm{NRB}})\) (or to the raw cube if you prefer).  
2) Apply KK/TDKK per pixel. citeturn0search6turn16search4  
3) Apply error-phase correction and scaling per pixel (Camp-style). citeturn0search7  
4) Treat the KK correction parameters (e.g., phase detrend coefficients, scaling factors) as **2D images** and apply SSD (or simpler TV) to those parameter maps, enforcing that correction terms change smoothly across morphology.

This is motivated by two facts documented in the literature:

- Errors from finite spectral range and surrogate NRB can lead to phase/amplitude artifacts, motivating correction strategies. citeturn0search7turn16search4  
- NRB variation and mismatch measurably affect retrieval quality and can influence downstream classification/analysis, motivating methods that are robust to NRB changes. citeturn16search21turn0search7  

This approach is “SSD-like spatial awareness added to a KK pipeline” without rewriting the physics.

**Full physics-informed SSD (when you want SSD to participate in Raman retrieval)**

If you want SSD to become part of NRB removal (and not just pre-denoising), you must modify the **forward model** in SSD’s optimization to match coherent mixing:

- Replace additive model \(Y=X+\varepsilon\) with BCARS forward intensity \(I \propto |\chi_{\mathrm{NR}} + \chi_{\mathrm{R}}|^2\). citeturn0search2turn16search4  
- Optimize over a representation of \(\chi_{\mathrm{R}}(\omega)\), which is complex-valued and constrained by causality (KK relations). citeturn0search6turn0search7  

A practical formulation that still “keeps KK” conceptually is to parameterize the resonant susceptibility by its imaginary (absorptive) part and compute the real part via a Hilbert-transform/Kramers–Kronig operator inside the optimization. One example:

- Let \(s(x,y,\omega)\) represent the target absorptive spectrum \(\propto \Im\{\chi^{(3)}_{\mathrm{R}}\}\).  
- Compute a dispersive component \(d=\mathcal{H}[s]\).  
- Form \(\chi_{\mathrm{R}} = d + i s\), and predict intensity:
  \[
  \widehat{I}(x,y,\omega)=\left|a(x,y)\chi_{\mathrm{NR}}(\omega) + \chi_{\mathrm{R}}(x,y,\omega)\right|^2,
  \]
  where \(a(x,y)\) is a slowly varying scale map that absorbs differences between the measured reference NRB and the in-sample NRB amplitude.

Then define an SSD-like objective over \(s\) (or over an untrained-network generator \(s_\theta\)):

\[
\min_{\theta, a}\ 
\sum_{x,y,\omega}\rho\!\left(I_{\mathrm{raw}}-\widehat{I}_{\theta,a}\right)
+\lambda\,\mathrm{SSTV}(s_\theta)
+\mu\,\mathrm{SD}(s_\theta)
+\gamma\,\mathrm{TV}(a).
\]

Here \(\rho\) should be likelihood-consistent (WLS/Poisson), \(\mathrm{SSTV}\) is spatio-spectral TV, and \(\mathrm{SD}\) is a spectral-distance regularizer computed in a **Raman-like** domain (i.e., on \(s\) or other NRB-suppressed representations), not on raw intensity mean images dominated by NRB. citeturn17search10turn17search0turn0search6turn16search4

This is, effectively, a physics-informed, self-supervised inverse-problem solver: “your network output is forced to be consistent with BCARS intensity formation plus KK causality.” Untrained-prior approaches of this family are broadly compatible with DIP-like ideas, even though the exact SSD implementation details are not fully accessible here. citeturn17search0turn8search1

## Implementation blueprint and validation experiments

**Minimum-viable implementation (recommended starting point)**

1) **Acquire**:
   - Raw BCARS cube \(I_{\mathrm{CARS}}(x,y,\omega)\).
   - A measured NRB reference \(I_{\mathrm{NRB}}(\omega)\) (or a reference cube). NRB measurements on reference media such as glass/water are explicitly discussed in BCARS NRB-removal literature. citeturn16search4turn0search7  
2) **Transform**: compute \(R=\log(I_{\mathrm{CARS}}/I_{\mathrm{NRB}})\) (or amplitude ratio).  
3) **SSD denoise**: run SSD on \(R\) as a full 3D cube, using SSTV + untrained prior; choose WLS/Poisson data term depending on your noise regime. citeturn17search10turn17search0turn17search24  
4) **Back-transform**: \(\widehat{I}_{\mathrm{CARS}}=I_{\mathrm{NRB}}\exp(\widehat{R})\).  
5) **KK/TDKK retrieval**: retrieve a Raman-like spectrum per pixel. citeturn0search6turn16search4  
6) **Error-phase correction**: apply Camp-style correction; optionally spatially regularize correction maps. citeturn0search7turn16search21  
7) **Compare baselines**: include leading deep-learning NRB-removal baselines (SpecNet, VECTOR, recurrent models, GANs) that operate on spectra; these are reviewed and benchmarked in recent work. citeturn16search4turn16search22turn16search2turn16search1  

**Validation strategy tailored to the “SSD + KK” hybrid**

Use at least three orthogonal checks:

- **Spectral validity**: On samples where spontaneous Raman is available, compare peak positions and relative peak areas in the retrieved Raman-like spectrum (KK/TDKK aims directly at retrieving the resonant imaginary component). citeturn0search6turn16search4  
- **Spatial fidelity**: Verify that morphology (edges, thin structures) is preserved; spatio-spectral regularizers are explicitly designed to preserve structure while removing noise. citeturn17search10turn17search27  
- **NRB robustness**: Evaluate sensitivity against NRB mismatch, which has been shown to influence Raman retrieval and downstream analyses. citeturn16search21turn0search7  

A useful practical benchmark is to test how well downstream unmixing/classification behaves under faster scanning / lower light (since SSD’s main intended benefit is enabling usable cubes at lower SNR without training data). Recent literature underscores that deep-learning NRB-removal models exist (SpecNet, VECTOR, LSTM/Bi-LSTM, GAN architectures), but they frequently process spectra as 1D sequences; an SSD-like approach can complement them by exploiting full 3D spatial–spectral correlation. citeturn16search4turn16search22turn16search2

## References

- Camp Jr., C.H.; Lee, Y.J.; Cicerone, M.T. “Quantitative, Comparable Coherent Anti-Stokes Raman Scattering (CARS) Spectroscopy: Correcting Errors in Phase Retrieval.” arXiv:1507.06543 (2015). citeturn0search7turn0search3  
- Liu, Y.; Lee, Y.J.; Cicerone, M.T. “Broadband CARS spectral phase retrieval using a time-domain Kramers–Kronig transform.” Optics Letters (2009). citeturn0search6  
- Vernuccio, F.; Broggio, E.; Sorrentino, S.; et al. “Non-resonant background removal in broadband CARS microscopy using deep-learning algorithms.” Scientific Reports (2024). citeturn16search4turn16search0  
- Valensise, C.M.; Giuseppi, A.; Vernuccio, F.; et al. “Removing non-resonant background from CARS spectra via deep learning (SpecNet).” APL Photonics (2020). citeturn16search22turn16search10  
- Wang, Z.; O’Dwyer, K.; Muddiman, R.; et al. “VECTOR: Very deep convolutional autoencoders for NRB removal in BCARS.” (Code repository and associated publication info). citeturn16search2turn16search16  
- Ulyanov, D.; Vedaldi, A.; Lempitsky, V. “Deep Image Prior.” CVPR (2018). citeturn17search0turn17search1  
- Representative spatio-spectral TV work and discussion in hyperspectral denoising. citeturn17search10turn17search5turn17search27turn17search24  
- Evidence of SSD Raman hyperspectral imaging code release (Code Ocean capsule citation). citeturn15search2turn10search0