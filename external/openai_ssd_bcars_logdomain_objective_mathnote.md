# SSD denoising in NRB-normalized BCARS domains (ratio vs log) — math note

This note answers:

- Does it make sense to run SSD on a normalized domain like \(I_\mathrm{CARS}/I_\mathrm{NRB}\) if you trust KK?
- Should you denoise in *ratio* or in *log* (or *sqrt*) domain?
- What SSD objective should be modified to be mathematically consistent?

## 1) KK/TDKK uses **log amplitude**, not raw intensity

Given a measured BCARS intensity spectrum \(I_\mathrm{CARS}(\omega)\) and a reference NRB spectrum \(I_\mathrm{ref}(\omega)\) (measured on a nonresonant sample), define the **normalized intensity ratio**

\[
S(\omega) \;=\; \frac{I_\mathrm{CARS}(\omega)}{I_\mathrm{ref}(\omega)}.
\]

In the usual CARS phase-retrieval setting, the spectral phase \(\phi(\omega)\) is obtained from the **Hilbert transform / KK integral of the log-amplitude**:

\[
\phi(\omega) = -\frac{1}{\pi}\,\mathcal{P}\!\int_{-\infty}^{\infty} \frac{\ln\sqrt{S(\omega')}}{\omega'-\omega}\,d\omega'.
\]

Then one reconstructs a complex susceptibility-like quantity via

\[
\tilde\chi^{(3)}(\omega) = \sqrt{S(\omega)}\,e^{i\phi(\omega)},
\qquad
\Im\{\tilde\chi^{(3)}\} = \sqrt{S(\omega)}\,\sin\phi(\omega).
\]

Therefore the natural variable to denoise *for KK stability* is

\[
r(\omega) \;:=\; \ln\sqrt{S(\omega)}
\;=\; \tfrac{1}{2}\ln\!\left(\frac{I_\mathrm{CARS}(\omega)}{I_\mathrm{ref}(\omega)}\right).
\]

## 2) Ratio vs log (and why log is usually better if you trust KK)

Let \(S_\mathrm{obs} = S + \delta S\) be the noisy ratio and \(r_\mathrm{obs} = r + \delta r\) the noisy log-amplitude.
A first-order expansion gives:

\[
\delta r \approx \frac{1}{2}\,\frac{\delta S}{S}.
\]

So *errors in KK’s input are essentially relative errors in the ratio*.

- Denoising **ratio** \(S\) with an **unweighted MSE** data term implicitly minimizes *absolute* error \(|\delta S|\), not *relative* error \(|\delta S|/S\). This can be suboptimal for phase retrieval.
- Denoising **log-amplitude** \(r\) directly targets what KK uses and compresses dynamic range, which also avoids the “mean image dominated by NRB” issue when SSD uses a band-mean prior image.

Potential downside of log:
\(\ln(\cdot)\) amplifies noise where \(I_\mathrm{CARS}\) is extremely small; in practice one uses a small offset \(\epsilon\) and/or avoids spectral edges.

## 3) SSD objective: minimal KK-consistent modification

Original SSD (additive) model:
\[
Y = X + S,
\]
\[
(\hat X,\hat S)=\arg\min_{X,S}\;\frac{1}{2}\|Y-X-S\|_F^2 + \lambda_R\,\mathcal R(X) + \lambda_S\|S\|_1.
\]

**KK-consistent choice of data**:
\[
Y := r_\mathrm{obs} = \tfrac{1}{2}\ln\!\left(\frac{I_\mathrm{CARS}+\epsilon}{I_\mathrm{ref}+\epsilon}\right).
\]

Then run SSD *unchanged* (same prior, same ADMM), and feed \(\hat r\) into KK/TDKK.

## 4) More rigorous: weighted fidelity in log domain (delta-method)

Assume photon-counting with read noise:
\[
I_\mathrm{obs}=I+\eta,\qquad \mathrm{Var}(\eta)\approx I+\sigma_\mathrm{read}^2.
\]

With \(r=\tfrac12\ln(I/I_\mathrm{ref})\), a first-order expansion yields
\[
\mathrm{Var}(r_\mathrm{obs})\approx \frac{I+\sigma_\mathrm{read}^2}{4I^2}.
\]

So a more statistically grounded SSD data term is WLS:
\[
\min_{X,S}\;\frac{1}{2}\|W\odot(Y-X-S)\|_F^2 + \lambda_R\,\mathcal R(X)+\lambda_S\|S\|_1,
\]
with elementwise weights \(W_{p,b}=1/\sigma_{r,p,b}\).

## 5) Even more rigorous: Poisson likelihood in intensity + SSD prior on r

Let \(z=r\) be the latent log-amplitude ratio. Predict intensity:
\[
\mu(z) = I_\mathrm{ref}\,e^{2z}.
\]

For Poisson measurements, the negative log-likelihood is
\[
\mathcal L(z) = \sum_{p,b}\Big(\mu_{p,b}(z) - I_{\mathrm{obs},p,b}\,\ln\mu_{p,b}(z)\Big)+\text{const}.
\]

A physics-consistent variational denoiser is:
\[
\min_z\; \mathcal L(z) + \lambda_R\,\mathcal R_{SSD}(z).
\]

This keeps KK intact (you still compute phase by Hilbert transform of \(z\)), but makes the data term consistent with photon noise.

## Practical recommendation if you “trust KK”

1. Compute \(r_\mathrm{obs}=\tfrac12\ln\big((I_\mathrm{CARS}+\epsilon)/(I_\mathrm{ref}+\epsilon)\big)\).
2. Run SSD on \(r_\mathrm{obs}\) (optionally WLS-weighted).
3. Apply KK/TDKK using \(\hat r\).

This is the lowest-risk, KK-aligned way to use SSD for BCARS.
