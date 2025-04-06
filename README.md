
# SHG Wavefront Shaping Simulation (Split-Step Beam Propagation)

This MATLAB code simulates **second-harmonic generation (SHG)** in a nonlinear medium using the **split-step Fourier method**, with optional **wave front shaping** to enhance SHG intensity in a user-defined region of interest (ROI).

It models the nonlinear propagation of a Gaussian (or User defined) beam and uses **wavefront shaping** (via a phase mask on superpixels) to maximize SHG intensity in a selected area.

---

## ✅ Features

- Gaussian beam propagation in a nonlinear medium.
- Adaptive phase mask optimization over discrete superpixels.
- Adjustable number of superpixels and phase steps.
- ROI-based enhancement of SHG using deterministic phase shaping.
- GPU acceleration via MATLAB's `gpuArray`.

---

## ▶️ How to Run the Code

1. **Open MATLAB** (with parallel computing toolbox enabled, and with CUDA enabled GPU).
2. Load and run the main script.

## 🔧 Adjustable Parameters and its default value

1. ROI Region (where SHG intensity is optimized):
- roi_row_start = 2;
- roi_col_start = 2;
- roi_row_end   = 9;
- roi_col_end   = 9;

2. Simulation Resolution:
- N    = 225;       % Grid size (NxN points)
- L    = 10e-3;     % Physical domain size in meters
- Nz   = 200;       % Number of z-steps
- Lz   = 0.01;      % Propagation distance

3. Wavelength and Refractive Indices
- lambda1 = 1064e-9;         % Fundamental wavelength
- lambda2 = lambda1 / 2;     % SH wavelength
- n1 = 1.8;                  % Fundamental index
- n2 = 1.8;                  % SH index
- Delta_k = 1e6;             % Phase mismatch (adjust to 0 for perfect phase matching)

4. Beam Properties
- w0 = 0.7e-3 * 5;           % Beam waist
- I0 = 2e8;                  % Intensity

5. Superpixel and Phase Mask Settings
- num_superpixels = 225;     % Total superpixels
- num_phases = 8;            % Discrete phase steps per superpixel


