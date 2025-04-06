
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
2. Load and run the main script:
   ```matlab
   main
