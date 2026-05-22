# EEG Macromolecular Crowding — D-serine Simulation Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Enw_8qovbdefOz6KcaA8k6bFf9ldSuLo)

## Overview

This repository contains the full Python implementation supporting the manuscript:

> **Inferring macromolecular crowding in autism and schizophrenia from EEG microfluctuations: a biophysical-computational framework with D-serine modulation**  
> Alvarado YJ, Delgado A, Cardozo-Urdaneta A, Lossada C, Quintero M, González-Paz L.  
> *Brain Research* (under review)

The framework:

1. Generates synthetic EEG-like signals from a neural mass model parameterised for **typically developing (TD)**, **autism spectrum disorder (ASD)**, and **schizophrenia (SCZ)** states.
2. Trains a **convolutional neural network (CNN)** on spectrogram representations to classify the three conditions.
3. Applies the trained CNN to **real resting-state EEG** recordings (loaded from public datasets).
4. Simulates the effect of **D-serine** (NMDA co-agonist) by modifying the spectral content of EEG signals.
5. Computes four **macromolecular crowding proxy metrics**: aperiodic exponent (χ), time constant (τ), multiscale entropy (MSE), and local gamma synchrony (PLV).

---

## Repository structure

```
.
├── dserine_eeg_crowding.py   # Main analysis script (Google Colab notebook converted to .py)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md
```

---

## Quick start

### Option 1 — Google Colab (recommended)

Click the badge above or open directly:  
[https://colab.research.google.com/drive/1Enw_8qovbdefOz6KcaA8k6bFf9ldSuLo](https://colab.research.google.com/drive/1Enw_8qovbdefOz6KcaA8k6bFf9ldSuLo)

### Option 2 — Local execution

```bash
git clone https://github.com/arianadelg/DSerine-ASD-EEG.git
cd DSerine-ASD-EEG
pip install -r requirements.txt
# Run in a Jupyter notebook environment (requires ipywidgets and Google Colab file upload)
jupyter notebook
```

> **Note**: The script uses `google.colab.files` for interactive file uploads and `ipywidgets` for the GUI. It is designed to run in Google Colab. For local use, the upload callbacks can be replaced with standard `open()` calls on local paths.

---

## Methods summary

### Neural mass model

A three-variable neural mass model describes the average activity of one excitatory (E) and two inhibitory populations (I₁ fast, I₂ slow). Condition-specific NMDA conductance parameters:

| Condition | g_ei | g_ee | low_freq_amp |
|-----------|------|------|-------------|
| TD        | 1.0  | 1.0  | 0.0         |
| ASD       | 0.5  | 0.9  | 0.1         |
| SCZ       | 0.2  | 0.75 | 0.4         |

Simulation: 100 segments × 10 s × 500 Hz per condition.

### CNN architecture

| Layer   | Details                                  |
|---------|------------------------------------------|
| Conv1   | 32 filters, 3×3, ReLU + BN + MaxPool 2×2 |
| Conv2   | 64 filters, 3×3, ReLU + BN + MaxPool 2×2 |
| Dense   | 64 units, ReLU, Dropout 0.5              |
| Output  | Softmax (3 classes)                      |

Trained with Adam (sparse categorical cross-entropy), batch size 64, max 20 epochs, early stopping (patience = 3). Seeds: NumPy = 42, TensorFlow = 42.

### D-serine simulation

$$s_{\text{post}} = s + (10^{8/20} - 1) \cdot s_{\gamma} + (10^{-5/20} - 1) \cdot s_{\text{slow}}$$

- **+8 dB** on gamma band (30–80 Hz)  
- **−5 dB** on slow-wave band (0.5–8 Hz)  
- 4th-order Butterworth band-pass filters

### Crowding metrics

| Metric | Description |
|--------|-------------|
| χ (aperiodic exponent) | FOOOF fit on Welch PSD (2–45 Hz), peak_width [1,6], max 2 peaks |
| τ (time constant) | Exponential fit to autocorrelation function (lags 0–0.5 s) |
| MSE (multiscale entropy AUC) | SampEn at scales 1–5, m=2, r=0.2×SD |
| PLV (gamma synchrony) | Phase-locking value on Hilbert-transformed gamma signal |

---

## Real EEG datasets

The script accepts ZIP archives containing EEG recordings in `.set` (EEGLAB), `.gdf`, or `.edf` format. The following public datasets were used in the associated manuscript:

| Group | Dataset | Reference |
|-------|---------|-----------|
| ASD   | Sheffield/ORDA resting-state EEG | Milne et al., *J Abnorm Psychol*, 2019 |
| SCZ   | Nigerian Schizophrenia EEG Dataset (NSzED) | Olateju et al., *arXiv:2311.18484*, 2023 |
| TD    | Matched controls from both datasets | — |

---

## Dependencies

See `requirements.txt`. Key packages:

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `mne` — EEG data loading and resampling
- `tensorflow` — CNN training
- `scikit-learn` — preprocessing and train/test split
- `fooof` — aperiodic exponent estimation
- `ipywidgets` — interactive UI (Colab)

---

## Citation

If you use this code, please cite:

```bibtex
@article{alvarado2025crowding,
  title   = {Inferring macromolecular crowding in autism and schizophrenia from {EEG} microfluctuations: 
             a biophysical-computational framework with {D}-serine modulation},
  author  = {Alvarado, Ysa{\'i}as J. and Delgado, Ariana and Cardozo-Urdaneta, Arlene 
             and Lossada, Carla and Quintero, M{\'a}ximo and Gonz{\'a}lez-Paz, Lenin},
  journal = {Brain Research},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Contact

- Lenin González-Paz: lgonzalezpaz@gmail.com  
- Ysaías J. Alvarado: alvaradoysaias@gmail.com
