# -*- coding: utf-8 -*-
"""
dserine_eeg_crowding.py
=======================
EEG Macromolecular Crowding — D-serine Simulation Framework

Implements the full analysis pipeline described in:
    Alvarado YJ, Delgado A, Cardozo-Urdaneta A, Lossada C, Quintero M,
    González-Paz L. "Inferring macromolecular crowding in autism and
    schizophrenia from EEG microfluctuations: a biophysical-computational
    framework with D-serine modulation." Brain Research (under review).

Repository: https://github.com/arianadelg/DSerine-ASD-EEG
Original Colab notebook: https://colab.research.google.com/drive/1Enw_8qovbdefOz6KcaA8k6bFf9ldSuLo

Pipeline:
    1. Generate synthetic EEG via a neural mass model (TD / ASD / SCZ).
    2. Train a CNN on spectrogram representations.
    3. Load real resting-state EEG (ZIP of .set/.gdf/.edf files).
    4. Simulate D-serine by modifying spectral content.
    5. Compute four macromolecular crowding proxy metrics:
       aperiodic exponent (chi), time constant (tau),
       multiscale entropy (MSE AUC), local gamma synchrony (PLV).

Requirements: see requirements.txt
Designed to run on Google Colab; ipywidgets GUI uses google.colab.files
for interactive file upload.

Authors:
    Ysaías J. Alvarado  <alvaradoysaias@gmail.com>
    Lenin González-Paz  <lgonzalezpaz@gmail.com>
    Instituto Venezolano de Investigaciones Científicas (IVIC)

License: MIT
"""

import sys, subprocess, importlib, time, threading, tempfile, os, io, zipfile, shutil
for pkg in ['mne', 'tensorflow', 'scikit-learn', 'matplotlib', 'ipywidgets', 'fooof']:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch, butter, filtfilt, hilbert
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import mne
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from google.colab import files
import fooof
import warnings
warnings.filterwarnings('ignore')

# Seeds
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
FS = 500
DURATION = 10
EPOCHS = 20
BATCH_SIZE = 64
NPERSEG = 256
NOVERLAP = 200
N_FREQ = NPERSEG // 2 + 1

GAMMA_GAIN_DB = 8.0
SLOWWAVE_ATT_DB = -5.0

SYNTHETIC_SAMPLES = 100
MAX_CROWDING_SAMPLES = 20  # límite para el análisis de hacinamiento por grupo

# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
class ProgressCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{self.params['epochs']} completed")

# ---------------------------------------------------------------------------
# Neural mass model
# ---------------------------------------------------------------------------
class RealisticNeuralMass:
    def __init__(self, g_nmda_ei=1.0, g_nmda_ee=1.0, low_freq_amp=0.0, fs=FS):
        self.fs = fs
        self.g_nmda_ei = g_nmda_ei
        self.g_nmda_ee = g_nmda_ee
        self.low_freq_amp = low_freq_amp
        self.tau_e = 0.005
        self.tau_i1 = 0.008
        self.tau_i2 = 0.050
        self.w_ee = 15.0
        self.w_ei1 = 12.0 * g_nmda_ei
        self.w_ei2 = 6.0
        self.w_i1e = -10.0
        self.w_i2e = -4.0
        self.p_base = 3.0
        self.sigma_p = 0.8
        self.noise_amp = 0.4

    def derivs(self, state, t):
        E, I1, I2 = state
        def sig(x, thr=2.5): return 1.0/(1.0+np.exp(-(x-thr)))
        I_slow = self.low_freq_amp * np.sin(2 * np.pi * 2.0 * t)
        p = self.p_base + self.sigma_p*np.random.randn() + I_slow
        I_syn_E = self.w_ee*self.g_nmda_ee*sig(E) + self.w_i1e*sig(I1) + self.w_i2e*sig(I2) + p
        I_syn_I1 = self.w_ei1*sig(E)
        I_syn_I2 = self.w_ei2*sig(E)
        dE = (-E + I_syn_E)/self.tau_e + self.noise_amp*np.random.randn()
        dI1 = (-I1 + I_syn_I1)/self.tau_i1 + self.noise_amp*np.random.randn()
        dI2 = (-I2 + I_syn_I2)/self.tau_i2 + self.noise_amp*np.random.randn()
        return np.array([dE, dI1, dI2])

    def simulate(self, duration_sec=10, seed=None):
        if seed is not None: np.random.seed(seed)
        dt = 1.0/self.fs
        steps = int(duration_sec*self.fs)
        state = np.array([0.1, 0.0, 0.0])
        eeg = np.zeros(steps)
        for i in range(steps):
            eeg[i] = state[0]
            state = state + dt*self.derivs(state, i*dt)
        return eeg

# ---------------------------------------------------------------------------
# Spectrogram in dB
# ---------------------------------------------------------------------------
def spectrogram_db(signal, fs=FS):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    return 10*np.log10(Sxx + 1e-10)

# ---------------------------------------------------------------------------
# Métricas de microfluctuaciones para hacinamiento macromolecular
# ---------------------------------------------------------------------------
def compute_aperiodic_exponent(signal, fs=FS, f_range=(2, 45)):
    try:
        freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
        fm = fooof.FOOOF(peak_width_limits=[1, 6], max_n_peaks=2, verbose=False)
        fm.fit(freqs=freqs, spectrum=psd, freq_range=f_range)
        return fm.aperiodic_params[1]
    except:
        freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
        idx = (freqs >= f_range[0]) & (freqs <= f_range[1])
        logf = np.log10(freqs[idx])
        logp = np.log10(psd[idx])
        slope, _ = np.polyfit(logf, logp, 1)
        return -slope

def compute_tau(signal, fs=FS, lag_max=0.5):
    lags = int(fs * lag_max)
    autocorr = np.correlate(signal - np.mean(signal), signal - np.mean(signal), mode='full')
    autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
    autocorr = autocorr[:lags]
    t_vals = np.arange(lags) / fs
    valid = (autocorr > 0.1)
    if np.sum(valid) < 5:
        return np.nan
    y = np.log(autocorr[valid])
    x = t_vals[valid]
    slope, _ = np.polyfit(x, y, 1)
    tau = -1.0 / slope if slope != 0 else np.nan
    return tau

def compute_multiscale_entropy(signal, fs=FS, scales=[1,2,3,4,5], m=2, r=0.2):
    def sample_entropy(x, m, r):
        N = len(x)
        if N < m+1:
            return 0.0
        # Vectorizado para acelerar (comparaciones con broadcasting)
        def _count_matches(template, data, r):
            return np.sum(np.max(np.abs(template - np.lib.stride_tricks.sliding_window_view(data, len(template))), axis=1) < r)
        templates = np.lib.stride_tricks.sliding_window_view(x, m)
        B = 0.0
        for t in templates:
            B += _count_matches(t, x, r) - 1
        B /= (N - m)
        templates_m1 = np.lib.stride_tricks.sliding_window_view(x, m+1)
        A = 0.0
        for t in templates_m1:
            A += _count_matches(t, x, r) - 1
        A /= (N - m - 1)
        if B == 0 or A == 0:
            return 0.0
        return -np.log(A / B)

    entropies = []
    for scale in scales:
        if scale == 1:
            coarse = signal
        else:
            coarse = np.mean(signal[:len(signal)//scale*scale].reshape(-1, scale), axis=1)
        r_scaled = r * np.std(coarse)
        if r_scaled == 0:
            entropies.append(0.0)
        else:
            entropies.append(sample_entropy(coarse, m, r_scaled))
    auc = np.trapz(entropies, scales)
    return auc, entropies

def compute_local_synchrony(signal, fs=FS, band='gamma'):
    if band == 'gamma':
        low, high = 30, 80
    elif band == 'beta':
        low, high = 13, 30
    else:
        low, high = 8, 13
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    analytic = hilbert(filtered)
    phase = np.angle(analytic)
    synchrony = np.abs(np.mean(np.exp(1j * phase)))
    return synchrony

def apply_dserine_signal(signal, fs=FS):
    nyq = fs/2
    b_g, a_g = butter(4, [30/nyq, 80/nyq], btype='band')
    gamma_comp = filtfilt(b_g, a_g, signal)
    b_s, a_s = butter(4, [0.5/nyq, 8/nyq], btype='band')
    slow_comp = filtfilt(b_s, a_s, signal)
    gamma_gain_lin = 10**(GAMMA_GAIN_DB/20)
    slow_att_lin = 10**(SLOWWAVE_ATT_DB/20)
    modified = signal + (gamma_gain_lin - 1)*gamma_comp + (slow_att_lin - 1)*slow_comp
    return modified

# ---------------------------------------------------------------------------
# Synthetic dataset generation (devuelve también señales)
# ---------------------------------------------------------------------------
def generate_dataset(n_samples=SYNTHETIC_SAMPLES, return_signals=False):
    params = {
        'TD':  (1.0, 1.0, 0.0),
        'ASD': (0.5, 0.9, 0.1),
        'SCZ': (0.2, 0.75, 0.4)
    }
    X_list, y_list, signals_list = [], [], []
    for label, (g_ei, g_ee, lf_amp) in params.items():
        for i in range(n_samples):
            model = RealisticNeuralMass(g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf_amp)
            eeg = model.simulate(DURATION, seed=i)
            X_list.append(spectrogram_db(eeg))
            y_list.append(label)
            if return_signals:
                signals_list.append(eeg)
    X = np.array(X_list)[..., np.newaxis]
    scaler = StandardScaler().fit(X.reshape(len(X), -1))
    X = scaler.transform(X.reshape(len(X), -1)).reshape(X.shape)
    le = LabelEncoder().fit(y_list)
    y = le.transform(y_list)
    if return_signals:
        return X, y, le.classes_, scaler, signals_list
    else:
        return X, y, le.classes_, scaler

# ---------------------------------------------------------------------------
# Simplified CNN
# ---------------------------------------------------------------------------
def create_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    model = create_cnn(X.shape[1:], len(np.unique(y)))
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test), verbose=0,
                        callbacks=[ProgressCallback(), early_stop])
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return model, history, acc*100

# ---------------------------------------------------------------------------
# Treatment simulation (espectrograma)
# ---------------------------------------------------------------------------
def apply_dserine_spectrogram(spec_db, freqs):
    spec_mod = spec_db.copy()
    idx_gamma = (freqs >= 30) & (freqs <= 80)
    spec_mod[idx_gamma, :] += GAMMA_GAIN_DB
    idx_slow = (freqs >= 0.5) & (freqs <= 8)
    spec_mod[idx_slow, :] += SLOWWAVE_ATT_DB
    return spec_mod

def simulate_treatment(g_ei, g_ee, lf_amp):
    n = 50
    pre_list, post_list = [], []
    f_ref = np.arange(N_FREQ) * (FS / NPERSEG)
    for i in range(n):
        model = RealisticNeuralMass(g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf_amp)
        eeg = model.simulate(DURATION, seed=i+10000)
        spec_pre = spectrogram_db(eeg)
        pre_list.append(spec_pre)
        spec_post = apply_dserine_spectrogram(spec_pre, f_ref)
        post_list.append(spec_post)
    X_pre = np.array(pre_list)[..., np.newaxis]
    X_post = np.array(post_list)[..., np.newaxis]
    return X_pre, X_post

def evaluate_shift(model, X_pre, X_post, scaler_baseline, classes):
    combined = np.concatenate([X_pre, X_post], axis=0)
    combined_flat = combined.reshape(combined.shape[0], -1)
    local_scaler = StandardScaler().fit(combined_flat)
    X_pre_norm = local_scaler.transform(X_pre.reshape(len(X_pre), -1)).reshape(X_pre.shape)
    X_post_norm = local_scaler.transform(X_post.reshape(len(X_post), -1)).reshape(X_post.shape)
    idx_td = np.where(classes == 'TD')[0][0]
    prob_pre = model.predict(X_pre_norm, verbose=0)
    prob_post = model.predict(X_post_norm, verbose=0)
    td_pre_clas = np.mean(np.argmax(prob_pre, axis=1) == idx_td) * 100
    td_post_clas = np.mean(np.argmax(prob_post, axis=1) == idx_td) * 100
    td_pre_prob = np.mean(prob_pre[:, idx_td]) * 100
    td_post_prob = np.mean(prob_post[:, idx_td]) * 100
    return td_pre_clas, td_post_clas, td_pre_prob, td_post_prob

# ---------------------------------------------------------------------------
# Real data processing
# ---------------------------------------------------------------------------
def load_real_eeg(path, duration=DURATION, fs=FS):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.set':
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    elif ext == '.gdf':
        raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    raw.resample(fs)
    signal = raw.get_data().mean(axis=0)
    needed = int(duration*fs)
    if len(signal) < needed:
        signal = np.pad(signal, (0, needed-len(signal)))
    else:
        signal = signal[:needed]
    return signal

def process_uploaded_zip(zip_bytes, label, return_signals=False):
    specs = []
    signals = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmpdir)
        for root, dirs, files in os.walk(tmpdir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ['.set', '.gdf', '.edf']:
                    continue
                full_path = os.path.join(root, fname)
                print(f"   Processing {full_path}...")
                try:
                    signal = load_real_eeg(full_path)
                    specs.append(spectrogram_db(signal))
                    if return_signals:
                        signals.append(signal)
                except Exception as e:
                    print(f"   ❌ Error processing {fname}: {e}")
    if return_signals:
        return specs, signals
    return specs

# ---------------------------------------------------------------------------
# Visualization functions (originales)
# ---------------------------------------------------------------------------
def force_2d(arr, n_freq=N_FREQ):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        total = len(arr)
        n_time = total // n_freq
        if n_time == 0:
            raise ValueError(f"Cannot reshape array of size {total} with N_FREQ={n_freq}")
        return arr[:n_time * n_freq].reshape(n_freq, n_time)
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Array has {arr.ndim} dimensions, expected 1 or 2.")

def compute_band_powers(spectrogram, bands, f):
    table = {}
    for band, (fmin, fmax) in bands.items():
        idx = (f >= fmin) & (f < fmax)
        table[band] = np.mean(spectrogram[idx, :])
    return table

def plot_training(history, title='Training curves'):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
    ax1.plot(history.history['loss'], label='Training')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Binary Crossentropy')
    ax1.legend(); ax1.grid(True, linestyle=':', alpha=0.6)
    ax2.plot(history.history['accuracy'], label='Training')
    ax2.plot(history.history['val_accuracy'], label='Validation')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True, linestyle=':', alpha=0.6)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def generate_raw_td(seed=0):
    model = RealisticNeuralMass(g_nmda_ei=1.0, g_nmda_ee=1.0, low_freq_amp=0.0)
    eeg = model.simulate(DURATION, seed=seed)
    return spectrogram_db(eeg)

def plot_synthetic_overview(td_raw, asd_pre_raw, asd_post_raw, scz_pre_raw, scz_post_raw):
    td_raw = force_2d(td_raw)
    asd_pre_raw = force_2d(asd_pre_raw)
    asd_post_raw = force_2d(asd_post_raw)
    scz_pre_raw = force_2d(scz_pre_raw)
    scz_post_raw = force_2d(scz_post_raw)

    time_step = (NPERSEG - NOVERLAP) / FS
    freq_res = FS / NPERSEG
    t_full = np.arange(td_raw.shape[1]) * time_step
    f = np.arange(td_raw.shape[0]) * freq_res

    max_time = 5.0
    idx_t = t_full <= max_time
    t = t_full[idx_t]
    td_raw = td_raw[:, idx_t]
    asd_pre_raw = asd_pre_raw[:, idx_t]
    asd_post_raw = asd_post_raw[:, idx_t]
    scz_pre_raw = scz_pre_raw[:, idx_t]
    scz_post_raw = scz_post_raw[:, idx_t]

    asd_combined = np.concatenate([asd_pre_raw.ravel(), asd_post_raw.ravel()]).reshape(-1, 1)
    scaler_asd = StandardScaler().fit(asd_combined)
    asd_pre_norm = scaler_asd.transform(asd_pre_raw.ravel().reshape(-1, 1)).reshape(asd_pre_raw.shape)
    asd_post_norm = scaler_asd.transform(asd_post_raw.ravel().reshape(-1, 1)).reshape(asd_post_raw.shape)

    scz_combined = np.concatenate([scz_pre_raw.ravel(), scz_post_raw.ravel()]).reshape(-1, 1)
    scaler_scz = StandardScaler().fit(scz_combined)
    scz_pre_norm = scaler_scz.transform(scz_pre_raw.ravel().reshape(-1, 1)).reshape(scz_pre_raw.shape)
    scz_post_norm = scaler_scz.transform(scz_post_raw.ravel().reshape(-1, 1)).reshape(scz_post_raw.shape)

    td_reshaped = td_raw.ravel().reshape(-1, 1)
    scaler_td = StandardScaler().fit(td_reshaped)
    td_norm = scaler_td.transform(td_reshaped).reshape(td_raw.shape)

    datasets_norm = [td_norm, asd_pre_norm, asd_post_norm, scz_pre_norm, scz_post_norm]
    titles = ['TD (untreated)', 'ASD pre-treatment', 'ASD post D‑serine', 'SCZ pre-treatment', 'SCZ post D‑serine']
    bands = {'δ': (0.5, 4), 'θ': (4, 8), 'α': (8, 13), 'β': (13, 30), 'γ': (30, 80)}

    fig = plt.figure(figsize=(25, 10))
    for i, (data, title) in enumerate(zip(datasets_norm, titles)):
        ax = fig.add_subplot(2, 5, i+1)
        im = ax.pcolormesh(t, f, data, shading='auto', cmap='jet')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        for band, (fmin, fmax) in bands.items():
            ax.axhline(y=fmin, color='white', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.text(t[-1]*0.95, (fmin+fmax)/2, band, color='white', fontsize=8, ha='right', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
        ax.set_ylim(0, 80)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    color_map = {'δ': 'blue', 'θ': 'cyan', 'α': 'green', 'β': 'orange', 'γ': 'red'}
    for i, (data, title) in enumerate(zip(datasets_norm, titles)):
        ax = fig.add_subplot(2, 5, i+6)
        for band, (fmin, fmax) in bands.items():
            idx_band = (f >= fmin) & (f < fmax)
            band_power = np.mean(data[idx_band, :], axis=0)
            ax.plot(t, band_power, color=color_map[band], label=band, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (normalized)')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, linestyle=':', alpha=0.5)

    fig.suptitle('Spectrograms and per‑band time evolution (first 5 s, intrasubject normalization)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    all_data_raw = [td_raw, asd_pre_raw, asd_post_raw, scz_pre_raw, scz_post_raw]
    table = {}
    for cond, raw_data in zip(titles, all_data_raw):
        table[cond] = compute_band_powers(raw_data, bands, f)
    df_table = pd.DataFrame(table).T[['δ', 'θ', 'α', 'β', 'γ']].round(2)
    display(HTML("<h4>📊 Average band power (absolute dB) – less negative = higher power</h4>"))
    display(HTML(df_table.to_html(classes='table table-striped', border=0, justify='center')))
    display(HTML(f"""
        <p style='color:#ccc;'>
        <b>Interpretation of table and figures:</b><br>
        • The <b>table</b> shows average absolute power in decibels (dB). <b>Less negative values indicate higher power</b> (e.g., –35 dB is stronger than –45 dB).<br>
        • <b>Spectrograms</b> (top row) are normalized within each pre/post pair to highlight relative changes. Warm colors (red) = higher activity, cool colors (blue) = lower activity.<br>
        • <b>Time courses</b> (bottom row) show the evolution of normalized band power. The normalization preserves relative differences; an upward shift of the red curve (gamma) after D‑serine reflects the same effect as a less negative dB value in the table.<br>
        • The simulated treatment applies a <b>{GAMMA_GAIN_DB} dB gain in gamma</b> and a <b>{abs(SLOWWAVE_ATT_DB)} dB attenuation in delta/theta</b>, which should appear as increased gamma (red) and reduced slow waves (blue) in both ASD and SCZ.
        </p>
    """))

def plot_separate_analysis(real_data, syn_data, bands, f, t_full):
    idx_t = t_full <= 5.0
    t = t_full[idx_t]

    color_map = {'δ': 'blue', 'θ': 'cyan', 'α': 'green', 'β': 'orange', 'γ': 'red'}
    classes = ['ASD', 'SCZ', 'TD']

    syn_panels = []
    for clase in classes:
        syn_pre, syn_post = syn_data[clase]
        syn_pre_5s = syn_pre[:, idx_t]
        syn_post_5s = syn_post[:, idx_t] if clase != 'TD' else None
        if clase == 'TD':
            syn_panels.append((syn_pre_5s, f'{clase} syn (untreated)'))
        else:
            syn_panels.append((syn_pre_5s, f'{clase} syn pre'))
            syn_panels.append((syn_post_5s, f'{clase} syn post'))

    fig_syn, axes_syn = plt.subplots(2, len(syn_panels), figsize=(4*len(syn_panels), 8))
    if len(syn_panels) == 1:
        axes_syn = np.array([[axes_syn[0]], [axes_syn[1]]])
    for i, (data, title) in enumerate(syn_panels):
        ax = axes_syn[0, i]
        im = ax.pcolormesh(t, f, data, shading='auto', cmap='jet')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        for band, (fmin, fmax) in bands.items():
            ax.axhline(y=fmin, color='white', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_ylim(0, 80)
        fig_syn.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax2 = axes_syn[1, i]
        for band, (fmin, fmax) in bands.items():
            idx = (f >= fmin) & (f < fmax)
            band_power = np.mean(data[idx, :], axis=0)
            ax2.plot(t, band_power, color=color_map[band], label=band, linewidth=1)
        ax2.set_title(title, fontsize=9)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Power (norm.)')
        ax2.legend(loc='upper right', fontsize=6)
        ax2.grid(True, linestyle=':', alpha=0.5)
    fig_syn.suptitle('Synthetic data (first 5 s, pre and post D‑serine)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    real_panels = []
    for clase in classes:
        real_pre, real_post = real_data[clase]
        real_pre_5s = real_pre[:, idx_t]
        real_post_5s = real_post[:, idx_t] if clase != 'TD' else None
        if clase == 'TD':
            real_panels.append((real_pre_5s, f'{clase} real (untreated)'))
        else:
            real_panels.append((real_pre_5s, f'{clase} real pre'))
            real_panels.append((real_post_5s, f'{clase} real post'))

    fig_real, axes_real = plt.subplots(2, len(real_panels), figsize=(4*len(real_panels), 8))
    if len(real_panels) == 1:
        axes_real = np.array([[axes_real[0]], [axes_real[1]]])
    for i, (data, title) in enumerate(real_panels):
        ax = axes_real[0, i]
        im = ax.pcolormesh(t, f, data, shading='auto', cmap='jet')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        for band, (fmin, fmax) in bands.items():
            ax.axhline(y=fmin, color='white', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_ylim(0, 80)
        fig_real.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax2 = axes_real[1, i]
        for band, (fmin, fmax) in bands.items():
            idx = (f >= fmin) & (f < fmax)
            band_power = np.mean(data[idx, :], axis=0)
            ax2.plot(t, band_power, color=color_map[band], label=band, linewidth=1)
        ax2.set_title(title, fontsize=9)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Power (norm.)')
        ax2.legend(loc='upper right', fontsize=6)
        ax2.grid(True, linestyle=':', alpha=0.5)
    fig_real.suptitle('Real data (first 5 s, pre and post D‑serine)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    for clase in classes:
        real_pre_raw, real_post_raw = real_data[clase]
        syn_pre_raw, syn_post_raw = syn_data[clase]
        raw_datasets = [syn_pre_raw, syn_post_raw, real_pre_raw, real_post_raw]
        titles = [f'{clase} syn pre', f'{clase} syn post', f'{clase} real pre', f'{clase} real post']
        table = {}
        for cond, raw in zip(titles, raw_datasets):
            table[cond] = compute_band_powers(raw, bands, f)
        df_table = pd.DataFrame(table).T[['δ', 'θ', 'α', 'β', 'γ']].round(2)
        display(HTML(f"<h4>📊 Average band power (absolute dB) – {clase}</h4>"))
        display(HTML(df_table.to_html(classes='table table-striped', border=0, justify='center')))
    display(HTML(f"""
        <p style='color:#ccc;'>
        <b>General interpretation:</b> Less negative values in the tables = higher absolute power.
        After D‑serine, an increase in gamma (closer to 0 dB) and a reduction in slow waves (more negative) are expected.
        </p>
    """))

def plot_comparison_bars(td_pre_real, td_post_real, td_pre_syn, td_post_syn,
                         prob_pre_real, prob_post_real, prob_pre_syn, prob_post_syn):
    labels = ['ASD pre-treatment', 'ASD post-treatment']
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - 1.5*width, [td_pre_real, td_post_real], width, label='Real (% classified TD)', color='#2e86ab')
    ax.bar(x - 0.5*width, [td_pre_syn, td_post_syn], width, label='Synthetic (% classified TD)', color='#a23b72')
    ax.bar(x + 0.5*width, [prob_pre_real, prob_post_real], width, label='Real (mean TD prob.)', color='#59a14f')
    ax.bar(x + 1.5*width, [prob_pre_syn, prob_post_syn], width, label='Synthetic (mean TD prob.)', color='#f28e2b')
    ax.set_ylabel('Percentage / Probability (%)')
    ax.set_title('Effect of D‑serine on ASD phenotype\n(shift toward TD)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()
    display(HTML("<p style='color:#ccc;'><b>Interpretation:</b> Bars show % of ASD samples classified as TD and mean TD probability. "
                 "An increase after D‑serine indicates normalization of the electrophysiological signature.</p>"))

# ---------------------------------------------------------------------------
# Interface & Data storage
# ---------------------------------------------------------------------------
DATA = {
    'synthetic': None,
    'real_X': [], 'real_y': [],
    'signals_synthetic': None,
    'signals_real': [],
    'cnn_syn': None, 'cnn_real': None, 'scaler_real': None
}
out = widgets.Output()

btn_synthetic = widgets.Button(description="🧪 Generate synthetic + D‑serine", button_style='info')
btn_real_asd = widgets.Button(description="Upload ASD (ZIP)", button_style='warning')
btn_real_td  = widgets.Button(description="Upload TD (ZIP)", button_style='warning')
btn_real_scz = widgets.Button(description="Upload SCZ (ZIP)", button_style='warning')
btn_process = widgets.Button(description="🧠 Process real data (full)", button_style='success')
btn_crowding = widgets.Button(description="🧬 Analyze crowding metrics", button_style='danger')
status_label = widgets.HTML(value="<b>Real data:</b> 0 samples (ASD:0, TD:0, SCZ:0)")

def update_status():
    counts = {c: DATA['real_y'].count(c) for c in ['ASD','TD','SCZ']}
    total = sum(counts.values())
    status_label.value = f"<b>Real data:</b> {total} samples (ASD:{counts['ASD']}, TD:{counts['TD']}, SCZ:{counts['SCZ']})"

def ensure_synthetic():
    if DATA['synthetic'] is None:
        print("Generating synthetic model...")
        X, y, classes, scaler, signals = generate_dataset(SYNTHETIC_SAMPLES, return_signals=True)
        DATA['synthetic'] = (X, y, classes, scaler)
        DATA['signals_synthetic'] = signals
        print(f"✅ Synthetic signals stored: {len(signals)} samples")
        model, history, acc = train_and_evaluate(X, y)
        DATA['cnn_syn'] = model
        print(f"✅ Synthetic accuracy: {acc:.1f}%")
        plot_training(history, 'CNN Training – Synthetic data')
    return DATA['cnn_syn'], DATA['synthetic'][2]

def create_upload_callback(label):
    def callback(b):
        with out:
            clear_output()
            print(f"Upload ZIP file with {label} data (.set/.gdf/.edf):")
        uploaded = files.upload()
        if not uploaded:
            with out:
                print("No file selected.")
            return
        zip_bytes = list(uploaded.values())[0]
        with out:
            print(f"Processing {label} files...")
        try:
            specs, signals = process_uploaded_zip(zip_bytes, label, return_signals=True)
        except Exception as e:
            with out:
                print(f"❌ Error: {e}")
            return
        with out:
            if not specs:
                print(f"⚠️ No valid .set/.gdf/.edf files found in the ZIP for {label}.")
            else:
                DATA['real_X'].extend(specs)
                DATA['real_y'].extend([label]*len(specs))
                DATA['signals_real'].extend(signals)
                print(f"✅ {label}: {len(specs)} samples added (raw signals stored).")
            update_status()
    return callback

btn_real_asd.on_click(create_upload_callback('ASD'))
btn_real_td.on_click(create_upload_callback('TD'))
btn_real_scz.on_click(create_upload_callback('SCZ'))

def on_process(b):
    with out:
        clear_output()
        missing = [c for c in ['ASD','TD','SCZ'] if DATA['real_y'].count(c) == 0]
        if missing:
            print(f"❌ Missing real data for: {', '.join(missing)}. Please upload the corresponding ZIP files.")
            return
        model_syn, classes_syn = ensure_synthetic()
        f_ref = np.arange(N_FREQ) * (FS / NPERSEG)
        bands = {'δ': (0.5, 4), 'θ': (4, 8), 'α': (8, 13), 'β': (13, 30), 'γ': (30, 80)}
        sample = next((DATA['real_X'][i] for i, etq in enumerate(DATA['real_y']) if etq), None)
        if sample is None:
            print("No real data to determine shape.")
            return
        n_time_cols = sample.shape[1]
        time_step = (NPERSEG - NOVERLAP) / FS
        t_full = np.arange(n_time_cols) * time_step
        f = f_ref

        real_data = {}
        syn_data = {}
        for clase in ['ASD','SCZ','TD']:
            indices = [i for i, etq in enumerate(DATA['real_y']) if etq == clase]
            real_specs = [DATA['real_X'][i] for i in indices]
            real_pre = np.array(real_specs)[..., np.newaxis]
            if clase == 'TD':
                real_post = real_pre.copy()
            else:
                real_post = np.array([apply_dserine_spectrogram(s, f_ref) for s in real_specs])[..., np.newaxis]

            if clase == 'ASD':
                g_ei, g_ee, lf = 0.5, 0.9, 0.1
            elif clase == 'SCZ':
                g_ei, g_ee, lf = 0.2, 0.75, 0.4
            else:
                g_ei, g_ee, lf = 1.0, 1.0, 0.0
            syn_pre, syn_post = simulate_treatment(g_ei, g_ee, lf)
            real_data[clase] = (real_pre[0,:,:,0], real_post[0,:,:,0])
            syn_data[clase] = (syn_pre[0,:,:,0], syn_post[0,:,:,0])

            if clase != 'TD':
                td_pre_clas, td_post_clas, td_pre_prob, td_post_prob = evaluate_shift(
                    model_syn, real_pre, real_post, DATA['synthetic'][3], classes_syn)
                print(f"➤ {clase} real – Before: {td_pre_clas:.1f}% TD (prob: {td_pre_prob:.1f}%)  →  After: {td_post_clas:.1f}% TD (prob: {td_post_prob:.1f}%)")

        plot_separate_analysis(real_data, syn_data, bands, f, t_full)
        print("\n✅ Full analysis completed.")

btn_process.on_click(on_process)

def on_synthetic(b):
    with out:
        clear_output()
        print("Generating synthetic dataset...")
        X, y, classes, scaler, signals = generate_dataset(SYNTHETIC_SAMPLES, return_signals=True)
        DATA['synthetic'] = (X, y, classes, scaler)
        DATA['signals_synthetic'] = signals
        print(f"✅ Synthetic signals stored: {len(signals)} samples")
        print("Training CNN...")
        model, history, acc = train_and_evaluate(X, y)
        DATA['cnn_syn'] = model
        print(f"✅ Accuracy: {acc:.1f}%")
        plot_training(history, 'CNN Training – Synthetic data')
        print("Simulating D‑serine on ASD synthetic...")
        X_pre_asd, X_post_asd = simulate_treatment(0.5, 0.9, 0.1)
        td_clas_pre, td_clas_post, td_prob_pre, td_prob_post = evaluate_shift(model, X_pre_asd, X_post_asd, scaler, classes)
        print(f"➤ ASD – Before: {td_clas_pre:.1f}% TD (prob: {td_prob_pre:.1f}%)  →  After: {td_clas_post:.1f}% TD (prob: {td_prob_post:.1f}%)")
        print("Simulating D‑serine on SCZ synthetic...")
        X_pre_scz, X_post_scz = simulate_treatment(0.2, 0.75, 0.4)
        scz_clas_pre, scz_clas_post, scz_prob_pre, scz_prob_post = evaluate_shift(model, X_pre_scz, X_post_scz, scaler, classes)
        print(f"➤ SCZ – Before: {scz_clas_pre:.1f}% TD (prob: {scz_prob_pre:.1f}%)  →  After: {scz_clas_post:.1f}% TD (prob: {scz_prob_post:.1f}%)")
        raw_td = generate_raw_td(seed=0)
        plot_synthetic_overview(raw_td,
                                X_pre_asd[0,:,:,0], X_post_asd[0,:,:,0],
                                X_pre_scz[0,:,:,0], X_post_scz[0,:,:,0])
btn_synthetic.on_click(on_synthetic)

# ---------------------------------------------------------------------------
# CORRECTED: Analyze crowding metrics button (optimizado) con figuras en inglés y tabla descriptiva
# ---------------------------------------------------------------------------
def compute_crowding_metrics_subset(signal_list, max_samples=MAX_CROWDING_SAMPLES, label=""):
    """Calcula métricas para un subconjunto limitado de señales e imprime progreso."""
    if len(signal_list) > max_samples:
        signal_list = signal_list[:max_samples]
        print(f"   {label}: using {max_samples} of {len(signal_list)+max_samples} signals")
    chi, tau, mse, sync = [], [], [], []
    for i, sig in enumerate(signal_list):
        print(f"   {label} sample {i+1}/{len(signal_list)}...", end='\r')
        chi.append(compute_aperiodic_exponent(sig))
        tau.append(compute_tau(sig))
        mse_auc, _ = compute_multiscale_entropy(sig)
        mse.append(mse_auc)
        sync.append(compute_local_synchrony(sig, band='gamma'))
    print(f"   {label}: completed ({len(signal_list)} signals)                    ")
    return np.array(chi), np.array(tau), np.array(mse), np.array(sync)

def on_crowding_analysis(b):
    with out:
        print("🔄 'Analyze crowding metrics' button pressed. Checking data...")
        if DATA['signals_synthetic'] is None or len(DATA['signals_real']) == 0:
            missing = []
            if DATA['signals_synthetic'] is None:
                missing.append("- Synthetic data not generated.")
            if len(DATA['signals_real']) == 0:
                missing.append("- No real signals loaded.")
            print("❌ Missing data:\n" + "\n".join(missing))
            return

        clear_output()
        print(f"✅ Calculating macromolecular crowding metrics... (limited to {MAX_CROWDING_SAMPLES} samples per group)")

        # Organizar señales reales por clase
        real_signals = {'ASD': [], 'SCZ': [], 'TD': []}
        for sig, etq in zip(DATA['signals_real'], DATA['real_y']):
            real_signals[etq].append(sig)

        # Señales sintéticas (orden: TD, ASD, SCZ)
        n = SYNTHETIC_SAMPLES
        syn_signals = {
            'TD': DATA['signals_synthetic'][0:n],
            'ASD': DATA['signals_synthetic'][n:2*n],
            'SCZ': DATA['signals_synthetic'][2*n:3*n]
        }

        # Aplicar D‑serina a señales reales (excepto TD)
        real_post = {}
        for clase in ['ASD', 'SCZ']:
            real_post[clase] = [apply_dserine_signal(s) for s in real_signals[clase]]
        real_post['TD'] = real_signals['TD']

        syn_post = {}
        for clase in ['ASD', 'SCZ']:
            syn_post[clase] = [apply_dserine_signal(s) for s in syn_signals[clase]]
        syn_post['TD'] = syn_signals['TD']

        metric_dict = {}
        # Reales
        for clase in ['TD', 'ASD', 'SCZ']:
            if clase == 'TD':
                chi, tau, mse, sync = compute_crowding_metrics_subset(
                    real_signals['TD'], label='TD real')
                metric_dict['TD real'] = {'chi': chi, 'tau': tau, 'mse': mse, 'sync': sync}
            else:
                chi_pre, tau_pre, mse_pre, sync_pre = compute_crowding_metrics_subset(
                    real_signals[clase], label=f'{clase} real pre')
                chi_post, tau_post, mse_post, sync_post = compute_crowding_metrics_subset(
                    real_post[clase], label=f'{clase} real post')
                metric_dict[f'{clase} real pre'] = {'chi': chi_pre, 'tau': tau_pre, 'mse': mse_pre, 'sync': sync_pre}
                metric_dict[f'{clase} real post'] = {'chi': chi_post, 'tau': tau_post, 'mse': mse_post, 'sync': sync_post}
        # Sintéticos
        for clase in ['TD', 'ASD', 'SCZ']:
            if clase == 'TD':
                chi, tau, mse, sync = compute_crowding_metrics_subset(
                    syn_signals['TD'], label='TD syn')
                metric_dict['TD syn'] = {'chi': chi, 'tau': tau, 'mse': mse, 'sync': sync}
            else:
                chi_pre, tau_pre, mse_pre, sync_pre = compute_crowding_metrics_subset(
                    syn_signals[clase], label=f'{clase} syn pre')
                chi_post, tau_post, mse_post, sync_post = compute_crowding_metrics_subset(
                    syn_post[clase], label=f'{clase} syn post')
                metric_dict[f'{clase} syn pre'] = {'chi': chi_pre, 'tau': tau_pre, 'mse': mse_pre, 'sync': sync_pre}
                metric_dict[f'{clase} syn post'] = {'chi': chi_post, 'tau': tau_post, 'mse': mse_post, 'sync': sync_post}

        # ---- FIGURA DE BOXPLOTS (con títulos y etiquetas en inglés) ----
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metric_names = ['Aperiodic exponent (χ)', 'Time constant (τ) [s]',
                        'Multiscale entropy (AUC)', 'Gamma synchrony (PLV)']
        keys = ['chi', 'tau', 'mse', 'sync']
        condition_order = ['TD real', 'TD syn',
                           'ASD real pre', 'ASD real post', 'ASD syn pre', 'ASD syn post',
                           'SCZ real pre', 'SCZ real post', 'SCZ syn pre', 'SCZ syn post']
        palette = plt.cm.tab10(np.linspace(0,1,len(condition_order)))
        for idx, (ax, key, name) in enumerate(zip(axes.flat, keys, metric_names)):
            positions = []
            data_vals = []
            labels = []
            for i, cond in enumerate(condition_order):
                if cond in metric_dict and key in metric_dict[cond]:
                    val = metric_dict[cond][key]
                    val = val[~np.isnan(val)]
                    if len(val) > 0:
                        positions.append(i+1)
                        data_vals.append(val)
                        labels.append(cond)
            bp = ax.boxplot(data_vals, positions=positions, widths=0.6, patch_artist=True)
            for patch, j in zip(bp['boxes'], range(len(positions))):
                patch.set_facecolor(palette[j % len(palette)])
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle=':', alpha=0.5)
        fig.suptitle('Macromolecular crowding metrics – real vs synthetic (pre/post D‑serine)', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # ---- TABLA CON ESTADÍSTICOS DESCRIPTIVOS (media ± std) ----
        rows = []
        for cond in condition_order:
            if cond in metric_dict:
                row = {'Condition': cond}
                for key in keys:
                    vals = metric_dict[cond][key]
                    vals_clean = vals[~np.isnan(vals)]
                    if len(vals_clean) > 0:
                        row[key] = f"{np.mean(vals_clean):.3f} ± {np.std(vals_clean):.3f}"
                    else:
                        row[key] = "N/A"
                rows.append(row)
        df_metrics = pd.DataFrame(rows)
        df_metrics = df_metrics[['Condition', 'chi', 'tau', 'mse', 'sync']]
        display(HTML("<h4>📊 Crowding metrics descriptive statistics (mean ± std)</h4>"))
        display(HTML(df_metrics.to_html(classes='table table-striped', border=0, justify='center', index=False)))

        # Texto interpretativo en inglés
        display(HTML("""
        <p style='color:#ccc;'>
        <b>Interpretation of metrics:</b><br>
        • <b>Aperiodic exponent χ:</b> a more negative χ indicates a steeper 1/f slope, reflecting slower synaptic integration times, potentially linked to a more viscous intracellular environment.<br>
        • <b>Time constant τ:</b> a larger τ suggests slower decay of postsynaptic potentials, consistent with delayed ion channel kinetics due to crowding.<br>
        • <b>Multiscale entropy (AUC):</b> lower complexity (smaller AUC) implies reduced dynamical degrees of freedom, possibly from molecular crowding constraints.<br>
        • <b>Gamma synchrony (PLV):</b> increased local phase locking in the gamma band may reflect hypersynchrony driven by loss of functional diversity.<br>
        D‑serine treatment is expected to partially normalize these markers, especially in ASD and SCZ.
        </p>
        """))

btn_crowding.on_click(on_crowding_analysis)

# ---------------------------------------------------------------------------
# Actualizador de estado y UI
# ---------------------------------------------------------------------------
def updater():
    while True:
        try: update_status()
        except: pass
        time.sleep(2)
threading.Thread(target=updater, daemon=True).start()

ui = widgets.VBox([
    widgets.HTML("<h2>🧠 D‑serine Study: real data upload + full processing + crowding metrics</h2>"),
    widgets.HBox([btn_synthetic, btn_process, btn_crowding]),
    widgets.HTML("<h4>Upload ZIP files (ASD, TD, SCZ) containing .set, .gdf or .edf recordings:</h4>"),
    widgets.HBox([btn_real_asd, btn_real_td, btn_real_scz]),
    status_label,
    out
])
display(ui)