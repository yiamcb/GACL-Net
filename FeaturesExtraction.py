import numpy as np
import mne  # For EEG processing and feature extraction
from scipy.signal import hilbert, coherence, welch
from tqdm import tqdm

eeg_data_reshaped = np.array(eeg_data).reshape(50 * 40, 33, 4000) 
mi_labels_reshaped = np.array(mi_labels).flatten() 

window_size = 1000
step_size = 500 
fs = 500

num_samples = eeg_data_reshaped.shape[0]
num_channels = eeg_data_reshaped.shape[1]
num_windows = (eeg_data_reshaped.shape[2] - window_size) // step_size + 1

features_matrix = np.zeros((num_samples, num_channels, num_windows, 8))  # 8 features per window

# Functions for feature calculations
def calculate_band_power(signal, fs, band):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[band_idx], freqs[band_idx])

def hilbert_huang_transform(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return np.mean(amplitude_envelope), np.std(amplitude_envelope)

def calculate_coherence(signal1, signal2, fs):
    f, Cxy = coherence(signal1, signal2, fs=fs, nperseg=fs*2)
    return np.mean(Cxy)

def calculate_erd_ers(signal, fs, motor_band):
    return calculate_band_power(signal, fs, motor_band)  # Placeholder for ERD/ERS calculation

def fractal_dimension(signal):
    return np.log(np.std(signal) / np.mean(signal))  # Placeholder calculation

def lyapunov_exponent(signal):
    return np.mean(np.diff(signal))  # Placeholder calculation

# Feature extraction process
for sample_idx in tqdm(range(num_samples), desc='Processing Samples'):
    for channel_idx in range(num_channels):
        channel_data = eeg_data_reshaped[sample_idx, channel_idx, :]

        for window_idx, start in enumerate(range(0, channel_data.shape[0] - window_size + 1, step_size)):
            window_data = channel_data[start:start + window_size]

            # Feature calculations for each window
            alpha_power = calculate_band_power(window_data, fs, band=(8, 13))
            beta_power = calculate_band_power(window_data, fs, band=(13, 30))
            mean_envelope, std_envelope = hilbert_huang_transform(window_data)
            fractal_dim = fractal_dimension(window_data)
            lyapunov_exp = lyapunov_exponent(window_data)

            if channel_idx < num_channels - 1:
                coherence_val = calculate_coherence(window_data, eeg_data_reshaped[sample_idx, channel_idx + 1, start:start + window_size], fs)
            else:
                coherence_val = calculate_coherence(window_data, window_data, fs)  # Self-coherence as fallback

            erd_ers_val = calculate_erd_ers(window_data, fs, motor_band=(8, 13))

            features_matrix[sample_idx, channel_idx, window_idx, :] = [
                alpha_power, beta_power, mean_envelope, std_envelope,
                coherence_val, erd_ers_val, fractal_dim, lyapunov_exp
            ]

print("Features matrix shape:", features_matrix.shape)