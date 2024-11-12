import numpy as np
from scipy.signal import filtfilt, firwin
import noisereduce as nr

def apply_bandpass_filter(data, lowcut, highcut, fs, numtaps=2048, window='hamming'):
    """
    Design and apply an FIR bandpass filter to the input data.

    Args:
        data (array): Input audio signal.
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        fs (float): Sampling frequency.
        numtaps (int): Number of filter taps (default: 2048).
        window (str): Window function for filter design (default: 'hamming').

    Returns:
        array: Filtered audio signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b = firwin(numtaps, [low, high], pass_zero=False, window=window)
    b /= np.sum(b)  # Normalize filter coefficients
    return filtfilt(b, 1.0, data)

def match_target_amplitude(y, target_dBFS):
    """
    Standardize volume of audio clip to target dBFS.

    Args:
        y (array): Audio time series.
        target_dBFS (float): Target dBFS level.

    Returns:
        array: Audio time series with standardized volume.
    """
    rms = (y ** 2).mean() ** 0.5
    scalar = 10 ** (target_dBFS / 20) / (rms + 1e-9)
    return y * scalar

def reduce_noise(y, sr):
    """
    Apply noise reduction on the audio signal.

    Args:
        y (array): Audio time series.
        sr (int): Sampling rate.

    Returns:
        array: Denoised audio signal.
    """
    epsilon = 1e-10
    return nr.reduce_noise(
        y=y + epsilon,
        sr=sr,
        stationary=False,
        n_fft=2048,
        prop_decrease=0.99,
        win_length=2048,
        hop_length=512,
        time_mask_smooth_ms=50,
        freq_mask_smooth_hz=50,
        n_jobs=-1
    )
