import numpy as np
import pywt
import librosa
import logging
from src.process_audio import AudioProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EventDetection:
    def __init__(self, target_sr=22050, threshold_factor=0.1, min_duration=0.5, apply_bp=False, bpll=150, bpul=750):
        self.target_sr = target_sr
        self.threshold_factor = threshold_factor
        self.min_duration = min_duration
        self.apply_bp = apply_bp
        self.bpll = bpll
        self.bpul = bpul
        self.audio_processor = AudioProcessor(target_sr=target_sr)

    def min_max_freq(self, y, sr, threshold_factor=0.01, fallback_min_freq=20, fallback_max_freq=20000):
        D = np.abs(librosa.stft(y))
        frequencies = librosa.fft_frequencies(sr=sr)
        average_magnitude = np.mean(D, axis=1)
        threshold = np.max(average_magnitude) * threshold_factor
        significant_indices = np.where(average_magnitude > threshold)[0]

        if len(significant_indices) == 0:
            logging.info("No significant frequencies found, using fallback values.")
            return fallback_min_freq, fallback_max_freq

        min_freq = frequencies[significant_indices[0]]
        max_freq = frequencies[significant_indices[-1]]

        if min_freq <= 0 or max_freq <= 0:
            logging.info(f"Detected min_freq={min_freq}, max_freq={max_freq}. Using fallback values.")
            return fallback_min_freq, fallback_max_freq

        return min_freq, max_freq

    def calculate_scales(self, y, wavelet, sr, num_scales=20):
        dt = 1 / sr
        min_freq, max_freq = self.min_max_freq(y, sr)
        min_scale = pywt.scale2frequency(wavelet, 1) / (max_freq * dt)
        max_scale = pywt.scale2frequency(wavelet, 1) / (min_freq * dt)

        if not np.isfinite(min_scale) or not np.isfinite(max_scale):
            raise ValueError("Computed scales are not finite.")

        return np.logspace(np.log10(min_scale), np.log10(max_scale), num=num_scales)

    def wavelet_transform(self, y, sr, wavelet='morl', scales=None):
        if scales is None:
            scales = self.calculate_scales(y, wavelet, sr)
        coeffs, _ = pywt.cwt(y, scales=scales, wavelet=wavelet)
        return coeffs

    def wavelet_features(self, coeffs):
        return np.sum(np.abs(coeffs), axis=0)

    def moving_average(self, wavelet_features, sr, window_size=0.15):
        window_size = int(sr * window_size)
        return np.convolve(wavelet_features, np.ones(window_size)/window_size, 'same')

    def smoothed_features(self, y, sr, wavelet='morl', window_size=0.15):
        coeffs = self.wavelet_transform(y, sr, wavelet=wavelet)
        features = self.wavelet_features(coeffs)
        return self.moving_average(features, sr, window_size=window_size)

    def detected_events(self, features, sr):
        adaptive_threshold = self.threshold_factor * np.mean(features)
        detected_events = features > adaptive_threshold
        event_indices = np.where(detected_events)[0]

        if len(event_indices) == 0:
            return []

        events = []
        start = event_indices[0]
        for i in range(1, len(event_indices)):
            if event_indices[i] != event_indices[i - 1] + 1:
                end = event_indices[i - 1]
                events.append((start, end))
                start = event_indices[i]
        events.append((start, event_indices[-1]))

        event_times = [(start / sr, end / sr) for start, end in events]
        return [(start, end) for start, end in event_times if (end - start) >= self.min_duration]

    def detected_times(self, file_path):
        y, y_denoised, sr = self.audio_processor.load_audio(
            file_path, bandpass=self.apply_bp, bandpass_lower_limit=self.bpll, bandpass_upper_limit=self.bpul
        )
        features = self.smoothed_features(y_denoised, sr, wavelet='mexh', window_size=0.15)
        event_times = self.detected_events(features=features, sr=sr)

        return y, y_denoised, sr, event_times
