import numpy as np
import pywt
import librosa
from src.process_audio import AudioProcessor  # Import AudioProcessor from process_audio.py

class EventDetection:
    def __init__(self, target_sr=22050, threshold_factor=0.1, min_duration=0.5):
        self.target_sr = target_sr
        self.threshold_factor = threshold_factor
        self.min_duration = min_duration
        self.audio_processor = AudioProcessor(target_sr=target_sr)  # Instantiate AudioProcessor

    def min_max_freq(self, y, sr, threshold_factor=0.01, fallback_min_freq=20, fallback_max_freq=20000):
        """
        Calculate the minimum and maximum significant frequencies in the audio signal.
        """
        D = np.abs(librosa.stft(y))
        frequencies = librosa.fft_frequencies(sr=sr)
        average_magnitude = np.mean(D, axis=1)
        threshold = np.max(average_magnitude) * threshold_factor
        significant_indices = np.where(average_magnitude > threshold)[0]

        if len(significant_indices) == 0:
            return fallback_min_freq, fallback_max_freq

        min_freq = frequencies[significant_indices[0]]
        max_freq = frequencies[significant_indices[-1]]

        if min_freq <= 0 or max_freq <= 0:
            return fallback_min_freq, fallback_max_freq

        return min_freq, max_freq

    def calculate_scales(self, y, wavelet, sr, num_scales=20):
        """
        Calculate wavelet scales based on significant frequencies.
        """
        dt = 1 / sr
        min_freq, max_freq = self.min_max_freq(y, sr)
        min_scale = pywt.scale2frequency(wavelet, [1])[0] / (max_freq * dt)
        max_scale = pywt.scale2frequency(wavelet, [1])[0] / (min_freq * dt)
        
        if not np.isfinite(min_scale) or not np.isfinite(max_scale):
            raise ValueError("Computed scales are not finite.")
        
        return np.logspace(np.log10(min_scale), np.log10(max_scale), num=num_scales)

    def wavelet_transform(self, y, sr, wavelet='morl', scales=None):
        """
        Apply Continuous Wavelet Transform (CWT) to the audio signal.
        """
        if scales is None:
            scales = self.calculate_scales(y, wavelet, sr)
        coeffs, _ = pywt.cwt(y, scales=scales, wavelet=wavelet)
        return coeffs

    def wavelet_features(self, coeffs):
        """
        Sum the absolute values of wavelet coefficients along scales.
        """
        return np.sum(np.abs(coeffs), axis=0)

    def moving_average(self, wavelet_features, sr, window_size=0.15):
        """
        Smooth wavelet features using a moving average filter.
        """
        window_size = int(sr * window_size)
        return np.convolve(wavelet_features, np.ones(window_size)/window_size, 'same')

    def smoothed_features(self, y, sr, wavelet='morl', window_size=0.15):
        """
        Compute smoothed wavelet features of the audio signal.
        """
        coeffs = self.wavelet_transform(y, sr, wavelet=wavelet)
        features = self.wavelet_features(coeffs)
        return self.moving_average(features, sr, window_size=window_size)

    def detected_events(self, features, sr):
        """
        Detect events in the smoothed features based on an adaptive threshold.
        """
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
        """
        Load audio using AudioProcessor, process it to detect events, 
        and return event start and end times along with corresponding audio segments.
        """
        y, y_denoised, sr = self.audio_processor.load_audio(file_path)  # Use AudioProcessor to load audio
        features = self.smoothed_features(y_denoised, sr, wavelet='mexh', window_size=0.15)
        event_times = self.detected_events(features=features, sr=sr)

        # Extract segments based on detected event times
        y_segments = []
        y_denoised_segments = []
        for start_time, end_time in event_times:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            y_segments.append(y[start_sample:end_sample])
            y_denoised_segments.append(y_denoised[start_sample:end_sample])

        return event_times, y_segments, y_denoised_segments, sr

# # Example usage
# filepath = "./data/test.wav"
# event_detector = EventDetection()
# event_times, y_segments, y_denoised_segments, _ = event_detector.detected_times(filepath)
# print("Detected Events:", event_times)
# print("Original Audio Segments:", y_segments[1].shape)
# print("Denoised Audio Segments:", y_denoised_segments[0].shape)
