import librosa
from src.utils import apply_bandpass_filter, match_target_amplitude, reduce_noise  # Import from utils

class AudioProcessor:
    def __init__(self, target_sr=22050, target_dBFS=-20):
        self.target_sr = target_sr
        self.target_dBFS = target_dBFS

    def load_audio(self, file_path, bandpass = False, bandpass_lower_limit = 1, bandpass_upper_limit = 8192):
        """
        Load and resample an audio file, then process with bandpass filtering, 
        volume matching, noise reduction, and normalization.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            tuple: Original and processed audio time series, and sampling rate.
        """
        y, sr = librosa.load(file_path, sr=self.target_sr)
        # Apply bandpass filter
        if bandpass:
            y = apply_bandpass_filter(y, bandpass_lower_limit, bandpass_upper_limit, sr)
        # Apply processing steps using functions from utils.py
        y_processed = match_target_amplitude(y, self.target_dBFS)
        y_processed = reduce_noise(y_processed, sr)
        y_processed = match_target_amplitude(y_processed, self.target_dBFS)
        
        return y, y_processed, sr

# # File path for testing
# filepath = "./data/test.wav"

# # Example usage
# processor = AudioProcessor()
# original_audio, processed_audio, sample_rate = processor.load_audio(filepath)

# print ("Original audio shape:", original_audio.shape)
# print ("Processed audio shape:", processed_audio.shape)
# print ("Sample rate:", sample_rate)
