import librosa
from src.utils import apply_bandpass_filter, match_target_amplitude, reduce_noise  # Import from utils

class AudioProcessor:
    def __init__(self, target_sr=22050, target_dBFS=-20, bandpass_lowcut=150, bandpass_highcut=600):
        self.target_sr = target_sr
        self.target_dBFS = target_dBFS
        self.bandpass_lowcut = bandpass_lowcut
        self.bandpass_highcut = bandpass_highcut

    def load_audio(self, file_path):
        """
        Load and resample an audio file, then process with bandpass filtering, 
        volume matching, noise reduction, and normalization.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            tuple: Original and processed audio time series, and sampling rate.
        """
        y, sr = librosa.load(file_path, sr=self.target_sr)
        
        # Apply processing steps using functions from utils.py
        y_processed = apply_bandpass_filter(y, self.bandpass_lowcut, self.bandpass_highcut, sr)
        y_processed = match_target_amplitude(y_processed, self.target_dBFS)
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
