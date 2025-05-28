import numpy as np
from src.utils import match_target_amplitude

class Segment:
    def __init__(self, segment_length=2.0, target_dBFS=-20):
        """
        Initialize the Segment class with default parameters.

        Args:
            segment_length (float): Desired segment length in seconds.
            target_dBFS (float): Target dBFS level for amplitude matching.
        """
        self.segment_length = segment_length
        self.target_dBFS = target_dBFS

    def pad_audio_with_clipping(self, audio, sample_rate):
        """
        Pad or truncate the audio segment to the target duration.

        Args:
            audio (np.array): The audio segment to pad or truncate.
            sample_rate (int): The sample rate of the audio.

        Returns:
            np.array: The padded or truncated audio segment.
        """
        target_samples = int(self.segment_length * sample_rate)
        current_samples = len(audio)

        if current_samples < target_samples:
            # Calculate padding required on both sides
            pad_length = target_samples - current_samples
            pad_left = audio[:pad_length // 2]
            pad_right = audio[-(pad_length - len(pad_left)):]
            
            padded_audio = np.concatenate([pad_left, audio, pad_right])
            return padded_audio[:target_samples]  # Ensure exact length if needed
        else:
            return audio[:target_samples]  # Truncate if longer than target duration

    def segment_audio(self, segments, sr):
        """
        Pad or truncate a list of segments with consistent amplitude and length.

        Args:
            segments (list of np.array): List of audio segments to process.
            sr (int): Sample rate of the audio.

        Returns:
            list of np.array: List of padded and amplitude-matched audio segments.
        """
        padded_segments = []
        for segment in segments:
            # Pad or truncate to target length and apply amplitude matching
            padded_segment = self.pad_audio_with_clipping(segment, sr)
            padded_segment = match_target_amplitude(padded_segment, self.target_dBFS)
            padded_segments.append(padded_segment)
        
        return padded_segments


# from event_detection import EventDetection
# filepath = "./data/test.wav"
# # Assuming `EventDetection` provides the original and denoised segments
# event_detector = EventDetection()
# event_times, y_segments, y_denoised_segments, sr = event_detector.detected_times(filepath)

# # Process segments
# segment_processor = Segment()


# # Pad and adjust amplitude for the segments
# padded_original_segments = segment_processor.segment_audio(y_segments, sr)
# padded_denoised_segments = segment_processor.segment_audio(y_denoised_segments, sr)

# # `padded_original_segments` and `padded_denoised_segments` are now lists of processed audio segments



# print(f"Original segments: {len(padded_original_segments)}")
# print(f"Denoised segments: {len(padded_denoised_segments)}")
# print(f"Segment duration: {len(padded_original_segments[0])} seconds")
