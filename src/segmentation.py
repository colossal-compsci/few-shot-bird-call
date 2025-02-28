import numpy as np
import os
import pandas as pd
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import firwin, filtfilt,lfilter,butter
#from src.utils import match_target_amplitude
from utils import match_target_amplitude
from pathlib import Path

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

# Functions from ipynotebook ------------------------------------------
def find_csv_file(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            return os.path.join(directory, file)
    return None

def load_metadata(csv_path):
    return pd.read_csv(csv_path)

def find_csv_file(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            return os.path.join(directory, file)
    return None

def save_segment(segment, sr, original_filename, start, end, species, output_dir):
    species_dir = os.path.join(output_dir, species)
    os.makedirs(species_dir, exist_ok=True)

    segment_filename = os.path.join(species_dir, f"{os.path.splitext(original_filename)[0]}_{int(start*1000)}_{int(end*1000)}.wav")
    sf.write(segment_filename, segment, sr)
    #print(f"Saved segment: {segment_filename}")

def load_metadata(csv_path):
    return pd.read_csv(csv_path)

def create_directory_structure(base_path):
    """
    Create the following folder structure:
    base_path/
        |---segmented_audio
            |---raw
            |---bandpass
            |---denoised
    """
    subfolders = ['segmented_audio']
    subsubfolders = ['raw', 'bandpass', 'denoised']

    for subfolder in subfolders:
        for subsubfolder in subsubfolders:
            folder_path = os.path.join(base_path, subfolder, subsubfolder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created directory: {folder_path}")

def fir_bandpass_filter(data, lowcut, highcut, fs, numtaps=2048, window='hamming'):
    nyquist = 0.5 * fs
    low = (lowcut) / nyquist
    high = (highcut) / nyquist
    b = firwin(numtaps, [low, high], pass_zero=False, window=window)
    y = filtfilt(b, 1.0, data)
    return y

def highpass_filter(data, cutoff_freq, fs, order=5):
    nyquist = 0.5 * fs
    cutoff = cutoff_freq / nyquist
    b, a = butter(order, cutoff, btype='high', analog=False, output='ba')
    y = filtfilt(b, a, data)
    return y

def noisereduce_wav(y, sr):
    epsilon = 1e-10
    y = nr.reduce_noise(y=y+epsilon,stationary=False, sr=sr,n_fft=2048,
                        prop_decrease=0.99,
                        win_length=2048,
                        hop_length=512,
                        time_mask_smooth_ms=50,
                        freq_mask_smooth_hz=50,
                        n_jobs=-1)
    return y

def normalize_amplitude(original, processed):
    original_rms = np.sqrt(np.mean(original**2))
    processed_rms = np.sqrt(np.mean(processed**2))
    if processed_rms == 0:  # Avoid division by zero
        return processed
    gain = original_rms / processed_rms
    return processed * gain

def match_target_amplitude(y, target_dBFS=-20):
    rms = (y ** 2).mean() ** 0.5
    scalar = 10 ** (target_dBFS / 20) / (rms + 1e-9)
    return y * scalar

def apply_fade(audio, sr, fade_duration=0.1):
    fade_samples = int(fade_duration * sr)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out

    return audio

def pad_audio_with_clipping(audio, target_duration, sample_rate, original_audio, start_idx, end_idx):
    """
    Pad the audio with clipping from just before and after the segment.

    Args:
    audio (np.array): The audio data to pad.
    target_duration (float): The desired duration in seconds.
    sample_rate (int): The sample rate of the audio.
    original_audio (np.array): The original audio data to use for padding.
    start_idx (int): The start index of the segment in the original audio.
    end_idx (int): The end index of the segment in the original audio.

    Returns:
    np.array: The padded audio data.
    """
    current_samples = len(audio)
    target_samples = int(target_duration * sample_rate)

    if current_samples < target_samples:
        pad_length = target_samples - current_samples
        pad_left = original_audio[max(0, start_idx - pad_length // 2):start_idx]
        pad_right = original_audio[end_idx:min(len(original_audio), end_idx + pad_length - len(pad_left))]

        # If not enough padding on either side, use remaining padding from the other side
        if len(pad_left) + len(pad_right) < pad_length:
            if len(pad_left) < pad_length // 2:
                pad_right = original_audio[end_idx:end_idx + pad_length - len(pad_left)]
            if len(pad_right) < pad_length // 2:
                pad_left = original_audio[start_idx - pad_length + len(pad_right):start_idx]

        padded_audio = np.concatenate([pad_left, audio, pad_right])
        #print(f"Padding audio from {current_samples} to {target_samples} samples")
        return padded_audio
    else:
        return audio[:target_samples]  # Truncate if longer than target duration

def process_and_segment_audio(audio, sr, df_file, output_base_path, fmin, fmax, segment_length=2.0):
    """
    Process and segment audio files based on metadata from a CSV file.

    Args:
    audio (np.array): The audio data.
    sr (int): The sample rate of the audio.
    df_file (pd.DataFrame): DataFrame containing metadata for the audio file.
    output_base_path (str): Base path for output directories.
    segment_length (float): Length of each segment in seconds.
    overlap_percent (float): Percentage of overlap between adjacent segments.
    """
    #print("Applying band-pass filter...")
    # filtered_audio = highpass_filter(audio, 150, sr)
    filtered_audio = fir_bandpass_filter(audio, fmin, fmax, sr)

    filtered_audio = match_target_amplitude(filtered_audio)

    #print("Applying noise reduction to entire audio...")
    denoised_audio = noisereduce_wav(filtered_audio, sr)

    denoised_audio = match_target_amplitude(denoised_audio)

    raw_dir = os.path.join(output_base_path, 'segmented_audio', 'raw')
    bandpass_dir = os.path.join(output_base_path, 'segmented_audio', 'bandpass')
    denoised_dir = os.path.join(output_base_path, 'segmented_audio', 'denoised')

    original_segments = []
    bandpass_segments = []
    denoised_segments = []

    for _, row in df_file.iterrows():
        start_time = round(row['start_time'], 2)
        end_time = round(row['end_time'], 2)
        species = row['species'].replace(" ", "_")
        duration = end_time - start_time

        #print(f"Processing segment for {species} from {start_time} to {end_time}")

        if duration <= segment_length:
            # For segments shorter than or equal to 1.5 seconds
            segment_starts = [
                start_time,
                # max(round(start_time - (segment_length - overlap_percent * segment_length), 2), 0),
                # min(round(start_time + (segment_length - overlap_percent * segment_length), 2), len(audio) / sr - segment_length)
            ]
            # segment_ends = [segment_starts[0] + duration, segment_starts[1] + duration ,min(segment_starts[2] + duration, end_time)]
            segment_ends = [segment_starts[0] + duration]
        else:
            # For segments longer than 1.5 seconds
            segment_starts = []
            current_start = start_time
            while current_start < end_time:
                segment_starts.extend([
                    # max(round(current_start - (segment_length - overlap_percent * segment_length), 2), 0),
                    current_start,
                    # min(round(current_start + (segment_length - overlap_percent * segment_length), 2), len(audio) / sr - segment_length)
                ])
                current_start += segment_length  # Move by segment_length each time

            # Remove duplicates and sort
            segment_starts = sorted(set(segment_starts))
            segment_ends = [min(start + segment_length, end_time) for start in segment_starts]

            # Remove the last segment if it's shorter than segment_length
            if segment_ends[-1] - segment_starts[-1] < segment_length:
                segment_starts.pop()
                segment_ends.pop()

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            #print(f"  Extracting segment from {seg_start} to {seg_end}")

            # Extract segments
            start_idx = int(seg_start * sr)
            end_idx = int(seg_end * sr)
            original_segment = audio[start_idx:end_idx]
            bandpass_segment = filtered_audio[start_idx:end_idx]
            denoised_segment = denoised_audio[start_idx:end_idx]

            # Pad segments if necessary
            original_segment = pad_audio_with_clipping(original_segment, segment_length, sr, audio, start_idx, end_idx)
            bandpass_segment = pad_audio_with_clipping(bandpass_segment, segment_length, sr, filtered_audio, start_idx, end_idx)
            denoised_segment = pad_audio_with_clipping(denoised_segment, segment_length, sr, denoised_audio, start_idx, end_idx)

            # Save raw segment
            save_segment(original_segment, sr, row['filename'], seg_start, seg_end, species, raw_dir)
            save_segment(bandpass_segment, sr, row['filename'], seg_start, seg_end, species, bandpass_dir)
            save_segment(denoised_segment, sr, row['filename'], seg_start, seg_end, species, denoised_dir)

            # add to growing lists 
            original_segments.append(original_segment)
            bandpass_segments.append(bandpass_segment)
            denoised_segments.append(denoised_segment)

    return original_segments, bandpass_segments, denoised_segments

def process_all_audio_files(df, base_audio_path, output_base_path, fmin=150, fmax=650, segment_length=2.0):

    grouped = df.groupby('filename')

    denoised_segments = []
    for filename, group_df in grouped:
        input_path = os.path.join(base_audio_path, filename)
        if os.path.exists(input_path):
            ##print(f"Processing file: {input_path}")
            #breakpoint()
            input_path = str(Path(input_path))

            audio, sr = librosa.load(input_path)
            _, _, ds = process_and_segment_audio(audio, sr, 
                                group_df, output_base_path, fmin = fmin, 
                                fmax=fmax, segment_length=segment_length)
        else:
            print(f"File not found: {input_path}")
            denoised_segments = []
        denoised_segments += ds

    return denoised_segments

def audio_from_event_times(raw_dir,processed_dir,species_label,bandpass_lower_limit,bandpass_upper_limit):
    """
    Load audio using AudioProcessor, process it to detect events,
    and return event start and end times along with corresponding audio segments.
    """
    csv_file_path = find_csv_file(raw_dir)

    if csv_file_path:
        metadata_df = load_metadata(csv_file_path)
        if metadata_df is not None:
            denoised_segments = process_all_audio_files(metadata_df, 
                                        raw_dir, processed_dir,
                                        fmin = bandpass_lower_limit,
                                        fmax = bandpass_upper_limit)

    #y, y_denoised, sr = self.audio_processor.load_audio(file_path)  # Use AudioProcessor to load audio
    #features = self.smoothed_features(y_denoised, sr, wavelet='mexh', window_size=0.15)
    #event_times = self.detected_events(features=features, sr=sr)

    ## Extract segments based on detected event times
    #y_segments = []
    #y_denoised_segments = []
    #for start_time, end_time in event_times:
        #start_sample = int(start_time * sr)
        #end_sample = int(end_time * sr)
        #y_segments.append(y[start_sample:end_sample])
        #y_denoised_segments.append(y_denoised[start_sample:end_sample])

    #return event_times, y_segments, y_denoised_segments, sr
    return denoised_segments # these should be padded too

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
