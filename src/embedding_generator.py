import numpy as np
import os
import tensorflow_hub as hub
from event_detection import EventDetection
from segmentation import Segment
from segmentation import audio_from_event_times 

class EmbeddingGenerator:
    def __init__(self, model_url='https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8', sr=32000, target_samples=160000):
        """
        Initialize the EmbeddingGenerator with the Bird Vocalization Classifier model.

        Args:
            model_url (str): URL to load the Bird Vocalization Classifier model.
            sr (int): Sample rate for the audio segments (default is 32000).
            target_samples (int): Target number of samples for each segment (default is 160000).
        """
        self.model = hub.load(model_url)
        self.sr = sr
        self.target_samples = target_samples

    def preprocess_segment(self, segment):
        """
        Preprocesses a single audio segment by padding or truncating to the target sample length.

        Args:
            segment (np.array): The audio segment to preprocess.

        Returns:
            np.array: Padded or truncated audio segment ready for embedding.
        """
        current_samples = len(segment)
        
        if current_samples < self.target_samples:
            # Calculate padding lengths and pad with zeros
            pad_length = self.target_samples - current_samples
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            padded_segment = np.pad(segment, (pad_left, pad_right), 'constant')
            return padded_segment
        else:
            # Truncate if longer than target duration
            return segment[:self.target_samples]

    def generate_embeddings(self, segments):
        """
        Generates embeddings for a list of audio segments using the Bird Vocalization Classifier model.

        Args:
            segments (list of np.array): List of preprocessed audio segments.

        Returns:
            np.array: Array of embeddings for each segment.
        """
        embeddings = []
        
        for segment in segments:
            # Preprocess each segment to match the model's target sample length
            processed_segment = self.preprocess_segment(segment)
            
            # Reshape to fit the model's expected input dimensions
            audio_input = processed_segment[np.newaxis, :]  # Shape (1, 160000)
            
            # Generate embeddings using the model
            result = self.model.infer_tf(audio_input)
            embedding = result['embedding']
            
            # Flatten and store the embedding
            embeddings.append(embedding.numpy().flatten())

        return np.array(embeddings)


# from src.embedding_generator import EmbeddingGenerator

# Initialize the event detector and segment processor
event_detector = EventDetection()
segment_processor = Segment()

# # Define the base path to your data file
# data_path = "./data/test.wav"

# # Detect events and get the denoised segments
#_, _, padded_denoised_segments, sr = event_detector.detected_times(data_path)
#padded_denoised_segments = segment_processor.segment_audio(padded_denoised_segments, sr)

#base_dir = '/home/jupyter/data/us_train_data/carolina_wren/raw/'
#base_dir = "/home/leah_colossal_com/us_bird_data"

species = ['white_throated_sparrow','northern_cardinal',
                'carolina_wren','eastern_towhee',
                'kentucky_warbler','kentucky_warbler_test']

# define the bandpass filters that should be used for each species, 
# this helps with denoising, but you can also just set to max limits
bp_filters = {'white_throated_sparrow':[1,8192],
                  'northern_cardinal':[1,8192],
                  'carolina_wren':[1,8192],
                  'eastern_towhee':[1,8192],
                  'kentucky_warbler':[1,8192]}
# test set kentucky warbler should have the same bp filter as train
bp_filters['kentucky_warbler_test'] = bp_filters['kentucky_warbler']

for species_idx in range(6):
    #species_idx = 5
    raw_dir = "/home/leah_colossal_com/us_bird_data/{}/raw/".format(species[species_idx])
    processed_dir = "/home/leah_colossal_com/us_bird_data/{}/processed/".format(species[species_idx])
    species_label = species[species_idx]

    # get bandpass limits for this particular species
    bandpass_lower_limit = bp_filters[species_label][0]
    bandpass_upper_limit = bp_filters[species_label][1]

    padded_denoised_segments = audio_from_event_times(raw_dir,processed_dir,species_label,bandpass_lower_limit,bandpass_upper_limit)
    # Initialize the EmbeddingGenerator
    embedding_generator = EmbeddingGenerator()

    # Generate and save embeddings for the denoised segments
    embeddings = embedding_generator.generate_embeddings(padded_denoised_segments)
    np.savez_compressed(os.path.join(processed_dir,"embeddings.npz"), *embeddings)

    # Generate and save labels for these embeddings
    labels = [species_label]*len(embeddings)
    np.save(os.path.join(processed_dir,"labels.npy"), labels)

# # `embeddings` now contains the embeddings for each padded and denoised segment
# print(f"Number of segments: {len(embeddings)}")
# print(f"Embedding shape: {embeddings[0].shape}")
