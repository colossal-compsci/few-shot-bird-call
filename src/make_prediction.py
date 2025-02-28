import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For saving and loading the PCA model

class MakePrediction:
    def __init__(self, n_components=142, model_folder="./model"):
        """
        Initialize MakePrediction with PCA configuration and model folder path.
        
        Args:
            n_components (int): Number of PCA components to reduce the embeddings to.
            model_folder (str): Path to the folder where model files are stored.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.model_folder = model_folder

    def fit_pca(self, embeddings):
        """Fits PCA on the known embeddings and saves the PCA model."""
        reduced_embeddings = self.pca.fit_transform(embeddings)
        joblib.dump(self.pca, os.path.join(self.model_folder, "pca_model.joblib"))
        print("PCA model fitted and saved successfully.")
        return reduced_embeddings

    def load_pca(self):
        """Loads the PCA model from the model folder."""
        pca_path = os.path.join(self.model_folder, "pca_model.joblib")
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
            print("PCA model loaded successfully.")
        else:
            raise FileNotFoundError("PCA model not found. Please run fit_pca first.")

    def transform_pca(self, embeddings):
        """Transforms embeddings using the previously fitted or loaded PCA."""
        # Ensure PCA is fitted by loading it if necessary
        if not hasattr(self.pca, "components_"):
            self.load_pca()
        return self.pca.transform(embeddings)

    @staticmethod
    def normalize_embeddings(embeddings):
        """Normalizes embeddings using L2 normalization."""
        return normalize(embeddings, axis=1, norm='l2')

    @staticmethod
    def softmax(scores):
        """Converts similarity scores to probabilities using softmax."""
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # Numerical stability
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def calculate_prototypes(embeddings, labels, use_median=True):
        """Calculates prototypes for each class using either median or trimmed mean."""
        unique_labels = np.unique(labels)
        prototypes = {}
        for label in unique_labels:
            label_embeddings = embeddings[labels == label]
            if use_median:
                prototypes[label] = np.median(label_embeddings, axis=0, keepdims=True)
            else:
                # Trimmed mean: remove the highest and lowest 10% of values
                sorted_embeddings = np.sort(label_embeddings, axis=0)
                trim_fraction = int(0.1 * label_embeddings.shape[0])
                trimmed_embeddings = sorted_embeddings[trim_fraction:-trim_fraction]
                prototypes[label] = np.mean(trimmed_embeddings, axis=0, keepdims=True)
        return prototypes

    @staticmethod
    def calculate_prototype_matrix(prototypes):
        """Converts a dictionary of prototypes to a matrix and a list of labels."""
        prototype_labels = []
        prototype_matrix = []
        for label, center in prototypes.items():
            prototype_labels.append(label)
            prototype_matrix.append(center)
        prototype_matrix = np.array(prototype_matrix).squeeze()
        return prototype_matrix, prototype_labels

    def save_prototype_matrix(self, prototypes):
        """Saves the prototype matrix and labels to files."""
        prototype_matrix, prototype_labels = self.calculate_prototype_matrix(prototypes)
        np.save(os.path.join(self.model_folder, "prototype_matrix.npy"), prototype_matrix)
        np.save(os.path.join(self.model_folder, "prototype_labels.npy"), prototype_labels)
        print("Prototype matrix and labels saved successfully.")

    def load_prototype_matrix(self):
        """Loads the prototype matrix and labels from files."""
        prototype_matrix_path = os.path.join(self.model_folder, "prototype_matrix.npy")
        prototype_labels_path = os.path.join(self.model_folder, "prototype_labels.npy")

        if os.path.exists(prototype_matrix_path) and os.path.exists(prototype_labels_path):
            prototype_matrix = np.load(prototype_matrix_path)
            prototype_labels = np.load(prototype_labels_path)
            print("Prototype matrix and labels loaded successfully.")
            return prototype_matrix, prototype_labels
        else:
            raise FileNotFoundError("Prototype matrix and labels not found in model folder.")

    def load_known_embeddings_and_labels(self):
        """Loads known embeddings and labels from the model folder."""
        embeddings_path = os.path.join(self.model_folder, "embeddings.npy")
        labels_path = os.path.join(self.model_folder, "labels.npy")

        if os.path.exists(embeddings_path) and os.path.exists(labels_path):
            known_embeddings = np.load(embeddings_path)
            known_labels = np.load(labels_path)
            print("Known embeddings and labels loaded successfully.")
            return known_embeddings, known_labels
        else:
            raise FileNotFoundError("Embeddings and labels files not found in model folder.")

    def prepare_prototypes(self):
        """Loads or calculates the prototype matrix and labels."""
        try:
            # Try loading the prototype matrix
            self.load_pca()  # Ensure PCA is loaded for transforming target embeddings
            return self.load_prototype_matrix()
        except FileNotFoundError:
            # If prototype matrix doesn't exist, calculate it from known embeddings
            known_embeddings, known_labels = self.load_known_embeddings_and_labels()
            reduced_embeddings = self.fit_pca(known_embeddings)
            normalized_embeddings = self.normalize_embeddings(reduced_embeddings)
            prototypes = self.calculate_prototypes(normalized_embeddings, known_labels)
            self.save_prototype_matrix(prototypes)
            return self.calculate_prototype_matrix(prototypes)

    def predict(self, target_embeddings, threshold=0.24, baseline_probability=1./6):
        """
        Classifies target embeddings based on similarity to prototypes.
        
        Args:
            target_embeddings (np.array): Embeddings of unknown samples.
            threshold (float): Confidence threshold for assigning known labels.
            baseline_probability (float): Baseline probability for equal probability (default is 1/6).
        
        Returns:
            pd.DataFrame: A DataFrame with predictions, confidence scores, and probabilities.
        """
        # Load or calculate the prototype matrix and labels
        prototype_matrix, prototype_labels = self.prepare_prototypes()
        
        # Transform and normalize target embeddings using the fitted PCA
        reduced_target_embeddings = self.transform_pca(target_embeddings)
        normalized_target_embeddings = self.normalize_embeddings(reduced_target_embeddings)

        # Compute cosine similarity between target embeddings and prototypes
        similarities = cosine_similarity(normalized_target_embeddings, prototype_matrix)

        # Convert similarities to probabilities
        probabilities = self.softmax(similarities)

        # Assign labels and calculate confidence scores based on the highest probability
        predicted_labels = []
        confidence_scores = []
        relative_confidences = []
        
        for prob in probabilities:
            max_prob = np.max(prob)
            if max_prob >= threshold:
                predicted_label = prototype_labels[np.argmax(prob)]
            else:
                predicted_label = 'unknown'
            predicted_labels.append(predicted_label)
            confidence_scores.append(max_prob)

            # Calculate and round relative confidence score to 2 significant figures
            relative_confidence = ((max_prob - baseline_probability) / baseline_probability)*100
            # relative_confidences.append("{:.2f}".format(relative_confidence))
            relative_confidences.append(round(relative_confidence, 2))
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame({
            'Predicted Label': predicted_labels,
            # 'Probability Score': confidence_scores,
            'Confidence Score': relative_confidences,
            # 'Probabilities': list(probabilities)
        })
        pd.options.display.float_format = '{:.2f}'.format
        return results_df

# import numpy as np
# from event_detection import EventDetection
# from segmentation import Segment
# from embedding_generator import EmbeddingGenerator
# from make_prediction import MakePrediction


# import os

# def main():
#     # Initialize classes
#     event_detector = EventDetection()
#     segment_processor = Segment()
#     embedding_generator = EmbeddingGenerator()
#     predictor = MakePrediction(n_components=142, model_folder="./model")

#     # Define the file path and extract the filename
#     file_path = "./data/test.wav"
#     filename = os.path.basename(file_path)

#     # Step 1: Detect events, process segments, and generate unknown embeddings
#     event_times, _, padded_denoised_segments, sr = event_detector.detected_times(file_path)
#     print(f"Total events detected: {len(padded_denoised_segments)}")

#     padded_denoised_segments = segment_processor.segment_audio(padded_denoised_segments, sr)
#     print("Segments processed.")

#     unknown_embeddings = embedding_generator.generate_embeddings(padded_denoised_segments)
#     print(f"Embeddings generated: {len(unknown_embeddings)}")

#     # Extract start and end times for each event
#     start_times = [start for start, _ in event_times]
#     end_times = [end for _, end in event_times]
#     event_numbers = list(range(1, len(unknown_embeddings) + 1))

#     # Step 2: Make predictions
#     results_df = predictor.predict(unknown_embeddings, threshold=0.24)
#     print("Predictions complete.")

#     # Add additional columns for event number, start time, end time, and filename
#     results_df['event_no'] = event_numbers
#     results_df['start_time'] = start_times
#     results_df['end_time'] = end_times
#     results_df['filename'] = filename

#     # Reorder the columns to place event info at the beginning
#     results_df = results_df[['filename','event_no', 'start_time', 'end_time', 'Predicted Label', 
#                              'Confidence Score']]
    
#     print(results_df)
#     print("Done.")

# # Run the main function only if this file is executed as a script
# if __name__ == "__main__":
#     main()
