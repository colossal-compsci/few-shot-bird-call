import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For saving and loading the PCA model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MakePrediction:
    def __init__(self, n_components=142, model_folder="./model"):
        """Initialize MakePrediction with PCA configuration and model folder path."""
        self.n_components = n_components
        self.model_folder = model_folder

    def fit_pca(self, embeddings):
        """Fits PCA on the known embeddings and saves the PCA model."""
        self.pca = PCA(n_components=min(self.n_components, embeddings.shape[0]))
        reduced_embeddings = self.pca.fit_transform(embeddings)
        joblib.dump(self.pca, os.path.join(self.model_folder, "pca_model.joblib"))
        logging.info("PCA model fitted and saved successfully.")
        return reduced_embeddings

    def load_pca(self):
        """Loads the PCA model from the model folder with fallback check."""
        pca_path = os.path.join(self.model_folder, "pca_model.joblib")
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
            logging.info("PCA model loaded successfully.")
        else:
            raise FileNotFoundError("PCA model not found. Please run fit_pca first.")

    def transform_pca(self, embeddings):
        """Normalize embeddings and transform using PCA."""
        if not hasattr(self, "pca") or not hasattr(self.pca, "components_"):
            self.load_pca()
        normalized_embeddings = normalize(embeddings, axis=1, norm="l2")
        return self.pca.transform(normalized_embeddings)

    def load_species_prototype(self):
        """Load the species prototype vector with fallback check."""
        path = os.path.join(self.model_folder, "species_prototype.npy")
        if os.path.exists(path):
            prototype = np.load(path)
            logging.info("Species prototype loaded successfully.")
            return prototype
        else:
            raise FileNotFoundError("Species prototype not found.")

    def load_threshold(self):
        """Load the classification threshold with fallback check."""
        path = os.path.join(self.model_folder, "threshold.npy")
        if os.path.exists(path):
            threshold = float(np.load(path))
            logging.info(f"Threshold loaded successfully: {threshold}")
            return threshold
        else:
            raise FileNotFoundError("Threshold not found.")

    def load_classifier(self):
        """Convenience method to load PCA, species prototype, and threshold together with verbose logging."""
        try:
            self.load_pca()
            self.species_prototype = self.load_species_prototype()
            self.threshold = 0.88 #self.load_threshold() 
            logging.info("Classifier loaded successfully with all components.")
        except Exception as e:
            logging.error(f"Error loading classifier components: {e}")
            raise

    def classify(self, unknown_embeddings, baseline_probability=1.0 / 6):
        if not hasattr(self, "species_prototype") or not hasattr(self, "threshold"):
            self.load_classifier()

        reduced_embeddings = self.transform_pca(unknown_embeddings)
        similarities = cosine_similarity(reduced_embeddings, self.species_prototype).flatten()

        predictions = [
            "species_of_interest" if sim >= self.threshold else "not_species_of_interest"
            for sim in similarities
        ]

        relative_confidences = [
            round(((score - baseline_probability) / baseline_probability) * 100, 2)
            for score in similarities
        ]

        results_df = pd.DataFrame({
            "Predicted Label": predictions,
            "Confidence Score": similarities,
            "Relative Confidence (%)": relative_confidences
        })

        logging.info("Classification completed for all embeddings.")
        return results_df
