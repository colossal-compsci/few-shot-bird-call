import numpy as np
from src.event_detection import EventDetection
from src.segmentation import Segment
from src.embedding_generator import EmbeddingGenerator
from src.make_prediction import MakePrediction


import os

def main():
    # Initialize classes
    event_detector = EventDetection()
    segment_processor = Segment()
    embedding_generator = EmbeddingGenerator()
    predictor = MakePrediction(n_components=142, model_folder="./model")

    # Define the file path and extract the filename
    file_path = "./data/test.wav"
    filename = os.path.basename(file_path)

    # Step 1: Detect events, process segments, and generate unknown embeddings
    event_times, _, padded_denoised_segments, sr = event_detector.detected_times(file_path)
    print(f"Total events detected: {len(padded_denoised_segments)}")

    padded_denoised_segments = segment_processor.segment_audio(padded_denoised_segments, sr)
    print("Segments processed.")

    unknown_embeddings = embedding_generator.generate_embeddings(padded_denoised_segments)
    print(f"Embeddings generated: {len(unknown_embeddings)}")

    # Extract start and end times for each event
    start_times = [start for start, _ in event_times]
    end_times = [end for _, end in event_times]
    event_numbers = list(range(1, len(unknown_embeddings) + 1))

    # Step 2: Make predictions
    results_df = predictor.predict(unknown_embeddings, threshold=0.24)
    print("Predictions complete.")

    # Add additional columns for event number, start time, end time, and filename
    results_df['event_no'] = event_numbers
    results_df['start_time'] = start_times
    results_df['end_time'] = end_times
    results_df['filename'] = filename

    # Reorder the columns to place event info at the beginning
    results_df = results_df[['filename','event_no', 'start_time', 'end_time', 'Predicted Label', 
                             'Confidence Score']]
    
    print(results_df)

    results_df.to_csv("results.csv", index=False)
    print("Results saved to results.csv.")
    print("Done.")

# Run the main function only if this file is executed as a script
if __name__ == "__main__":
    main()