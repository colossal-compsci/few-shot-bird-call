import os
import argparse
import pandas as pd
import logging
from src.event_detection import EventDetection
from src.segmentation import Segment
from src.embedding_generator import EmbeddingGenerator
from src.make_prediction import MakePrediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_folder(base_folder, params_list, output_file='results.csv', model_folder="./model"):
    segment_processor = Segment()
    embedding_generator = EmbeddingGenerator()
    predictor = MakePrediction(model_folder=model_folder)
    predictor.load_classifier()

    results = []
    total_files_processed = 0

    for params in params_list:
        tf = params['tf']
        md = params['md']
        apply_bp = params['apply_bp']
        bpll = params['bpll']
        bpul = params['bpul']

        event_detector = EventDetection(threshold_factor=tf, min_duration=md, apply_bp=apply_bp, bpll=bpll, bpul=bpul)

        for root, dirs, files in os.walk(base_folder):
            for current_file in files:
                if current_file.lower().endswith(('.wav', '.mp3')):
                    current_file_path = os.path.join(root, current_file)
                    file_dir = os.path.dirname(current_file_path)
                    logging.info(f'Processing {current_file_path}')
                    total_files_processed += 1

                    try:
                        y, y_denoised, sr, event_times = event_detector.detected_times(current_file_path)

                        padded_denoised_segments = segment_processor.segment_audio(
                            [y_denoised[int(start*sr):int(end*sr)] for start, end in event_times], sr)
                        unknown_embeddings = embedding_generator.generate_embeddings(padded_denoised_segments)

                        results_df = predictor.classify(unknown_embeddings)

                        for i, (start_time, end_time) in enumerate(event_times):
                            result_entry = {
                                'filepath': file_dir,
                                'filename': current_file,
                                'event_no': i + 1,
                                'start_time': round(start_time, 2),
                                'end_time': round(end_time, 2),
                                'Predicted Label': results_df.iloc[i]['Predicted Label'],
                                'Confidence Score': results_df.iloc[i]['Confidence Score']
                            }
                            results.append(result_entry)

                    except Exception as e:
                        logging.error(f'Error processing {current_file_path}: {e}')

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=['filename', 'event_no'])
        df.to_csv(output_file, index=False)
        logging.info(f'Full workflow results saved to {output_file}')
        logging.info(f'Total files processed: {total_files_processed}')
        print(f'Total files processed: {total_files_processed}')
        print(f'Results saved to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process folder for event detection and prediction.")
    parser.add_argument("--base_folder", "-BF", type=str, default="./data", help="Input base folder with audio files.")
    parser.add_argument("--output_file", "-O", type=str, default="results.csv", help="Output CSV file name.")

    args = parser.parse_args()

    params_list = [
        {"tf": 0.1, "md": 1, "apply_bp": True, "bpll": 150, "bpul": 600}
    ]

    model_folder = "./model"
    process_folder(args.base_folder, params_list, output_file=args.output_file, model_folder=model_folder)

# if __name__ == "__main__":
#     # base_folder = "/home/leah_colossal_com/tbp_dataset/balanced_dataset/test_data"
#     base_folder = "/home/jupyter/data/raw_data/tooth_billed_pigeon_raw"
#     params_list = [
#         {"tf": 0.15, "md": 1, "apply_bp": True, "bpll": 150, "bpul": 600}
#     ]
#     model_folder = "./model"
#     process_folder(base_folder, params_list, model_folder=model_folder)

# tf 0.15 and md 1 with threshold 0.88 gave the best