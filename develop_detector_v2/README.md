# Bird Call Detection and Classification

This repository provides a pipeline to detect, segment, and classify bird calls from an audio file. It uses machine learning models to process audio data, generate embeddings, and classify bird calls, producing a `results.csv` file with detailed information for each detected event.

## How to Run

### Steps:

1. **Download develop_detector_v2**

```bash
   cd downloaded folder
```

2. **Create a Conda Environment**

After downloading the repository, create a new conda environment named birdcall with Python 3.11:

```bash
conda create -n birdcall python=3.11 -y
```
Activate the environment:

```bash
conda activate birdcall
```

3. **Install the Requirements**
Install the required dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```

4. **Run the Main Script**
Run the main script to start the detection and classification process. This will analyze test.wav (located in the data folder) and generate a results.csv file with the detection results.

```bash
python main.py
```

To run a custom audio folder and specify the name of the output CSV file:

```bash
python main.py -BF /path/to/folder/ -O csv_filename.csv
```

Replace `/path/to/folder/` with the path to your directory containing audio files, and `csv_filename.csv` with your desired output file name.

## Output

The `results.csv` file contains the following columns for each detected event:

- **filepath**: Path to the audio file.
- **filename**: Name of the input audio file.
- **event_no**: Sequential number for each detected event.
- **start_time**: Start time of the event in seconds.
- **end_time**: End time of the event in seconds.
- **Predicted Label**: The predicted bird species label.
- **Confidence Score**: Confidence score for the prediction.



### Example Output

The generated `results.csv` might look like this for the test file `test.wav`:

### Example Output Table

| filepath   | filename  | event_no | start_time | end_time | Predicted Label       | Confidence Score |
|------------|-----------|----------|------------|----------|-----------------------|------------------|
| ./data     | test.wav  | 1        | 0.62       | 1.75     | species_of_interest   | 0.91021246       |
| ./data     | test.wav  | 2        | 3.10       | 4.52     | species_of_interest   | 0.88730020       |
| ./data     | test.wav  | 3        | 5.99       | 7.37     | species_of_interest   | 0.93345630       |
| ./data     | test.wav  | 4        | 9.01       | 10.37    | species_of_interest   | 0.91329980       |



This `results.csv` file contains all detected events with details on the time intervals, predicted bird species, and confidence scores.
