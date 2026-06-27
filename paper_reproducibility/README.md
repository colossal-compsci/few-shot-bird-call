# Paper reproducibility code

Code accompanying *An Automated Pipeline for Few-Shot Bird Call Classification, a Case
Study with the Tooth-Billed Pigeon* (Ecology and Evolution). It reproduces the analyses
added during revision: the preprocessing ablation and the event-detector localization /
efficiency comparison.

## Conservation notice
The tooth-billed pigeon (*Didunculus strigirostris*) is critically endangered. **Its field
recordings and all identifying metadata — recording filenames, dates, sites, GPS, and
per-call timestamps — are withheld** and are NOT included in this repository. In the code,
recordings are referred to only by anonymized labels (RECORDING_A..E), and data locations
are environment variables. To run the tooth-billed pigeon analyses you must supply your own
licensed dataset. The Eastern Towhee example (`run_ablation.py`) uses publicly available
Xeno-Canto recordings and is fully runnable.

## Environment variables (set before running)
- `TBP_PIPELINE_REPO` — path to the core pipeline (the `src/` of this repository)
- `TBP_DATA_ROOT` — root of your local dataset (ignored by git)
- `TBP_TEST_FILES` — folder of full test recordings (ignored by git)

## Setup
```
pip install librosa soundfile noisereduce pywt openpyxl scikit-learn pandas tensorflow tensorflow_hub
```
The Perch / Google bird-vocalization-classifier model is downloaded from TensorFlow Hub on
first run.

## Files
See `REPRODUCIBILITY.md` for the full description of each script, the run order, and the
expected results. Result CSVs and trained artifacts are not committed (see `.gitignore`).
