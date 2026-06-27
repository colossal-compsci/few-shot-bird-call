# TBP Preprocessing Ablation — Reproducibility Guide

This folder reproduces the preprocessing ablation reported in the Discussion (§4) of the manuscript:
preprocessing (bandpass + denoise) has little effect on the **classifier** but is **essential for the
event detector** to localize calls in field recordings.

## 1. What's here

| File | Purpose |
|------|---------|
| `build_clean_model.py` | Builds the held-out TBP prototype model (TBP = 36 single-recording training calls only; the held-out test recording is excluded). Output → `model_clean/`. |
| `run_ablation.py` | Eastern Towhee ablation: classify already-segmented calls, raw vs denoised (clean dataset). |
| `run_ablation_tbp.py` | TBP-domain ablation: 5-fold CV on the labeled field segments, raw vs denoised. |
| `run_detector_recall.py` | **Event-detector** ablation: raw vs denoised on the 5 test recordings; reports events, durations, coverage, and recovery of annotated calls. → `detector_recall.csv` |
| `run_window_compare.py` | Windows the (large) raw events into 2 s segments, classifies with `model_clean/`, compares raw vs denoised at the annotated call locations. → `window_compare.csv` |
| `run_endtoend.py` | End-to-end (event detect → classify) counts per recording. → `endtoend_results.npy` |
| `run_localization_iou.py` | **Localization quality**: temporal IoU + boundary error between detected events and annotated calls, raw vs denoised. → `localization_iou.csv` |
| `run_pure_window.py` | Pure sliding-window localization (NO event detector), raw vs denoised: recall, missed, false-positive intervals, IoU. → `pure_window.csv` |
| `run_timing.py` | Runtime / resource comparison of the four pipelines (classifier-query counts + wall-clock). → `timing.csv` |
| `event_detection_review.ipynb` | Interactive: listen to each detected event (denoised vs raw), inspect predictions, write/annotate `event_detections_review.csv`. |
| `data/` | Symlink to `TBP_project/data` (ET + TBP segmented audio). |
| `embeddings/` | Cached Perch embeddings (auto-created; delete to recompute). |

## 2. Environment

```bash
# Python 3.10. Core deps (TensorFlow + tensorflow_hub already present in the base env):
pip install librosa soundfile noisereduce pywt openpyxl scikit-learn pandas
```

- The Perch / Google bird-vocalization-classifier model is downloaded from TensorFlow Hub on first run
  (~50 s, needs internet). It is cached afterward.
- Code reuses the repo pipeline in `TBP_project/few-shot-bird-call/src/` (added to `sys.path` by each script).

## 3. Inputs and ground truth

- **Test recordings:** `${TBP_TEST_FILES}/` (4 mixed field recordings A–D + the
  held-out a held-out tooth-billed pigeon recording, recording E).
- **Ground truth:** `test_files/raw_data.xlsx`, sheet `Test_set`, column **`Moe Label 1`** (TBP / PIP / WTP /
  CCFD / NTBP). Call start/end times are the **last two numbers in each filename, in milliseconds**
  (e.g. `..._<start_ms>_<end_ms>.wav` (ms)). The manumea call times (4 calls) come from its training-segment
  filenames and are hard-coded in the scripts.
- **Held-out note:** the held-out recording's segments exist in the training folder, so `build_clean_model.py`
  **excludes** them; the 36 training calls are all from the single training recording (paper setting).

## 4. Run order

```bash
cd .

# 1) Build the clean held-out TBP model (required by run_window_compare.py)
python3 build_clean_model.py

# 2) Event-detector localization ablation (the headline result)
python3 run_detector_recall.py            # -> detector_recall.csv

# 3) Windowed classification at annotated calls, raw vs denoised
python3 run_window_compare.py             # -> window_compare.csv

# 4) Localization quality (temporal IoU) and pipeline comparison (paper Table 5)
python3 run_localization_iou.py           # -> localization_iou.csv  (detector: denoised IoU~0.76 vs raw ~0.04)
python3 run_pure_window.py                # -> pure_window.csv       (no-detector window baseline)
python3 run_timing.py                     # -> timing.csv           (classifier-query counts + runtime)

# (optional) classifier-only ablations and end-to-end counts
python3 run_ablation.py                   # Eastern Towhee
python3 run_ablation_tbp.py               # TBP 5-fold CV
python3 run_endtoend.py                   # end-to-end detect->classify

# (optional) listen to events interactively
jupyter notebook event_detection_review.ipynb
```

## 5. Expected results (what the manuscript reports)

- **Detector localization** (`run_detector_recall.py`): **denoised → 111 localized events** (median ~1–2 s)
  that isolate every annotated call; **raw → 7 giant events**, each covering ~100% of its recording. Both
  "recover" all calls by overlap, but raw cannot localize (a single event spans the whole file).
- **Windowed classification** (`run_window_compare.py`): once raw's giant events are windowed into 2 s
  segments, **both conditions recover all 7 true TBP calls** with comparable precision on the 25 hard
  negatives (denoised 4 false-TBP, raw 1) — i.e., the classifier is robust once calls are segmented.
- **Classifier-only ablations**: Eastern Towhee and TBP metrics are essentially unchanged raw vs denoised
  (Perch front end is robust to preprocessing).
- **Localization quality** (`run_localization_iou.py`): mean temporal IoU **denoised 0.76** (boundary error
  ~0.5 s) vs **raw 0.04** (~45 s) — without preprocessing the detector cannot recover call timings.
- **Pure window baseline** (`run_pure_window.py`): window+raw recovers only **10/32** calls (IoU 0.06, 9 FP
  intervals); window+denoised recovers 32/32 but localizes poorly (IoU 0.15, 10 FP) — far below the detector.
- **Runtime / resource** (`run_timing.py`, paper Table 5): event detector + denoised = best localization
  (IoU 0.76) at **111 classifier queries / ~84 s**, vs sliding window = **699 queries / ~270–286 s**
  (≈6× more queries, 3–9× slower). Absolute times are CPU- and hardware-dependent; the query counts are not.

**Conclusion:** preprocessing is essential to *end-to-end* performance because it enables reliable event
detection/segmentation, even though the downstream classifier is itself insensitive to it.

## 6. Notes / caveats

- The "recall = 1.00 for raw" in `detector_recall.csv` is a **degenerate artifact** — one giant event
  overlaps every annotated interval. Read it together with the `coverage_pct` / `max_event_dur_s` columns.
- TBP detection counts in `run_window_compare.py` use the clean held-out model; the older `run_endtoend.py`
  uses the repo's the bundled model (includes the held-out recording) and is for illustrative counts only.
- Determinism: Perch embeddings and PCA are deterministic; `run_ablation_tbp.py` fixes the CV seed (`SEED=0`).
