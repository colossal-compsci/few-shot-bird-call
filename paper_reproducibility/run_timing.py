"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
"""
Runtime / resource comparison of the four localization pipelines, on the 5 test recordings.
Measures, per pipeline: preprocessing time, detection time, number of classifier (Perch) calls,
classification time, and total wall-clock. Perch model load is one-time and excluded.

Pipelines:
  A) event detector + denoised   (preprocess -> wavelet detect -> classify detected events)
  B) event detector + raw        (load raw -> wavelet detect -> classify detected events)
  C) sliding window + denoised    (preprocess -> classify every 2s/0.5s-hop window)
  D) sliding window + raw         (load raw -> classify every 2s/0.5s-hop window)
"""
import os, sys, glob, time
import numpy as np, pandas as pd
SRC = "${TBP_PIPELINE_REPO}/src"
REPO = "."
sys.path.insert(0, SRC)
from event_detection import EventDetection
from make_prediction import MakePrediction
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub, librosa

TEST = "${TBP_TEST_FILES}"
PERCH_URL = ('https://www.kaggle.com/models/google/bird-vocalization-classifier/'
             'TensorFlow2/bird-vocalization-classifier/8')
PERCH_SR, PERCH_SAMPLES, THR, WIN, HOP = 32000, 160000, 0.24, 2.0, 0.5

print("Loading Perch...", flush=True)
model = hub.load(PERCH_URL)
det = EventDetection()
pred = MakePrediction(n_components=142, model_folder=os.path.join(REPO, "model_clean"))
proto, plabels = pred.prepare_prototypes()

def perch_in(seg, sr):
    if sr != PERCH_SR: seg = librosa.resample(seg, orig_sr=sr, target_sr=PERCH_SR)
    if len(seg) < PERCH_SAMPLES:
        p = PERCH_SAMPLES - len(seg); seg = np.pad(seg, (p//2, p-p//2))
    return seg[:PERCH_SAMPLES].astype(np.float32)

def classify_time(segs, sr):
    """Return (n_calls, seconds) for embedding+classifying the segments."""
    if not segs: return 0, 0.0
    t0 = time.time()
    embs = np.array([model.infer_tf(perch_in(s, sr)[np.newaxis, :])["embedding"].numpy().flatten() for s in segs])
    red = normalize(pred.transform_pca(embs), axis=1, norm="l2")
    pred.softmax(cosine_similarity(red, proto))
    return len(segs), time.time() - t0

files = sorted(glob.glob(os.path.join(TEST, "*.wav")) + glob.glob(os.path.join(TEST, "*.mp3")))

# warm up (first inference includes XLA compile); excluded from timing
classify_time([np.zeros(PERCH_SAMPLES, np.float32)], PERCH_SR)

agg = {p: {"preprocess": 0.0, "detect": 0.0, "classify": 0.0, "n_calls": 0} for p in "ABCD"}
for f in files:
    # load raw (and time it)
    t0 = time.time(); y, _ = librosa.load(f, sr=det.target_sr); t_loadraw = time.time() - t0
    # preprocess (bandpass + denoise) on top of load
    t0 = time.time(); _, y_den, sr = det.audio_processor.load_audio(f); t_pre = time.time() - t0
    tot = len(y) / sr
    win_starts = []
    t = 0.0
    while t < tot:
        win_starts.append((t, min(t + WIN, tot))); t += HOP

    # A: detector + denoised
    t0 = time.time(); evA = det.detected_events(det.smoothed_features(y_den, sr, wavelet="mexh", window_size=0.15), sr); tdA = time.time() - t0
    nA, tcA = classify_time([y_den[int(s*sr):int(e*sr)] for s, e in evA], sr)
    agg["A"]["preprocess"] += t_pre; agg["A"]["detect"] += tdA; agg["A"]["classify"] += tcA; agg["A"]["n_calls"] += nA
    # B: detector + raw
    t0 = time.time(); evB = det.detected_events(det.smoothed_features(y, sr, wavelet="mexh", window_size=0.15), sr); tdB = time.time() - t0
    nB, tcB = classify_time([y[int(s*sr):int(e*sr)] for s, e in evB], sr)
    agg["B"]["preprocess"] += t_loadraw; agg["B"]["detect"] += tdB; agg["B"]["classify"] += tcB; agg["B"]["n_calls"] += nB
    # C: window + denoised
    nC, tcC = classify_time([y_den[int(s*sr):int(e*sr)] for s, e in win_starts], sr)
    agg["C"]["preprocess"] += t_pre; agg["C"]["classify"] += tcC; agg["C"]["n_calls"] += nC
    # D: window + raw
    nD, tcD = classify_time([y[int(s*sr):int(e*sr)] for s, e in win_starts], sr)
    agg["D"]["preprocess"] += t_loadraw; agg["D"]["classify"] += tcD; agg["D"]["n_calls"] += nD
    print(f"  {os.path.basename(f)[:34]:<36} eventsA={nA:>3} eventsB={nB:>2} windows={nC}", flush=True)

names = {"A": "Event detector + denoised", "B": "Event detector + raw",
         "C": "Sliding window + denoised", "D": "Sliding window + raw"}
print("\n===================== RUNTIME / RESOURCE (5 recordings, ~5 min 54 s audio) =====================")
print(f"{'pipeline':<28}{'classifier calls':>17}{'preproc(s)':>12}{'detect(s)':>11}{'classify(s)':>12}{'total(s)':>10}")
rows = []
for p in "ABCD":
    a = agg[p]; total = a["preprocess"] + a["detect"] + a["classify"]
    print(f"{names[p]:<28}{a['n_calls']:>17}{a['preprocess']:>12.2f}{a['detect']:>11.2f}{a['classify']:>12.2f}{total:>10.2f}")
    rows.append({"pipeline": names[p], **a, "total_s": total})
pd.DataFrame(rows).to_csv(os.path.join(REPO, "timing.csv"), index=False)
print("\nSaved -> timing.csv")
