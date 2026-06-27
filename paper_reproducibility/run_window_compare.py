"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
"""
Fair raw-vs-denoised comparison at annotated call locations.

Raw detection yields a few giant events, so we WINDOW every detected event into 2 s windows
(1 s hop) for both conditions, classify each window with the TBP 6-species Perch+cosine model,
and then ask, at each of the 28 annotated call locations (overlap), whether the call is
classified correctly / as TBP. Ground truth = 'Moe Label 1' (start/end ms in filename).
Only Recording A has true TBP calls; B/C/D are hard negatives (PIP/WTP/CCFD).
"""
import os, sys, glob
import numpy as np, pandas as pd
REPO = "${TBP_PIPELINE_REPO}"
sys.path.insert(0, os.path.join(REPO, "src"))
from event_detection import EventDetection
from make_prediction import MakePrediction
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub

TEST = "${TBP_TEST_FILES}"
PERCH_URL = ('https://www.kaggle.com/models/google/bird-vocalization-classifier/'
             'TensorFlow2/bird-vocalization-classifier/8')
PERCH_SR, PERCH_SAMPLES, THR, TARGET = 32000, 160000, 0.24, "tooth_billed_pigeon"
REC = {"RECORDING_A": "A", "RECORDING_B": "B",
       "RECORDING_C": "C", "RECORDING_D": "D",
       "RECORDING_E": "E"}
LMAP = {"TBP": "tooth_billed_pigeon", "PIP": "pacific_imperial_pigeon",
        "WTP": "white_throated_pigeon", "CCFD": "crimson_crowned_fruit_dove", "NTBP": "not_tbp"}

print("Loading Perch...", flush=True)
model = hub.load(PERCH_URL)
det = EventDetection()
# Clean held-out model (TBP = 36 'Confirmed' calls only; manumea test recording excluded)
pred = MakePrediction(n_components=142, model_folder=os.path.join(os.path.dirname(__file__), "model_clean"))
proto, plabels = pred.prepare_prototypes(); plabels = list(plabels)
import librosa

df = pd.read_excel(os.path.join(TEST, "raw_data.xlsx"), "Test_set")
df["base"] = df["True File Name "].str.replace(r"_\d+_\d+\.wav$", "", regex=True)
m = df["True File Name "].str.extract(r"_(\d+)_(\d+)\.wav$").astype(float)
df["gs"], df["ge"] = m[0] / 1000.0, m[1] / 1000.0
df["true"] = df["Moe Label 1"].str.split().str[0].map(LMAP)

# held-out manumea test recording: 4 TBP calls, times from the segment filenames (ms -> s)
MANUMEA = "RECORDING_E"
df = pd.concat([df, pd.DataFrame([
    {"base": MANUMEA, "gs": s / 1000.0, "ge": e / 1000.0, "true": "tooth_billed_pigeon"}
    for s, e in []  # call intervals withheld (load from private annotations)])], ignore_index=True)

def perch(seg, sr):
    if sr != PERCH_SR: seg = librosa.resample(seg, orig_sr=sr, target_sr=PERCH_SR)
    if len(seg) < PERCH_SAMPLES:
        p = PERCH_SAMPLES - len(seg); seg = np.pad(seg, (p // 2, p - p // 2))
    return seg[:PERCH_SAMPLES].astype(np.float32)

def windows(s, e):
    if e - s <= 2.0: return [(s, e)]
    out, t = [], s
    while t < e:
        out.append((t, min(t + 2.0, e))); t += 1.0
    return out

def classify(segs, sr):
    if not segs: return []
    emb = np.array([model.infer_tf(perch(s, sr)[np.newaxis, :])["embedding"].numpy().flatten() for s in segs])
    probs = pred.softmax(cosine_similarity(normalize(pred.transform_pca(emb), axis=1, norm="l2"), proto))
    return [(plabels[int(np.argmax(p))] if np.max(p) >= THR else "unknown", float(np.max(p))) for p in probs]

def overlap(a, b, c, d): return a < d and c < b

results = {}
for base, g in df.groupby("base"):
    f = glob.glob(os.path.join(TEST, base + ".*"))[0]
    y, yd, sr = det.audio_processor.load_audio(f)
    gts = list(zip(g["gs"], g["ge"], g["true"]))
    for cond, sig in [("denoised", yd), ("raw", y)]:
        ev = det.detected_events(det.smoothed_features(sig, sr, wavelet="mexh", window_size=0.15), sr)
        wins = [w for s, e in ev for w in windows(s, e)]
        preds = classify([sig[int(s * sr):int(e * sr)] for s, e in wins], sr)
        results.setdefault(cond, []).append((REC[base], len(ev), len(wins), gts, list(zip(wins, preds))))

rows = []
for cond in ["denoised", "raw"]:
    for rec, n_ev, n_win, gts, wp in results[cond]:
        for gs, ge, true in gts:
            ov = [(lab, c) for (ws, we), (lab, c) in wp if overlap(ws, we, gs, ge)]
            pred_tbp = any(lab == TARGET for lab, _ in ov)
            correct = any(lab == true for lab, _ in ov)
            rows.append({"cond": cond, "rec": rec, "true": true, "pred_tbp": pred_tbp,
                         "correct": correct, "n_windows_overlap": len(ov)})
R = pd.DataFrame(rows)
R.to_csv(os.path.join(os.path.dirname(__file__), "window_compare.csv"), index=False)

print("\n=========== WINDOWED CLASSIFICATION AT ANNOTATED CALLS (raw vs denoised) ===========")
print("Events / windows per recording:")
for cond in ["denoised", "raw"]:
    parts = [f"{rec}:{n_ev}ev/{n_win}win" for rec, n_ev, n_win, _, _ in results[cond]]
    print(f"  {cond:<9} " + "  ".join(parts))
print()
for cond in ["denoised", "raw"]:
    s = R[R.cond == cond]
    tbp = s[s.true == TARGET]; neg = s[s.true != TARGET]
    print(f"{cond:<9}: TBP recall={tbp.pred_tbp.sum()}/{len(tbp)} (Rec A)   "
          f"false-TBP on hard negatives={neg.pred_tbp.sum()}/{len(neg)}   "
          f"overall correct-species={s.correct.sum()}/{len(s)}")
print("\nPer hard-negative recording, false-TBP rate:")
for cond in ["denoised", "raw"]:
    for rec in ["B", "C", "D"]:
        s = R[(R.cond == cond) & (R.rec == rec)]
        print(f"  {cond:<9} Rec {rec}: false-TBP={s.pred_tbp.sum()}/{len(s)}")
print("\nSaved -> window_compare.csv")
