"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
"""
Event-detector recall: does the wavelet detector recover the annotated calls, and how does
raw vs denoised preprocessing change this? Ground truth from test_files/raw_data.xlsx (Test_set);
call start/end (ms) parsed from the filename, 'Moe Label 1' is the true species.
"""
import os, sys, re, glob
import numpy as np, pandas as pd
REPO = "${TBP_PIPELINE_REPO}"
sys.path.insert(0, os.path.join(REPO, "src"))
from event_detection import EventDetection
TEST_DIR = "${TBP_TEST_FILES}"

df = pd.read_excel(os.path.join(TEST_DIR, "raw_data.xlsx"), "Test_set")
df["base"] = df["True File Name "].str.replace(r"_\d+_\d+\.wav$", "", regex=True)
m = df["True File Name "].str.extract(r"_(\d+)_(\d+)\.wav$").astype(float)
df["gt_start"] = m[0] / 1000.0   # ms -> s
df["gt_end"]   = m[1] / 1000.0
df["label"]    = df["Moe Label 1"].str.split().str[0]   # 'PIP (with..)' -> 'PIP'

# Additional pure-TBP test recording: 'RECORDING_E.wav'.
# Its 4 call times come from the segment filenames (ms -> s). NOTE: this file also appears in
# the training data, so it is valid for the (unsupervised) detector-localization test only.
MANUMEA = "RECORDING_E"
manumea_calls = []  # call intervals withheld (load from private annotations)
df = pd.concat([df, pd.DataFrame([
    {"base": MANUMEA, "gt_start": s / 1000.0, "gt_end": e / 1000.0, "label": "TBP"}
    for s, e in manumea_calls])], ignore_index=True)

def find_file(base):
    for ext in (".wav", ".mp3", ".WAV"):
        p = os.path.join(TEST_DIR, base + ext)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(TEST_DIR, base + ".*"))
    return hits[0] if hits else None

def overlaps(ds, de, gs, ge):
    return ds < ge and gs < de

detector = EventDetection()
print(f"{'recording':<34}{'cond':<10}{'#events':>8}{'GT recovered':>14}")
rows = []
for base, g in df.groupby("base"):
    f = find_file(base)
    if not f:
        print("  !! file not found:", base); continue
    y, y_den, sr = detector.audio_processor.load_audio(f)
    gts = list(zip(g["gt_start"], g["gt_end"], g["label"]))
    rec_len = len(y) / sr
    for cond, sig in [("denoised", y_den), ("raw", y)]:
        feats = detector.smoothed_features(sig, sr, wavelet="mexh", window_size=0.15)
        ev = detector.detected_events(feats, sr)
        durs = [e - s for s, e in ev]
        med_dur = float(np.median(durs)) if durs else 0.0
        max_dur = float(max(durs)) if durs else 0.0
        cover = (sum(durs) / rec_len * 100) if rec_len else 0.0
        rec = sum(1 for gs, ge, _ in gts if any(overlaps(ds, de, gs, ge) for ds, de in ev))
        tbp_rec = sum(1 for gs, ge, lab in gts if lab == "TBP" and any(overlaps(ds, de, gs, ge) for ds, de in ev))
        n_tbp = sum(1 for _, _, lab in gts if lab == "TBP")
        print(f"{base[:33]:<34}{cond:<9}{len(ev):>7}{rec:>7}/{len(gts):<3}  "
              f"med/max dur={med_dur:.1f}/{max_dur:.1f}s  covers={cover:.0f}%")
        rows.append({"base": base, "cond": cond, "rec_len_s": round(rec_len, 1), "n_events": len(ev),
                     "med_event_dur_s": round(med_dur, 2), "max_event_dur_s": round(max_dur, 2),
                     "coverage_pct": round(cover, 1), "gt_total": len(gts), "gt_recovered": rec,
                     "tbp_total": n_tbp, "tbp_recovered": tbp_rec})

r = pd.DataFrame(rows)
print("\n================ EVENT-DETECTOR RECALL (overlap with annotated calls) ================")
for cond in ["denoised", "raw"]:
    s = r[r.cond == cond]
    print(f"{cond:<9}: events={s.n_events.sum():>4}  all-call recall={s.gt_recovered.sum()}/{s.gt_total.sum()} "
          f"({s.gt_recovered.sum()/s.gt_total.sum():.2f})  "
          f"TBP-call recall={s.tbp_recovered.sum()}/{s.tbp_total.sum()}")
r.to_csv(os.path.join(os.path.dirname(__file__), "detector_recall.csv"), index=False)
print("\nSaved -> detector_recall.csv")
