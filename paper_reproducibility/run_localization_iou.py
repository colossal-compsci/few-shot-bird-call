"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
"""
Localization quality of the event detector: does WITHOUT preprocessing recover call start/end
times as precisely as WITH preprocessing? Instead of binary overlap (which raw passes only via
giant whole-file events), we measure temporal IoU and boundary errors between each detected event
and each annotated call. Ground truth = raw_data.xlsx + the held-out manumea call times.
"""
import os, sys, glob
import numpy as np, pandas as pd
REPO = "${TBP_PIPELINE_REPO}"
sys.path.insert(0, os.path.join(REPO, "src"))
from event_detection import EventDetection
TEST = "${TBP_TEST_FILES}"

df = pd.read_excel(os.path.join(TEST, "raw_data.xlsx"), "Test_set")
df["base"] = df["True File Name "].str.replace(r"_\d+_\d+\.wav$", "", regex=True)
m = df["True File Name "].str.extract(r"_(\d+)_(\d+)\.wav$").astype(float)
df["gs"], df["ge"] = m[0] / 1000.0, m[1] / 1000.0
df = pd.concat([df, pd.DataFrame([
    {"base": "RECORDING_E", "gs": s/1000.0, "ge": e/1000.0}
    for s, e in []  # call intervals withheld (load from private annotations)])], ignore_index=True)

def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0

det = EventDetection()
rows = []
for base, g in df.groupby("base"):
    f = glob.glob(os.path.join(TEST, base + ".*"))[0]
    y, yd, sr = det.audio_processor.load_audio(f)
    gts = list(zip(g["gs"], g["ge"]))
    for cond, sig in [("denoised", yd), ("raw", y)]:
        ev = det.detected_events(det.smoothed_features(sig, sr, wavelet="mexh", window_size=0.15), sr)
        for gs, ge in gts:
            if ev:
                ious = [iou(ds, de, gs, ge) for ds, de in ev]
                k = int(np.argmax(ious)); best = ious[k]; ds, de = ev[k]
                serr, eerr = abs(ds - gs), abs(de - ge)
            else:
                best, serr, eerr = 0.0, np.nan, np.nan
            rows.append({"base": base, "cond": cond, "best_iou": best,
                         "start_err_s": serr, "end_err_s": eerr})

R = pd.DataFrame(rows)
R.to_csv(os.path.join(os.path.dirname(__file__), "localization_iou.csv"), index=False)
print("=============== EVENT LOCALIZATION QUALITY (temporal IoU vs annotated calls) ===============")
print(f"{'condition':<24}{'mean IoU':>10}{'median IoU':>12}{'mean |start err|':>18}{'mean |end err|':>16}")
for cond in ["denoised", "raw"]:
    s = R[R.cond == cond]
    print(f"{('denoised (WITH preproc)' if cond=='denoised' else 'raw (NO preproc)'):<24}"
          f"{s.best_iou.mean():>10.3f}{s.best_iou.median():>12.3f}"
          f"{s.start_err_s.mean():>17.2f}s{s.end_err_s.mean():>15.2f}s")
print("\nPer-recording mean best-IoU:")
print(f"{'recording':<34}{'denoised':>10}{'raw':>8}")
for base in sorted(R.base.unique()):
    d = R[(R.base==base)&(R.cond=='denoised')].best_iou.mean()
    r = R[(R.base==base)&(R.cond=='raw')].best_iou.mean()
    print(f"  {base[:31]:<32}{d:>10.3f}{r:>8.3f}")
print("\nSaved -> localization_iou.csv")
