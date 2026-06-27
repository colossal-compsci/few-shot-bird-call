"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
"""
Pure sliding-window localization (NO event detector) on raw audio.

Slide a 2 s window (0.5 s hop) across each recording, classify every window with the clean
TBP model; a window is "call-positive" if it predicts a known species (max prob >= threshold).
Merge consecutive positive windows into detected intervals, then compare to the annotated calls:
  - recall / missed calls
  - false-positive intervals (detected where no annotated call)
  - localization quality (temporal IoU + boundary error) for matched calls
Run for raw and (for reference) denoised. Compare against the event-detector results
(detector+denoised IoU ~0.76; detector+raw ~0.04).
"""
import os, sys, glob
import numpy as np, pandas as pd
REPO = "."
SRC = "${TBP_PIPELINE_REPO}/src"
sys.path.insert(0, SRC)
from event_detection import EventDetection
from make_prediction import MakePrediction
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub, librosa

TEST = "${TBP_TEST_FILES}"
PERCH_URL = ('https://www.kaggle.com/models/google/bird-vocalization-classifier/'
             'TensorFlow2/bird-vocalization-classifier/8')
PERCH_SR, PERCH_SAMPLES, THR = 32000, 160000, 0.24
WIN, HOP = 2.0, 0.5

print("Loading Perch...", flush=True)
model = hub.load(PERCH_URL)
det = EventDetection()
pred = MakePrediction(n_components=142, model_folder=os.path.join(REPO, "model_clean"))
proto, plabels = pred.prepare_prototypes(); plabels = list(plabels)

df = pd.read_excel(os.path.join(TEST, "raw_data.xlsx"), "Test_set")
df["base"] = df["True File Name "].str.replace(r"_\d+_\d+\.wav$", "", regex=True)
m = df["True File Name "].str.extract(r"_(\d+)_(\d+)\.wav$").astype(float)
df["gs"], df["ge"] = m[0]/1000.0, m[1]/1000.0
df = pd.concat([df, pd.DataFrame([
    {"base": "RECORDING_E", "gs": s/1000.0, "ge": e/1000.0}
    for s, e in []  # call intervals withheld (load from private annotations)])], ignore_index=True)

def perch(seg, sr):
    if sr != PERCH_SR: seg = librosa.resample(seg, orig_sr=sr, target_sr=PERCH_SR)
    if len(seg) < PERCH_SAMPLES:
        p = PERCH_SAMPLES - len(seg); seg = np.pad(seg, (p//2, p-p//2))
    return seg[:PERCH_SAMPLES].astype(np.float32)

def iou(a0,a1,b0,b1):
    inter = max(0.0, min(a1,b1)-max(a0,b0)); uni=(a1-a0)+(b1-b0)-inter
    return inter/uni if uni>0 else 0.0

def merge(intervals, gap=HOP*1.5):
    if not intervals: return []
    intervals = sorted(intervals); out=[list(intervals[0])]
    for s,e in intervals[1:]:
        if s <= out[-1][1]+gap: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return [tuple(x) for x in out]

def positives(sig, sr):
    """Sliding-window: return list of (start,end) windows predicted as a known species."""
    wins=[]; t=0.0; tot=len(sig)/sr
    starts=[]
    while t < tot:
        starts.append((t, min(t+WIN, tot))); t+=HOP
    segs=[sig[int(s*sr):int(e*sr)] for s,e in starts]
    embs=np.array([model.infer_tf(perch(s,sr)[np.newaxis,:])["embedding"].numpy().flatten() for s in segs])
    red=normalize(pred.transform_pca(embs), axis=1, norm="l2")
    probs=pred.softmax(cosine_similarity(red, proto))
    pos=[starts[i] for i in range(len(starts)) if np.max(probs[i])>=THR]
    return merge(pos)

rows=[]
for base, g in df.groupby("base"):
    f=glob.glob(os.path.join(TEST, base+".*"))[0]
    y, yd, sr = det.audio_processor.load_audio(f)
    gts=list(zip(g["gs"], g["ge"]))
    for cond, sig in [("raw", y), ("denoised", yd)]:
        intervals = positives(sig, sr)
        matched=set()
        for gi,(gs,ge) in enumerate(gts):
            best=0.0; berr=(np.nan,np.nan)
            for (ds,de) in intervals:
                v=iou(ds,de,gs,ge)
                if v>best: best=v; berr=(abs(ds-gs),abs(de-ge))
            rows.append({"base":base,"cond":cond,"kind":"call","best_iou":best,
                         "start_err":berr[0],"end_err":berr[1],"recovered":best>0})
            if best>0:
                for ii,(ds,de) in enumerate(intervals):
                    if iou(ds,de,gs,ge)>0: matched.add(ii)
        fp = len(intervals)-len(matched)
        rows.append({"base":base,"cond":cond,"kind":"summary","n_intervals":len(intervals),"fp_intervals":fp})
        print(f"  {base[:30]:<32}{cond:<9} intervals={len(intervals):>3}  FP_intervals={fp}", flush=True)

R=pd.DataFrame(rows)
R.to_csv(os.path.join(REPO,"pure_window.csv"), index=False)
calls=R[R.kind=="call"]
print("\n========== PURE SLIDING-WINDOW LOCALIZATION (no event detector) ==========")
print(f"{'condition':<12}{'recall':>10}{'missed':>8}{'mean IoU':>10}{'med IoU':>9}{'mean start/end err':>22}{'FP intervals':>14}")
for cond in ["raw","denoised"]:
    c=calls[calls.cond==cond]; rec=c.recovered.mean(); miss=(~c.recovered).sum()
    fp=R[(R.cond==cond)&(R.kind=="summary")].fp_intervals.sum()
    print(f"{cond:<12}{rec*len(c)/len(c):>9.2f} ({c.recovered.sum()}/{len(c)})"
          f"{miss:>8}{c.best_iou.mean():>10.3f}{c.best_iou.median():>9.3f}"
          f"{c.start_err.mean():>11.2f}s/{c.end_err.mean():.2f}s{fp:>10}")
print("\nReference (event detector): denoised IoU~0.76, raw IoU~0.04")
print("Saved -> pure_window.csv")
