"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
"""
E2 (TBP) — Preprocessing ablation on the noisy TBP-domain field recordings.

The labeled TBP *test* set is not available locally (only the unlabeled 'uncertain' pool),
so we use the labeled TBP-domain field segments (6 species, ~350 clips) and run stratified
5-fold cross-validation, comparing:
  - 'denoised' = WITH preprocessing (bandpass + denoise + normalize)
  - 'raw'      = WITHOUT preprocessing
Every other step matches the paper (Perch embeddings -> PCA(95% var) -> L2 norm -> median
prototypes -> cosine -> softmax -> recall-0.9 threshold for TBP on each fold's train split).
Metric: binary tooth-billed-pigeon-vs-rest, averaged over folds.

Caveat: folds are split at the call level (not recording level), so absolute numbers may be
optimistic, but the raw-vs-denoised DELTA is the quantity of interest and any leakage is
symmetric across the two conditions.
"""
import os, glob, time
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data", "train_data")
TARGET = "tooth_billed_pigeon"
PERCH_URL = ('https://www.kaggle.com/models/google/bird-vocalization-classifier/'
             'TensorFlow2/bird-vocalization-classifier/8')
PERCH_SR = 32000
PERCH_SAMPLES = 160000
RECALL_TARGET = 0.90
N_FOLDS = 5
SEED = 0

def list_clips(cond):
    paths, labels = [], []
    for sp in sorted(os.listdir(DATA)):
        d = os.path.join(DATA, sp, "segmented_audio", cond)
        if not os.path.isdir(d):
            continue
        for w in sorted(glob.glob(os.path.join(d, "*.wav"))):
            paths.append(w); labels.append(sp)
    return paths, np.array(labels)

def load_audio_perch(path, librosa):
    y, _ = librosa.load(path, sr=PERCH_SR)
    if len(y) < PERCH_SAMPLES:
        pad = PERCH_SAMPLES - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))
    else:
        y = y[:PERCH_SAMPLES]
    return y.astype(np.float32)

def embed(paths, model, librosa, cache):
    if os.path.exists(cache):
        return np.load(cache)
    embs = []
    for i, p in enumerate(paths):
        x = load_audio_perch(p, librosa)[np.newaxis, :]
        embs.append(model.infer_tf(x)["embedding"].numpy().flatten())
        if (i + 1) % 50 == 0:
            print(f"    embedded {i+1}/{len(paths)}", flush=True)
    arr = np.array(embs); np.save(cache, arr); return arr

def l2(x):
    from sklearn.preprocessing import normalize
    return normalize(x, axis=1, norm="l2")

def softmax(s):
    e = np.exp(s - np.max(s, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def eval_fold(tr_emb, tr_lab, te_emb, te_lab):
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    pca = PCA(n_components=0.95, svd_solver="full")
    trr = l2(pca.fit_transform(tr_emb)); ter = l2(pca.transform(te_emb))
    cls = np.unique(tr_lab)
    protos = np.array([np.median(trr[tr_lab == c], axis=0) for c in cls])

    def predict(red):
        probs = softmax(cosine_similarity(red, protos))
        return cls[np.argmax(probs, axis=1)], np.max(probs, axis=1)

    tr_pred, tr_maxp = predict(trr)
    m = tr_lab == TARGET
    hits = tr_maxp[m & (tr_pred == TARGET)]
    n_t = int(m.sum()); needed = int(np.ceil(RECALL_TARGET * n_t))
    thr = np.sort(hits)[::-1][needed - 1] if len(hits) >= needed else (float(hits.min()) if len(hits) else 0.0)

    te_pred, te_maxp = predict(ter)
    y_true = (te_lab == TARGET).astype(int)
    y_pred = ((te_pred == TARGET) & (te_maxp >= thr)).astype(int)
    return y_true, y_pred

def run_condition(cond, model, librosa):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
    paths, labels = list_clips(cond)
    cdir = os.path.join(os.path.dirname(__file__), "embeddings"); os.makedirs(cdir, exist_ok=True)
    emb = embed(paths, model, librosa, os.path.join(cdir, f"tbp_{cond}.npy"))
    print(f"  [{cond}] segments={len(paths)}  TBP={int((labels==TARGET).sum())}", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    yt_all, yp_all = [], []
    for tr_idx, te_idx in skf.split(emb, labels):
        yt, yp = eval_fold(emb[tr_idx], labels[tr_idx], emb[te_idx], labels[te_idx])
        yt_all.append(yt); yp_all.append(yp)
    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    return {
        "cond": cond,
        "recall": recall_score(yt, yp, zero_division=0),
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
        "tp": int(((yt==1)&(yp==1)).sum()), "fp": int(((yt==0)&(yp==1)).sum()),
        "fn": int(((yt==1)&(yp==0)).sum()), "n": len(yt), "n_pos": int(yt.sum()),
    }

def main():
    import librosa, tensorflow_hub as hub
    print("Loading Perch...", flush=True); t0 = time.time()
    model = hub.load(PERCH_URL); print(f"Perch loaded in {time.time()-t0:.0f}s", flush=True)
    res = {}
    for cond in ["denoised", "raw"]:
        print(f"=== condition: {cond} ===", flush=True)
        res[cond] = run_condition(cond, model, librosa)
    print("\n======= TBP ABLATION (tooth-billed pigeon vs rest, 5-fold CV) =======")
    print(f"{'condition':<24}{'recall':>9}{'accuracy':>10}{'precision':>11}{'F1':>8}")
    for cond in ["denoised", "raw"]:
        r = res[cond]; name = "denoised (WITH preproc)" if cond=="denoised" else "raw (NO preproc)"
        print(f"{name:<24}{r['recall']:>9.3f}{r['accuracy']:>10.3f}{r['precision']:>11.3f}{r['f1']:>8.3f}")
    d, w = res["denoised"], res["raw"]
    print("\nDelta (denoised - raw):")
    print(f"  recall {d['recall']-w['recall']:+.3f}  accuracy {d['accuracy']-w['accuracy']:+.3f}  "
          f"precision {d['precision']-w['precision']:+.3f}  F1 {d['f1']-w['f1']:+.3f}")
    print(f"  TP/FP/FN denoised={d['tp']}/{d['fp']}/{d['fn']}  raw={w['tp']}/{w['fp']}/{w['fn']}  "
          f"(n={d['n']}, TBP positives={d['n_pos']})")
    np.save(os.path.join(os.path.dirname(__file__), "ablation_results_tbp.npy"), res)
    print("\nSaved -> ablation_results_tbp.npy")

if __name__ == "__main__":
    main()
