"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
"""
E2 — Preprocessing ablation on the Eastern Towhee (ET) dataset.

Compares the full classification pipeline using:
  - 'denoised' clips  = WITH preprocessing (bandpass + denoise + normalize)   [paper setting]
  - 'raw' clips       = WITHOUT preprocessing
keeping every other step identical (Perch embeddings -> PCA(95% var) -> L2 norm ->
median prototypes -> cosine similarity -> softmax -> recall-0.9 threshold on ET train).

Metric: binary Eastern-Towhee-vs-rest on the test set (recall, accuracy, precision, F1),
matching the paper's reported ET numbers (recall 0.979, accuracy 0.988, F1 0.968) for the
'denoised' condition, which serves as a reproduction sanity check.
"""
import os, glob, sys, time
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")
TARGET = "eastern_towhee"
TRAIN_SPECIES = ["carolina_wren", "eastern_towhee", "kentucky_warbler",
                 "northern_cardinal", "white_throated_sparrow"]
PERCH_URL = ('https://www.kaggle.com/models/google/bird-vocalization-classifier/'
             'TensorFlow2/bird-vocalization-classifier/8')
PERCH_SR = 32000
PERCH_SAMPLES = 160000  # 5 s
RECALL_TARGET = 0.90

def norm_label(folder_name):
    """Map a species folder name to its canonical label (test KW folder has a _test suffix)."""
    return "kentucky_warbler" if folder_name.startswith("kentucky_warbler") else folder_name

def list_clips(split_dir, cond):
    """Return (paths, labels) for one split ('us_train_data'/'us_test_data') and condition."""
    paths, labels = [], []
    base = os.path.join(DATA, split_dir)
    for sp_folder in sorted(os.listdir(base)):
        cond_dir = os.path.join(base, sp_folder, "processed", "segmented_audio", cond)
        if not os.path.isdir(cond_dir):
            continue
        wavs = sorted(glob.glob(os.path.join(cond_dir, "*", "*.wav")))
        for w in wavs:
            paths.append(w)
            labels.append(norm_label(sp_folder))
    return paths, np.array(labels)

def load_audio_perch(path, librosa):
    y, _ = librosa.load(path, sr=PERCH_SR)  # resample to Perch's expected rate
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
        e = model.infer_tf(x)["embedding"].numpy().flatten()
        embs.append(e)
        if (i + 1) % 25 == 0:
            print(f"    embedded {i+1}/{len(paths)}", flush=True)
    arr = np.array(embs)
    np.save(cache, arr)
    return arr

def l2(x):
    from sklearn.preprocessing import normalize
    return normalize(x, axis=1, norm="l2")

def softmax(s):
    e = np.exp(s - np.max(s, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def run_condition(cond, model, librosa):
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

    tr_paths, tr_labels = list_clips("us_train_data", cond)
    te_paths, te_labels = list_clips("us_test_data", cond)
    print(f"  [{cond}] train clips={len(tr_paths)}  test clips={len(te_paths)}", flush=True)

    cdir = os.path.join(os.path.dirname(__file__), "embeddings")
    os.makedirs(cdir, exist_ok=True)
    tr_emb = embed(tr_paths, model, librosa, os.path.join(cdir, f"train_{cond}.npy"))
    te_emb = embed(te_paths, model, librosa, os.path.join(cdir, f"test_{cond}.npy"))

    # PCA capturing 95% variance (paper's rule), fit on train only
    pca = PCA(n_components=0.95, svd_solver="full")
    tr_red = l2(pca.fit_transform(tr_emb))
    te_red = l2(pca.transform(te_emb))
    n_comp = pca.n_components_

    # median prototypes per class
    classes = np.unique(tr_labels)
    protos = np.array([np.median(tr_red[tr_labels == c], axis=0) for c in classes])

    def predict(red):
        sims = cosine_similarity(red, protos)
        probs = softmax(sims)
        pred_idx = np.argmax(probs, axis=1)
        max_prob = np.max(probs, axis=1)
        return classes[pred_idx], max_prob

    # threshold = lowest max-prob achieving recall>=0.9 for ET on TRAIN
    tr_pred, tr_maxp = predict(tr_red)
    et_mask = tr_labels == TARGET
    et_hits = tr_maxp[et_mask & (tr_pred == TARGET)]  # ET-train clips correctly argmax'd to ET
    n_et = int(et_mask.sum())
    needed = int(np.ceil(RECALL_TARGET * n_et))
    if len(et_hits) >= needed:
        thr = np.sort(et_hits)[::-1][needed - 1]  # keep top-'needed' -> recall 0.9
    else:
        thr = float(et_hits.min()) if len(et_hits) else 0.0
    print(f"  [{cond}] PCA n_components={n_comp}  ET train clips={n_et}  threshold={thr:.4f}", flush=True)

    # apply to test, binary ET-vs-rest
    te_pred, te_maxp = predict(te_red)
    y_true = (te_labels == TARGET).astype(int)
    y_pred = ((te_pred == TARGET) & (te_maxp >= thr)).astype(int)
    return {
        "cond": cond, "n_components": int(n_comp), "threshold": float(thr),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_test": len(y_true), "n_pos": int(y_true.sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }

def main():
    import librosa
    import tensorflow_hub as hub
    print("Loading Perch...", flush=True)
    t0 = time.time()
    model = hub.load(PERCH_URL)
    print(f"Perch loaded in {time.time()-t0:.0f}s", flush=True)

    results = {}
    for cond in ["denoised", "raw"]:
        print(f"=== condition: {cond} ===", flush=True)
        results[cond] = run_condition(cond, model, librosa)

    print("\n================ ABLATION RESULTS (Eastern Towhee, binary) ================")
    hdr = f"{'condition':<24}{'recall':>9}{'accuracy':>10}{'precision':>11}{'F1':>8}{'thr':>8}{'PCs':>6}"
    print(hdr)
    for cond in ["denoised", "raw"]:
        r = results[cond]
        name = "denoised (WITH preproc)" if cond == "denoised" else "raw (NO preproc)"
        print(f"{name:<24}{r['recall']:>9.3f}{r['accuracy']:>10.3f}{r['precision']:>11.3f}"
              f"{r['f1']:>8.3f}{r['threshold']:>8.3f}{r['n_components']:>6}")
    d, w = results["denoised"], results["raw"]
    print("\nDelta (denoised - raw):")
    print(f"  recall:   {d['recall']-w['recall']:+.3f}   accuracy: {d['accuracy']-w['accuracy']:+.3f}")
    print(f"  TP/FP/FN  denoised={d['tp']}/{d['fp']}/{d['fn']}   raw={w['tp']}/{w['fp']}/{w['fn']}"
          f"   (test n={d['n_test']}, ET positives={d['n_pos']})")
    np.save(os.path.join(os.path.dirname(__file__), "ablation_results.npy"), results)
    print("\nSaved -> ablation_results.npy")

if __name__ == "__main__":
    main()
