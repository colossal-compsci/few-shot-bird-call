"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
"""
Build a clean TBP prototype model that EXCLUDES the held-out test recording.

The TBP training calls are the 36 segments from the single 'Confirmed_tooth_billed_pigeon'
recording (paper setting). The 4 'RECORDING_E' segments are dropped so that
the manumea test recording is fully independent. Comparison species are used as-is. Outputs a
prototype model (PCA + median prototypes) to TBP_experiments/model_clean/.
"""
import os, sys, glob
import numpy as np
REPO = "${TBP_PIPELINE_REPO}"
sys.path.insert(0, os.path.join(REPO, "src"))
TRAIN = "${TBP_DATA_ROOT}/data/train_data"
OUT = os.path.join(os.path.dirname(__file__), "model_clean")
N_COMPONENTS = 142
EXCLUDE_SUBSTR = "RECORDING_E"   # held-out test recording
PERCH_URL = ('https://www.kaggle.com/models/google/bird-vocalization-classifier/'
             'TensorFlow2/bird-vocalization-classifier/8')
PERCH_SR, PERCH_SAMPLES = 32000, 160000

def main():
    import librosa, tensorflow_hub as hub, joblib
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    os.makedirs(OUT, exist_ok=True)

    paths, labels = [], []
    for sp in sorted(os.listdir(TRAIN)):
        for w in sorted(glob.glob(os.path.join(TRAIN, sp, "segmented_audio", "denoised", "*.wav"))):
            if EXCLUDE_SUBSTR in os.path.basename(w):
                continue
            paths.append(w); labels.append(sp)
    labels = np.array(labels)
    print("Train segments per species (after exclusion):")
    for sp in np.unique(labels):
        print(f"  {sp}: {(labels==sp).sum()}")
    assert not any(EXCLUDE_SUBSTR in p for p in paths), "held-out file leaked into training!"

    print("Loading Perch...", flush=True)
    model = hub.load(PERCH_URL)
    def emb_one(p):
        y, _ = librosa.load(p, sr=PERCH_SR)
        if len(y) < PERCH_SAMPLES:
            pad = PERCH_SAMPLES - len(y); y = np.pad(y, (pad//2, pad-pad//2))
        return model.infer_tf(y[:PERCH_SAMPLES].astype(np.float32)[np.newaxis, :])["embedding"].numpy().flatten()
    embs = np.array([emb_one(p) for p in paths])
    print("embeddings:", embs.shape)

    pca = PCA(n_components=N_COMPONENTS)
    red = normalize(pca.fit_transform(embs), axis=1, norm="l2")
    classes = np.unique(labels)
    protos = np.array([np.median(red[labels == c], axis=0) for c in classes])

    joblib.dump(pca, os.path.join(OUT, "pca_model.joblib"))
    np.save(os.path.join(OUT, "prototype_matrix.npy"), protos)
    np.save(os.path.join(OUT, "prototype_labels.npy"), classes)
    np.save(os.path.join(OUT, "embeddings.npy"), embs)
    np.save(os.path.join(OUT, "labels.npy"), labels)
    print(f"Saved clean model -> {OUT}  (classes: {list(classes)})")

if __name__ == "__main__":
    main()
