"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
"""
End-to-end preprocessing ablation on the TBP test recordings.

For each full test recording in ../test_files/, we run the wavelet EVENT DETECTOR on:
  - 'denoised' signal = WITH preprocessing (bandpass 150-600 Hz + denoise + normalize)
  - 'raw' signal      = WITHOUT preprocessing
then segment each detected event, embed with Perch, and classify with the TBP 6-species
cosine-similarity model (model artifacts in few-shot-bird-call/model/).

This tests the hypothesis that preprocessing's value is in the DETECTION stage (cleaner,
fewer candidate events on field audio) rather than in classifying already-segmented calls.
Reports, per condition: number of detected events, number classified as tooth-billed pigeon,
and the per-file breakdown. (Also speaks to Reviewer C3 / E1: the event detector's behavior.)
"""
import os, sys, glob, time
import numpy as np

HERE = os.path.dirname(__file__)
REPO = "${TBP_PIPELINE_REPO}"
sys.path.insert(0, os.path.join(REPO, "src"))
TEST_DIR = "${TBP_TEST_FILES}"
MODEL_DIR = os.path.join(REPO, "model")
PERCH_URL = ('https://www.kaggle.com/models/google/bird-vocalization-classifier/'
             'TensorFlow2/bird-vocalization-classifier/8')
PERCH_SR = 32000
PERCH_SAMPLES = 160000
SEG_LEN = 2.0
THRESHOLD = 0.24       # model's default decision threshold (same for both conditions)
TARGET = "tooth_billed_pigeon"

def perch_input(seg, sr, librosa):
    """Resample a detected segment to Perch's rate and pad/truncate to 5 s."""
    if sr != PERCH_SR:
        seg = librosa.resample(seg, orig_sr=sr, target_sr=PERCH_SR)
    if len(seg) < PERCH_SAMPLES:
        pad = PERCH_SAMPLES - len(seg)
        seg = np.pad(seg, (pad // 2, pad - pad // 2))
    else:
        seg = seg[:PERCH_SAMPLES]
    return seg.astype(np.float32)

def main():
    import librosa
    import tensorflow_hub as hub
    from event_detection import EventDetection
    from make_prediction import MakePrediction
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity

    print("Loading Perch...", flush=True); t0 = time.time()
    model = hub.load(PERCH_URL); print(f"Perch loaded in {time.time()-t0:.0f}s", flush=True)

    detector = EventDetection()           # default target_sr=22050, bandpass 150-600 via AudioProcessor
    predictor = MakePrediction(n_components=142, model_folder=MODEL_DIR)
    proto_matrix, proto_labels = predictor.prepare_prototypes()
    proto_labels = list(proto_labels)

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.wav")) + glob.glob(os.path.join(TEST_DIR, "*.mp3")))
    print(f"Test recordings: {[os.path.basename(f) for f in files]}", flush=True)

    def classify(segments, sr):
        """Embed + classify a list of audio segments; return list of (label, conf)."""
        if not segments:
            return []
        embs = np.array([model.infer_tf(perch_input(s, sr, librosa)[np.newaxis, :])["embedding"].numpy().flatten()
                         for s in segments])
        red = normalize(predictor.transform_pca(embs), axis=1, norm="l2")
        sims = cosine_similarity(red, proto_matrix)
        probs = predictor.softmax(sims)
        out = []
        for p in probs:
            mp = float(np.max(p)); lab = proto_labels[int(np.argmax(p))] if mp >= THRESHOLD else "unknown"
            out.append((lab, mp))
        return out

    summary = {"denoised": {"events": 0, "tbp": 0}, "raw": {"events": 0, "tbp": 0}}
    per_file = []
    for f in files:
        name = os.path.basename(f)
        try:
            y, y_den, sr = detector.audio_processor.load_audio(f)
        except Exception as e:
            print(f"  !! could not load {name}: {e}", flush=True); continue
        row = {"file": name}
        for cond, signal in [("denoised", y_den), ("raw", y)]:
            feats = detector.smoothed_features(signal, sr, wavelet="mexh", window_size=0.15)
            ev = detector.detected_events(feats, sr)
            segs = [signal[int(s*sr):int(e*sr)] for s, e in ev]
            preds = classify(segs, sr)
            n_tbp = sum(1 for lab, _ in preds if lab == TARGET)
            row[cond] = {"n_events": len(ev), "n_tbp": n_tbp,
                         "tbp_conf": sorted([round(c, 3) for l, c in preds if l == TARGET], reverse=True)}
            summary[cond]["events"] += len(ev); summary[cond]["tbp"] += n_tbp
            print(f"  [{name[:40]:<40}] {cond:<9} events={len(ev):>3}  TBP={n_tbp}", flush=True)
        per_file.append(row)

    print("\n=========== END-TO-END ABLATION (event detect -> classify) ===========")
    print(f"{'condition':<24}{'total events':>14}{'TBP detections':>16}")
    for cond in ["denoised", "raw"]:
        print(f"{('denoised (WITH preproc)' if cond=='denoised' else 'raw (NO preproc)'):<24}"
              f"{summary[cond]['events']:>14}{summary[cond]['tbp']:>16}")
    print("\nPer-file detail:")
    for r in per_file:
        print(f"  {r['file']}")
        for cond in ["denoised", "raw"]:
            print(f"     {cond:<9} events={r[cond]['n_events']:>3} TBP={r[cond]['n_tbp']} "
                  f"conf={r[cond]['tbp_conf'][:6]}")
    np.save(os.path.join(HERE, "endtoend_results.npy"), {"summary": summary, "per_file": per_file})
    print("\nSaved -> endtoend_results.npy")

if __name__ == "__main__":
    main()
