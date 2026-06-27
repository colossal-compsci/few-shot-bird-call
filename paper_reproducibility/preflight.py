"""
NOTE (conservation): The tooth-billed pigeon is critically endangered. Its field
recordings and all identifying metadata (recording filenames, dates, sites, GPS, and
per-call timestamps) are WITHHELD. Recording identifiers in this file are anonymized
(RECORDING_A..E) and data paths are environment variables; fill them from your own
private dataset to run. See the Data Accessibility Statement in the paper.
"""
import time, numpy as np
t0=time.time()
import tensorflow_hub as hub
url='https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8'
print("Loading Perch from:", url, flush=True)
model=hub.load(url)
print("Model loaded in %.1fs"%(time.time()-t0), flush=True)
# single inference: 5s @ 32kHz = 160000 samples
x=np.zeros((1,160000),dtype=np.float32)
r=model.infer_tf(x)
emb=r['embedding'].numpy()
print("Embedding shape:", emb.shape, "| OK")
