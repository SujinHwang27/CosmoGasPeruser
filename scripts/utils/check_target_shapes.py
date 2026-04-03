import numpy as np
import os

files = [
    "data/feature_discovery/micro_classifier_params.npy",
    "data/processed/wavelet_db8_l6_d12/1/data.npy",
    "data/processed/windowed_dct_256_128_32/1/data.npy"
]

for f in files:
    if os.path.exists(f):
        data = np.load(f, mmap_mode='r')
        print(f"{f}: {data.shape}")
    else:
        print(f"{f}: File not found")
