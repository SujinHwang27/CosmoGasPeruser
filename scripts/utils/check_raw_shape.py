import numpy as np
import os

def check_raw_shape():
    path = "data/preprocessed/Sherwood_z0.3_inf/1/flux.npy"
    if os.path.exists(path):
        data = np.load(path, mmap_mode='r')
        print(f"Raw data shape: {data.shape}")
        print(f"Total elements: {data.size}")
    else:
        print(f"File not found: {path}")

if __name__ == "__main__":
    check_raw_shape()
