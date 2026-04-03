import numpy as np
import os

def check_shapes():
    paths = [
        "data/processed/wavelet_db8_l6_d12/1/data.npy",
        "data/processed/windowed_dct_256_128_32/1/data.npy",
        "data/feature_discovery/base_data/params_wavelet.npy",
        "data/feature_discovery/base_data/params_dct.npy"
    ]
    
    for path in paths:
        if os.path.exists(path):
            data = np.load(path, mmap_mode='r')
            print(f"{path} shape: {data.shape}")
        else:
            print(f"{path} does not exist.")

if __name__ == "__main__":
    check_shapes()
