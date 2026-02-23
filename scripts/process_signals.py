import os
import numpy as np
import argparse
from src.core.data import DataIngestor
from src.core.transforms import WindowedDCTTransform, WaveletTransform
from tqdm import tqdm

def process_data(base_path, output_dir, transform_type='wavelet', **transform_params):
    """
    Processes flux data: F -> A = 1 - F, then applies transform.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {base_path}...")
    ingestor = DataIngestor(base_path, filename="flux.npy", num_classes=4)
    X, y = ingestor.load()
    
    # Step 1: Compute Absorption Field A = 1 - F
    print("Computing absorption field A = 1 - F...")
    A = 1.0 - X
    
    # Step 2: Apply Transform
    if transform_type == 'wavelet':
        print(f"Applying Wavelet transform ({transform_params.get('wavelet', 'db8')})...")
        transformer = WaveletTransform(**transform_params)
    elif transform_type == 'windowed_dct':
        print(f"Applying Windowed DCT transform...")
        transformer = WindowedDCTTransform(**transform_params)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    X_transformed = transformer.fit_transform(A)
    print(f"Transformed shape: {X_transformed.shape}")
    
    # Step 3: Save results back into class-specific directories for the micro-classifiers
    # Actually, the user wants n classifiers trained on sample i from each class.
    # It might be easier to save the whole transformed matrix and then slice it.
    
    num_samples_per_class = len(X) // 4
    for c in range(1, 5):
        class_dir = os.path.join(output_dir, str(c))
        os.makedirs(class_dir, exist_ok=True)
        
        start = (c-1) * num_samples_per_class
        end = c * num_samples_per_class
        
        np.save(os.path.join(class_dir, "data.npy"), X_transformed[start:end])
        print(f"Saved class {c} to {class_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/preprocessed/Sherwood_z0.3_inf", help="Input data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed features")
    parser.add_argument("--type", type=str, choices=['wavelet', 'windowed_dct'], default='wavelet', help="Transform type")
    parser.add_argument("--wavelet", type=str, default="db8", help="Wavelet family")
    parser.add_argument("--level", type=int, default=6, help="Wavelet decomposition level")
    parser.add_argument("--drop_levels", type=int, nargs='*', default=[1, 2], help="Levels to drop (e.g. 1 2 for D1, D2)")
    
    args = parser.parse_args()
    
    params = {}
    if args.type == 'wavelet':
        params = {'wavelet': args.wavelet, 'level': args.level, 'drop_levels': args.drop_levels}
    elif args.type == 'windowed_dct':
        params = {'window_len': 256, 'hop_size': 128, 'n_coeffs': 32}
        
    process_data(args.input, args.output, transform_type=args.type, **params)
