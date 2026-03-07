import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def merge_plots():
    # Updated paths for reorganized directory structure
    path1 = "data/feature_discovery/experiments/wavelet_k5_test/classifier_clusters_umap.png"
    path2 = "data/feature_discovery/experiments/wavelet_k8_primary/classifier_clusters_umap.png"
    output = "data/feature_discovery/comparisons/k_stability/k5_vs_k8_comparison.png"
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"Required plots not found at {path1} or {path2}")
        return

    img1 = mpimg.imread(path1)
    img2 = mpimg.imread(path2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(img1)
    ax1.set_title("K=5 Clusters", fontsize=20)
    ax1.axis('off')
    
    ax2.imshow(img2)
    ax2.set_title("K=8 Clusters (Discovery scale)", fontsize=20)
    ax2.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Saved side-by-side comparison to {output}")

if __name__ == "__main__":
    merge_plots()
