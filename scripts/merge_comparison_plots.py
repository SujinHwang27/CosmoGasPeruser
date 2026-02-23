import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def merge_plots():
    path1 = "data/feature_discovery/clustering_results_k5/classifier_clusters_umap.png"
    path2 = "data/feature_discovery/clustering_results/classifier_clusters_umap.png"
    output = "data/feature_discovery/k5_vs_k8_comparison.png"
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print("Required plots not found.")
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
    plt.savefig(output, dpi=150)
    print(f"Saved side-by-side comparison to {output}")

if __name__ == "__main__":
    merge_plots()
