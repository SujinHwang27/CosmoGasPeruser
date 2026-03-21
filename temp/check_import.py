import sys
import os
sys.path.append(os.getcwd())
try:
    from src.viz import plot_spatial_cluster_map
    print("Success: plot_spatial_cluster_map imported")
except ImportError as e:
    print(f"Error: {e}")
    import src.viz
    print(f"Available attributes in src.viz: {[attr for attr in dir(src.viz) if not attr.startswith('__')]}")
