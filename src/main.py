import yaml
import argparse
import os
import numpy as np
import mlflow
from dotenv import load_dotenv
from core.data import DatasetFactory

# Load environment variables
load_dotenv()
from core.transforms import PCATransform, DCTTransform, FisherTransform
from core.models import SimpleTransformerClassifier, train_model
from core.viz import plot_training_curves
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

def create_dataloader(X, y, batch_size=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def run_experiment(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Pattern 2: Remote Tracking Support
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = config.get('mlflow', {}).get('experiment', config.get('name', 'Default'))
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=config.get('name')):
        print(f"--- Running Experiment: {config.get('name', 'unnamed')} ---")
        mlflow.log_params(config.get('mlflow', {}).get('params', {}))
        
        # 1. Load Data
        X, y = DatasetFactory.get_dataset(
            config['data'], 
            filename=config.get('data_filename', 'flux.npy')
        )
        print(f"Loaded data: {X.shape}, classes: {len(np.unique(y))}")
        mlflow.log_param("input_shape", X.shape)
        
        # 2. Apply Transformations
        for transform_cfg in config.get('transforms', []):
            t_type = transform_cfg['type']
            print(f"Applying transform: {t_type}")
            
            if t_type == 'pca':
                t = PCATransform(**transform_cfg.get('params', {}))
            elif t_type == 'dct':
                t = DCTTransform(**transform_cfg.get('params', {}))
            elif t_type == 'fisher':
                t = FisherTransform(**transform_cfg.get('params', {}))
            else:
                raise ValueError(f"Unknown transform: {t_type}")
                
            X = t.fit_transform(X, y)
            print(f"New shape: {X.shape}")
            mlflow.log_param(f"transform_{t_type}", transform_cfg.get('params'))

        # 3. Handle Model Selection
        model_cfg = config.get('model', {})
        if model_cfg.get('type') == 'transformer':
            # Split Data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            train_loader = create_dataloader(X_train, y_train)
            val_loader = create_dataloader(X_val, y_val)

            constructor_keys = ['num_classes', 'd_model', 'nhead', 'num_layers']
            constructor_params = {k: v for k, v in model_cfg.get('params', {}).items() if k in constructor_keys}
            
            model = SimpleTransformerClassifier(
                input_dim=X.shape[1],
                **constructor_params
            )
            mlflow.log_params(constructor_params)

            model, history = train_model(
                model, 
                train_loader, 
                val_loader,
                epochs=model_cfg.get('params', {}).get('epochs', 10),
                lr=model_cfg.get('params', {}).get('lr', 1e-3)
            )
            
            # Log metrics per epoch (simplified)
            for i in range(len(history['train_loss'])):
                mlflow.log_metric("train_loss", history['train_loss'][i], step=i)
                mlflow.log_metric("val_loss", history['val_loss'][i], step=i)
                mlflow.log_metric("val_acc", history['val_acc'][i], step=i)
            
            plot_path = f"reports/{config.get('name', 'exp')}_plot.png"
            plot_training_curves(history, save_path=plot_path)
            mlflow.log_artifact(plot_path)
        
        print(f"Experiment {config.get('name')} completed. Track results in MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    
    run_experiment(args.config)
