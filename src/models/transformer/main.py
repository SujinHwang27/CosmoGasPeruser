from load_data import load_dataset
from transformer_utils import (
    SimpleTransformerClassifier,
    create_dataloader,
    train_model,
    evaluate_model
)
from plot import plot_training_curves


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset("data/processed/Sherwood_z0.3_inf/dct_full")

    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_val, y_val)

    model = SimpleTransformerClassifier(
        input_dim=20,
        num_classes=4,
        d_model=64,
        nhead=4,
        num_layers=2
    )

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=20,
        lr=1e-3,
        device="cpu"
    )

    # KPI extraction
    test_metrics = evaluate_model(model, X_test, y_test)
    print("\n=== FINAL TEST METRICS ===")
    for k, v in test_metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    plot_training_curves(history)


if __name__ == "__main__":
    main()
