import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from data_loader import load_data, load_wavelength



import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    spectra, labels = load_data(2.4)
    
    # 2. Split data into train and test sets
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, 
        labels, 
        test_size=0.2,  
        random_state=42,  
        stratify=labels  # maintain class distribution
    )
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    # Define the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(objective="multi:softmax", num_class=4, eval_metric="mlogloss", use_label_encoder=False)

    # Train the model
    xgb_clf.fit(X_train, y_train)

    # Predictions
    y_pred = xgb_clf.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(xgb.XGBClassifier(objective="multi:softmax", num_class=4, eval_metric="mlogloss"),
                            param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Best parameters
    print("Best Parameters:", grid_search.best_params_)

    # Evaluate the best model
    best_xgb = grid_search.best_estimator_
    y_pred_best = best_xgb.predict(X_test)

    print("Tuned Accuracy:", accuracy_score(y_test, y_pred_best))
    print("Tuned Classification Report:\n", classification_report(y_test, y_pred_best))


if __name__ == "__main__" : 
    main()