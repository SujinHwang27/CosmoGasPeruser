from sklearn.ensemble import RandomForestClassifier
import numpy as np



rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Sort and visualize top features
importances = rf.feature_importances_
feature_names = np.array(["Feature " + str(i) for i in range(X_train.shape[1])])

# Sort features by importance
sorted_indices = np.argsort(importances)[::-1]
for i in sorted_indices[:10]:  # Show top 10 features
    print(f"{feature_names[i]}: {importances[i]:.4f}")



# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
# import joblib

# class RandomForestModel:
#     def __init__(self, task_type='classification', **kwargs):
#         """
#         Initialize Random Forest model
#         Args:
#             task_type (str): 'classification' or 'regression'
#             **kwargs: Additional parameters for RandomForestClassifier/Regressor
#         """
#         self.task_type = task_type
#         if task_type == 'classification':
#             self.model = RandomForestClassifier(**kwargs)
#         else:
#             self.model = RandomForestRegressor(**kwargs)
    
#     def train(self, X, y):
#         """
#         Train the Random Forest model
#         Args:
#             X (array-like): Training features
#             y (array-like): Target values
#         """
#         self.model.fit(X, y)
    
#     def predict(self, X):
#         """
#         Make predictions using the trained model
#         Args:
#             X (array-like): Features to predict on
#         Returns:
#             array-like: Predictions
#         """
#         return self.model.predict(X)
    
#     def evaluate(self, X, y):
#         """
#         Evaluate the model's performance
#         Args:
#             X (array-like): Test features
#             y (array-like): True target values
#         Returns:
#             dict: Dictionary containing evaluation metrics
#         """
#         predictions = self.predict(X)
#         if self.task_type == 'classification':
#             return {
#                 'accuracy': accuracy_score(y, predictions),
#                 'report': classification_report(y, predictions)
#             }
#         else:
#             return {
#                 'mse': mean_squared_error(y, predictions),
#                 'rmse': np.sqrt(mean_squared_error(y, predictions))
#             }
    
#     def save_model(self, filepath):
#         """
#         Save the trained model to disk
#         Args:
#             filepath (str): Path where to save the model
#         """
#         joblib.dump(self.model, filepath)
    
#     def load_model(self, filepath):
#         """
#         Load a trained model from disk
#         Args:
#             filepath (str): Path to the saved model
#         """
#         self.model = joblib.load(filepath) 