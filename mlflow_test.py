import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)



# set the tracking server's uri (otherwise log will be saved in local
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# set new MLflow experiment name
mlflow.set_experiment("MLflow Quickstart")

# Start MLflow run
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    
    # infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))


    # log the model (params and metric)
    model_info =  mlflow.sklearn.log_model(
            sk_model = lr,
            name="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
    )    # automatic signature generation with input_example during model logging

    # set tag for the model (brief message about the model)
    mlflow.set_logged_model_tags(
            model_info.model_id, {"Training Info": "Basic LR model for iris data"}
    )

    # Load the model back for predictions, with pyfunc flavor
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    iris_feature_names = datasets.load_iris().feature_names


    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    result[:4]






