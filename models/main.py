import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import umap
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from data_loader import load_data, load_wavelength
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], 'GPU')



import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    spectra, labels = load_data(2.4)
    print(labels)
    
    # 2. Split data into train and test sets
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, 
        labels, 
        test_size=0.1,  
        random_state=42,  
        stratify=labels  # maintain class distribution
    )
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    # Reshape your data for CNN (20000 samples, 1064 features, 1 channel)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the CNN model
    model = models.Sequential()

    # Add convolutional layers
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))

    # Flatten the output and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))  # 4 classes

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
    print(f"Test accuracy: {test_acc:.3f}")

    # Predict and evaluate
    y_pred = model.predict(X_test_reshaped)
    y_pred_classes = tf.argmax(y_pred, axis=1)

    # Print classification report
    print(classification_report(y_test, y_pred_classes))

if __name__ == "__main__":
    main()