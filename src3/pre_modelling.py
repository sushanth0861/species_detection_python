from src3.constants import BASE_DIR
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load data
    x_train_path = os.path.join(BASE_DIR, "Resized_xTrain.npy")
    y_train_path = os.path.join(BASE_DIR, "Resized_yTrain.npy")

    # Load the data
    X_train_data = np.load(x_train_path)
    y_train_data = np.load(y_train_path)

    # CAREFUL HERE
    # X_train_data = X_train_data[:1000]
    # y_train_data = y_train_data[:1000]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_data, y_train_data, test_size=0.1, random_state=65
    )

    # Preprocess training and validation data & scaling
    X_train = X_train.astype("float32") / 255
    X_val = X_val.astype("float32") / 255

    return X_train, X_val, y_train, y_val