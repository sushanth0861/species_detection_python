import json
import os
import glob
from keras import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import ResNet101, DenseNet121
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from src3.constants import *
from datetime import datetime


def train_model(model_name, X_train, y_train, X_val, y_val, f1_metrics):
    if model_name.lower() == "resnet":
        base_model = ResNet101(include_top=False, input_shape=(32, 32, 3))
    elif model_name.lower() == "densenet":
        base_model = DenseNet121(include_top=False, input_shape=(32, 32, 3))
    else:
        raise ValueError("Invalid model name. Choose either 'resnet' or 'densenet'.")

    # Define the sequential model
    model = Sequential(name=model_name)
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(14, activation="softmax"))

    # Compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"]
    )

    # Print model summary
    model.summary()

    # Define checkpoint to save the best model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint = ModelCheckpoint(
        f"model_{model_name}_clahe_gs_{timestamp}.h5",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    # Fit the model using the validation generator
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=10,
        callbacks=[f1_metrics, checkpoint]
    )

    # Extract F1 scores, precision, and recall from f1_metrics
    f1_scores = f1_metrics.val_f1s
    val_precisions = f1_metrics.val_precisions
    val_recalls = f1_metrics.val_recalls

    # Include F1 scores, precision, and recall in the history dictionary
    history.history['val_f1'] = f1_scores
    history.history['val_precision'] = val_precisions
    history.history['val_recall'] = val_recalls

    with open(f'history_{model_name}_clahe_gs.json', 'w') as f:
        json.dump(history.history, f)

    return model, history
