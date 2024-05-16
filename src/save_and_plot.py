import json
import pandas as pd
import matplotlib.pyplot as plt

def save_and_plot(history, model):
    history_df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Plot loss and validation loss
    history_df[['loss', 'val_loss']].plot(ax=axes[0])
    axes[0].set_xlabel('Epoch')
    axes[0].set_title('Loss')

    # Plot accuracy and validation accuracy
    history_df[['accuracy', 'val_accuracy']].plot(ax=axes[1])
    axes[1].set_xlabel('Epoch')
    axes[1].set_title('Accuracy')

    # Plot F1 score, precision, and recall
    history_df[['val_f1', 'val_precision', 'val_recall']].plot(ax=axes[2])
    axes[2].set_xlabel('Epoch')
    axes[2].set_title('F1, Precision, Recall')

    # Print last epoch's metrics
    last_epoch = len(history_df) - 1
    train_loss = history_df['loss'].iloc[last_epoch]
    val_loss = history_df['val_loss'].iloc[last_epoch]
    train_accuracy = history_df['accuracy'].iloc[last_epoch]
    val_accuracy = history_df['val_accuracy'].iloc[last_epoch]
    val_f1 = history_df['val_f1'].iloc[last_epoch]
    val_precision = history_df['val_precision'].iloc[last_epoch]
    val_recall = history_df['val_recall'].iloc[last_epoch]

    plt.tight_layout()
    plt.show()

    print(f"{model} model performance:".title())
    print(f"Training Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation F1 Score: {val_f1}")
    print(f"Validation Precision: {val_precision}")
    print(f"Validation Recall: {val_recall}")