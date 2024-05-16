from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Define class mapping
class_mapping = {
    'opossum': 19, 'empty': 0, 'squirrel': 3, 'rabbit': 8, 'rodent': 4,
    'raccoon': 13, 'deer': 1, 'coyote': 11, 'bobcat': 16, 'cat': 17,
    'skunk': 14, 'dog': 18, 'fox': 10, 'mountain_lion': 22
}

labels = sorted(class_mapping, key=lambda x: class_mapping[x])

#Function to create confusion matrix heatmap
def plot_confusion_matrix(conf_matrix, labels):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Display labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = conf_matrix.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()

def get_confusion_matrix(model, X_test, y_test):
    # Evaluate the model on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

    return plot_confusion_matrix(conf_matrix, labels)
