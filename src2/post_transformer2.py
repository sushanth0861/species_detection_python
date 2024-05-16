import matplotlib.pyplot as plt
import numpy as np
import os

transformed_data_directory = 'content/transformed_data_clahe/'
file_path = os.path.join(transformed_data_directory, 'Resized_xTrain.npy')
def plot_images(cols, rows):
    figure = plt.figure(figsize=(5 * cols, 3 * rows))

    for i in range(cols * rows):
        image_array = np.load(file_path)[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.imshow(image_array)

    plt.show()

cols = 4
rows = 4
plot_images(cols, rows)