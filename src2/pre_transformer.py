import matplotlib.pyplot as plt
import cv2
import pandas as pd
from constants import TRAIN_FILE_PATH

def plot_images(train_df, cols, rows):
    figure = plt.figure(figsize=(5 * cols, 3 * rows))

    for i in range(cols * rows):
        image_path = train_df.loc[i, 'file_name']
        image = cv2.imread(f'content/transformed_content/train_images/{image_path}')
        figure.add_subplot(rows, cols, i + 1)
        plt.imshow(image)

    plt.show()

train_df = pd.read_csv(TRAIN_FILE_PATH)
cols = 4
rows = 4
plot_images(train_df, cols, rows)
