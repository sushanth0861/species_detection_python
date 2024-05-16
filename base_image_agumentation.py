import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from src3.constants import CONTENT_ROOT

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.25 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img_path, ang_range, shear_range, trans_range, brightness=0):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ang_rot = np.random.uniform(ang_range[0], ang_range[1]) - (ang_range[1] - ang_range[0]) / 2
    rows, cols, ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    tr_x = np.random.uniform(trans_range[0], trans_range[1]) - (trans_range[1] - trans_range[0]) / 2
    tr_y = np.random.uniform(trans_range[0], trans_range[1]) - (trans_range[1] - trans_range[0]) / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    shear_val = np.random.uniform(shear_range[0], shear_range[1]) - (shear_range[1] - shear_range[0]) / 2
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pts2 = np.float32([[5 + shear_val, 5], [20 + shear_val, 5 + shear_val], [5, 20]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    if brightness == 1:
        img = augment_brightness_camera_images(img)

    return img

def save_transformed_images(input_img_path, output_dir, num_images, ang_range, shear_range, trans_range, brightness=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_images):
        transformed_img = transform_image(input_img_path, ang_range, shear_range, trans_range, brightness)
        output_img_path = os.path.join(output_dir, f"transformed_{i}.jpg")
        cv2.imwrite(output_img_path, cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))

def display_transformed_images(output_dir, num_images):
    plt.figure(figsize=(20, 20))
    gs1 = gridspec.GridSpec(10, 10)
    gs1.update(wspace=0.01, hspace=0.02)  # set the spacing between axes.
    
    transformed_image_paths = sorted(os.listdir(output_dir))
    for i in range(num_images):
        if i >= len(transformed_image_paths):
            break
        img_path = os.path.join(output_dir, transformed_image_paths[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax1 = plt.subplot(gs1[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

        plt.subplot(10, 10, i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.show()
    time.sleep(5)  # Display images for 5 seconds
    plt.close()    # Close the matplotlib window after 5 seconds

def apply_transformations_to_images(input_dir, output_dir, num_images, ang_range, shear_range, trans_range, brightness=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming the images are JPEG or PNG files
            input_img_path = os.path.join(input_dir, filename)

            for i in range(num_images):
                transformed_img = transform_image(input_img_path, ang_range, shear_range, trans_range, brightness)
                output_img_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_transformed_{i}.jpg")
                cv2.imwrite(output_img_path, cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))

# Usage example:
input_dir = os.path.join(CONTENT_ROOT, "test_image_dump1")
output_dir =  os.path.join(CONTENT_ROOT, "test_image_dump2")
num_images = 1  # Number of transformed images to display
ang_range = [-5, 5]  # Range of angles for rotation
shear_range = [-5, 5]  # Range of shear values
trans_range = [-5, 5]  # Range of translation values
brightness = 1  # Augment brightness
display_transformed_images(input_dir, 5)
apply_transformations_to_images(input_dir, output_dir, num_images, ang_range, shear_range, trans_range, brightness)
display_transformed_images(output_dir, 5)
