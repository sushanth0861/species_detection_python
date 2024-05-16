#Source: https://www.kaggle.com/code/xhlulu/exploration-and-preprocessing-for-keras-224x224?scriptVersionId=8423348&cellId=11
#Some code is inspired from above mentioned source.
import cv2
import numpy as np
import pandas as pd
import math
import os
import sys

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def pad_and_resize_cv(image_path, dataset, desired_size=32):
    img = cv2.imread(f'content/transformed_content/{dataset}_images/{image_path}.jpg')
    
    pad_width = get_pad_width(img, max(img.shape))
    padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    
    resized = cv2.resize(padded, (desired_size,)*2).astype('uint8')
    
    return resized

def transform_dataset(train_df, sample_df):

    train_resized_images = [pad_and_resize_cv(image_id, 'train') for image_id in train_df['id']]
    test_resized_images  = [pad_and_resize_cv(image_id, 'test') for image_id in sample_df['Id']]

    X_train = np.stack(train_resized_images)
    X_test = np.stack(test_resized_images)

    target_dummies = pd.get_dummies(train_df['category_id'])
    train_label = target_dummies.columns.values
    y_train = target_dummies.values

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)

    base_directory = 'content'
    transformed_data_directory = os.path.join(base_directory, 'transformed_data')
        
    # Create the 'transformed_data' directory if it doesn't exist
    if not os.path.exists(transformed_data_directory):
        os.makedirs(transformed_data_directory)

    # Save files, replacing if they already exist
    np.save(os.path.join(transformed_data_directory, 'Resized_xTrain.npy'), X_train)
    print("Saving content/transformed_data/Resized_xTrain.npy")
    np.save(os.path.join(transformed_data_directory, 'Resized_yTrain.npy'), y_train)
    print("Saving content/transformed_data/Resized_yTrain.npy")
    np.save(os.path.join(transformed_data_directory, 'Resized_xTest.npy'), X_test)
    print("Saving content/transformed_data/Resized_xTest.npy")

# if __name__ == "__main__":
#     train_file_path = sys.argv[1]
#     sample_file_path = sys.argv[2]
    
#     # Load train and sample datasets
#     train_df = pd.read_csv(train_file_path)
#     sample_df = pd.read_csv(sample_file_path)
    
#     # Process the datasets
#     transform_dataset(train_df, sample_df)
