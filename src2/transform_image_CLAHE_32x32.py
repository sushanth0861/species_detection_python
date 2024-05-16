import cv2
import numpy as np
import pandas as pd
import math
import os

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def apply_clahe_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    clahe_image = clahe.apply(gray)
    # return clahe_image
    return cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(img_lab)
    img_l = clahe.apply(l)
    img_clahe = cv2.merge((img_l, a, b))
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)

    return img_clahe

def pad_and_resize_cv(image_path, dataset, desired_size=32):
    img = cv2.imread(f'content/transformed_content/{dataset}_images/{image_path}.jpg',cv2.IMREAD_COLOR)
    
    clahe_img = apply_clahe(img)
    
    pad_width = get_pad_width(clahe_img, max(clahe_img.shape))
    padded = np.pad(clahe_img, pad_width=pad_width, mode='constant', constant_values=0)
    
    resized = cv2.resize(padded, (desired_size,)*2).astype('uint8')
    
    return resized

def transform_dataset(train_df, sample_df):

    train_resized_images = [pad_and_resize_cv(image_id, 'train') for image_id in train_df['id']]#[:10000]]
    test_resized_images  = [pad_and_resize_cv(image_id, 'test') for image_id in sample_df['Id']]#[:10000]]

    X_train = np.stack(train_resized_images)
    X_test = np.stack(test_resized_images)

    target_dummies = pd.get_dummies(train_df['category_id'])
    train_label = target_dummies.columns.values
    y_train = target_dummies.values

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)

    base_directory = 'content'
    transformed_data_directory = os.path.join(base_directory, 'transformed_data_clahe')
        
    # Create the 'transformed_data' directory if it doesn't exist
    if not os.path.exists(transformed_data_directory):
        os.makedirs(transformed_data_directory)

    # Save files, replacing if they already exist
    np.save(os.path.join(transformed_data_directory, 'Resized_xTrain.npy'), X_train)
    print("Saving content/transformed_data_clahe/Resized_xTrain.npy")
    np.save(os.path.join(transformed_data_directory, 'Resized_yTrain.npy'), y_train)
    print("Saving content/transformed_data_clahe/Resized_yTrain.npy")
    np.save(os.path.join(transformed_data_directory, 'Resized_xTest.npy'), X_test)
    print("Saving content/transformed_data_clahe/Resized_xTest.npy")

# if __name__ == "__main__":
#     train_file_path = sys.argv[1]
#     sample_file_path = sys.argv[2]
    
#     # Load train and sample datasets
#     train_df = pd.read_csv(train_file_path)
#     sample_df = pd.read_csv(sample_file_path)
    
#     # Process the datasets
#     transform_dataset(train_df, sample_df)