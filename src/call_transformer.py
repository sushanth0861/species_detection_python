import time
import pandas as pd
from transform_image_into_32x32 import transform_dataset
from constants import SAMPLE_FILE_PATH, TRAIN_FILE_PATH

start_time = time.time()

train_df = pd.read_csv(TRAIN_FILE_PATH)
sample_df = pd.read_csv(SAMPLE_FILE_PATH)
transform_dataset(train_df, sample_df)

end_time = time.time()
total_time = end_time - start_time
print("Total time taken for transforming images is", total_time, "seconds")