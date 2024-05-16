import os

# Constants
CONTENT_ROOT = r"C:\Users\susha\Documents\Workspace\ML_project\species_detection_python\content"
SAMPLE_FILE_PATH = os.path.join(CONTENT_ROOT,"transformed_content","sample_submission.csv")
TRAIN_FILE_PATH = os.path.join(CONTENT_ROOT,"transformed_content","train.csv")

BASE_DIR = os.path.join(CONTENT_ROOT, "transformed_data1")
BATCH_SIZE = 64
