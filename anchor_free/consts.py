TRAINER_EXPERIMENT_NAME = "002_segmentation"
TRAINER_EXPERIMENT_VERSION = "001_test"
TRAINER_FAST_DEV_RUN = False

TRAINER_MAX_EPOCHS = 1000
TRAINER_MONITOR = "jaccard/val"
TRAINER_MONITOR_MODE = "max"
TRAINER_EARLY_STOPPING_PATIENCE = 100

TRAINER_ACCELERATOR = "gpu"
TRAINER_DEVICES = 1
TRAINER_PRECISION = 16

DATA_NUM_WORKERS = 2
DATA_BATCH_SIZE = 4 if TRAINER_FAST_DEV_RUN == False else 2
DATA_IMAGE_SIZE_DETECTION = 640
DATA_IMAGE_SIZE_SEGMENTATION = (272, 400)  # (544, 800)
DATA_MAX_BOXES_COUNT = 200

MODEL_CLASSIFICATION_THRESHOLD = 0.5
