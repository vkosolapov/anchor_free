TRAINER_EXPERIMENT_NAME = "003_detection"
TRAINER_EXPERIMENT_VERSION = "001_test"
TRAINER_FAST_DEV_RUN = False

TRAINER_MAX_EPOCHS = 1000
TRAINER_MONITOR = "map_05/val"
TRAINER_MONITOR_MODE = "max"
TRAINER_EARLY_STOPPING_PATIENCE = 100

TRAINER_ACCELERATOR = "tpu"
TRAINER_DEVICES = 1
TRAINER_PRECISION = 16

DATA_NUM_WORKERS = 8
DATA_BATCH_SIZE = 4 if TRAINER_FAST_DEV_RUN == False else 2
DATA_IMAGE_SIZE = 640
DATA_MAX_BOXES_COUNT = 200

MODEL_CLASSIFICATION_THRESHOLD = 0.5
