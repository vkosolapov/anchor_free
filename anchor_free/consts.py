TRAINER_EXPERIMENT_NAME = "001_classification"
TRAINER_EXPERIMENT_VERSION = "004_res_mlp"
TRAINER_FAST_DEV_RUN = False

TRAINER_MAX_EPOCHS = 1000
TRAINER_MONITOR = "accuracy/val"
TRAINER_MONITOR_MODE = "max"
TRAINER_EARLY_STOPPING_PATIENCE = 30

TRAINER_ACCELERATOR = "gpu"
TRAINER_DEVICES = 1
TRAINER_PRECISION = 16

DATA_NUM_WORKERS = 2
DATA_BATCH_SIZE = 8 if TRAINER_FAST_DEV_RUN == False else 2
DATA_MAX_BOXES_COUNT = 200
DATA_IMAGE_SIZE_DETECTION = 640
DATA_IMAGE_SIZE_SEGMENTATION = (288, 416)  # (544, 800)
DATA_IMAGE_SIZE_INSTANCE = (512, 256)  # (2048, 1024)

MODEL_CLASSIFICATION_THRESHOLD = 0.5
MODEL_NMS = True
MODEL_NMS_THRESHOLD = 0.7
