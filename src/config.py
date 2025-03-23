import pathlib
import numpy as np

IMAGE_SHAPE = (224, 224)
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 1000
LEARNING_RATE = 0.0002
EPOCHS = 300
PATIENCE = 15
CLASSIFIER_DROPOUT_RATE = 0.1

TRAIN_DIR = r"E:\ImageNet1K\Train"
VAL_DIR = r"E:\ImageNet1K\Val"

ENCODER_WEIGHTS_PATH = r"E:\ImageNet1K\encoder_Weights_66_ECALayer_combined_loss.h5"

hidden_units = []

target_height = 224
target_width = 224
CROP_HEIGHT = 160
CROP_WIDTH = 160

# ---------------------------------------
# Hyperparameters & Configuration of LiteAE
# ---------------------------------------
INPUT_SHAPE = (224, 224, 3)
ENCODER_FILTERS = [34, 52, 66]
DROPOUT_RATE = 0.15
LEAKY_RELU_ALPHA = 0.3
KERNEL_SIZE = (3, 3)
STRIDES = (2, 2)
OUTPUT_CHANNELS = 3  # RGB

CLASS_NAMES = np.array([
    item.name for item in pathlib.Path(TRAIN_DIR).glob("*")
    if item.name != "LICENSE.txt"
])

#DeepResNet Block Parameters
DEEPRESNET_CONFIG = {
    "stage1": {
        "num_filters": 110,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "additional_layers": 2,
        "dropout_rate": 0.03,
        "residual_dropout_rate_1": 0.0,
        "residual_dropout_rate_2": 0.025,
    },
    "stage2": {
        "num_filters": 180,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "additional_layers": 2,
        "dropout_rate": 0.03,
        "residual_dropout_rate_1": 0.0,
        "residual_dropout_rate_2": 0.025,
    },
    "stage3": {
        "num_filters": 340,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "additional_layers": 2,
        "dropout_rate": 0.03,
        "residual_dropout_rate_1": 0.0,
        "residual_dropout_rate_2": 0.025,
    }
}
