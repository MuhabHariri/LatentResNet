import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
from src.config import *
from src.config import CROP_HEIGHT, CROP_WIDTH, target_height, target_width
from src.layers import ECALayer
import matplotlib.pyplot as plt

# ------------------------------
##Basic Augmentation Functions
# ------------------------------

def random_flip(image):
    """Randomly flip the image horizontally."""
    return tf.image.random_flip_left_right(image)


def random_rotate(image):
    """Randomly rotate the image within a range of -20 to 20 degrees."""
    angle = tf.random.uniform([], minval=-20*np.pi/180, maxval=20*np.pi/180)
    return tfa.image.rotate(image, angle)


def random_shear(image):
    """Apply random shear transformation along both X and Y axes."""
    with tf.device('/cpu:0'):
        shear_angle = tf.random.uniform([], minval=-0.25, maxval=0.25)
        replace = tf.constant([0, 0, 0], dtype=tf.float32)
        image = tfa.image.shear_x(image, shear_angle, replace)
        image = tfa.image.shear_y(image, shear_angle, replace)
    return image


def random_translate(image):
    """Translate image randomly within ±25% of its dimensions."""
    height, width, _ = image.shape
    width_shift = tf.random.uniform([], -width * 0.25, width * 0.25)
    height_shift = tf.random.uniform([], -height * 0.25, height * 0.25)
    translations = [width_shift, height_shift]
    return tfa.image.translate(image, translations)


def random_zoom(image):
    """Apply random zoom with padding/cropping to return to original size."""
    zoom_factor = tf.random.uniform([], 0.75, 1.25)
    new_height = tf.cast(target_height * zoom_factor, tf.int32)
    new_width = tf.cast(target_width * zoom_factor, tf.int32)
    image = tf.image.resize(image, [new_height, new_width])
    return tf.image.resize_with_crop_or_pad(image, target_height, target_width)


def random_crop_and_pad(image):
    """Randomly crop a portion of the image, then pad to original size."""
    crop_size = [CROP_HEIGHT, CROP_WIDTH, 3]
    image = tf.image.random_crop(image, size=crop_size)
    return tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

# ------------------------------
##Color Augmentation Functions
# ------------------------------

def random_adjust_brightness(image):
    """Randomly adjust image brightness."""
    delta = tf.random.uniform([], 0.06, 0.14)
    return tf.image.adjust_brightness(image, delta)


def random_channel_shift(image):
    """Apply random shift to all RGB channels."""
    delta = tf.random.uniform([], -50, 50) / 255.0
    return tf.clip_by_value(image + delta, 0.0, 1.0)


def random_saturation(image):
    """Randomly adjust saturation."""
    saturation_factor = tf.random.uniform([], 1.0, 1.2)
    return tf.image.adjust_saturation(image, saturation_factor)


def random_contrast(image):
    """Randomly adjust contrast."""
    contrast_factor = tf.random.uniform([], 1.2, 1.6)
    return tf.image.adjust_contrast(image, contrast_factor)

# ------------------------------
##Composite / Complex Augmentations
# ------------------------------

def custom_random_translate(image, width_shift_range, height_shift_range):
    """Translate image with custom shift ranges."""
    width_shift = tf.random.uniform([], *width_shift_range)
    height_shift = tf.random.uniform([], *height_shift_range)
    return tfa.image.translate(image, [width_shift, height_shift])


def combined_random_augmentations_with_order(image):
    """Apply a random set of brightness, contrast, saturation, channel shift, and translation in random order."""
    height, width, _ = image.shape
    width_shift_range = [width * 0.1, width * 0.25]
    height_shift_range = [height * 0.1, height * 0.25]

    augmentations = [
        lambda img: tf.image.adjust_brightness(img, tf.random.uniform([], 0.06, 0.2)),
        lambda img: tf.clip_by_value(img + tf.random.uniform([], -80, 80) / 255.0, 0.0, 1.0),
        lambda img: tf.image.adjust_saturation(img, tf.random.uniform([], 1, 1.6)),
        lambda img: tf.image.adjust_contrast(img, tf.random.uniform([], 1.2, 1.9)),
        lambda img: custom_random_translate(img, width_shift_range, height_shift_range),
    ]

    np.random.shuffle(augmentations)

    for aug in augmentations:
        image = aug(image)

    return image

# ------------------------------
##Custom Mask-Based Augmentation
# ------------------------------

def random_black_rectangle(image):
    """Apply a random black rectangle mask to a portion of the image."""
    height, width, _ = tf.unstack(tf.shape(image))
    dtype = image.dtype

    rect_height = tf.random.uniform([], 45, 60, dtype=tf.int32)
    rect_width = tf.random.uniform([], 80, 100, dtype=tf.int32)
    start_y = tf.random.uniform([], 0, height - rect_height, dtype=tf.int32)
    start_x = tf.random.uniform([], 0, width - rect_width, dtype=tf.int32)

    mask = tf.ones((height, width, 1), dtype=dtype)
    y_range = tf.range(start_y, start_y + rect_height)
    x_range = tf.range(start_x, start_x + rect_width)
    yx_indices = tf.stack(tf.meshgrid(y_range, x_range, indexing='ij'), axis=-1)
    flat_indices = tf.reshape(yx_indices, [-1, 2])
    updates = tf.zeros((rect_height * rect_width, 1), dtype=dtype)
    mask = tf.tensor_scatter_nd_update(mask, flat_indices, updates)
    mask = tf.tile(mask, [1, 1, 3])

    return image * mask



def augment(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    def apply_augmentation(i, image):
        return tf.switch_case(
            branch_index=i,
            branch_fns={
                0: lambda: random_flip(image),
                1: lambda: combined_random_augmentations_with_order(image),
                2: lambda: random_black_rectangle(image),
                3: lambda: random_shear(image),
                4: lambda: random_crop_and_pad(image),
                5: lambda: random_translate(image),
                6: lambda: random_rotate(image),
                7: lambda: random_adjust_brightness(image),
                8: lambda: random_zoom(image),
                9: lambda: random_channel_shift(image)
            }
        )

    aug_idx = tf.random.uniform([], minval=0, maxval=10, dtype=tf.int32)
    image = apply_augmentation(aug_idx, image)

    return image, label


def plot_and_save_images(images, labels, epoch, save_dir="plots"):
    plt.figure(figsize=(12, 8))
    for i, (image, label) in enumerate(zip(images, labels)):
        if i == 8:  
            break
        plt.subplot(2, 4, i + 1)
        plt.imshow(image)
        plt.title(CLASS_NAMES[label])
        plt.axis('off')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"epoch_{epoch}_augmented_images.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()
