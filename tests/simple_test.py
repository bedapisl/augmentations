import os
import cv2
import numpy as np
import time

import augmentations


def get_image_files(directory='tests/data'):
    return [os.path.join(directory, filename) for filename in os.listdir(directory)]


def get_augmentations_list():
    return [augmentations.Rotate(1.0, -30, 30),
            augmentations.Crop(1.0, 0.8),
            augmentations.Resize(512, 512),
            augmentations.Flip(0.5, True),
            augmentations.BrightnessContrast(0.5, -16, 16, 0.5, 1.5),
            augmentations.HueSaturation(0.5, -18, 18, -10, 10),
            augmentations.BGR2RGB()]


def memory_leak_evaluation():

    augmentations_list = get_augmentations_list()
    augmentor = augmentations.Augmentations(get_image_files(), augmentations_list, batch_size=8, num_threads=8)

    while True:
        augmentor.backend.simple_get_example()


def stress_evaluation():
    augmentations_list = get_augmentations_list()
    image_files = get_image_files('/data/logo_detection/dataset_version_10/train/images')
    augmentor = augmentations.Augmentations(image_files, augmentations_list, batch_size=8, num_threads=2)

    while True:
        for batch in augmentor.epoch():
            time.sleep(0.5)


if __name__ == "__main__":
    stress_evaluation()

