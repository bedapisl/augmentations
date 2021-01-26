import os
import cv2
import numpy as np

from augmentations import Augmentations, Flip


def get_image_files(directory='tests/data'):
    return [os.path.join('tests/data', filename) for filename in os.listdir('tests/data')]


def test_image_loading(epochs=3):
    transformations_list = []
    augmentations = Augmentations(get_image_files(), transformations_list, batch_size=1)

    for epoch in range(epochs):
        image_count = 0
        for i, (batch, image_file) in enumerate(zip(augmentations.epoch(), get_image_files())):
            image = batch[0]
            cv2_image = cv2.imread(image_file)
            print(f'Image number {i}, shape: {cv2_image.shape}')

            if cv2_image.shape != image.shape:
                print(f'Expected shape: {cv2_image.shape}, actual shape: {image.shape}')

            assert cv2_image.shape == image.shape
            assert (cv2_image == image).all()

            image_count += 1

        assert len(get_image_files()) == image_count


def test_flip():
    transformations_list = [Flip(1.0, True)]

    augmentations = Augmentations(get_image_files(), transformations_list, batch_size=1)

    image_count = 0
    for i, (batch, image_file) in enumerate(zip(augmentations.epoch(), get_image_files())):
        image = batch[0]
        cv2_image = cv2.imread(image_file)
        flipped_image = np.fliplr(cv2_image)

        if flipped_image.shape != image.shape:
            print(f'Expected shape: {flipped_image.shape}, actual shape: {image.shape}')

        assert flipped_image.shape == image.shape
        assert (flipped_image == image).all()

        image_count += 1

    assert len(get_image_files()) == image_count


if __name__ == "__main__":
    test_image_loading(10000000)
    print('----------------------------------------------------------')
    print('\n\n\n')
    test_flip()

