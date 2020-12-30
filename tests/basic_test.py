import os
import cv2

from augmentations import Augmentations


def get_image_files(directory='tests/data'):
    return [os.path.join('tests/data', filename) for filename in os.listdir('tests/data')]


def test_image_loading():
    augmentations = Augmentations(get_image_files(), {})

    image_count = 0
    for image, image_file in zip(augmentations.epoch(), get_image_files()):
        cv2_image = cv2.imread(image_file)

        assert cv2_image.shape == image.shape
        assert (cv2_image == image).all()

        image_count += 1

    assert len(get_image_files()) == image_count


if __name__ == "__main__":
    test_image_loading()


