from typing import Dict, List, Tuple
import time

from augmentations_backend import AugmentationsBackend, Flip, Crop, Rotate, BrightnessContrast, HueSaturation, Resize, BGR2RGB

__all__ = [Flip, Crop, Rotate, BrightnessContrast, HueSaturation, Resize, BGR2RGB]


def boxes_to_points(boxes: List[Tuple[Tuple[float, float]]]):
    points = [point for box in boxes for point in box]
    points = [(float(x[0]), float(x[1])) for x in points]
    return points


def points_to_boxes(points: List[Tuple[float, float]]):
    return zip(points[0::2], points[1::2])


class Augmentations:
    def __init__(self, images: List, augmentations_list: List, num_threads: int=2, output_queue_size: int=5, batch_size: int=64, input_type='image'):
        config = {}
        config['num_threads'] = num_threads
        config['output_queue_size'] = output_queue_size
        self.batch_size = batch_size

        assert batch_size > 0

        if config['output_queue_size'] % config['num_threads'] > 0:
            config['output_queue_size'] += config['num_threads'] - (config['output_queue_size'] % config['num_threads'])

        self.input_type = input_type

        # convert data to format acceptable by backend
        if input_type == 'image':
            converted_images = [([image_path], []) for image_path in images]
        elif input_type == 'images':
            converted_images = [(image_paths, []) for image_path in images]
        elif input_type == 'image_point':
            converted_images = [([image_path], [point]) for image_path, point in images]
        elif input_type == 'image_points':
            converted_images = [([image_path], points) for image_path, points in images]
        elif input_type == 'images_points':
            converted_images = images
        elif input_type == 'image_boxes':
            converted_images = [([image_path], boxes_to_points(boxes)) for image_path, boxes in images]

        import pdb
        pdb.set_trace()
        self.backend = AugmentationsBackend(converted_images, config, augmentations_list)


    def convert_back(self, example):
        if self.input_type == 'image':
            converted_example = example[0][0]
        elif self.input_type == 'images':
            converted_example = example[0]
        elif self.input_type == 'image_point':
            converted_example = (example[0][0], example[1][0])
        elif self.input_type == 'image_points':
            converted_example = (example[0][0], example[1])
        elif self.input_type == 'images_points':
            converted_example = (example[0][0], points_to_boxes(example[1]))
       
        return converted_example    


    def epoch(self):

        print('Python: Starting epoch')
        self.backend.start_epoch()
        print('Python: Epoch started')

        example = self.backend.get_example()

        batch = []
        
        while example:
            print('Python: example processed')
            batch.append(self.convert_back(example))

            if len(batch) == self.batch_size:
                yield batch
                batch = []

            example = self.backend.get_example()

        if batch:
            yield batch
