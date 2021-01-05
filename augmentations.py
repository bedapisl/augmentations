from typing import Dict, List
import time

from augmentations_backend import AugmentationsBackend, Flip

__all__ = [Flip]


class Augmentations:
    def __init__(self, images: List, config: Dict, augmentations_list: List, input_type='image'):
        config['num_threads'] = 2
        config['output_queue_size'] = 5

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
        elif input_type == 'images_points':
            converted_images = images

        self.backend = AugmentationsBackend(converted_images, config, augmentations_list)


    def convert_back(self, example):
        if self.input_type == 'image':
            converted_example = example[0][0]
        elif self.input_type == 'images':
            converted_example = example[0]
        elif self.input_type == 'image_point':
            converted_example = (example[0][0], example[0][1])
        elif self.input_type == 'images_points':
            converted_example = example
       
        return converted_example    


    def epoch(self):

        print('Python: Starting epoch')
        self.backend.start_epoch()
        print('Python: Epoch started')

        example = self.backend.get_example()
        
        while example:
            print('Python: example processed')
            yield self.convert_back(example)

            example = self.backend.get_example()
