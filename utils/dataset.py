import numpy as np
import tensorflow as tf
import os

from utils.data_utils import load_image_jpg_to_YUV, get_im_paths
from utils.color_discretizer import ColorDiscretizer


class Dataset(object):
    def __init__(self, train_path, val_path, color_discretizer, name="", filter=None):
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.color_discretizer = color_discretizer

        self.filter = filter
        self.train_ex_paths = get_im_paths(self.train_path)
        self.val_ex_paths = get_im_paths(self.val_path)

    def get_dataset_batched(self, is_test, config,seed = 0):
        def gen():
            return self.gen_images(self.val_path if is_test else self.train_path, is_test, config,seed)

        dataset = tf.data.Dataset.from_generator(generator=gen,
                                                 output_types=(tf.float32,
                                                               tf.int32,
                                                               tf.float32,
                                                               tf.bool))
        batch_size = config.batch_size
        batched_dataset = dataset.batch(batch_size)
        batched_dataset = batched_dataset.prefetch(1)
        return batched_dataset

    def gen_images(self, directory, is_test, config,seed = 0):
        images_paths = os.listdir(directory)
        if self.filter is None:
            global_images_paths = [os.path.join(directory, impath) for impath in images_paths]
        else:
            global_images_paths = [os.path.join(directory, impath) for impath in images_paths if impath in self.filter]
        images_paths_utf = [impath.encode('utf-8') for impath in
                            global_images_paths]  # need to encode in bytes to pass it to tf.py_func

        images_paths_utf.sort()
        if not is_test:
            np.random.seed(seed)
            np.random.shuffle(images_paths_utf)

        for impath in images_paths_utf:
            image_Yscale, image_UVscale, mask = load_image_jpg_to_YUV(impath, is_test, config)
            categorized_image, weights = self.color_discretizer.categorize(image_UVscale, return_weights=True)
            yield image_Yscale, categorized_image, weights, mask
