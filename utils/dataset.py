import numpy as np
import tensorflow as tf

from data_utils import load_image_jpg

class Dataset:
    def __init__(self):
        raise NotImplementedError
    def get_dataset_batched(self,is_test,config):
        def gen():
            return self.gen_images(is_test, config)

        dataset = tf.data.Dataset.from_generator(generator=gen,
                                                 output_types=(tf.float32, tf.float32))
        batch_size = config.batch_size
        batched_dataset = dataset.batch(batch_size)
        batched_dataset = batched_dataset.prefetch(1)
        return batched_dataset
    def gen_images(self):
        raise NotImplementedError

class FiteredSubDirDataset:
    def __init__(self,directory,name="",filt=None):
        self.name = name
        self.dir = directory
        self.filter = filt

    def gen_images(directory, is_test, config):
        images_paths = os.listdir(directory)
        if self.filter is None:
            global_images_paths = [os.path.join(directory, impath) for impath in images_paths]
        else:
            global_images_paths = [os.path.join(directory,impath) for impath in images_paths if impath in self.filter]
        images_paths_utf = [impath.encode('utf-8') for impath in
                            global_images_paths]  # need to encode in bytes to pass it to tf.py_func

        images_paths_utf.sort()
        if not is_test:
            np.random.seed(0)
            np.random.shuffle(images_paths_utf)

        for impath in images_paths_utf:
            image = load_image_jpg(impath, is_test, config).astype(np.dtype("float32"))
            image_greyscale = np.mean(image, axis=2).reshape([image.shape[0], image.shape[1], 1])
            yield image_greyscale, image
