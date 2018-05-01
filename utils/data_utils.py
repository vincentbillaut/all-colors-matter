import tensorflow as tf
import os
import numpy as np
from scipy.ndimage import imread


def get_dataset_batched(directory, is_test, config):
    def gen():
        return gen_images(directory, is_test, config)

    dataset = tf.data.Dataset.from_generator(generator=gen,
                                             output_types=(tf.float32, tf.float32))
    batch_size = config.batch_size
    batched_dataset = dataset.batch(batch_size)
    batched_dataset = batched_dataset.prefetch(1)

    return batched_dataset


def gen_images(directory, is_test, config):
    images_paths = os.listdir(directory)
    global_images_paths = [os.path.join(directory, impath) for impath in images_paths]
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


def load_image_jpg(impath, is_test, config):
    impath = impath.decode('utf-8')
    image = imread(impath)

    if not is_test:
        flip = np.random.random()
        if flip < 0.5:
            image = image[:, ::-1, :]
    padded_image = pad_image_to_size(image, config.image_shape)
    return padded_image


def get_im_paths(path):
    ex_paths = []
    for ex_name in os.listdir(path):
        ex_path = os.path.join(path, ex_name)
        ex_paths.append(ex_path)
    return ex_paths


def pad_image_to_size(image, shape):
    new_image = np.zeros(shape=shape)
    new_image[:image.shape[0], :image.shape[1], :] = image
    return new_image
