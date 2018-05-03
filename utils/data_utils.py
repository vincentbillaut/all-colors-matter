import tensorflow as tf
import os
import numpy as np
from scipy.ndimage import imread


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
