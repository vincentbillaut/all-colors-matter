import os

import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave
from utils.color_utils import RGB_to_YUV, YUV_to_RGB


def load_image_jpg_to_YUV(impath, is_test, config):
    """
    Loads a jpg into a np array of the image.
    :param impath: path to the image's jpg file.
    :param is_test: switch; if true, randomly flips the image horizontally.
    :param config: contains the target image shape.
    :return: A tuple of the image in YUV format and the Y channel of the image.
    """
    if impath is bytes:
        impath = impath.decode('utf-8')
    image = RGB_to_YUV(imread(impath)).astype(np.dtype("float32"))

    if not is_test:
        flip = np.random.random()
        if flip < 0.5:
            image = image[:, ::-1, :]
            # pass
    padded_image = pad_image_to_size(image, config.image_shape)
    return padded_image[:, :, :1], padded_image[:, :, 1:]


def dump_YUV_image_to_jpg(YUV_image, path):
    RGB_image = YUV_to_RGB(YUV_image)
    imsave(path, RGB_image, 'png')


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
