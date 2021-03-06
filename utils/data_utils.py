import os

import numpy as np
import matplotlib

matplotlib.use("agg")
from matplotlib.pyplot import imread, imsave
from utils.color_utils import RGB_to_YUV, YUV_to_RGB


def load_image_jpg_to_YUV(impath, transf, config):
    """Loads a jpg into a np array of the image.

    Parameters
    ----------
    impath : str
        Path to the image's jpg file.
    transf : fun
        Function that takes an image and outputs a transformed image.
        This function should come from a DataAugmenter object.
    config : Config
        contains the target image shape.

    Returns
    -------
        A tuple of the image in YUV format and the Y channel of the image.

    """
    """
    .
    :param impath: path to the image's jpg file.
    :param config: contains the target image shape.
    :return: A tuple of the image in YUV format and the Y channel of the image.
    """
    if isinstance(impath,bytes):
        impath = impath.decode('utf-8')
    image = imread(impath).astype(np.dtype("float32"))
    image_transf = transf(image)

    padded_image, mask = pad_image_to_size(image_transf, config.image_shape)
    YUV_padded_image = RGB_to_YUV(padded_image)
    return YUV_padded_image[:, :, :1], YUV_padded_image[:, :, 1:], mask


def dump_YUV_image_to_jpg(YUV_image, path):
    RGB_image = YUV_to_RGB(YUV_image).astype("uint8")
    imsave(fname=path, arr=RGB_image,format = 'png')


def get_im_paths(path):
    ex_paths = []
    for ex_name in os.listdir(path):
        ex_path = os.path.join(path, ex_name)
        ex_paths.append(ex_path)
    return ex_paths


def pad_image_to_size(image, shape):
    new_image = np.zeros(shape=shape)
    mask = np.zeros(shape=[shape[0], shape[1]])
    new_image[:image.shape[0], :image.shape[1], :] = image
    mask[:image.shape[0], :image.shape[1]] = 1
    return new_image, mask
