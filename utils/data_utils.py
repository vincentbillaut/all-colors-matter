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


################################################################################
# util functions for translating RGB to YUV and vice versa inspired from:
# https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
# and a little bit modified

# input is a RGB numpy array with shape (height,width,3), can be uint,int,float
# or double, values expected in the range 0..255
# output is a double YUV numpy array with shape (height,width,3), values in the
# range 0..255
def RGB_to_YUV(rgb):

    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    assert(rgb.shape==yuv.shape)
    return yuv

#input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
#output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
def YUV_to_RGB(yuv):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304
    assert(rgb.shape==yuv.shape)
    return rgb
################################################################################
