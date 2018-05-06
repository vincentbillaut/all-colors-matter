import json

import numpy as np
import tensorflow as tf

from utils.color_utils import variance_UV_channels
from utils.data_utils import dump_YUV_image_to_jpg, load_image_jpg_to_YUV


class Config(object):
    def __init__(self, config_path):
        input_dict = json.load(open(config_path))
        self.__dict__.update(input_dict)
        self.config_name = config_path.split('/')[-1]


class ColoringModelForGAN(object):
    def __init__(self, config):
        self.config = config
        self.params = {}

        self.add_placeholders()
        self.add_model(self.image_Yscale)
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()

    def add_placeholders(self):
        """ Create placeholder variables.
        """
        self.image_Yscale = tf.placeholder(dtype=tf.float32, shape=[None, 320, 320, 1])
        self.image_UVscale = tf.placeholder(dtype=tf.float32, shape=[None, 320, 320, 2])

    def add_model(self, images):
        """ Add Tensorflow ops to get scores from inputs.
        """
        pass

    def add_loss_op(self):
        """ Add Tensorflow op to compute loss.
        """
        self.loss = tf.Variable(1.)
        raise NotImplementedError

    def add_train_op(self):
        """ Add Tensorflow op to run one iteration of SGD.
        """
        self.train_op = None
        raise NotImplementedError

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions.
        """
        self.pred_color_image = None
        pass

    def pred_color_one_image(self, sess, image_path, out_jpg_path):
        image_Yscale, image_UVscale = load_image_jpg_to_YUV(image_path, is_test=False, config=self.config)
        image_Yscale = image_Yscale.reshape([1] + self.config.image_shape[:2] + [1])
        image_UVscale = image_UVscale.reshape([1] + self.config.image_shape[:2] + [2])

        feed = {self.image_Yscale: image_Yscale,
                self.image_UVscale: image_UVscale,
                }

        loss, pred_color_image = sess.run([self.loss, self.pred_color_image],
                                          feed_dict=feed)

        print('\nprediction loss = {}'.format(loss))
        predicted_YUV_image = np.concatenate([image_Yscale, pred_color_image], axis=3)
        print("variance in UV channels, predicted image: {}".format(variance_UV_channels(predicted_YUV_image[0, ...])))
        dump_YUV_image_to_jpg(predicted_YUV_image[0, :, :, :], out_jpg_path + "_pred.png")

        true_YUV_image = np.concatenate([image_Yscale, image_UVscale], axis=3)
        print("variance in UV channels, true image: {}".format(variance_UV_channels(true_YUV_image[0, ...])))
        dump_YUV_image_to_jpg(true_YUV_image[0, :, :, :], out_jpg_path + "_true.png")
