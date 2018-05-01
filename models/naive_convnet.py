import tensorflow as tf
from models.model import Config, Model
import numpy as np


class NaiveConvConfig(Config):
    def __init__(self):
        super().__init__()


class NaiveConvModel(Model):
    def __init__(self, config):
        super().__init__(config)

    def add_placeholders(self):
        """ Create placeholder variables.
        """
        pass

    def add_model(self):
        """ Add Tensorflow ops to get scores from inputs.
        """
        init = tf.contrib.layers.xavier_initializer()

        W1 = tf.Variable(init(shape=(3, 3, 1, 10)), dtype=tf.float32)
        conv1 = tf.nn.conv2d(input=self.image_greyscale, filter=W1, strides=(1, 1, 1, 1), padding="SAME")
        a1 = tf.nn.relu(conv1)

        W2 = tf.Variable(init(shape=(3, 3, 10, 10)), dtype=tf.float32)
        conv2 = tf.nn.conv2d(input=a1, filter=W2, strides=(1, 1, 1, 1), padding="SAME")
        a2 = tf.nn.relu(conv2)

        W3 = tf.Variable(init(shape=(3, 3, 10, 3)), dtype=tf.float32)
        self.conv3 = tf.nn.conv2d(input=a2, filter=W3, strides=(1, 1, 1, 1), padding="SAME")

    def add_loss_op(self):
        """ Add Tensorflow op to compute loss.
        """
        self.loss = tf.nn.l2_loss(self.pred_color_image - self.image)

    def add_train_op(self):
        """ Add Tensorflow op to run one iteration of SGD.
        """
        opt = tf.train.AdamOptimizer(learning_rate=.01)
        self.train_op = opt.minimize(self.loss)

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions.
        """
        self.pred_color_image = tf.nn.sigmoid(self.conv3) * 255.
