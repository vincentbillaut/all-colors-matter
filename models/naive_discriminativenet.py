import tensorflow as tf

from models.discriminative_net import DiscriminativeModel
from models.coloringmodel import ColoringModel


class NaiveDiscriminativeNet(DiscriminativeModel):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def add_placeholders(self):
        """ Create placeholder variables.
        """
        pass

    def add_model(self, images):
        """ Add Tensorflow ops to get scores from inputs.
        """
        init = tf.contrib.layers.xavier_initializer()

        ## First layer
        conv11 = tf.layers.conv2d(images,
                                  filters=16, kernel_size=3, strides=(1, 1),
                                  padding="SAME", kernel_initializer=init)
        a11 = tf.nn.relu(conv11)

        conv12 = tf.layers.conv2d(a11,
                                  filters=16, kernel_size=3, strides=(1, 1),
                                  padding="SAME", kernel_initializer=init)
        a12 = tf.nn.relu(conv12)

        batchnorm1 = tf.layers.batch_normalization(a12)
        maxpool1 = tf.layers.max_pooling2d(inputs=batchnorm1,
                                           pool_size=2,
                                           strides=2)

        ## Second layer
        conv21 = tf.layers.conv2d(maxpool1,
                                  filters=32, kernel_size=3, strides=(1, 1),
                                  padding="SAME", kernel_initializer=init)
        a21 = tf.nn.relu(conv21)

        conv22 = tf.layers.conv2d(a21,
                                  filters=32, kernel_size=3, strides=(1, 1),
                                  padding="SAME", kernel_initializer=init)
        a22 = tf.nn.relu(conv22)

        batchnorm2 = tf.layers.batch_normalization(a22)
        maxpool2 = tf.layers.max_pooling2d(inputs=batchnorm2,
                                           pool_size=2,
                                           strides=2)

        ## Third layer
        conv31 = tf.layers.conv2d(maxpool2,
                                  filters=64, kernel_size=3, strides=(1, 1),
                                  padding="SAME", kernel_initializer=init)
        a31 = tf.nn.relu(conv31)

        conv32 = tf.layers.conv2d(a31,
                                  filters=64, kernel_size=3, strides=(1, 1),
                                  padding="SAME", kernel_initializer=init)
        a32 = tf.nn.relu(conv32)

        batchnorm3 = tf.layers.batch_normalization(a32)

        features = tf.reduce_mean(batchnorm3, axis=(1, 2))
        return tf.layers.dense(features, 2)

    def add_loss_op(self):
        """ Add Tensorflow op to compute loss.
        """
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.scores)

    def add_train_op(self):
        """ Add Tensorflow op to run one iteration of SGD.
        """
        opt = tf.train.AdamOptimizer(learning_rate=.01)
        self.train_op = opt.minimize(self.loss)

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions.
        """
        self.pred = tf.argmax(self.scores, axis=1)
