import tensorflow as tf

from models.coloringmodel import ColoringModel


class NaiveConvColoringModel(ColoringModel):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def add_placeholders(self):
        """ Create placeholder variables.
        """
        pass

    def add_model(self):
        """ Add Tensorflow ops to get scores from inputs.
        """
        init = tf.contrib.layers.xavier_initializer()
        conv1 = tf.layers.conv2d(self.image_Yscale, filters=50, kernel_size=3, strides=(1, 1), padding="SAME",
                                 kernel_initializer=init)
        a1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(a1, filters=50, kernel_size=3, strides=(1, 1), padding="SAME", kernel_initializer=init)
        a2 = tf.nn.relu(conv2)

        self.conv3 = tf.layers.conv2d(a2, filters=2, kernel_size=3, strides=(1, 1), padding="SAME",
                                      kernel_initializer=init)

    def add_loss_op(self):
        """ Add Tensorflow op to compute loss.
        """
        self.loss = tf.nn.l2_loss(self.pred_color_image - self.image_UVscale)

    def add_train_op(self):
        """ Add Tensorflow op to run one iteration of SGD.
        """
        opt = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.train_op = opt.minimize(self.loss)

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions.
        """
        self.pred_color_image = tf.nn.sigmoid(self.conv3) * 255.
