import tensorflow as tf

from models.coloringmodel import ColoringModel


class NaiveConvColoringModel(ColoringModel):
    def __init__(self, config, dataset,name="NaiveConvNet",seed = 42):
        super().__init__(config, dataset,name,seed)

    def add_model(self):
        """ Add Tensorflow ops to get scores from inputs.
        """
        init = tf.contrib.layers.xavier_initializer(seed=4)
        conv1 = tf.layers.conv2d(self.image_Yscale, filters=50, kernel_size=3, strides=(1, 1), padding="SAME",
                                 kernel_initializer=init)
        a1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(a1, filters=50, kernel_size=3, strides=(1, 1), padding="SAME", kernel_initializer=init)
        a2 = tf.nn.relu(conv2)

        conv3 = tf.layers.conv2d(a2, filters=self.n_categories, kernel_size=3, strides=(1, 1), padding="SAME",
                                 kernel_initializer=init)

        self.scores = conv3

    def add_train_op(self):
        """ Add Tensorflow op to run one iteration of SGD.
        """
        opt = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.train_op = opt.minimize(self.loss)

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions.
        """
        self.pred_image_categories = tf.nn.softmax(self.scores)
