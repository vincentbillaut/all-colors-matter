import tensorflow as tf

from models.coloringmodel import ColoringModel


class NextConvColoringModel(ColoringModel):
    def __init__(self, config, dataset, name="NaiveConvNet", seed=42):
        super().__init__(config, dataset, name, seed)

    def add_model(self):
        """ Add Tensorflow ops to get scores from inputs.
        """
        init = tf.contrib.layers.xavier_initializer(seed=4)

        conv11 = tf.layers.conv2d(self.image_Yscale, filters=32, kernel_size=3, strides=2, padding="SAME",
                                  kernel_initializer=init)
        a11 = tf.nn.relu(conv11)
        conv12 = tf.layers.conv2d(a11, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)
        a12 = tf.nn.relu(conv12)
        maxpool1 = tf.layers.max_pooling2d(a12, 2, 2)
        bn1 = tf.layers.batch_normalization(maxpool1)


        conv21 = tf.layers.conv2d(bn1, filters=64, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)
        a21 = tf.nn.relu(conv21)
        conv22 = tf.layers.conv2d(a21, filters=64, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)
        a22 = tf.nn.relu(conv22)
        maxpool2 = tf.layers.max_pooling2d(a22, 2, 2)
        bn2 = tf.layers.batch_normalization(maxpool2)


        conv31 = tf.layers.conv2d(bn2, filters=64, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)
        a31 = tf.nn.relu(conv31)
        conv32 = tf.layers.conv2d(a31, filters=64, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)
        a32 = tf.nn.relu(conv32)
        bn3 = tf.layers.batch_normalization(a32)


        upconv41 = tf.layers.conv2d_transpose(bn3, filters=32, kernel_size=3, strides=2, padding="SAME",
                                              kernel_initializer=init)
        a41 = tf.nn.relu(upconv41)
        conv42 = tf.layers.conv2d(a41, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)
        a42 = tf.nn.relu(conv42)
        conv43 = tf.layers.conv2d(a42, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)
        a43 = tf.nn.relu(conv43)


        upconv51 = tf.layers.conv2d_transpose(a43, filters=16, kernel_size=3, strides=2, padding="SAME",
                                              kernel_initializer=init)
        a51 = tf.nn.relu(upconv51)
        conv52 = tf.layers.conv2d(a51, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)
        a52 = tf.nn.relu(conv52)
        conv53 = tf.layers.conv2d(a52, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)
        a53 = tf.nn.relu(conv53)

        upconv61 = tf.layers.conv2d_transpose(a53, filters=16, kernel_size=3, strides=2, padding="SAME",
                                              kernel_initializer=init)

        self.scores = tf.layers.conv2d(upconv61, filters=self.n_categories, kernel_size=1, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init)


    def add_train_op(self):
        """ Add Tensorflow op to run one iteration of SGD.
        """
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        self.train_op = opt.minimize(self.loss)

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions.
        """
        self.pred_image_categories = tf.nn.softmax(self.scores)
