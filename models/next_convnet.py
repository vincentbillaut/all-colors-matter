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
                                  kernel_initializer=init,name="CONV_1_1")
        a11 = tf.nn.relu(conv11,name="RELU_1_1")
        conv12 = tf.layers.conv2d(a11, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name="CONV_1_2")
        a12 = tf.nn.relu(conv12,name="RELU_1_2")
        maxpool1 = tf.layers.max_pooling2d(a12, 2, 2,name="MAXPOOL_1")
        bn1 = tf.layers.batch_normalization(maxpool1,name = "BATCHNORM_1")


        conv21 = tf.layers.conv2d(bn1, filters=64, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name="CONV_2_1")
        a21 = tf.nn.relu(conv21,name="RELU_2_1")
        conv22 = tf.layers.conv2d(a21, filters=64, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name="CONV_2_2")
        a22 = tf.nn.relu(conv22,name="RELU_2_2")
        maxpool2 = tf.layers.max_pooling2d(a22, 2, 2,name = "MAXPOOL_2")
        bn2 = tf.layers.batch_normalization(maxpool2,name = "BATCHNORM_2")


        conv31 = tf.layers.conv2d(bn2, filters=64, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name = "CONV_3_1")
        a31 = tf.nn.relu(conv31,name = "RELU_3_1")
        conv32 = tf.layers.conv2d(a31, filters=64, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name = "CONV_3_2")
        a32 = tf.nn.relu(conv32,name = "RELU_3_2")
        bn3 = tf.layers.batch_normalization(a32,name = "BATCHNORM_3")


        upconv41 = tf.layers.conv2d_transpose(bn3, filters=32, kernel_size=3, strides=2, padding="SAME",
                                              kernel_initializer=init,name = "UPCONV_4_1")
        a41 = tf.nn.relu(upconv41,name = "RELU_4_1")
        conv42 = tf.layers.conv2d(a41, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name = "CONV_4_2")
        a42 = tf.nn.relu(conv42,name="RELU_4_2")
        conv43 = tf.layers.conv2d(a42, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name = "CONV_4_3")
        a43 = tf.nn.relu(conv43,name = "RELU_4_3")


        upconv51 = tf.layers.conv2d_transpose(a43, filters=16, kernel_size=3, strides=2, padding="SAME",
                                              kernel_initializer=init,name = "UPCONV_5_1")
        a51 = tf.nn.relu(upconv51,name = "RELU_5_1")
        conv52 = tf.layers.conv2d(a51, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name = "CONV_5_2")
        a52 = tf.nn.relu(conv52,name = "RELU_5_2")
        conv53 = tf.layers.conv2d(a52, filters=32, kernel_size=3, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name = "CONV_5_3")
        a53 = tf.nn.relu(conv53,name = "RELU_5_3")

        upconv61 = tf.layers.conv2d_transpose(a53, filters=16, kernel_size=3, strides=2, padding="SAME",
                                              kernel_initializer=init,name="UPCONV_6_1")

        self.scores = tf.layers.conv2d(upconv61, filters=self.n_categories, kernel_size=1, strides=(1, 1), padding="SAME",
                                  kernel_initializer=init,name = "CONV_6_2")


    def add_train_op(self):
        """ Add Tensorflow op to run one iteration of SGD.
        """
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr,name="AdamOpt")
        self.train_op = opt.minimize(self.loss)

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions.
        """
        self.pred_image_categories = tf.nn.softmax(self.scores,name="SCORES")
