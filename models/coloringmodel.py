import json

import numpy as np
import tensorflow as tf

from utils.color_utils import variance_UV_channels
from utils.data_utils import dump_YUV_image_to_jpg, load_image_jpg_to_YUV
from utils.progbar import Progbar


class Config(object):
    def __init__(self, config_path):
        input_dict = json.load(open(config_path))
        self.__dict__.update(input_dict)
        self.config_name = config_path.split('/')[-1]


class ColoringModel(object):
    def __init__(self, config, dataset):
        self.config = config
        self.params = {}
        self.dataset = dataset

        self.add_dataset()
        self.add_placeholders()
        self.add_model()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()

    def add_dataset(self):
        train_dataset = self.dataset.get_dataset_batched(False, self.config)
        val_dataset = self.dataset.get_dataset_batched(True, self.config)
        # iterator just needs to know the output types and shapes of the datasets
        self.iterator = tf.data.Iterator.from_structure(output_types=(tf.float32, tf.float32),
                                                        output_shapes=([None, 320, 320, 1],
                                                                       [None, 320, 320, 2]))
        self.image_Yscale, self.image_UVscale = self.iterator.get_next()
        self.train_init_op = self.iterator.make_initializer(train_dataset)
        self.test_init_op = self.iterator.make_initializer(val_dataset)

    def add_placeholders(self):
        """ Create placeholder variables.
        """
        pass

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

    def run_epoch(self, sess, epoch_number=0):
        nbatches = len(self.dataset.train_ex_paths)
        prog = Progbar(target=min(nbatches, self.config.max_batch))
        batch = 0

        sess.run(self.train_init_op)
        while True:
            try:
                # feed = {self.dropout_placeholder: self.config.dropout,
                #         self.lr_placeholder: lr_schedule.lr,
                #         self.is_training: self.config.use_batch_norm}

                loss, _ = sess.run([self.loss, self.train_op],
                                   )  # feed_dict=feed)
                batch += self.config.batch_size

            except tf.errors.OutOfRangeError:
                break

            if batch > self.config.max_batch:
                break

            prog.update(batch, values=[("loss", loss)])

        self.pred_color_one_image(sess, "data/iccv09Data/images/0000382.jpg",
                                  "outputs/0000382_epoch{}".format(epoch_number))

    def train_model(self, sess):
        for ii in range(self.config.n_epochs):
            i = ii + 1
            print("\nRunning epoch {}:".format(i))
            self.run_epoch(sess, i)

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
