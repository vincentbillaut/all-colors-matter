import json
import os
import pickle

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

        if self.max_batch == -1:
            self.max_batch = np.inf


class ColoringModel(object):
    def __init__(self, config, dataset, name=None, seed=42):
        self.seed = seed
        self.config = config
        self.params = {}
        self.dataset = dataset
        self.name = name

    def _build_model(self):
        self.add_dataset()
        self.add_placeholders()
        self.add_model()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()

    def _build_new_graph_session(self):

        # Create new computation graph
        self.graph = tf.Graph()

        with self.graph.as_default():
            # Set graph-level random seed
            tf.set_random_seed(self.seed)

            # Build the model
            self._build_model()

        # Create new session
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=conf)

    def add_dataset(self):
        train_dataset = self.dataset.get_dataset_batched(False, self.config)
        val_dataset = self.dataset.get_dataset_batched(True, self.config)

        # iterator just needs to know the output types and shapes of the datasets
        self.iterator = tf.data.Iterator.from_structure(output_types=(tf.float32, tf.float32),
                                                        output_shapes=(
                                                        [None, self.config.image_shape[0], self.config.image_shape[1],
                                                         1],
                                                        [None, self.config.image_shape[0], self.config.image_shape[1],
                                                         2]))
        self.image_Yscale, self.image_UVscale = self.iterator.get_next()
        self.train_init_op = self.iterator.make_initializer(train_dataset)
        self.test_init_op = self.iterator.make_initializer(val_dataset)

    def add_placeholders(self):
        """ Create placeholder variables.
        """
        pass

    def add_model(self):
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

    def run_epoch(self, epoch_number=0):
        nbatches = len(self.dataset.train_ex_paths)
        prog = Progbar(target=min(nbatches, self.config.max_batch))
        batch = 0

        with self.graph.as_default():
            self.session.run(self.train_init_op)
            while True:
                try:
                    # feed = {self.dropout_placeholder: self.config.dropout,
                    #         self.lr_placeholder: lr_schedule.lr,
                    #         self.is_training: self.config.use_batch_norm}

                    loss, _ = self.session.run([self.loss, self.train_op],
                                               )  # feed_dict=feed)
                    batch += self.config.batch_size

                except tf.errors.OutOfRangeError:
                    break

                if self.config.max_batch > 0 and batch > self.config.max_batch:
                    break

            prog.update(batch, values=[("loss", loss)])

        self.pred_color_one_image("data/iccv09Data/images/0000382.jpg",
                                  "outputs/0000382_epoch{}".format(epoch_number))

    def train_model(self, warm_start=False):
        if not warm_start:
            self._build_new_graph_session()
            with self.graph.as_default():
                self.session.run(tf.global_variables_initializer())
        for ii in range(self.config.n_epochs):
            i = ii + 1
            print("\nRunning epoch {}:".format(i))
            self.run_epoch(i)

    def pred_color_one_image(self, image_path, out_jpg_path):
        image_Yscale, image_UVscale = load_image_jpg_to_YUV(image_path, is_test=False, config=self.config)
        image_Yscale = image_Yscale.reshape([1] + self.config.image_shape[:2] + [1])
        image_UVscale = image_UVscale.reshape([1] + self.config.image_shape[:2] + [2])

        feed = {self.image_Yscale: image_Yscale,
                self.image_UVscale: image_UVscale,
                }

        loss, pred_color_image = self.session.run([self.loss, self.pred_color_image],
                                                  feed_dict=feed)

        print('\nprediction loss = {}'.format(loss))
        predicted_YUV_image = np.concatenate([image_Yscale, pred_color_image], axis=3)
        print("variance in UV channels, predicted image: {}".format(variance_UV_channels(predicted_YUV_image[0, ...])))
        dump_YUV_image_to_jpg(predicted_YUV_image[0, :, :, :], out_jpg_path + "_pred.png")

        true_YUV_image = np.concatenate([image_Yscale, image_UVscale], axis=3)
        print("variance in UV channels, true image: {}".format(variance_UV_channels(true_YUV_image[0, ...])))
        dump_YUV_image_to_jpg(true_YUV_image[0, :, :, :], out_jpg_path + "_true.png")

    def save(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Save current model."""
        if model_name is None:
            model_name = self.name

        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Create Saver
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())

        # Save model kwargs needed to rebuild model
        with open(os.path.join(model_dir, "model_config.pkl"), 'wb') as f:
            pickle.dump(self.config, f)

        # Save graph
        saver.save(self.session, os.path.join(model_dir, model_name))
        if verbose:
            print("[{0}] Model saved as <{1}>".format(self.name, model_name))

    def load(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Load model"""
        model_name = model_name or self.name
        model_dir = os.path.join(save_dir, model_name)

        # Load model kwargs needed to rebuild model
        with open(os.path.join(model_dir, "model_config.pkl"), 'rb') as f:
            self.config = pickle.load(f)

        # Create new graph, build network, and start session
        self._build_new_graph_session()

        # Initialize variables
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())

        # Load saved checkpoint to populate trained parameters
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.session, ckpt.model_checkpoint_path)
            if verbose:
                print("[{0}] Loaded model <{1}>".format(self.name, model_name))
        else:
            raise Exception("[{0}] No model found at <{1}>".format(
                self.name, model_name))
