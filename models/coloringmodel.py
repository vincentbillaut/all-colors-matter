import json
import os
import random
from datetime import datetime
from shutil import copyfile

import numpy as np
import tensorflow as tf

from utils.data_utils import dump_YUV_image_to_jpg, load_image_jpg_to_YUV
from utils.progbar import Progbar


class Config(object):
    def __init__(self, config_path):
        input_dict = json.load(open(config_path))
        self.__dict__.update(input_dict)
        self.config_name = config_path.split('/')[-1]

        self.output_path = "outputs/" + "{:%Y%m%d_%H%M%S}-".format(datetime.now()) + \
                           hex(random.getrandbits(16))[2:].zfill(4) + "/"
        os.mkdir(self.output_path)

        copyfile(config_path, os.path.join(self.output_path, "config.json"))
        print("Saving model outputs to", self.output_path)


class ColoringModel(object):
    def __init__(self, config, dataset, name=None, seed=42):
        self.seed = seed
        self.config = config
        self.params = {}
        self.dataset = dataset
        self.name = name
        self.n_categories = self.dataset.color_discretizer.n_categories

    def _build_model(self):
        self.add_dataset()
        self.add_model()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_tensorboard_op()

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
        train_dataset = self.dataset.get_dataset_batched(False, self.config, shuffle=True, seed=self.seed)
        val_dataset = self.dataset.get_dataset_batched(True, self.config)

        # iterator just needs to know the output types and shapes of the datasets
        self.iterator = tf.data.Iterator.from_structure(output_types=(tf.float32,
                                                                      tf.int32,
                                                                      tf.float32,
                                                                      tf.bool),
                                                        output_shapes=([None, self.config.image_shape[0],
                                                                        self.config.image_shape[1], 1],
                                                                       [None, self.config.image_shape[0],
                                                                        self.config.image_shape[1]],
                                                                       [None, self.config.image_shape[0],
                                                                        self.config.image_shape[1]],
                                                                       [None, self.config.image_shape[0],
                                                                        self.config.image_shape[1]]))

        self.image_Yscale, self.categorized_image, self.weights, self.mask = self.iterator.get_next()
        self.train_init_op = self.iterator.make_initializer(train_dataset)
        self.test_init_op = self.iterator.make_initializer(val_dataset)

    def add_model(self):
        """ Add Tensorflow ops to get scores from inputs, stores it in self.scores.
        """
        raise NotImplementedError

    def add_loss_op(self):
        """ Add Tensorflow op to compute loss.
        """
        self.ytrue_onehot = tf.one_hot(self.categorized_image, depth=self.n_categories)
        # Remark: in the original paper, they used soft-encoding on the 5 nearest neighbors
        # instead of one-hot encoding, cf bottom page 5.

        self.mask_weights = self.weights * tf.cast(self.mask, tf.float32)
        self.reshaped_mask_weights = tf.reshape(self.mask_weights,
                                                shape=[-1, self.config.image_shape[0], self.config.image_shape[1], 1])

        self.loss = - tf.reduce_mean(
            tf.log(tf.nn.softmax(self.scores)) * self.ytrue_onehot * self.reshaped_mask_weights)

    def add_train_op(self):
        """ Add Tensorflow op to run one iteration of SGD, stores it in self.train_op.
        """
        raise NotImplementedError

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions, stores it in self.pred_image_categories.
        """
        raise NotImplementedError

    def add_tensorboard_op(self):
        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

    def run_epoch(self, epoch_number=0, val_type="single"):
        nbatches = len(self.dataset.train_ex_paths)
        target_progbar = nbatches if self.config.max_batch <= 0 else min(nbatches, self.config.max_batch)
        prog = Progbar(target=target_progbar)
        batch = 0

        with self.graph.as_default():
            self.session.run(self.train_init_op)
            while True:
                try:
                    loss, summary, _ = self.session.run([self.loss, self.summary_op, self.train_op],
                                                        )  # feed_dict=feed)
                    self.writer.add_summary(summary, epoch_number * target_progbar + batch)
                    batch += self.config.batch_size

                except tf.errors.OutOfRangeError:
                    break

                if 0 < self.config.max_batch < batch:
                    break

                prog.update(batch, values=[("loss", loss)])
            if val_type == "single":
                self.pred_color_one_image("data/test_pic.jpeg",
                                          os.path.join(self.config.output_path,
                                                       "samplepic_epoch{}".format(epoch_number)),
                                          epoch_number)
            if val_type == "full":
                self.pred_validation_set(epoch_number)

    def train_model(self, warm_start=False, val_type="single"):
        if not warm_start:
            self._build_new_graph_session()
            with self.graph.as_default():
                self.session.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.config.output_path + "train/", graph=self.graph)
        for ii in range(self.config.n_epochs):
            i = ii + 1
            print("\nRunning epoch {}/{}:".format(i, self.config.n_epochs))
            self.run_epoch(i, val_type=val_type)
            self.dataset.iterating_seed += 1

    def pred_color_one_image(self, image_path, out_jpg_path=None, epoch_number=0):
        image_Yscale, image_UVscale, mask = load_image_jpg_to_YUV(image_path, is_test=False, config=self.config)
        categorized_image, weights = self.dataset.color_discretizer.categorize(image_UVscale, return_weights=True)
        feed = {self.image_Yscale: image_Yscale.reshape([1] + self.config.image_shape[:2] + [1]),
                self.categorized_image: categorized_image.reshape([1] + self.config.image_shape[:2]),
                self.weights: weights.reshape([1] + self.config.image_shape[:2]),
                self.mask: mask.reshape([1] + self.config.image_shape[:2])
                }

        loss, pred_image_categories = self.session.run([self.loss, self.pred_image_categories],
                                                       feed_dict=feed)

        if out_jpg_path is not None:
            print('\nprediction loss = {}'.format(loss))
            pred_UVimage = self.dataset.color_discretizer.UVpixels_from_distribution(pred_image_categories)
            pred_UVimage = np.reshape(pred_UVimage, newshape=pred_UVimage.shape[1:])
            predicted_YUV_image = np.concatenate([image_Yscale, pred_UVimage], axis=2) * np.reshape(mask, (
                mask.shape[0], mask.shape[1], 1))
            dump_YUV_image_to_jpg(predicted_YUV_image, out_jpg_path + "_pred.png")

            if epoch_number == 1:
                true_YUV_image = np.concatenate([image_Yscale, image_UVscale], axis=2)
                dump_YUV_image_to_jpg(true_YUV_image, out_jpg_path + "_groundtruth.png")
        return loss

    def pred_validation_set(self, epoch_number=0):
        print("Validating epoch {} ...".format(epoch_number))
        self.writer_val = tf.summary.FileWriter(self.config.output_path + "test/", graph=self.graph)
        nbatches = len(self.dataset.val_ex_paths)
        if hasattr(self.config, "max_batch_val"):
            target_progbar = nbatches if self.config.max_batch_val <= 0 else min(nbatches, self.config.max_batch_val)
        else:
            target_progbar = nbatches
        prog = Progbar(target=target_progbar)
        batch = 0

        with self.graph.as_default():
            self.session.run(self.test_init_op)
            while True:
                try:
                    loss, summary, pred_image_categories = self.session.run(
                        [self.loss, self.summary_op, self.pred_image_categories], )
                    self.writer_val.add_summary(summary, epoch_number * target_progbar + batch)
                    batch += self.config.batch_size

                except tf.errors.OutOfRangeError:
                    break

                if 0 < self.config.max_batch < batch:
                    break

                prog.update(batch, values=[("loss", loss)])

    def save(self, save_dir=None, verbose=True):
        """Save current model."""
        if save_dir is None:
            save_dir = os.path.join(self.config.output_path, "checkpoints")
            os.mkdir(save_dir)

        # Create Saver
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())

        # Save graph
        saver.save(self.session, os.path.join(save_dir, "model"))
        if verbose:
            print("Model saved in {}".format(save_dir))

    def load(self, output_dir, verbose=True):
        """Load model"""
        save_dir = os.path.join(output_dir, "checkpoints")
        config_dir = os.path.join(output_dir, "config.json")

        # Load the dumped model config
        self.config = Config(config_dir)

        # Create new graph, build network, and start session
        self._build_new_graph_session()

        # Initialize variables
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())

        # Load saved checkpoint to populate trained parameters
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.join(save_dir))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.session, ckpt.model_checkpoint_path)
            if verbose:
                print("[{0}] Loaded model <{1}>".format(self.name, save_dir))
        else:
            raise Exception("[{0}] No model found at <{1}>".format(
                self.name, save_dir))
