import tensorflow as tf
from utils.data_utils import get_dataset_batched, get_im_paths
from utils.progbar import Progbar


class Config(object):
    def __init__(self):
        # param_dict = input
        self.train_path = "data/iccv09Data/images"
        self.val_path = "data/iccv09Data/images"
        self.batch_size = 16
        self.image_shape = (320, 320, 3)


class Model(object):
    def __init__(self, config):
        self.config = config
        self.params = {}

        self.load_data()
        self.add_dataset()
        self.add_placeholders()
        self.add_model()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()

    def load_data(self):
        self.train_ex_paths = get_im_paths(self.config.train_path)
        self.val_ex_paths = get_im_paths(self.config.val_path)

    def add_dataset(self):
        train_dataset = get_dataset_batched(self.config.train_path, False, self.config)
        val_dataset = get_dataset_batched(self.config.val_path, True, self.config)
        # iterator just needs to know the output types and shapes of the datasets
        self.iterator = tf.contrib.data.Iterator.from_structure(output_types=(tf.float32, tf.float32),
                                                                output_shapes=([None, 240, 320, 1],
                                                                               [None, 240, 320, 3]))
        self.image_greyscale, self.image = self.iterator.get_next()
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

    def add_train_op(self):
        """ Add Tensorflow op to run one iteration of SGD.
        """
        pass

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions.
        """
        pass

    def run_epoch(self, sess):
        nbatches = len(self.train_ex_paths)
        prog = Progbar(target=nbatches)
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

            prog.update(batch, values=[("loss", loss)])
