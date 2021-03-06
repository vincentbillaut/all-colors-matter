import tensorflow as tf

from utils.progbar import Progbar


class DiscriminativeModel(object):
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
                                                        output_shapes=([None, 320, 320, 3],
                                                                       [None]))
        self.image_YUVscale, self.label = self.iterator.get_next()
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
        print("\nRunning epoch {}:".format(epoch_number))
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

    def train_model(self, sess):
        for ii in range(self.config.n_epochs):
            i = ii + 1
            self.run_epoch(sess, i)
