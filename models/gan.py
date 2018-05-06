from models.coloringmodel import ColoringModel
from models.discriminative_net import DiscriminativeModel
import tensorflow as tf

from utils.progbar import Progbar


class GAN(object):
    def __init__(self, config, dataset, coloring_model, discriminative_model):
        self.config = config
        self.dataset = dataset
        self.coloring_model = coloring_model
        self.coloring_model = ColoringModel(config)
        self.discriminative_model = discriminative_model
        self.discriminative_model = DiscriminativeModel(config)

        self.n_training_operations = 100
        self.add_loss()
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

    def run_epoch_discriminative(self, sess, epoch_number=0):
        print("\nRunning Discriminative epoch {}:".format(epoch_number))

        nbatches = len(self.dataset.train_ex_paths)
        prog = Progbar(target=min(nbatches, self.config.max_batch))
        batch = 0

        sess.run(self.train_init_op)
        while True:
            try:
                # feed = {self.dropout_placeholder: self.config.dropout,
                #         self.lr_placeholder: lr_schedule.lr,
                #         self.is_training: self.config.use_batch_norm}

                loss, _ = sess.run([self.discriminative_net_loss, self.discriminative_net_train_op],
                                   )  # feed_dict=feed)
                batch += self.config.batch_size

            except tf.errors.OutOfRangeError:
                break

            if batch > self.config.max_batch:
                break

            prog.update(batch, values=[("loss", loss)])

    def run_epoch_generative(self, sess, epoch_number=0):
        print("\nRunning Generative epoch {}:".format(epoch_number))

        nbatches = len(self.dataset.train_ex_paths)
        prog = Progbar(target=min(nbatches, self.config.max_batch))
        batch = 0

        sess.run(self.train_init_op)
        while True:
            try:
                # feed = {self.dropout_placeholder: self.config.dropout,
                #         self.lr_placeholder: lr_schedule.lr,
                #         self.is_training: self.config.use_batch_norm}

                loss, _ = sess.run([self.generative_net_loss, self.generative_net_train_op],
                                   )  # feed_dict=feed)
                batch += self.config.batch_size

            except tf.errors.OutOfRangeError:
                break

            if batch > self.config.max_batch:
                break

            prog.update(batch, values=[("loss", loss)])

    def add_loss(self):
        real_images, real_imagesYchannel = None, None

        recolorized_images = self.coloring_model.add_model(real_imagesYchannel)
        recolorized_images_scores = self.discriminative_model.add_model(recolorized_images)
        real_images_scores = self.discriminative_model.add_model(real_images)

        # Optimizing the discriminative net
        self.discriminative_net_loss = -tf.reduce_mean(tf.log(real_images_scores) +
                                                       tf.log(1 - recolorized_images_scores))

        # Optimizing the generating net
        self.generative_net_loss = - tf.reduce_mean(tf.log(recolorized_images_scores))

    def add_train_op(self):
        discriminative_net_opt = tf.train.AdamOptimizer(learning_rate=.01)
        self.discriminative_net_train_op = discriminative_net_opt.minimize(self.discriminative_net_loss)

        generative_net_opt = tf.train.AdamOptimizer(learning_rate=.01)
        self.generative_net_train_op = generative_net_opt.minimize(self.generative_net_loss)
