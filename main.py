import tensorflow as tf
from models.model import Model, Config
from models.naive_convnet import NaiveConvConfig, NaiveConvModel

if __name__ == "__main__":
    config = NaiveConvConfig()
    model = NaiveConvModel(config)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())
        model.run_epoch(sess)
