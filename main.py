import tensorflow as tf
from models.model import Model, Config

if __name__ == "__main__":
    config = Config()
    model = Model(config)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())
        model.run_epoch(sess)
