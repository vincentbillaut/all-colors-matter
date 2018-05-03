import tensorflow as tf
from models.model import Config
from models.naive_convnet import NaiveConvModel
from utils.dataset import Dataset

if __name__ == "__main__":
    config = Config("configs/config.json")
    dataset = Dataset(config.train_path, config.val_path)
    model = NaiveConvModel(config, dataset=dataset)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())
        model.run_epoch(sess)
