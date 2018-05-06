import tensorflow as tf
from models.coloringmodel import Config
from models.naive_convnet import NaiveConvColoringModel
from utils.color_discretizer import ColorDiscretizer
from utils.dataset import Dataset

if __name__ == "__main__":
    config = Config("configs/config.json")

    cd = ColorDiscretizer()
    imdir = "data/iccv09Data/images/"
    cd.train(imdir, 30)

    dataset = Dataset(config.train_path, config.val_path, cd)
    model = NaiveConvColoringModel(config, dataset=dataset)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())
        model.train_model(sess)
