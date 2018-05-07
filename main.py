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
    model.train_model()
    model.save("test_save")
    # model.load("test_save")
    # model.pred_color_one_image("data/iccv09Data/images/0000382.jpg",
    #                           "outputs/0000382_epoch{}".format("loaded"))
    # print("Training from loaded")
    # model.train_model(warm_start=True)
