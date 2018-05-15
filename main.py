import argparse

from models.coloringmodel import Config
from models.naive_convnet import NaiveConvColoringModel
from models.next_convnet import NextConvColoringModel
from utils.color_discretizer import ColorDiscretizer
from utils.dataset import Dataset

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Run a model from a given config file.')
    argparser.add_argument('--config', type=str, default="configs/config.json")
    args = argparser.parse_args()

    config = Config(args.config)

    cd = ColorDiscretizer(max_categories=config.max_categories)
    cd.train(config.cd_train_path, 30)

    dataset = Dataset(config.train_path, config.val_path, cd)

    if hasattr(config, "model"):
        if config.model == "NextConvColoringModel":
            model = NextConvColoringModel(config, dataset=dataset)
        elif config.model == "NaiveConvColoringModel":
            model = NaiveConvColoringModel(config, dataset=dataset)
        else:
            model = NaiveConvColoringModel(config, dataset=dataset)
    else:
        model = NaiveConvColoringModel(config, dataset=dataset)

    model.train_model()
    model.save("test_save")
    # model.load("test_save")
    # model.pred_color_one_image("data/iccv09Data/images/0000382.jpg",
    #                           "outputs/0000382_epoch{}".format("loaded"))
    # print("Training from loaded")
    # model.train_model(warm_start=True)
