import argparse
import os
import shutil
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.misc import imresize
import numpy as np
from tqdm import tqdm


def prepare_dataset(ds_path, name, begin_filter=None, splits_props=None, splits_names=None, resize_params=None,
                    overwrite_dir=True):
    """

    :param ds_path: dataset path
    :param name: name of the dataset directory prefix
    :param begin_filter: filter to take images that begin by given strings, None for no filter
    :param splits_props: proportion of images to assign to each split
    :param splits_names: names of the splits
    :param resize_params: dictionary that provides
        "method" : "naive",‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’ :  downsampling method
        "output_max": int tuple, max size of the output
    :param overwrite_dir: If true, overwrite existing images of the same dataset
    :return:
    """
    if splits_props is None and splits_names is None:
        split_list = ["train", "val", "test"]
        split_props = [.7, .15, .15]
    else:
        assert splits_props is not None and splits_names is not None and len(splits_props) == len(splits_names)
        split_list = splits_names
        split_props = splits_props
    if begin_filter is None:
        subfil = [""]
    else:
        subfil = begin_filter

    print("Filtering")
    impaths = [x for x in tqdm(os.listdir(ds_path)) if
               any([x.startswith(y) for y in subfil]) and (x.lower().endswith("jpg") or x.lower().endswith("jpeg"))]
    impaths = [x for x in impaths if (x.lower().endswith("jpg") or x.lower().endswith("jpeg"))]

    for spname in split_list:
        if not os.path.isdir(name + "_" + spname):
            os.mkdir(name + "_" + spname)
        else:
            if overwrite_dir:
                shutil.rmtree(name + "_" + spname)
                os.mkdir(name + "_" + spname)
    split_ass = np.random.choice(len(split_props), p=split_props, size=len(impaths)).reshape((-1,))

    if resize_params is None:
        print("Copying...")
        for im, sa in zip(tqdm(impaths), split_ass):
            shutil.copy(os.path.join(ds_path, im), os.path.join(name + "_" + split_list[sa], im))
    else:
        print("Processing...")
        x, y = resize_params["output_max"]
        method = resize_params["method"]
        for im, sa in zip(tqdm(impaths), split_ass):
            image = imread(os.path.join(ds_path, im))
            if len(image.shape) == 3:
                image = image[:, :, :3]

                ratio_x = image.shape[0] / x
                ratio_y = image.shape[1] / y
                max_ratio = max(ratio_x, ratio_y)
                if max_ratio > 1:
                    if method == "naive":
                        stride = int(np.ceil(max_ratio))
                        new_image = image[::stride, ::stride, :]
                    else:
                        new_x = int(image.shape[0] / max_ratio)
                        new_y = int(image.shape[1] / max_ratio)
                        new_image = imresize(image, (new_x, new_y, 3), interp=method)
                else:
                    new_image = image

                imsave(os.path.join(name + "_" + split_list[sa], im), new_image)
            else:
                print("Skipping image {} because of incorrect shape {}".format(im, image.shape))

    print("Done")


def plot_dataset_stats(ds_path, fignum=0):
    impath = [x for x in os.listdir(ds_path) if (x.lower().endswith("jpg") or x.lower().endswith("jpeg"))]
    sizes_x = list()
    sizes_y = list()
    for im in tqdm(impath):
        image = imread(os.path.join(ds_path, im))
        sizes_x.append(image.shape[0])
        sizes_y.append(image.shape[1])
    plt.figure(fignum)
    plt.hist(sizes_y, alpha=0.7, label="y", bins=20)
    plt.axvline(np.mean(sizes_y), label="ym", color="green")
    plt.hist(sizes_x, alpha=0.7, label="x", bins=20)
    plt.axvline(np.mean(sizes_x), label="xm", color="purple")
    plt.legend()
    plt.title(ds_path)


if __name__ == "__main__":
    resize_params = {
        "method": "lanczos",
        "output_max": (512, 512)
    }
    argparser = argparse.ArgumentParser(description='Resize and prepare a dataset.')
    argparser.add_argument('--inpath', type=str, default=None)
    argparser.add_argument('--outprefix', type=str, default=None)
    args = argparser.parse_args()
    if os.path.isdir("SUN2012/Images"):
        prepare_dataset("SUN2012/Images", "sun_inet", begin_filter=["b_beach_sun", "c_coast", "s_sandbar"],
                        resize_params=resize_params)
    if os.path.isdir("imagenet"):
        for i, subinet in enumerate(os.listdir("imagenet")):
            prepare_dataset(os.path.join("imagenet", subinet), "sun_inet", resize_params=resize_params, overwrite_dir=False)

    if args.inpath is not None:
        if args.outprefix is not None:
            if os.path.isdir(args.inpath):
                prepare_dataset(args.inpath, args.outprefix, resize_params=resize_params)
