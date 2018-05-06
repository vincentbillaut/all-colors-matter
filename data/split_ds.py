import os
import shutil
from matplotlib.pyplot import imread
from scipy.misc import imsave
import numpy as np
from tqdm import tqdm

def prepare_dataset(ds_path,name,begin_filter=None,splits_props= None,splits_names = None,resize_params = None,overwrite_dir = True):
    if splits_props is None and splits_names is None:
        split_list = ["train","val","test"]
        split_props = [.7,.15,.15]
    else:
        assert splits_props is not None and splits_names is not None and len(splits_props)==len(splits_names)
        split_list = splits_names
        split_props = splits_props
    if begin_filter is None:
        subfil = [""]
    else:
        subfil = begin_filter

    print("Filtering")
    impaths = [x for x in tqdm(os.listdir(ds_path))if any([x.startswith(y) for y in subfil])]

    for spname in split_list:
        if not os.path.isdir(name+"_"+spname):
            os.mkdir(name+"_"+spname)
        else:
            if overwrite_dir:
                shutil.rmtree(name+"_"+spname)
                os.mkdir(name + "_" + spname)
    split_ass = np.random.choice(len(split_props), p=split_props, size=len(impaths)).reshape((-1,))

    if resize_params is None:
        print("Copying...")
        for im,sa in zip(tqdm(impaths),split_ass):
            shutil.copy(os.path.join(ds_path,im),os.path.join(name+"_"+split_list[sa],im))
    else:
        print("Processing...")
        x,y = resize_params["output_max"]
        method = resize_params["method"]
        for im, sa in zip(tqdm(impaths), split_ass):
            image = imread(os.path.join(ds_path,im))
            image = image[:,:,:3]
            if method =="naive":
                stride_x = int(np.ceil(image.shape[0]/x))
                stride_y = int(np.ceil(image.shape[1] / y))
                stride = max(stride_x,stride_y)
                new_image = image[::stride,::stride,:]
            imsave(os.path.join(name+"_"+split_list[sa],im),new_image)


    print("Done")


if __name__=="__main__":
    resize_params = {
        "method" : "naive",
        "output_max" : (320,320)
    }
    prepare_dataset("SUN2012/Images","sun_s1",begin_filter=["b_beach_sun","c_coast","s_sandbar"],resize_params = resize_params)