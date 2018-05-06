import os
import shutil
import numpy as np
from tqdm import tqdm

def split_dataset(ds_path,name,begin_filter=None,splits_props= None,splits_names = None):
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

    print("Copying...")
    for spname in split_list:
        os.mkdir(name+"_"+spname)
    split_ass = np.random.choice(len(split_props),p = split_props,size = len(impaths)).reshape((-1,))
    for im,sa in zip(tqdm(impaths),split_ass):
        shutil.copy(os.path.join(ds_path,im),os.path.join(name+"_"+split_list[sa],im))

    print("Done")


if __name__=="__main__":
    split_dataset("SUN2012/Images","sun_s1",begin_filter=["b_beach_sun","c_coast","s_sandbar"])