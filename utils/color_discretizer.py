import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread

from utils.color_utils import RGB_to_YUV


class ColorDiscretizer(object):
    def __init__(self, nbins=30, threshold=5):
        self.nbins = nbins
        self.threshold = threshold

        self.xedges = np.linspace(0, 255., self.nbins + 1)
        self.yedges = np.linspace(0, 255., self.nbins + 1)

    def train(self, imdir="data/iccv09Data/images/", n_images=None):
        Us = []
        Vs = []

        if n_images is None:
            n_images = len(os.listdir(imdir))

        for impath in os.listdir(imdir)[:n_images]:
            imfullpath = os.path.join(imdir, impath)
            image = imread(imfullpath).astype(np.dtype("float32"))
            YUVimage = RGB_to_YUV(image)
            UVimage = YUVimage[:, :, 1:]
            UVpixels = np.reshape(UVimage, newshape=[-1, 2])
            Us.extend(UVpixels[:, 0])
            Vs.extend(UVpixels[:, 1])

        bins = np.linspace(0, 255., self.nbins + 1)
        self.heatmap, _, _ = np.histogram2d(Us, Vs, bins=[bins, bins])

        self.xycategories_to_indices_map = {}
        self.indices_to_xycategories_map = {}
        index = 1
        for xcategory in range(self.nbins):
            for (ycategory, heatscore) in enumerate(self.heatmap[xcategory, :]):
                if heatscore > self.threshold:
                    self.xycategories_to_indices_map[(xcategory, ycategory)] = index
                    self.indices_to_xycategories_map[index] = (xcategory, ycategory)
                    index += 1
                else:
                    self.xycategories_to_indices_map[(xcategory, ycategory)] = 0
                    self.indices_to_xycategories_map[0] = (xcategory, ycategory)

        self.n_categories = index

        self.categories_mean_pixels = np.zeros([self.n_categories, 2])
        for index in range(1, self.n_categories):
            xcategory, ycategory = self.indices_to_xycategories_map[index]
            self.categories_mean_pixels[index, :] = [(self.xedges[xcategory] + self.xedges[xcategory + 1]) / 2,
                                                     (self.yedges[ycategory] + self.yedges[ycategory + 1]) / 2]

    def plot_heatmap(self):
        logheatmap = np.log10(self.heatmap)
        extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]

        plt.imshow(logheatmap.T, extent=extent, origin='lower')
        plt.colorbar()
        plt.xlim([0, 255])
        plt.ylim([0, 255])
        plt.show()

    def categorize(self, UVpixels):
        Upixels = UVpixels[:, 0]
        Vpixels = UVpixels[:, 1]

        Upixels_categories = np.searchsorted(self.xedges[:-1], Upixels) - 1
        Vpixels_categories = np.searchsorted(self.yedges[:-1], Vpixels) - 1

        return [self.xycategories_to_indices_map[xycategories] for xycategories in
                zip(Upixels_categories, Vpixels_categories)]

    def UVpixels_from_distribution(self, distribution):
        """
        Returns mean pixels from Npixels distributions over the color categories.
        :param distribution: matrix of size Npixels * n_categories.
        """
        distribution /= np.sum(distribution, axis=1).reshape([-1, 1])
        return np.dot(distribution, self.categories_mean_pixels)