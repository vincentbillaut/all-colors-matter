import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread

from utils.color_utils import RGB_to_YUV, YUV_to_RGB


class ColorDiscretizer(object):
    def __init__(self, nbins=10, threshold=.000001, weighting_lambda=.2, max_categories=None):
        self.nbins = nbins
        self.threshold = threshold
        self.weighting_lambda = weighting_lambda

        self.xedges = np.linspace(-.450, .450, self.nbins + 1)
        self.yedges = np.linspace(-.650, .650, self.nbins + 1)
        self.max_categories = max_categories

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

        self.heatmap, _, _ = np.histogram2d(Us, Vs, bins=[self.xedges, self.yedges])
        self.heatmap /= np.sum(self.heatmap)

        self.xycategories_to_indices_map = {}
        self.indices_to_xycategories_map = {}
        self.category_frequency = {}
        index = 0

        if self.max_categories is not None:
            sorted_values = np.sort(np.ravel(self.heatmap))[::-1]
            num_categ = sorted_values.shape[0] - np.searchsorted(sorted_values[::-1], self.threshold)
            if num_categ > self.max_categories:
                self.threshold = (sorted_values[self.max_categories - 1] + sorted_values[self.max_categories]) * 0.5

        # First pass; giving unique ids to colors above threshold
        for xcategory in range(self.nbins):
            for (ycategory, heatscore) in enumerate(self.heatmap[xcategory, :]):
                if heatscore > self.threshold:
                    self.xycategories_to_indices_map[(xcategory, ycategory)] = index
                    self.indices_to_xycategories_map[index] = (xcategory, ycategory)
                    self.category_frequency[index] = heatscore
                    index += 1

        self.n_categories = index

        # Second pass, mapping rare colors to frequent ones; updating frequencies
        for xcategory in range(self.nbins):
            for (ycategory, heatscore) in enumerate(self.heatmap[xcategory, :]):
                if not heatscore > self.threshold:
                    closest_class = min(range(self.n_categories),
                                        key=lambda k: (self.indices_to_xycategories_map[k][0] - xcategory) ** 2 +
                                                      (self.indices_to_xycategories_map[k][1] - ycategory) ** 2)
                    self.category_frequency[closest_class] += heatscore
                    self.xycategories_to_indices_map[(xcategory, ycategory)] = closest_class


        # compute the weights associated with every pixel class
        self.weights = {k: 1. / (proba * (1. - self.weighting_lambda) + self.weighting_lambda / self.n_categories) for
                        (k, proba) in self.category_frequency.items()}

        normalization_factor = sum([self.weights[k] * self.category_frequency[k] for k in self.weights])
        self.weights = {k: weight / normalization_factor for k, weight in self.weights.items()}

        self.categories_mean_pixels = np.zeros([self.n_categories, 2])
        for index in range(1, self.n_categories):
            xcategory, ycategory = self.indices_to_xycategories_map[index]
            self.categories_mean_pixels[index, :] = [(self.xedges[xcategory] + self.xedges[xcategory + 1]) / 2,
                                                     (self.yedges[ycategory] + self.yedges[ycategory + 1]) / 2]

    def plot_heatmap(self):
        hm = np.copy(self.heatmap)
        hm[hm < self.threshold] = 0
        logheatmap = np.log10(hm)
        extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]

        plt.figure(figsize=(15, 10))
        plt.subplot(222)
        plt.imshow(logheatmap.T, extent=extent, origin='lower')
        plt.colorbar()
        plt.ylim([-.650, .650])
        plt.xlim([-.650, .650])
        plt.title("Frequency map (log-scale)")

        plt.subplot(224)
        weights_matrix = np.zeros([self.nbins, self.nbins])
        for k in self.weights:
            weights_matrix[self.indices_to_xycategories_map[k]] = self.weights[k]
        logweights_matrix = np.log10(weights_matrix)
        plt.imshow(logweights_matrix.T, extent=extent, origin='lower')
        plt.colorbar()
        plt.ylim([-.650, .650])
        plt.xlim([-.650, .650])
        plt.title("Weight map (log-scale)")

        plt.subplot(221)
        color_matrix = np.zeros([self.nbins, self.nbins, 3]) + 255.
        for k in self.weights:
            yuv = np.zeros([1, 1, 3]) + .5
            yuv[..., 1:] = self.categories_mean_pixels[k]
            color_matrix[self.indices_to_xycategories_map[k][1], self.indices_to_xycategories_map[k][0], :] = YUV_to_RGB(yuv)
        plt.imshow(color_matrix / 255., extent=extent, origin='lower')
        plt.ylim([-.650, .650])
        plt.xlim([-.650, .650])
        plt.title("Color map")

        plt.subplot(223)
        plt.imshow(-logheatmap.T, extent=extent, origin='lower')
        plt.colorbar()
        plt.ylim([-.650, .650])
        plt.xlim([-.650, .650])
        plt.title("Inverse-frequency map (log-scale)")

        plt.tight_layout()

    def categorize(self, UVpixels, return_weights=False):
        """
        From a list of UV pixels, returns the indices of the categories they fall in. If asked, returns a list of the
        associated categories frequencies.
        :param UVpixels:
        :return:
        """
        Upixels = UVpixels[..., 0]
        flatUpixels = Upixels.reshape([-1])
        Vpixels = UVpixels[..., 1]
        flatVpixels = Vpixels.reshape([-1])

        Upixels_categories = np.searchsorted(self.xedges[:-1], flatUpixels) - 1
        Vpixels_categories = np.searchsorted(self.yedges[:-1], flatVpixels) - 1

        if return_weights:
            return np.reshape(np.array([self.xycategories_to_indices_map[xycategories] for xycategories in
                                        zip(Upixels_categories, Vpixels_categories)]), Upixels.shape), \
                   np.reshape(np.array([self.weights[self.xycategories_to_indices_map[xycategories]] for xycategories in
                                        zip(Upixels_categories, Vpixels_categories)]), Upixels.shape)
        else:
            return np.reshape(np.array([self.xycategories_to_indices_map[xycategories] for xycategories in
                                        zip(Upixels_categories, Vpixels_categories)]), Upixels.shape)

    def UVpixels_from_distribution(self, distribution, temperature=1):
        """
        Returns mean pixels from Npixels distributions over the color categories.
        :param temperature: temperature of the annealed probability distribution.
        :param distribution: matrix of size Npixels * Mpixels * n_categories.
        """
        temp_distribution = np.exp(np.log(distribution + 1e-8) / temperature)
        newshape = list(distribution.shape)
        newshape[-1] = 1
        temp_distribution /= np.sum(temp_distribution, axis=-1).reshape(newshape)

        return np.dot(temp_distribution, self.categories_mean_pixels)
