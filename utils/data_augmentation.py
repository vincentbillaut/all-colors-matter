import numpy as np
from utils.color_utils import YUV_to_RGB, RGB_to_YUV

class DataAugmenter(object):
    def __init__(self, rand_variances = [], n_crops = 0, crop_param = .7, do_flip = False, keep_original = True):
        self.n_rand = len(rand_variances)
        self.rand_vars = rand_variances
        self.n_crop = n_crops
        self.crop_param = crop_param
        self.do_flip = do_flip
        self.keep_orig = keep_original
        self.n = int(self.keep_orig) + int(self.do_flip) + self.n_rand + self.n_crop


    def get_n(self):
        """Returns the number of images that the Augmenter induces.
        """
        return self.n


    def transf_generator(self, image, seed):
        """Generator, which yields the transformed versions of the input image.

        Parameters
        ----------
        image : np.ndarray
            Input image (RGB-encoded).
        seed : str
            Seed to give for random transformations.
            Usually we'll give the image's path.

        Will first yield the original image itself (if keep_original is True),
        then the noised images, then the cropped images.
        """
        # original image
        if self.keep_orig:
            yield image
        # flipped image
        if self.do_flip:
            yield self.gen_flipped(image)
        # noised images
        for i,v in enumerate(self.rand_vars):
            yield self.gen_noised(image, v, seed+str(i+int(self.keep_orig)+int(self.do_flip)))
        # cropped images
        for i in range(self.n_crop):
            yield self.gen_cropped(image, self.crop_param, seed+str(i+int(self.keep_orig)+int(self.do_flip)+self.n_rand))


    def get_transformation(self, i, seed):
        """Return the ith transformation of the DataAugmenter instance.
        Uselful when using it as an argument to another function (e.g. if
        the image is read at a different place than the data_augmenter is
        used)

        Parameters
        ----------
        Same as get_transformed(...), except that we output a function that
        takes an image as an argument.
        """
        return lambda x: self.get_transformed(x, i, seed)


    def get_transformed(self, image, i, seed):
        """Random access generator, which gives the image to which we applied
        the Augmenter's ith transformation.

        Parameters
        ----------
        image : np.ndarray
            Input image (RGB-encoded).
        i : int
            Index of the Augmenter's transformation to apply.
            Must be between 0 and self.n
        seed : str
            Seed to give for random transformations.
            Usually we'll give the image's path.

        Returns
        -------
        np.ndarray
            Transformed image.

        """
        assert(i < self.n)
        if i < int(self.keep_orig):
            return image
        elif i < int(self.keep_orig) + int(self.do_flip):
            return self.gen_flipped(image)
        elif i < int(self.keep_orig) + int(self.do_flip) + self.n_rand:
            return self.gen_noised(image, self.rand_vars[i - int(self.keep_orig) - int(self.do_flip)], seed+str(i))
        else:
            return self.gen_cropped(image, self.crop_param, seed+str(i))


    def gen_flipped(self, img):
        """Flips the image on the horizontal axis.
        """
        return img[:, ::-1, :]


    def gen_noised(self, input_img, v, seed='0'):
        """Adds noise to the luminance channel of an image.

        Parameters
        ----------
        input_img : np.ndarray
            Input image to add noise to, RGB-encoded ([0,1]).
        v : float
            Standard deviation parameter, for the noise.
        seed : str
            Seed to give for random transformations.

        Returns
        -------
        np.ndarray
            Image with added noise to the luminance channel, RGB-encoded ([0,1]).

        """
        np.random.seed(hash(seed) % 2**32)
        if np.max(input_img) > 1:
            img = input_img / 255.
        yuv_img = RGB_to_YUV(img)
        yuv_img[:,:,0] += v*np.random.rand(img.shape[0], img.shape[1])
        output_img = YUV_to_RGB(yuv_img)
        return np.clip(output_img, 0, 1)


    def gen_cropped(self, input_img, alpha, seed='0'):
        """Crops the image by a given ratio, both vertically and horizontally.

        Parameters
        ----------
        input_img : np.ndarray
            Input image.
        alpha : float
            Ratio by which to crop the image.
            Usually take about 0.7 or 0.8.
        seed : str
            Seed to give for random transformations.

        Returns
        -------
        np.ndarray
            Cropped image.

        """
        np.random.seed(hash(seed) % 2**32)
        H, W, _ = input_img.shape
        new_H, new_W = int(H*alpha), int(W*alpha)
        new_y0, new_x0 = np.random.randint(0,H-new_H+1), np.random.randint(0,W-new_W+1)
        return input_img[new_y0:(new_y0+new_H),new_x0:(new_x0+new_W),:]
