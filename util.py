""" util.py

Functions to import and manipulate JPG images.


Author: Brian Tuan
Last Modified: February 11, 2017

"""

import numpy as np
from os import listdir
from skimage import color, io


def import_jpg(fname):
    return io.imread(fname)


def rgb_to_grayscale(img):
    return color.rgb2gray(img)


def import_img_dir(dirname, rgb2gray=True):
    dirname = dirname if '/' in dirname else dirname + '/'
    imgs = [import_jpg(dirname + f) for f in listdir(dirname) if f.endswith('.jpg')]

    if rgb2gray:
        imgs = [rgb_to_grayscale(i) for i in imgs]

    return np.array(imgs) * 255


def integral_image(arr):
    """ Returns integral image over last two axes. Assume first index is image index. """
    return arr.cumsum(axis=-1).cumsum(axis=-2)

