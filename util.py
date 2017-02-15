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
    return np.rint(color.rgb2gray(img) * 255)


def import_img_dir(dirname, rgb2gray=True):
    dirname = dirname if '/' in dirname else dirname + '/'
    images = [import_jpg(dirname + f) for f in listdir(dirname) if f.endswith('.jpg')]

    if rgb2gray:
        images = [rgb_to_grayscale(i) for i in images]

    return np.array(images)


def integral_image(arr):
    """ Returns integral image over last two axes. Assume first index is image index. """
    return arr.cumsum(axis=-1).cumsum(axis=-2)


def draw_bounding_boxes(arr, bounding_boxes, w, h, fpath=None):
    # TODO: Merge intersecting bounding boxes


    for x1, y1 in bounding_boxes:
        x2, y2 = x1 + w, y1 + h
        arr[x1: x2, y1] = np.full(w, 255, dtype=np.int64)
        arr[x1: x2, y2 - 1] = np.full(w, 255, dtype=np.int64)
        arr[x1, y1: y2] = np.full(h, 255, dtype=np.int64)
        arr[x2 - 1, y1: y2] = np.full(h, 255, dtype=np.int64)

    io.imsave('image.jpg', arr)

