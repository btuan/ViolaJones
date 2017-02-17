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


def boxes_intersect(bbox1, bbox2):
    """ Take axis-aligned boxes defined by lowest x and y coordinates. """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1 += w1 // 2
    y1 += h1 // 2
    x2 += w2 // 2
    y2 += h2 // 2

    return abs(x1 - x2) * 2 < w1 + w2 and abs(y1 - y2) * 2 < h1 + h2


def draw_bounding_boxes(arr, bounding_boxes, w, h, merge_intersections=False, fpath=None):
    boxes_to_draw = []

    if merge_intersections:
        # TODO: make this a weighted average
        for x1, y1 in bounding_boxes:
            intersecting_box = None
            for ind, (x2, y2, w2, h2) in enumerate(boxes_to_draw):
                if boxes_intersect((x1, y1, w, h), (x2, y2, w2, h2)):
                    intersecting_box = ind
                    break

            if intersecting_box is None:
                boxes_to_draw.append((x1, y1, w, h))
            else:
                x2, y2, w2, h2 = boxes_to_draw[intersecting_box]
                x_lo = min(x1, x2)
                y_lo = min(y1, y2)
                new_width = max(x1 + w, x2 + w2) - x_lo
                new_height = max(y1 + h, y2 + h2) - y_lo
                boxes_to_draw[intersecting_box] = (x_lo, y_lo, new_width, new_height)

        for x1, y1, w1, h1 in boxes_to_draw:
            x2, y2 = x1 + w1, y1 + h1
            arr[x1: x2, y1] = np.full(w1, 255, dtype=np.int64)
            arr[x1: x2, y2 - 1] = np.full(w1, 255, dtype=np.int64)
            arr[x1, y1: y2] = np.full(h1, 255, dtype=np.int64)
            arr[x2 - 1, y1: y2] = np.full(h1, 255, dtype=np.int64)

        io.imsave('image.jpg', arr)
    else:
        for x1, y1 in bounding_boxes:
            x2, y2 = x1 + w, y1 + h
            arr[x1: x2, y1] = np.full(w, 255, dtype=np.int64)
            arr[x1: x2, y2 - 1] = np.full(w, 255, dtype=np.int64)
            arr[x1, y1: y2] = np.full(h, 255, dtype=np.int64)
            arr[x2 - 1, y1: y2] = np.full(h, 255, dtype=np.int64)

        io.imsave('image.jpg', arr)

