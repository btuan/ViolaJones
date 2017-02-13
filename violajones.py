""" violajones.py


Author: Brian Tuan
Last Modified: February 13, 2017

"""

import click
import numpy as np

from util import import_img_dir, import_jpg, integral_image

"""
FEATURE PROCESSING FUNCTIONS
============================

Each feature can be uniquely specified by a 6-tuple containing the following parameters:
    (x_position, y_position, width, height, polarity, threshold)

We use the following four Haar wavelet rectangular feature archetypes:
    - Type 1      - Type 2      - Type 3      - Type 4
        ++++++++      ++++0000      ++++0000      0000++++0000
        ++++++++      ++++0000      ++++0000      0000++++0000
        00000000      ++++0000      0000++++      0000++++0000
        00000000      ++++0000      0000++++      0000++++0000

In addition, for any given dimension of the filter, where:
    D is the length of the image along that dimension,
    K is the length of the filter along that dimension, and
    S is the length of the stride (assumed to be equal in all directions,
the number of filters that fit is: 1 + (D - K) / S. We do not pad the input in this implementation.

In each feature generating function, we iterate over each possible (x, y) location and each possible
filter size stemming from that location. Consequently, many filters are created.
"""


def gen_type1(width, height, stride=1):
    """ Type 1 filters must have an even height.
    Reference points:
        D   C
        E   B
        F   A

    Formula: A + C + 2E - 2B - D - F
    """
    features = []
    for w in range(1, width, 1):
        for h in range(2, height, 2):
            for x in range(0, 1 + (width - w) // stride, stride):
                for y in range(0, 1 + (height - h) // stride, stride):
                    A = (x + w - 1, y + h - 1)
                    B = (x + w - 1, y + (h // 2) - 1)
                    C = (x + w - 1, y - 1)
                    D = (x - 1, y - 1)
                    E = (x - 1, y + (h // 2) - 1)
                    F = (x - 1, y + h - 1)

                    add = [i for i in [A, C, E, E] if i[0] >= 0 and i[1] >= 0]
                    sub = [i for i in [B, B, D, F] if i[0] >= 0 and i[1] >= 0]
                    features.append((tuple(add), tuple(sub)))

    return features


def gen_type2(width, height, stride=1):
    """ Type 2 filters must have even width.
    Reference points:
        D   E   F
        C   B   A

    Formula: A + C + 2E - 2B - D - F
    """
    features = []
    for w in range(2, width, 2):
        for h in range(1, height, 1):
            for x in range(0, 1 + (width - w) // stride, stride):
                for y in range(0, 1 + (height - h) // stride, stride):
                    A = (x + w - 1, y + h - 1)
                    B = (x + (w // 2) - 1, y + h - 1)
                    C = (x - 1, y + h - 1)
                    D = (x - 1, y - 1)
                    E = (x + (w // 2) - 1, y - 1)
                    F = (x + w - 1, y - 1)

                    add = [i for i in [A, C, E, E] if i[0] >= 0 and i[1] >= 0]
                    sub = [i for i in [B, B, D, F] if i[0] >= 0 and i[1] >= 0]
                    features.append((tuple(add), tuple(sub), 1, 0))

    return features


def gen_type3(width, height, stride=1):
    """ Type 3 filters must have even width and even height.
    Reference points:
        I   H   G
        F   E   D
        C   B   A

    Formula: A + C + 4E + G + I - 2B - 2D - 2F - 2H
    """
    features = []
    for w in range(2, width, 2):
        for h in range(2, height, 2):
            for x in range(0, 1 + (width - w) // stride, stride):
                for y in range(0, 1 + (height - h) // stride, stride):
                    A = (x + w - 1, y + h - 1)
                    B = (x + (w // 2) - 1, y + h - 1)
                    C = (x - 1, y + h - 1)
                    D = (x + w - 1, y + (h // 2) - 1)
                    E = (x + (w // 2) - 1, y + (h // 2) - 1)
                    F = (x - 1, y + (h // 2) - 1)
                    G = (x + w - 1, y - 1)
                    H = (x + (w // 2) - 1, y - 1)
                    I = (x - 1, y - 1)

                    add = [i for i in [A, C, E, E, E, E, G, I] if i[0] >= 0 and i[1] >= 0]
                    sub = [i for i in [B, B, D, D, F, F, H, H] if i[0] >= 0 and i[1] >= 0]
                    features.append((tuple(add), tuple(sub)))

    return features


def gen_type4(width, height, stride=1):
    """ Type 4 filters must have a width that is a multiple of 3.
    Reference points:
        E   F   G   H
        D   C   B   A

    Formula: A + 2C + E + 2G - 2B - D - 2F - H
    """
    features = []
    for w in range(3, width, 3):
        for h in range(1, height, 1):
            for x in range(0, 1 + (width - w) // stride, stride):
                for y in range(0, 1 + (height - h) // stride, stride):
                    A = (x + w - 1, y + h - 1)
                    B = (x + 2 * (w // 3) - 1, y + h - 1)
                    C = (x + (w // 3) - 1, y + h - 1)
                    D = (x - 1, y + h - 1)
                    E = (x - 1, y - 1)
                    F = (x + (w // 3) - 1, y - 1)
                    G = (x + 2 * (w // 3) - 1, y - 1)
                    H = (x + w - 1, y - 1)

                    add = [i for i in [A, C, C, E, G, G] if i[0] >= 0 and i[1] >= 0]
                    sub = [i for i in [B, B, D, F, F, H] if i[0] >= 0 and i[1] >= 0]
                    features.append((tuple(add), tuple(sub)))

    return features


def generate_features(width, height, stride=1, verbose=False):
    """ Generate features based on integral image representation.
    Each feature is represented by points to add, points to subtract, polarity, and threshold.
    """

    features = []

    if verbose:
        print("Generating type 1 features...")
    features.extend(gen_type1(width, height, stride))

    if verbose:
        print("Generating type 2 features...")
    # features.extend(gen_type2(width, height, stride))

    if verbose:
        print("Generating type 3 features...")
    # features.extend(gen_type3(width, height, stride))

    if verbose:
        print("Generating type 4 features...\n")
    # features.extend(gen_type4(width, height, stride))

    return features


def train_features(features, faces, background, faces_dist, background_dist):
    """ Train polarity and threshold for each feature on the test set. """
    together = np.concatenate((faces, background))
    for ind, feature in enumerate(features):
        add, sub = feature[0], feature[1]

        # TODO: Normalize the indicator
        indicator = np.concatenate((faces_dist, -1 * background_dist))
        to_add = together[:, [x[0] for x in add], [y[1] for y in add]].sum(axis=-1)
        to_sub = together[:, [x[0] for x in sub], [y[1] for y in sub]].sum(axis=-1)
        scores = to_add - to_sub
        sort_perm = scores.argsort()

        # Calculate optimal polarity and threshold.
        t_face, t_back = faces_dist.sum(), background_dist.sum()
        scores, indicator_perm = scores[sort_perm], indicator[sort_perm]

        s_face = 0 if indicator_perm[0] < 0 else indicator_perm[0]
        s_back = 0 if indicator_perm[0] >= 0 else indicator_perm[0]
        error_min = min(s_face + t_back - s_back, s_back + t_face - s_face)
        polarity_min = -1 if error_min == s_face + t_back - s_back else 1
        threshold = together[0]
        for j in range(1, scores.shape[0]):
            if indicator_perm[j] < 0:
                s_back -= indicator_perm[j]
            else:
                s_face += indicator_perm[j]

            left = s_face + t_back - s_back
            right = s_back + t_face - s_face
            error = min(left, right)
            if error < error_min:
                error_min = error
                polarity_min = -1 if left < right else 1
                threshold = together[j]

        features[ind] = tuple((add, sub, polarity_min, threshold, error_min))

    features.sort(key=lambda x: x[-1])
    return features


def construct_cascade():
    pass


@click.command()
@click.option("-f", "--faces", help="Path to directory containing face examples.", required=True)
@click.option("-b", "--background", help="Path to directory containing background examples.", required=True)
@click.option("-t", "--test", help="Test image.", default=None)
@click.option("-v", "--verbose", default=False, is_flag=True, help="Toggle for verbosity.")
def run(faces, background, test, verbose):
    if verbose:
        print("Importing face examples from: {} ...".format(faces))
    faces = integral_image(import_img_dir(faces))

    if verbose:
        print("Importing background examples from: {} ...\n".format(background))
    background = integral_image(import_img_dir(background))

    features = generate_features(64, 64, stride=8, verbose=verbose)

    if verbose:
        print("Training {} features...".format(len(features)))
    faces_dist_init = np.full((faces.shape[0]), 1 / (faces.shape[0] + background.shape[0]))
    background_dist_init = np.full((background.shape[0]), 1 / (faces.shape[0] + background.shape[0]))
    trained_features = train_features(features, faces, background, faces_dist_init, background_dist_init)

    from pprint import PrettyPrinter as P
    p = P(indent=4)
    # p.pprint(trained_features[:10])

    # p.pprint([i for i in trained_features if i[-3] == 1][:10])

    # if test:
    #     print(import_jpg(test).shape)






if __name__ == "__main__":
    run()
