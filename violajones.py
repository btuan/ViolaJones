""" violajones.py


Author: Brian Tuan
Last Modified: February 13, 2017

"""

import click
from multiprocessing import cpu_count, Pool
import numpy as np
import json

from util import import_img_dir, import_jpg, integral_image, draw_bounding_boxes


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


def gen_type1(width, height, stride=1, increment=1):
    """ Type 1 filters must have an even height.
    Reference points:
        D   C
        E   B
        F   A

    Formula: A + C + 2E - 2B - D - F
    """
    features = []
    for w in range(1, width, 1 * increment):
        for h in range(2, height, 2 * increment):
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


def gen_type2(width, height, stride=1, increment=1):
    """ Type 2 filters must have even width.
    Reference points:
        D   E   F
        C   B   A

    Formula: A + C + 2E - 2B - D - F
    """
    features = []
    for w in range(2, width, 2 * increment):
        for h in range(1, height, 1 * increment):
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


def gen_type3(width, height, stride=1, increment=1):
    """ Type 3 filters must have even width and even height.
    Reference points:
        I   H   G
        F   E   D
        C   B   A

    Formula: A + C + 4E + G + I - 2B - 2D - 2F - 2H
    """
    features = []
    for w in range(2, width, 2 * increment):
        for h in range(2, height, 2 * increment):
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


def gen_type4(width, height, stride=1, increment=1):
    """ Type 4 filters must have a width that is a multiple of 3.
    Reference points:
        E   F   G   H
        D   C   B   A

    Formula: A + 2C + E + 2G - 2B - D - 2F - H
    """
    features = []
    for w in range(3, width, 3 * increment):
        for h in range(1, height, 1 * increment):
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


def generate_features(width, height, stride=1, increment=1, verbose=False):
    """ Generate features based on integral image representation.
    Each feature is represented by points to add, points to subtract, polarity, and threshold.
    """

    features = []

    if verbose:
        print("Generating type 1 features...")
    features.extend(gen_type1(width, height, stride=stride, increment=increment))

    if verbose:
        print("Generating type 2 features...")
    features.extend(gen_type2(width, height, stride=stride, increment=increment))

    if verbose:
        print("Generating type 3 features...")
    features.extend(gen_type3(width, height, stride=stride, increment=increment))

    if verbose:
        print("Generating type 4 features...\n")
    features.extend(gen_type4(width, height, stride=stride, increment=increment))

    return features

"""
FEATURE EVALUATION AND PROCESSING FUNCTIONS
===========================================
"""


def eval_feature(feature, data):
    add, sub = feature[0], feature[1]
    to_add = data[:, [x[0] for x in add], [y[1] for y in add]].sum(axis=-1)
    to_sub = data[:, [x[0] for x in sub], [y[1] for y in sub]].sum(axis=-1)
    return to_add - to_sub


def _train_features(features, together, indicator, t_face, t_back, verbose=False):
    """ Train polarity and threshold for each feature on the test set. """
    for ind, feature in enumerate(features):
        if verbose and ind % 1000 == 0:
            print('\rTrained {} features...'.format(ind), end='')
        add, sub = feature[0], feature[1]

        scores = eval_feature(feature, together)
        sort_perm = scores.argsort()
        scores, indicator_perm = scores[sort_perm], indicator[sort_perm]

        s_face = 0 if indicator_perm[0] < 0 else indicator_perm[0]
        s_back = 0 if indicator_perm[0] >= 0 else -1 * indicator_perm[0]
        error_min = min(s_face + t_back - s_back, s_back + t_face - s_face)
        polarity_min = +1 if error_min == s_face + t_back - s_back else -1
        threshold = scores[0]
        for j in range(1, scores.shape[0]):
            if indicator_perm[j] < 0:
                s_back -= indicator_perm[j]
            else:
                s_face += indicator_perm[j]

            left = s_face + t_back - s_back
            right = s_back + t_face - s_face
            try:
                assert(left >= 0)
                assert(right >= 0)
            except AssertionError as e:
                print(e)
                print(t_face, s_face)
                print(t_back, s_back)
                print(t_face + t_back, left, right)

            error = min(left, right)
            if error < error_min:
                error_min = error
                polarity_min = +1 if left < right else -1
                threshold = scores[j]

        features[ind] = tuple((add, sub, polarity_min, threshold, abs(error_min)))

    if verbose:
        print('\rFinished training {} features.'.format(len(features)))

    return features


def train_features(features, faces, background, faces_dist, background_dist, threadpool, verbose=False):
    norm = faces_dist.sum() + background_dist.sum()
    faces_dist /= norm
    background_dist /= norm
    t_face, t_back = faces_dist.sum(), background_dist.sum()
    together = np.concatenate((faces, background))
    indicator = np.concatenate((faces_dist, -1 * background_dist))

    NUM_PROCS = cpu_count() * 3
    args = []
    chunk = len(features) // NUM_PROCS
    for cpu in range(cpu_count()):
        if cpu + 1 == NUM_PROCS:
            args.append((
                features[cpu * chunk:], together, indicator, t_face, t_back, False
            ))
        else:
            args.append((
                features[cpu * chunk: (cpu + 1) * chunk], together, indicator, t_face,
                t_back, False
            ))

    result = [y for x in threadpool.starmap_async(_train_features, args).get() for y in x]
    result.sort(key=lambda x: x[-1])
    return result


"""
ADABOOST ENSEMBLE FUNCTIONS
"""


def calculate_ensemble_error(classifiers, alphas, threshold, faces, background):
    face_scores = np.zeros(faces.shape[0])
    background_scores = np.zeros(background.shape[0])
    for ind, classifier in enumerate(classifiers):
        _, _, polarity, theta, _ = classifier
        face_scores += alphas[ind] * np.sign(polarity * (eval_feature(classifier, faces) - theta))
        background_scores += alphas[ind] * np.sign(polarity * (eval_feature(classifier, background) - theta))

    face_scores -= threshold
    background_scores -= threshold

    false_negatives = face_scores < 0
    false_positives = background_scores >= 0

    error = (false_negatives.sum() + false_positives.sum()) / (faces.shape[0] + background.shape[0] + 1e-100)
    return error, face_scores, background_scores, false_negatives, false_positives


def construct_boosted_classifier(features, faces, background, threadpool, target_false_pos_rate=0.3, verbose=False):
    eps = 1E-100
    classifiers, alphas = [], []

    faces_dist = np.full((faces.shape[0]), 1 / (faces.shape[0] + background.shape[0]))
    background_dist = np.full((background.shape[0]), 1 / (faces.shape[0] + background.shape[0]))

    while True:
        # Take classifier with minimum error on the distribution.
        add, sub, polarity, theta, err = train_features(
            features, faces, background, faces_dist, background_dist, threadpool
        )[0]
        if verbose:
            print("\nSelected", add, sub, polarity, theta, err)

        err += eps
        classifiers.append((add, sub, polarity, theta, err))
        alphas.append((1 / 2) * np.log((1 - err) / err))
        zt = 2 * np.sqrt(err * (1 - err))

        ht = np.sign(polarity * (eval_feature((add, sub), faces) - theta) + eps)
        faces_dist = (faces_dist / zt) * np.exp(-1 * alphas[-1] * ht)
        # faces_dist *= (err / (1 - err)) ** (1 * (ht >= 0))

        ht = np.sign(polarity * (eval_feature((add, sub), background) - theta) + eps)
        background_dist = (background_dist / zt) * np.exp(+1 * alphas[-1] * ht)
        # background_dist *= (err / (1 - err)) ** (1 * (ht < 0))

        # _, face_scores, _, false_negatives, _ = calculate_ensemble_error(classifiers, alphas, 0, faces, background)
        # threshold = np.amin(face_scores[false_negatives]) if false_negatives.sum() > 0 else 0
        # Calculate threshold
        # TODO: double check that this is correct
        face_scores = np.zeros(faces.shape[0])
        for ind, classifier in enumerate(classifiers):
            _, _, polarity, theta, _ = classifier
            face_scores += alphas[ind] * np.sign(polarity * (eval_feature(classifier, faces) - theta))
        threshold = np.amin(face_scores)

        error, _, _, _, false_positives = calculate_ensemble_error(classifiers, alphas, threshold, faces, background)
        false_positive_rate = false_positives.sum() / background.shape[0]

        if verbose:
            print("Boosted classifier has {} features with ensemble false positive rate {:0.5f} and error {:0.5f}.".
                  format(len(classifiers), false_positive_rate, error)
                  )
        if false_positive_rate < target_false_pos_rate:
            break

    return classifiers, alphas, threshold, error


"""
CASCADE FUNCTIONS
"""


def evaluate_cascade_error(cascade, faces, background, verbose=False):
    for step in cascade:
        classifiers, alphas, threshold = step

        error, _, _, false_negatives, false_positives = calculate_ensemble_error(
            classifiers, alphas, threshold, faces, background
        )

        faces = faces[~false_negatives]
        background = background[false_positives]

    return faces, background


def construct_classifier_cascade(features, faces, background, verbose=False):
    if verbose:
        print("Training {} features...".format(len(features)))

    NUM_PROCS = cpu_count() * 3
    pool = Pool(processes=NUM_PROCS)
    cascade = []
    num_initial_background = background.shape[0]

    # while True:
    for _ in range(10):
        if verbose:
            print("\nBOOSTING ROUND {}".format(len(cascade) + 1))
            print("================")
        classifiers, alphas, threshold, error = construct_boosted_classifier(
            features, faces, background, pool, target_false_pos_rate=0.4, verbose=verbose
        )

        cascade.append((classifiers, alphas, threshold))
        faces, background = evaluate_cascade_error(cascade, faces, background, verbose=verbose)
        if verbose:
            print("Boosting concluded with {} classifiers and remaining background proportion: {:0.5f}".format(
                len(classifiers), background.shape[0] / num_initial_background
            ))
            print("================")
        # if background.shape[0] / num_initial_background < 0.01:
        #     break

    return cascade


def get_cascade_prediction(cascade, integral_images, face_indices, verbose=False):
    if verbose:
        print("Evaluating cascade in {} image patches.".format(integral_images.shape[0]))

    for ind, step in enumerate(cascade):
        classifiers, alphas, theta = step
        threshold = sum(alphas)
        _, scores, _, negatives, _ = calculate_ensemble_error(
            classifiers, alphas, +0.25 * threshold, integral_image(integral_images), integral_images[:1]
            # classifiers, alphas, +0.375 * threshold, integral_images, integral_images[:1]
            # classifiers, alphas, +0.4627 * threshold, integral_images, integral_images[:1]
            # classifiers, alphas, +0.35 * threshold, integral_images, integral_images[:1] # For stringent
            # classifiers, alphas, 0, integral_images, integral_images[:1] # Default case
        )
        integral_images, face_indices = integral_images[~negatives], face_indices[~negatives]

        if verbose:
            print("After {} cascade steps, {} potential faces.".format(ind + 1, integral_images.shape[0]))

    return face_indices


@click.command()
@click.option("-f", "--faces", help="Path to directory containing face examples.", required=True)
@click.option("-b", "--background", help="Path to directory containing background examples.", required=True)
@click.option("-l", "--load", help="Load saved cascade configuration.", default=None)
@click.option("-t", "--test", help="Test image.", default=None)
@click.option("-v", "--verbose", default=False, is_flag=True, help="Toggle for verbosity.")
def run(faces, background, load, test, verbose):
    if load is None:
        stride = 2
        increment = 4
        if verbose:
            print("Importing face examples from: {} ...".format(faces))
        faces = integral_image(import_img_dir(faces))

        if verbose:
            print("Importing background examples from: {} ...\n".format(background))
        background = integral_image(import_img_dir(background))

        features = generate_features(64, 64, stride=stride, increment=increment, verbose=verbose)
        cascade = construct_classifier_cascade(features, faces, background, verbose=verbose)

        with open('cascade_save.json', 'w') as f:
            json.dump(cascade, f)
    else:
        stride = 3
        with open(load, 'r') as f:
            cascade = json.load(f)

        original_image = import_jpg(test)
        # test_image = integral_image(original_image)
        bounding_boxes = []
        # integrated_image = np.empty_like(original_image)
        integral_images = []
        indices = []
        for x in range(0, original_image.shape[0] - 64, stride):
            for y in range(0, original_image.shape[1] - 64, stride):
                bounding_boxes.append((x, y))
                indices.append(len(indices))
                integral_images.append(original_image[x: x + 64, y: y + 64])
                # integral_images.append(integral_image(original_image[x: x + 64, y: y + 64]))

        face_indices = get_cascade_prediction(cascade, np.array(integral_images), np.array(indices), verbose=verbose)
        draw_bounding_boxes(original_image, np.array(bounding_boxes)[face_indices], 64, 64)


if __name__ == "__main__":
    run()
