""" adaboost.py

Author: Brian Tuan
Last Modified: February 13, 2017

"""

import numpy as np


def construct_boosted_classifier(classifiers, error_func, pos_set, neg_set, threshold=0):
    error = 0
    classifiers = {'classifiers': [], 'coefficients': [], 'threshold': 0}

    return classifiers, error



