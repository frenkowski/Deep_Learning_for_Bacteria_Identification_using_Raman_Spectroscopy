import math
import os


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.python.keras.utils.np_utils import normalize


def load_dataset(filename, folder):
    return np.load(os.path.join(folder, filename))


def shuffle(X, y):
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], y[permutation]


def extract_subset_by_classes(X_in, y_in, classes):
    X_out, y_out = [], []

    for (value, label) in zip(X_in, y_in):
        if label in classes:
            X_out.append(value)
            y_out.append(label)

    return np.array(X_out), np.array(y_out)


def plot_confusion_matrix(y_true, y_predicted, labels=[], ax=None, output=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15, 12))

    cm = confusion_matrix(y_true, y_predicted)
    cm = 100 * cm / cm.sum(axis=1)[:, np.newaxis]

    # ax.xaxis.tick_top()

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    ).plot(include_values=True, cmap=plt.cm.Blues, values_format='0.0f', ax=ax, xticks_rotation='vertical')

    if output is not None:
        plt.savefig(output)
