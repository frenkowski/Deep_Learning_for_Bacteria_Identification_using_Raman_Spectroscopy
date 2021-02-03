import os


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def load_dataset(filename, folder):
    return os.path.join(folder, filename)


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

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_predicted),
        display_labels=labels,
    ).plot(include_values=True, cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

    if output is not None:
        plt.savefig(output)
