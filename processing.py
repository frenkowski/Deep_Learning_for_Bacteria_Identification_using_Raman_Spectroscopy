import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow import keras
from tensorflow_docs import plots as tfplots

import utils


def preprocess_dataset(name, path, classes=None, shuffle=True, expand_dims=False, one_hot_encode=False, verbose=2):
    X = utils.load_dataset('X_{}.npy'.format(name), path)
    y = utils.load_dataset('y_{}.npy'.format(name), path)

    if verbose > 0:
        print('\n> Input ({}) data:'.format(name))
        print(' - X shape: {}\n - Y shape: {}'.format(X.shape, y.shape))

    X, y = utils.extract_subset_by_classes(X, y, classes)
    num_classes = len(set(y))
    if verbose > 1:
        print('> Extracted {} classes'.format(num_classes))

    if verbose > 0:
        print('\n> Filtered ({}) data:'.format(name))
        print(' - X shape: {}\n - Y shape: {}'.format(X.shape, y.shape))

    if shuffle:
        X, y = utils.shuffle(X, y)
        if verbose > 1:
            print('> Shuffling dataset')

    if expand_dims:
        # Reshape X to: (X.shape[0], X.shape[1], 1)
        X = np.expand_dims(X, axis=-1)
        if verbose > 1:
            print('> Expanding dimensions')

    if not one_hot_encode:
        return [X, y]

    # One hot encoding procedure
    y_mapping = { k: i for i, k in enumerate(set(y)) }
    y = list(map(lambda x: y_mapping[x], y))
    y = keras.utils.to_categorical(y, num_classes)
    if verbose > 1:
        print('> Applying one-hot encoding to y values')

    return X, y, y_mapping


def grid_search_summary(grid):
    print('\n - Best parameters set found on development set:')
    print(grid.best_score_, grid.best_params_)

    print('\n - Grid scores on development set:')
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print('{:0.3f} (+/-{:0.03}) for {}'.format(mean, std * 2, params))


def save_history(model, history, output, persist_model=True):
    if persist_model:
        print('\n - Saving model and weights to file')
        model.save(os.path.join(output, 'model', 'model.h5'))
        model.save_weights(os.path.join(output, 'model', 'model_weights.h5'))

    print(' - Saving network training plots')
    histories = {'': history}
    accuracy_plotter = tfplots.HistoryPlotter(metric='accuracy', smoothing_std=0)
    loss_plotter = tfplots.HistoryPlotter(metric='loss', smoothing_std=0)

    plt.figure(figsize=(15, 10))
    accuracy_plotter.plot(histories)
    plt.savefig(os.path.join(output, 'accuracy_plot.pdf'))

    plt.figure(figsize=(15, 10))
    loss_plotter.plot(histories)
    plt.savefig(os.path.join(output, 'loss_plot.pdf'))


def performance_summary(y_test, y_predicted, output, y_mapping=None, y_labels=None):
    scores = {}
    scores['Accuracy'] = accuracy_score(y_test, y_predicted)
    scores['Precision'] = precision_score(y_test, y_predicted, average='macro')
    scores['Recall'] = recall_score(y_test, y_predicted, average='macro')
    scores['F1'] = f1_score(y_test, y_predicted, average='macro')
    print(scores)

    print(' - Detailed classification report:')
    if y_mapping is not None:
        y_test = list(map(y_mapping, y_test))
        y_predicted = list(map(y_mapping, y_predicted))
    detailed_report = classification_report(y_test, y_predicted)

    print(detailed_report, end='\n')
    with open(os.path.join(output, 'classification_report.txt'), 'w') as out:
        out.writelines(detailed_report)

    print(' - Saving confusion matrix')
    utils.plot_confusion_matrix(
        y_test,
        y_predicted,
        labels=y_labels,
        output=os.path.join(output, 'confusion_matrix.pdf')
    )
