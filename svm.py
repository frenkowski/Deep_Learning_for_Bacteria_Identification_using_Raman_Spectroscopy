import os


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def plot_confusion_matrix(y_true, y_predicted, labels=[], ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 12))

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_predicted),
        display_labels=labels
    ).plot(include_values=True, cmap=plt.cm.Blues, ax=ax)


dataset_folder = os.path.join(os.getcwd(), 'dataset')
def load_dataset(filename):
    return os.path.join(dataset_folder, filename)


def extract_subset_by_classes(X_in, y_in, classes):
    X_out = []
    y_out = []

    for (value, label) in zip(X_in, y_in):
        if label in classes:
            X_out.append(value)
            y_out.append(label)

    return np.array(X_out), np.array(y_out)


X = np.load(load_dataset('X_finetune.npy'))
y = np.load(load_dataset('y_finetune.npy'))

X_test = np.load(load_dataset('X_test.npy'))
y_test = np.load(load_dataset('y_test.npy'))

print('X shape: {}\nY shape: {}'.format(X.shape, y.shape))
print('X test shape: {}\nY test shape: {}'.format(X_test.shape, y_test.shape))

valid_labels = [
    *range(14, 21+1),   # From MRSA 1 to S. lugdunensis
    *range(25, 29+1),   # From Group A Strep. to Group G Strep.
    6,                  # E. faecalis 1
    7,                  # E. faecalis 2
    19                  # S. enterica
]

X, y = extract_subset_by_classes(X, y, valid_labels)
X_test, y_test = extract_subset_by_classes(X_test, y_test, valid_labels)

print('X shape: {}\nY shape: {}'.format(X.shape, y.shape))
print('X test shape: {}\nY test shape: {}'.format(X_test.shape, y_test.shape))

y_indexes = {k: i for i, k in enumerate(set(y))}
y_test_indexes = {k: i for i, k in enumerate(set(y_test))}

num_classes = len(y_indexes)
input_shape = (X.shape[1], 1)

# ------------------
print('### SVM Model ###')
metrics = ['accuracy']
tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]

print('> Grid search:')
for metric in metrics:
    print(' - Tuning hyper-parameters for {} metric\n'.format(metric))
    grid_search = GridSearchCV(
        SVC(), tuned_parameters, scoring=metric
    )
    grid_search.fit(X, y)

    print(' - Best parameters set found on development set:')
    print(grid_search.best_params_)

    print('\n - Grid scores on development set:')
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print('{:0.3f} (+/-{:0.03}) for {}'.format(mean, std * 2, params))

    print('\n - Detailed classification report:')
    y_true, y_pred = y_test, grid_search.predict(X_test)
    print(classification_report(y_true, y_pred), end='\n\n')

classifier = SVC()
classifier.set_params(**grid_search.best_params_)
classifier.fit(X, y)

print(' > Predicting')
y_predicted = classifier.predict(X_test)
print(y_predicted)

scores = {}
scores['Accuracy'] = accuracy_score(y_test, y_predicted)
scores['Precision'] = precision_score(y_test, y_predicted, average='macro')
scores['Recall'] = recall_score(y_test, y_predicted, average='macro')
scores['F1'] = f1_score(y_test, y_predicted, average='macro')

print(scores)

plot_confusion_matrix(y_test, y_predicted, labels=y_test_indexes.keys())
plt.show()
