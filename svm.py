import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from definitions import antibiotics, bacteria_antibiotics, bacteria_list
import processing


np.random.seed(2021)

dataset_folder = os.path.join(os.getcwd(), 'dataset', 'bacteria')
output_folder = os.path.join(os.getcwd(), 'output', 'svm')

X, y = processing.preprocess_dataset('finetune', dataset_folder, classes=bacteria_list.keys())

num_classes = len(set(y))
input_shape = (X.shape[1], 1)

X_test, y_test = processing.preprocess_dataset('test', dataset_folder, classes=bacteria_list.keys())

print(set(y_test))

# ------------------

print('### PCA ###')
pca = PCA(n_components=20)
pca.fit(X)

X = pca.transform(X)
X_test = pca.transform(X_test)

print('### SVM Model ###')
metric = 'accuracy'
tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]

print('> Grid search:')
print(' - Tuning hyper-parameters for \'{}\' metric\n'.format(metric))
grid_search = GridSearchCV(SVC(), tuned_parameters, cv=2, verbose=2, scoring=metric, refit=False)
grid_search.fit(X, y)

processing.grid_search_summary(grid_search)

print('\n - Fitting SVM with best parameters from grid search')
classifier = SVC()
classifier.set_params(**grid_search.best_params_)
classifier.fit(X, y)

print('\n> Predicting 15 class isolates')
y_predicted = list(map(int, classifier.predict(X_test)))
y_test = list(map(int, y_test))

processing.performance_summary(
    y_test,
    y_predicted,
    output_folder,
    y_mapping=lambda x: bacteria_list[x],
    y_labels=bacteria_list.values()
)

print('\n> Predicting antibiotic treatments')
antibiotic_predicted = list(map(lambda x: bacteria_antibiotics[x], y_predicted))
antibiotic_test = list(map(lambda x: bacteria_antibiotics[x], y_test))

processing.performance_summary(
    antibiotic_predicted,
    antibiotic_test,
    y_mapping=lambda x: antibiotics[x],
    y_labels=np.take(list(antibiotics.values()), list(set(antibiotic_test))),
    output=os.path.join(output_folder, 'antibiotic')
)
