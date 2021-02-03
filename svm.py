import os


import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


from definitions import bacteria_list
import utils


np.random.seed(2021)

dataset_folder = os.path.join(os.getcwd(), 'dataset', 'bacteria')
output_folder = os.path.join(os.getcwd(), 'output', 'svm')

X = np.load(utils.load_dataset('X_reference.npy', dataset_folder))
y = np.load(utils.load_dataset('y_reference.npy', dataset_folder))

X_test = np.load(utils.load_dataset('X_test.npy', dataset_folder))
y_test = np.load(utils.load_dataset('y_test.npy', dataset_folder))

print('Input data:')
print(' - X shape: {}\n - Y shape: {}'.format(X.shape, y.shape))
print(' - X test shape: {}\n - Y test shape: {}'.format(X_test.shape, y_test.shape))

X, y = utils.extract_subset_by_classes(X, y, bacteria_list.keys())
X_test, y_test = utils.extract_subset_by_classes(X_test, y_test, bacteria_list.keys())

print('Valid data:')
print(' - X shape: {}\n - Y shape: {}'.format(X.shape, y.shape))
print(' - X test shape: {}\n - Y test shape: {}'.format(X_test.shape, y_test.shape))

num_classes = len(set(y))
print()

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

print(' - Best parameters set found on development set:')
print(grid_search.best_score_, grid_search.best_params_)

print('\n - Grid scores on development set:')
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print('{:0.3f} (+/-{:0.03}) for {}'.format(mean, std * 2, params))

print('\n - Fitting SVM with best parameters from grid search')
classifier = SVC()
classifier.set_params(**grid_search.best_params_)
classifier.fit(X, y)

print('\n - Predicting')
y_predicted = classifier.predict(X_test)

scores = {}
scores['Accuracy'] = accuracy_score(y_test, y_predicted)
scores['Precision'] = precision_score(y_test, y_predicted, average='macro')
scores['Recall'] = recall_score(y_test, y_predicted, average='macro')
scores['F1'] = f1_score(y_test, y_predicted, average='macro')
print(scores)

print('\n - Detailed classification report:')
detailed_report = classification_report(y_test, y_predicted)

print(detailed_report, end='\n\n')
with open(os.path.join(output_folder, 'classification_report.txt'), 'w') as out:
    out.writelines(detailed_report)

print(' - Generating confusion matrix...')
utils.plot_confusion_matrix(
    y_test,
    y_predicted,
    labels=bacteria_list.values(),
    output=os.path.join(output_folder, 'confusion_matrix.pdf')
)
