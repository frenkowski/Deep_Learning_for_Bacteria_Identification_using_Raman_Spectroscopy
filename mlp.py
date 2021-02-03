import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow_docs import plots as tfplots
from tensorflow.keras import layers, models


from definitions import bacteria_list
import utils


def build_model(units, hidden_layers, dropout_rate, optimizer, init_mode, regularizer_mode=None, **kwargs):
    model = models.Sequential()

    model.add(keras.Input(shape=input_shape))
    model.add(layers.Flatten())

    # Variable number of hidden layers
    for i in range(hidden_layers):
        model.add(layers.Dense(units // (i+1), activation='relu', kernel_initializer=init_mode, kernel_regularizer=regularizer_mode))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


np.random.seed(2021)
tf.random.set_seed(2021)

dataset_folder = os.path.join(os.getcwd(), 'dataset', 'bacteria')
output_folder = os.path.join(os.getcwd(), 'output', 'mlp')

X = np.load(utils.load_dataset('X_finetune.npy', dataset_folder))
y = np.load(utils.load_dataset('y_finetune.npy', dataset_folder))

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
input_shape = (X.shape[1], 1)

X, y = utils.shuffle(X, y)

# Map y values (6, 7, 14, 15, ...) to ordinal index (0, 1, 2, 3, ..., 14)
y_indices = {k: i for i, k in enumerate(set(y))}
y_test_indices = {k: i for i, k in enumerate(set(y_test))}

y = list(map(lambda x: y_indices[x], y))
y = keras.utils.to_categorical(y, num_classes)

y_test = list(map(lambda x: y_test_indices[x], y_test))
y_test = keras.utils.to_categorical(y_test, num_classes)
print()

print('### MLP Model ###')
metric = 'accuracy'
tuned_parameters = {
    'epochs': [50],
    'batch_size': [16, 32],
    'units': [256, 512],
    'hidden_layers': [1, 2],
    'optimizer': ['adam'],
    'init_mode': ['glorot_uniform'],
    'dropout_rate': [0.1, 0.3]
}

print('> Grid search:')
print(' - Tuning hyper-parameters for \'{}\' metric\n'.format(metric))
grid_search = GridSearchCV(
    KerasClassifier(build_fn=build_model, verbose=0),
    tuned_parameters,
    cv=3,
    n_jobs=1,
    verbose=2
)

print(' - ', end='')
grid_search.fit(X, y, callbacks=[
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        min_delta=0.01,
        restore_best_weights=True
    )
])

print(' - Best parameters set found on development set:')
print(grid_search.best_score_, grid_search.best_params_)

print('\n - Grid scores on development set:')
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print('{:0.3f} (+/-{:0.03}) for {}'.format(mean, std * 2, params))

print('\n - Fitting MLP with best parameters from grid search')
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False)
print(' - X train shape: {}\n - Y train shape: {}'.format(X_train.shape, y_train.shape))
print(' - X val shape: {}\n - Y val shape: {}'.format(X_val.shape, y_val.shape))

best_model = build_model(regularizer_mode=tf.keras.regularizers.l2(0.001), **grid_search.best_params_)
best_model.summary()
plot_model(best_model, to_file=os.path.join(output_folder, 'model.pdf'), show_shapes=True)

batch_size = grid_search.best_params_['batch_size']
epochs = grid_search.best_params_['epochs']
history = best_model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.01,
            restore_best_weights=True
        )
    ]
)

print(' - Generating network training plots...')
histories = {'': history}
accuracy_plotter = tfplots.HistoryPlotter(metric='accuracy', smoothing_std=0)
loss_plotter = tfplots.HistoryPlotter(metric='loss', smoothing_std=0)

plt.figure(figsize=(15, 10))
accuracy_plotter.plot(histories)
plt.savefig(os.path.join(output_folder, 'accuracy_plot.pdf'))

plt.figure(figsize=(15, 10))
loss_plotter.plot(histories)
plt.savefig(os.path.join(output_folder, 'loss_plot.pdf'))

print(' - Predicting')
y_predicted = np.argmax(best_model.predict(X_test), axis=-1)
y_test = np.argmax(y_test, axis=-1)

scores = {}
scores['Accuracy'] = accuracy_score(y_test, y_predicted)
scores['Precision'] = precision_score(y_test, y_predicted, average='macro')
scores['Recall'] = recall_score(y_test, y_predicted, average='macro')
scores['F1'] = f1_score(y_test, y_predicted, average='macro')
print(scores)

print('\n - Detailed classification report:')
y_test = list(map(lambda x: list(y_test_indices.keys())[x], y_test))
y_predicted = list(map(lambda x: list(y_test_indices.keys())[x], y_predicted))
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
