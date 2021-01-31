import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow_docs import plots as tfplots
from tensorflow.keras import layers, models


def plot_confusion_matrix(y_true, y_predicted, labels=[], ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 12))

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true.argmax(axis=1), y_predicted.argmax(axis=1)),
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

# Reshape X to: (X.shape[0], X.shape[1], 1)
# X = np.expand_dims(X, axis=-1)
# Reshape X_train to: (X.shape[0], X.shape[1], 1)
# X_test = np.expand_dims(X_test, axis=-1)

y = list(map(lambda x: y_indexes[x], y))
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

y_test = list(map(lambda x: y_test_indexes[x], y_test))
y_test = keras.utils.to_categorical(y_test, num_classes)

print('X train shape: {}\nY train shape: {}'.format(X_train.shape, y_train.shape))
print('X val shape: {}\nY val shape: {}'.format(X_val.shape, y_val.shape))

# -----------------------------

def build_model(units, hidden_layers, optimizer='adam', init_mode='glorot_normal', dropout_rate=0.3, regularizer_mode=None):
    model = models.Sequential([
        keras.Input(shape=input_shape),
        layers.Flatten(),
    ])

    # Variable number of hidden layers
    for i in range(hidden_layers):
        model.add(layers.Dense(units // (i+1), activation='relu', kernel_initializer=init_mode, kernel_regularizer=regularizer_mode))
        model.add(layers.Dropout(dropout_rate))

    # Output
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

parameters_range = {
    'epochs': [50],
    'batch_size': [16, 32],
    'units': [256, 512],
    'hidden_layers': [1, 2],
    'optimizer': ['adam'],
    'init_mode': ['glorot_uniform'],
    'dropout_rate': [0.0, 0.3],
    'regularizer_mode':[None, (tf.keras.regularizers.l2(0.01))]
}

accuracy_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=5,
    min_delta=0.01,
    restore_best_weights=True
)

grid_search = GridSearchCV(
    estimator=KerasClassifier(build_fn=build_model, verbose=0),
    param_grid=parameters_range,
    cv=3,
    n_jobs=-1,
    verbose=1
)

model = grid_search.fit(X_train, y_train, callbacks=[accuracy_early_stop])
print("Best: %f using %s" % (model.best_score_, model.best_params_))

means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']
params = model.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# epochs = 50
# batch_size = 16

# model.compile(
#     optimizer=model_optimizer,
#     loss=model_loss,
#     metrics=['accuracy', *additional_metrics]
# )

# history = model.fit(
#     X_train,
#     y_train,
#     epochs=epochs,
#     batch_size=batch_size,
#     validation_data=(X_val, y_val),
#     callbacks=[accuracy_early_stop]
# )

# acc_plotter = tfplots.HistoryPlotter(metric='accuracy', smoothing_std=0)

# histories = {
#     'Conv1D/MaxPool1D': history
# }

# plt.figure(figsize=(15, 10))
# acc_plotter.plot(histories)

y_predicted = model.predict(X_test)

scores = {}
scores['Accuracy'] = accuracy_score(y_test.argmax(axis=-1), y_predicted.argmax(axis=-1))
scores['Precision'] = precision_score(y_test.argmax(axis=-1), y_predicted.argmax(axis=-1), average='macro')
scores['Recall'] = recall_score(y_test.argmax(axis=-1), y_predicted.argmax(axis=-1), average='macro')
scores['F1'] = f1_score(y_test.argmax(axis=-1), y_predicted.argmax(axis=-1), average='macro')

print(scores)

plot_confusion_matrix(y_test, y_predicted, labels=y_test_indexes.keys())
plt.show()
