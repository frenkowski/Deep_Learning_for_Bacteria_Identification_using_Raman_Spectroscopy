import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from definitions import antibiotics, bacteria_antibiotics, bacteria_list
import processing


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

X, y, y_indices = processing.preprocess_dataset('finetune', dataset_folder,
    classes=bacteria_list.keys(),
    one_hot_encode=True
)

print(y_indices)

num_classes = len(y_indices)
input_shape = (X.shape[1], 1)

X_test, y_test, y_test_indices = processing.preprocess_dataset('test', dataset_folder,
    classes=bacteria_list.keys(),
    one_hot_encode=True
)

# -------

print('### MLP Model ###')

metric = 'accuracy'
tuned_parameters = {
    'epochs': [100],
    'batch_size': [64],
    'units': [256],
    'hidden_layers': [1],
    'dropout_rate': [0.5],
    'optimizer': ['adam'],
    'init_mode': ['glorot_uniform'],
}

print('> Grid search:')
print(' - Tuning hyper-parameters for \'{}\' metric\n'.format(metric))
grid_search = GridSearchCV(
    KerasClassifier(build_fn=build_model, verbose=0),
    tuned_parameters,
    cv=2,
    n_jobs=2,
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

processing.grid_search_summary(grid_search)

best_params = grid_search.best_params_

# -------

print('\n> Fitting MLP with best parameters from grid search')
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False)
print(' - X train shape: {}\n - Y train shape: {}'.format(X_train.shape, y_train.shape))
print(' - X val shape: {}\n - Y val shape: {}'.format(X_val.shape, y_val.shape))
print()

best_model = build_model(regularizer_mode=None, **best_params)
best_model.summary()
plot_model(best_model, to_file=os.path.join(output_folder, 'model', 'model.pdf'), show_shapes=True)

batch_size = best_params['batch_size']
epochs = best_params['epochs']
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

processing.save_history(best_model, history, output=output_folder)

print('\n> Predicting 15 class isolates')
y_predicted = np.argmax(best_model.predict(X_test), axis=-1)
y_test = np.argmax(y_test, axis=-1)

processing.performance_summary(
    y_test,
    y_predicted,
    y_mapping=lambda x: list(y_indices.values())[x],
    y_labels=bacteria_list.values(),
    output=output_folder
)

y_predicted = list(map(lambda x: list(y_indices.keys())[x], y_predicted))
y_test = list(map(lambda x: list(y_indices.keys())[x], y_test))

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
