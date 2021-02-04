import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model

from definitions import antibiotics, bacteria_antibiotics, bacteria_list
import processing


def build_model(conv_layers, filters, kernel_size, units, dropout_rate, optimizer, init_mode, regularizer_mode=None, **kwargs):
    model = models.Sequential()

    # Input
    model.add(keras.Input(shape=input_shape))

    # Conv
    for i in range(conv_layers):
        model.add(layers.Conv1D(filters * (i+1), kernel_size=kernel_size, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(2, strides=2, padding='same'))

    # FCN
    model.add(layers.Flatten())
    model.add(layers.Dense(units, activation='relu', kernel_initializer=init_mode, kernel_regularizer=regularizer_mode))
    model.add(layers.Dropout(dropout_rate))

    # Output
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
output_folder = os.path.join(os.getcwd(), 'output', 'cnn')

X, y, y_indices = processing.preprocess_dataset('reference', dataset_folder,
    classes=bacteria_list.keys(),
    expand_dims=True,
    one_hot_encode=True
)

num_classes = len(y_indices)
input_shape = (X.shape[1], 1)

X_finetune, y_finetune, y_finetune_indices = processing.preprocess_dataset('finetune', dataset_folder,
    classes=bacteria_list.keys(),
    expand_dims=True,
    one_hot_encode=True
)

X_test, y_test, y_test_indices = processing.preprocess_dataset('test', dataset_folder,
    classes=bacteria_list.keys(),
    expand_dims=True,
    one_hot_encode=True
)
print()

# -------

print('### CNN Model ###')

metric = 'accuracy'
tuned_parameters = {
    'epochs': [50],
    # 'batch_size': [32, 64],
    # 'conv_layers': [1, 2],
    # 'filters': [16, 32],
    # 'kernel_size': [3, 5],
    # 'units': [512, 1024],
    # 'dropout_rate': [0.3, 0.5],

    'batch_size': [64],
    'conv_layers': [3],
    'filters': [8],
    'kernel_size': [5],
    'units': [256],
    'dropout_rate': [0.2],
    'optimizer': ['adam'],
    'init_mode': ['glorot_uniform'],
}

print('> Grid search:')
print(' - Tuning hyper-parameters for \'{}\' metric\n'.format(metric))
grid_search = GridSearchCV(
    KerasClassifier(build_fn=build_model, verbose=0),
    tuned_parameters,
    cv=2,
    n_jobs=3,
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

# -------

print('\n> Fine tuning best CNN model from grid search')
X_train, X_val, y_train, y_val = train_test_split(X_finetune, y_finetune, shuffle=False)
print(' - X train shape: {}\n - Y train shape: {}'.format(X_train.shape, y_train.shape))
print(' - X val shape: {}\n - Y val shape: {}'.format(X_val.shape, y_val.shape))
print()

base_model = grid_search.best_estimator_.model
base_model.trainable = False
# base_model.summary()

x = base_model.get_layer('flatten').output
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(num_classes, activation='softmax')(x)

finetune_model = Model(inputs=base_model.input, outputs=x)
finetune_model.summary()

finetune_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

plot_model(finetune_model, to_file=os.path.join(output_folder, 'model', 'model.pdf'), show_shapes=True)

batch_size = grid_search.best_params_['batch_size']
epochs = grid_search.best_params_['epochs']
history = finetune_model.fit(
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

processing.save_history(finetune_model, history, output=output_folder)

print('\n> Predicting 30 class isolates')
y_predicted = np.argmax(finetune_model.predict(X_test), axis=-1)
y_test = np.argmax(y_test, axis=-1)

processing.performance_summary(
    y_test,
    y_predicted,
    y_mapping=y_indices.values(),
    y_labels=bacteria_list.values(),
    output=output_folder
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
