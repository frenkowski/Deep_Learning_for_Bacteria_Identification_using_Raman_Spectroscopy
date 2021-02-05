import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.utils.vis_utils import plot_model
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from definitions import antibiotics, bacteria_antibiotics, bacteria_list
import processing


np.random.seed(2021)
tf.random.set_seed(2021)

dataset_folder = os.path.join(os.getcwd(), 'dataset', 'bacteria')
output_folder = os.path.join(os.getcwd(), 'output', 'cnn')

X, y, y_indices = processing.preprocess_dataset('finetune', dataset_folder,
    classes=bacteria_list.keys(),
    expand_dims=True,
    one_hot_encode=True
)

num_classes = len(y_indices)
input_shape = (X.shape[1], 1)

X_test, y_test, y_test_indices = processing.preprocess_dataset('test', dataset_folder,
    classes=bacteria_list.keys(),
    expand_dims=True,
    one_hot_encode=True
)
print()

# -------

print('### CNN Model (Transfer Learning) ###')

print('\n> Fine tuning best CNN model')
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, test_size=0.2)
print(' - X train shape: {}\n - Y train shape: {}'.format(X_train.shape, y_train.shape))
print(' - X val shape: {}\n - Y val shape: {}'.format(X_val.shape, y_val.shape))
print()

base_model = tf.keras.models.load_model(os.path.join(output_folder, 'model', 'model.h5'))
base_model.trainable = False
base_model.summary()

x = base_model.get_layer('flatten').output
x = layers.Dense(90, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(num_classes, activation='softmax')(x)

finetune_model = Model(inputs=base_model.input, outputs=x)
finetune_model.summary()

finetune_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

plot_model(finetune_model, to_file=os.path.join(output_folder, 'model', 'model.pdf'), show_shapes=True)

batch_size = 64
epochs = 50
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

print('\n> Predicting 15 class isolates')
y_predicted = np.argmax(finetune_model.predict(X_test), axis=-1)
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
