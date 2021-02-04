import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model

from definitions import bacteria_list, antibiotics, bacteria_antibiotics
import processing


dataset_folder = os.path.join(os.getcwd(), 'dataset', 'bacteria')
output_folder = os.path.join(os.getcwd(), 'output', 'cnn')

X_test, y_test, y_test_indices = processing.preprocess_dataset('test', dataset_folder,
    classes=bacteria_list.keys(),
    expand_dims=True,
    one_hot_encode=True
)

model = tf.keras.models.load_model(os.path.join(output_folder, 'model', 'model.h5'))
model.summary()

print('\n - Predicting')
y_predicted = np.argmax(model.predict(X_test), axis=-1)
y_test = np.argmax(y_test, axis=-1)

# processing.performance_summary(
#     y_test,
#     y_predicted,
#     y_mapping=y_indices.values(),
#     y_labels=bacteria_list.values(),
#     output=output_folder
# )

antibiotic_predicted = list(map(lambda x: bacteria_antibiotics[x], y_predicted))
antibiotic_test = list(map(lambda x: bacteria_antibiotics[x], y_test))

processing.performance_summary(
    antibiotic_predicted,
    antibiotic_test,
    y_mapping=antibiotics.values(),
    y_labels=antibiotics.values(),
    output=output_folder
)
