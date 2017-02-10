"""
Author: Abner Ayala-Acevedo

Dataset: Kaggle Dataset Dogs vs Cats
https://www.kaggle.com/c/dogs-vs-cats/data
- test folder unlabelled data

Example: Dogs vs Cats (Directory Structure)
test_dir/
    test/
        001.jpg
        002.jpg
        ...
        cat001.jpg
        cat002.jpg
        ...

If you need using a multi-class classification model change binary_cross_entropy to categorical_cross_entropy
"""

import sys
import os
import csv
import glob
import shutil
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import probas_to_classes
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras.models import model_from_json

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# parameters dependent on your dataset: modified to your example
img_width, img_height = 299, 299  # must match the fix size of your train image sizes. 600, 150 for text_images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).

# default paths
model_name = 'model.json'
model_weights = 'top_model_weights.h5'
results_name = 'predictions.csv'


def classify(trained_model_dir, test_data_dir, results_path):
    # load json and create model
    json_file = open(os.path.join(trained_model_dir, model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(trained_model_dir, model_weights))

    # Read Data
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                      target_size=(img_width, img_height),
                                                      batch_size=batch_size,
                                                      shuffle=False)

    # Calculate class posteriors probabilities
    y_probabilities = model.predict_generator(test_generator,
                                              val_samples=test_generator.nb_sample)
    # Calculate class labels
    y_classes = probas_to_classes(y_probabilities)
    filenames = [filename.split('/')[1] for filename in test_generator.filenames]
    ids = [filename.split('.')[0] for filename in filenames]

    # save results as a csv file in the specified results directory
    with open(os.path.join(results_path, results_name), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(('id', 'class0_prob', 'class1_prob', 'label'))
        writer.writerows(zip(ids, y_probabilities[:, 0], y_probabilities[:, 1], y_classes))

    # # semi-supervise learning: save the classified test data in their respective train class folder to increase
    # # your training dataset to include training
    # validation_data_dir = os.path.join(test_data_dir, '../validation')
    # validation_datagen = ImageDataGenerator(rescale=1. / 255)
    # validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
    #                                                               target_size=(img_width, img_height),
    #                                                               batch_size=batch_size,
    #                                                               shuffle=False)
    # labels_dictionary = dict((value, key) for key, value in validation_generator.class_indices.items())
    # for i, filename in enumerate(glob.glob(os.path.join(test_data_dir, 'test', '*.*'))):
    #     if y_probabilities[i, y_classes[i]] > .9995:
    #         shutil.copy(filename, os.path.join(test_data_dir, '../train', labels_dictionary.get(y_classes[i])))


if __name__ == '__main__':
    if not len(sys.argv) == 4:
        print('Arguments must match:\npython code/classify.py <model_dir/> <test_dir/> <results_dir/>')
        print('Example: python code/classify.py model/dogs_cats data/dogs_cats/test/ results/dogs_cats/')
        sys.exit(2)
    else:
        model_dir = os.path.abspath(sys.argv[1])
        test_dir = os.path.abspath(sys.argv[2])
        results_dir = os.path.abspath(sys.argv[3])
        os.makedirs(results_dir, exist_ok=True)

    classify(model_dir, test_dir, results_dir)  # train model

    # release memory
    k.clear_session()
