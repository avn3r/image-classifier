'''
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
'''

import sys
import os
import csv
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
nb_test_samples = 12500  # Total number of test samples. change based on your data
img_width, img_height = 512, 512  # must match the fix size of your train image sizes. 600, 150 for text_images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).

# default paths
model_name = 'model.json'
model_weights = 'model_weights.h5'
results_name = 'predictions.csv'


def classify(trained_model_dir, test_data_dir, results_path):
    # load json and create model
    json_file = open(trained_model_dir + model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(trained_model_dir + model_weights)

    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # with open("results/dogs_cats/predictions.csv", "w") as file:
    #     csv_writer = csv.writer(file)
    #     csv_writer.writerow(('id', 'labels'))
    #     for _, _, images in os.walk(test_data_dir):
    #         for im in images:
    #             pic_id = im.split(".")[0]
    #             img = load_img(test_data_dir + im)
    #             img = imresize(img, size=(img_height, img_width))
    #             test_x = img_to_array(img).reshape(img_height, img_width, 3)
    #             test_x = test_x.reshape((1,) + test_x.shape)
    #             test_generator = test_datagen.flow(test_x,
    #                                                batch_size=1,
    #                                                shuffle=False)
    #             prediction = model.predict_generator(test_generator, 1)[0][0]
    #             csv_writer.writerow((pic_id, prediction))

    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                      target_size=(img_width, img_height),
                                                      batch_size=batch_size,
                                                      shuffle=False)
    # fine-tune the model
    y_probabilities = model.predict_generator(test_generator,
                                          val_samples=test_generator.nb_sample)
    y_classes = probas_to_classes(y_probabilities)
    ids = [x.split("/")[1].split('.')[0] for x in test_generator.filenames]

    with open(results_path + results_name, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(('id', 'class1_prob', 'label'))
        writer.writerows(zip(ids, y_probabilities[:, 1], y_classes))

    print(y_classes)

if __name__ == '__main__':
    if not len(sys.argv) == 4:
        print('Arguments must match:\npython code/classify.py <model_dir/> <test_dir/> <results_dir/>')
        print('Example: python code/classify.py model/dogs_cats data/dogs_cats/test/ results/dogs_cats/')
        sys.exit(2)
    else:
        model_dir = sys.argv[1]
        test_dir = sys.argv[2]
        results_dir = sys.argv[3]

    classify(model_dir, test_dir, results_dir)  # train model

    # release memory
    k.clear_session()
