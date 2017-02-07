'''
Author: Abner Ayala-Acevedo

This script based on examples provided in the keras documentation and a blog.
"Building powerful image classification models using very little data"
from blog.keras.io.

Dataset: Subset of Kaggle Dataset
https://www.kaggle.com/c/dogs-vs-cats/data
- cat pictures index 0-999 in data/train/cats
- cat pictures index 1000-1400 in data/validation/cats
- dogs pictures index 0-999 in data/train/dogs
- dog pictures index 1000-1400 in data/validation/dogs

Example: Dogs vs Cats (Directory Structure)
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...

Example has 1000 training examples for each class, and 400 validation examples for each class.
The data folder already contains the dogs vs cat data you simply need to run script. For the dogs_cats classification
you can find a model already trained in the model folder. Feel free to create your own data.
'''

import sys
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.applications import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
from keras.models import Model

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# parameters dependent on your dataset: modified to your example
nb_train_samples = 2000  # Total number of train samples. NOT including augmented images
nb_validation_samples = 800  # Total number of train samples. NOT including augmented images.
nb_classes = 2  # number of classes
img_width, img_height = 512, 512  # change based on the shape/structure of your images

# hyper parameters for model
based_model_last_block_layer_number = 172  # value is based on based model selected.
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum


def train(train_data_dir, validation_data_dir, model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    base_model = InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

    # Top Model Block
    x = base_model.output
    x = BatchNormalization(axis=3)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze: this is used to define based_model_last_block_layer_number
    for i, layer in enumerate(model.layers):  # comment these two lines once the correct based_model_last_block_layer
        print(i, layer.name)  # has been selected

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolution layers
    for layer in base_model.layers:
        layer.trainable = False

    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       shear_range=.2,
                                       zoom_range=.2,
                                       cval=.2,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
                                                        # save_to_dir=data_dir + '/preview',
                                                        # save_prefix='aug',
                                                        # save_format='jpeg')
    # use the above 3 commented lines if you want to save and look at how the data augmentations look like

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                            target_size=(img_height, img_width),
                                                            batch_size=batch_size,
                                                            class_mode='categorical')

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc
    top_weights_path = model_path + 'top_model_weights.h5'
    callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ]

    # Train Simple CNN
    model.fit_generator(train_generator,
                        samples_per_epoch=nb_train_samples,
                        nb_epoch=nb_epoch / 5,
                        validation_data=validation_generator,
                        nb_val_samples=nb_validation_samples,
                        callbacks=callbacks_list)

    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch is selected and not the last one.
    model.load_weights(top_weights_path)

    # verbose
    print("\nStarting to Fine Tune Model\n")

    # based_model_last_block_layer_number points to the layer in your model you want to train.
    # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will used the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.
    for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(optimizer=SGD(lr=learn_rate, momentum=momentum),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc
    final_weights_path = model_path + 'model_weights.h5'
    callbacks_list = [
        ModelCheckpoint(final_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ]

    # fine-tune the model
    model.fit_generator(train_generator,
                        samples_per_epoch=nb_train_samples,
                        nb_epoch=nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=nb_validation_samples,
                        callbacks=callbacks_list)

    # save model
    model_json = model.to_json()
    with open(model_path + 'model.json', 'w') as json_file:
        json_file.write(model_json)


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print('Arguments must match:\npython code/fine_tune.py <data_dir/> <model_dir/>')
        print('Example: python code/fine_tune.py data/dogs_cats/ model/dog_cats/')
        sys.exit(2)
    else:
        data_dir = sys.argv[1]
        train_dir = data_dir + '/train'  # change to your train path. Inside, each class should have it's own folder
        validation_dir = data_dir + '/validation'  # validation path. Inside, each class should have it's own folder
        model_dir = sys.argv[2]
        
    train(train_dir, validation_dir, model_dir)  # train model

    # release memory
    k.clear_session()
