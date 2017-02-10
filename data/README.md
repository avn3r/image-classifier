# Image Classification: Fine-Tune CNN Model via Transfer Learning
Althought model has been optimized for specific image classification task, this code can be used as a generic image classifier for any problem.

## Requirements:
This project was built using Ubuntu 16.10, Ananconda, Keras, and Tensorflow. The code has been tested with both CPU (64GB RAM computer) and GPU (2x Nvidia GeForce GTX 1080). To clone one of the two enviorments provided follow the instructions below.

1. Download [__Anaconda__](https://www.continuum.io/downloads)
2. Download [__Dataset AND Models__](https://my.pcloud.com/publink/show?code=VZvzMlZyPYO1hSn92LXdzmGNr9y1j7qKDzX): only the dogs vs. cats data is provided, since the text_images data is not public.
3. Clone Environment: (CPU): `conda env create -f cpu-environment.yml`
3. Clone Environment: (GPU): `conda env create -f gpu-environment.yml`

## Data: 
This empty directory is created to store the dataset if desired. Download the dataset from the requirements section and place it inside this folder. 
_Example:_ `data/dogs_cats/..`