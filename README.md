# Research-Deepfake

Dataset:

download below dataset and unzip to ./data/raw

Fake images: 1-million-fake-faces (https://www.kaggle.com/datasets/tunguz/1-million-fake-faces)

Real images: flickrfaceshq-dataset-ffhq (https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)

Source code:

face_resize.py: resize raw image to 256x256

face_crop.py: extract face images from raw image

feature.py:extract fd,mfs,lac,entropy,mean,std feature from face image

analyse.py:sampling from features for distribution analyzing

Execute demo:

cd ./scripts

sh pipline.sh
