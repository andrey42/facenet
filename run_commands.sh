#!/bin/bash

export PYTHONPATH=/work/facenet/src

python3 /work/facenet/src/validate_on_lfw.py ~/datasets/lfw/lfw_mtcnnpy_160 ~/facenet_models/20180402-114759 --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization

for N in {1..4}; do python3 /work/facenet/src/align/align_dataset_mtcnn.py ~/datasets/lfw/raw ~/datasets/lfw/lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done

python3 src/classifier.py TRAIN ~/datasets/lfw/lfw_mtcnnpy_160/ ~/facenet_models/20180402-114759/20180402-114759.pb ~/facenet_models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset

python3 src/classifier.py CLASSIFY ~/datasets/lfw/lfw_mtcnnpy_160 ~/facenet_models/20180402-114759/20180402-114759.pb ~/facenet_models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_datase

