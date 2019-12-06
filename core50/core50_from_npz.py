"""
Author:         Dennis Broekhuizen, Tilburg University
Credits:        Giacomo Spigler, pyMeta: https://github.com/spiglerg/pyMeta
Description:    Load the CORe50 dataset into memory and create a distribution
                of ClassificationTasks.
"""

import pickle as pkl
import os
import numpy as np
import tensorflow as tf
import cv2

from pyMeta.core50.core50_task import ClassificationTaskCORe50
from pyMeta.core.task_distribution import TaskDistribution

core50_images = []

def load_npz_file(path_to_dataset):
    """
    Loads the CORe50 npz file into memory.
    Requirements:
      - Path to folder with CORe50 files: paths.pkl and core50_imgs.npz
    """
    # Load the paths file.
    pkl_file = open(path_to_dataset+'paths.pkl', 'rb')
    paths = pkl.load(pkl_file)

    # Load the image file.
    imgs = np.load(path_to_dataset+'core50_imgs.npz')['x']

    return imgs, paths


def process_npz_img(npz_index):
    return np.asarray(core50_images[npz_index], dtype=np.float32)


def create_core50_from_npz_task_distribution(path_to_dataset,
                                             batch_size=32,
                                             num_training_samples_per_class=10,
                                             num_test_samples_per_class=-1,
                                             num_training_classes=20,
                                             meta_batch_size=5):

    imgs, paths = load_npz_file(path_to_dataset)

    global core50_images
    core50_images = imgs

    def get_session_objects(session_num, path_file):
        session_indexes = []
        session_labels = []
        for index, path in enumerate(path_file):
            splitted_path = path.split('/')
            if splitted_path[0] == 's'+str(session_num):
                for i in range(1, 51):
                    if splitted_path[1] == 'o'+str(i):
                        session_indexes.append(index)
                        session_labels.append(i)
        return session_indexes, session_labels


    def dataset_from_npz(session_nums, path_file):
        # Object index numbers in npz file.
        X_indexes = []

        # Object labels.
        y = []

        # Background (session) labels.
        b = []

        for session_num in session_nums:
            session_indexes, session_labels = get_session_objects(session_num, path_file)
            X_indexes.extend(session_indexes)
            y.extend(session_labels)
            for i in range(len(session_indexes)):
                b.append(session_num)

        X_indexes = np.asarray(X_indexes, dtype=np.int32)
        y = np.asarray(y, dtype=np.int32)
        b = np.asarray(b, dtype=np.int32)

        return X_indexes, y, b


    # Pre-define backround sessions to use.
    all_sessions = []
    for i in range(1, 12):
        all_sessions.append(i)

    X_indexes, y, b = dataset_from_npz(session_nums=all_sessions, path_file=paths)

    # Split indexes: first 40 objects train set & last 10 objects for test set.
    train_indexes = np.where(y<=40)[0]
    test_indexes = np.where(y>40)[0]

    # Split the dataset.
    trainX = X_indexes[train_indexes]
    trainY = y[train_indexes]
    trainB = b[train_indexes]

    testX = X_indexes[test_indexes]
    testY = y[test_indexes]
    testB = b[test_indexes]


    # Create ClassificationTask objects
    metatrain_tasks_list = [ClassificationTaskCORe50(trainX,
                                               trainY,
                                               num_training_samples_per_class,
                                               num_test_samples_per_class,
                                               num_training_classes,
                                               split_train_test=-1,
                                               input_parse_fn=process_npz_img, # defaults to num_train / (num_train+num_test)
                                               background_labels=trainB)]
    metatest_tasks_list = [ClassificationTaskCORe50(testX,
                                              testY,
                                              num_training_samples_per_class,
                                              num_test_samples_per_class,
                                              num_training_classes,
                                              split_train_test=-1,
                                              input_parse_fn=process_npz_img,
                                              background_labels=testB)]

    # Create TaskDistribution objects that wrap the ClassificationTask objects to produce meta-batches of tasks
    metatrain_task_distribution = TaskDistribution(tasks=metatrain_tasks_list,
                                                   task_probabilities=[1.0],
                                                   batch_size=meta_batch_size,
                                                   sample_with_replacement=True)

    metatest_task_distribution = TaskDistribution(tasks=metatest_tasks_list,
                                                  task_probabilities=[1.0],
                                                  batch_size=meta_batch_size,
                                                  sample_with_replacement=True)

    # TODO: split into validation and test!
    return metatrain_task_distribution, metatest_task_distribution, metatest_task_distribution
