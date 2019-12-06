"""
Author:         Dennis Broekhuizen, Tilburg University
Credits:        Giacomo Spigler, pyMeta: https://github.com/spiglerg/pyMeta
Description:    Make CORe50 specific classification tasks.
"""

from pyMeta.core.task import ClassificationTaskFromFiles

import numpy as np


class ClassificationTaskCORe50(ClassificationTaskFromFiles):
    def __init__(self, X, y, num_training_samples_per_class=-1, num_test_samples_per_class=-1, num_training_classes=-1,
                 split_train_test=0.8, input_parse_fn=None, background_labels=None):

        self.background_labels = background_labels

        super().__init__(X=X, y=y,
                         num_training_samples_per_class=num_training_samples_per_class,
                         num_test_samples_per_class=num_test_samples_per_class,
                         num_training_classes=num_training_classes, split_train_test=split_train_test, input_parse_fn=input_parse_fn)

    def reset(self):
        # Substract all unique classes.
        classes_to_use = list(set(self.y))

        # Check if train classes >= 1 and choose classes to use.
        if self.num_training_classes >= 1:
            classes_to_use = np.random.choice(classes_to_use, self.num_training_classes, replace=False)

        # Pre-define backgrounds to use for the test set.
        test_bkg = [3, 7, 10]
        test_bkg_indices = []
        for i in range(len(test_bkg)):
            test_bkg_indices.extend(np.where(self.background_labels==test_bkg[i])[0])
        test_bkg_indices = np.asarray(test_bkg_indices)

        # Pre-define backgrounds to use for the train set.
        train_bkg = [1, 2, 4, 5, 6, 8, 9, 11]
        train_bkg_indices = []
        for i in range(len(train_bkg)):
            train_bkg_indices.extend(np.where(self.background_labels==train_bkg[i])[0])
        train_bkg_indices = np.asarray(train_bkg_indices)

        # Define lists to save train and test samples (indices).
        self.train_indices = []
        self.test_indices = []

        # For each class, take list of indices, sample k.
        for c in classes_to_use:
            # List of class indices equal to current class in loop.
            class_indices = np.where(self.y==c)[0]

            # Check and use the intersection between current class and background indices.
            all_train_indices = list(set(list(train_bkg_indices)).intersection(set(list(class_indices))))
            all_test_indices = list(set(list(test_bkg_indices)).intersection(set(list(class_indices))))

            # Randomly choose train and test samples of current class.
            if self.num_training_samples_per_class >= 1:
                all_train_indices = np.random.choice(all_train_indices,
                                                     self.num_training_samples_per_class,
                                                     replace=False)
            if self.num_test_samples_per_class >= 1:
                all_test_indices = np.random.choice(all_test_indices,
                                                    self.num_test_samples_per_class,
                                                    replace=False)

            # Add samples of class to list of all samples.
            self.train_indices.extend(all_train_indices)
            self.test_indices.extend(all_test_indices)

        # Randomly shuffle the final train and test sample lists.
        np.random.shuffle(self.train_indices)
        np.random.shuffle(self.test_indices)

        # Rename train and test indices so that they run in [0, num_of_classes).
        self.classes_ids = list(classes_to_use)
