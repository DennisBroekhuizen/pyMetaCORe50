"""
Author:         Dennis Broekhuizen, Tilburg University
Credits:        Giacomo Spigler, pyMeta: https://github.com/spiglerg/pyMeta
Description:    Train the CORe50 dataset in a meta-learning setting.
"""

from pyMeta.metalearners.reptile import ReptileMetaLearner
from pyMeta.metalearners.fomaml import FOMAMLMetaLearner
from pyMeta.metalearners.implicit_maml import iMAMLMetaLearner

# CORe50 specific imports
from pyMeta.core50.core50_from_npz import create_core50_from_npz_task_distribution
from pyMeta.core50.core50_network import make_core50_cnn_model

import sys, os
import time
import numpy as np
import tensorflow as tf

from absl import app, flags

# Force the batchnormalization layers to use statistics from the current minibatch only, instead of learnt accumulated
# statistics.
tf.keras.backend.set_learning_phase(1)

# Tensorflow 2.0 GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



FLAGS = flags.FLAGS

# Dataset and model options
flags.DEFINE_string('dataset', 'core50', 'Default is core50 in this example file.')
flags.DEFINE_string('metamodel', 'fomaml', 'fomaml or reptile or imaml')

flags.DEFINE_integer('num_output_classes', 5, 'Number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_train_samples_per_class', 5, 'Number of samples per class used in classification (e.g. 5-shot classification).')
flags.DEFINE_integer('num_test_samples_per_class', 15, 'Number of samples per class used in testing (e.g., evaluating a model trained on k-shots, on a different set of samples).')

# Meta-training options
flags.DEFINE_integer('num_outer_metatraining_iterations', 10000, 'Number of iterations in the outer (meta-training) loop.')
flags.DEFINE_integer('meta_batch_size', 5, 'Meta-batch size: number of tasks sampled at each meta-iteration.')
flags.DEFINE_float('meta_lr', 0.0001, 'Learning rate of the meta-optimizer ("outer" step size). Default 0.001 for FOMAML, 1.0 for Reptile') # 0.1 for omniglot

flags.DEFINE_integer('num_validation_batches', 10, 'Number of batches to sample from, and average over, when validating the performance of the model at regular intervals.')

# implicit-MAML (iMAML) specific options
flags.DEFINE_float('imaml_lambda_reg', 2.0, 'Value of lambda for the inner-loop L2 regularizer wrt to the initial parameters. Only used by iMAML. Original values are 2.0 for Omniglot and 0.5 for MiniImageNet.')
flags.DEFINE_integer('imaml_cg_steps', 5, 'Number of steps to run the iMAML optimizer for, in order to estimate the per-task meta-gradient. E.g., this usually refers to the number of iterations of Conjugate Gradient.')

# Inner-training options
flags.DEFINE_integer('num_inner_training_iterations', 10, 'Number of gradient descent steps to perform for each task in a meta-batch (inner steps).')
flags.DEFINE_integer('inner_batch_size', -1, 'Batch size: number of task-specific points sampled at each inner iteration. If <0, then it defaults to num_train_samples_per_class*num_output_classes.')
flags.DEFINE_float('inner_lr', 0.01, 'Learning rate of the inner optimizer. Default 0.01 for FOMAML, 1.0 for Reptile')

# Logging, saving, and testing options
flags.DEFINE_integer('save_every_k_iterations', 1000, 'The model is saved every k iterations.')
flags.DEFINE_integer('test_every_k_iterations', 100, 'The performance of the model is evaluated every k iterations.')
flags.DEFINE_string('model_save_filename', 'saved/model', 'Path + filename where to save the model to.')

flags.DEFINE_integer('seed', '100', 'random seed.')



def main(argv):
    if FLAGS.inner_batch_size < 0:
        FLAGS.inner_batch_size = FLAGS.num_train_samples_per_class * FLAGS.num_output_classes
    FLAGS.dataset.lower()
    FLAGS.metamodel.lower()

    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)


    def custom_sparse_categorical_cross_entropy_loss(y_true, y_pred):
        ## Implementation of sparse_categorial_cross_entropy_loss based on categorical_crossentropy,
        ## to work-around the limitation of the former when computing 2nd order derivatives (in the current
        ## Tensorflow implementation)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), FLAGS.num_output_classes)
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)


    # Create the dataset and network model
    if FLAGS.dataset == "core50":
        metatrain_task_distribution, metaval_task_distribution, metatest_tasks_distribution = \
                        create_core50_from_npz_task_distribution('datasets/core50/',
                        num_training_samples_per_class=FLAGS.num_train_samples_per_class,
                        num_test_samples_per_class=FLAGS.num_test_samples_per_class,
                        num_training_classes=FLAGS.num_output_classes,
                        meta_batch_size=FLAGS.meta_batch_size)

        model = make_core50_cnn_model(FLAGS.num_output_classes)
        optim = tf.keras.optimizers.SGD(lr=FLAGS.inner_lr)
        if FLAGS.metamodel == "reptile":
            optim = tf.keras.optimizers.Adam(lr=FLAGS.inner_lr, beta_1=0.0)
        loss_function = custom_sparse_categorical_cross_entropy_loss
        metrics = ['sparse_categorical_accuracy']

    else:
        print("ERROR: training task not recognized [", FLAGS.dataset, "]")
        sys.exit()


    # Setup the meta-learner
    if FLAGS.metamodel == 'reptile':
        optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.meta_lr)
        metalearner = ReptileMetaLearner(model=model,
                                         optimizer=optimizer,
                                         name="ReptileMetaLearner")

    elif FLAGS.metamodel == 'fomaml':
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.meta_lr)  # , beta_1=0.0)
        metalearner = FOMAMLMetaLearner(model=model,
                                        optimizer=optimizer,
                                        name="FOMAMLMetaLearner")
    elif FLAGS.metamodel == 'imaml':
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.meta_lr)  # , beta_1=0.0)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.meta_lr)
        metalearner = iMAMLMetaLearner(model=model,
                                      optimizer=optimizer,
                                      lambda_reg = FLAGS.imaml_lambda_reg, #0.5, #2.0,
                                      n_iters_optimizer = FLAGS.imaml_cg_steps,
                                      name="iMAMLMetaLearner")


    # The model should be compiled AFTER being wrapped by a meta-learner, as the meta-learner may add special ops
    # or regularizers to the model.
    model.compile(optimizer=optim,
                  loss=loss_function,
                  metrics=metrics)

    model.summary()
    print("Meta model: ", FLAGS.metamodel)
    print("Problem: ", FLAGS.dataset)
    print()
    print("Output classes:", FLAGS.num_output_classes)
    print("Train samples per class:", FLAGS.num_train_samples_per_class)
    print("Test samples per class:", FLAGS.num_test_samples_per_class)
    print()
    print("Outer metatraining iterations:", FLAGS.num_outer_metatraining_iterations)
    print("Meta batch size:", FLAGS.meta_batch_size)
    print("Meta learning rate:", FLAGS.meta_lr)
    print()
    print("Validation batches:", FLAGS.num_validation_batches)
    print()
    print("Inner training iterations:", FLAGS.num_inner_training_iterations)
    print("Inner batch size:", FLAGS.inner_batch_size)
    print("Inner learning rate:", FLAGS.inner_lr)



    metalearner.initialize()



    # Main meta-training loop: for each outer iteration, we will sample a number of training tasks, then train on each of
    # them (inner training loop) while recording their final test performance to track training. After all tasks in the
    # meta-batch have been observed, the model is updated in the outer loop, and we proceed to the next outer iteration.
    # Note that the focus is shifted on the outer training loop, with the inner one consisting of traditional
    # single-task training.
    last_time = time.time()
    for outer_iter in range(FLAGS.num_outer_metatraining_iterations+1):
        meta_batch = metatrain_task_distribution.sample_batch()

        # META-TRAINING over batch
        # TODO: inefficient; we are solving each task sequentially, when we should rather do it in parallel
        # However it may be better to do it this way for few-shot classification problems, where few inner iterations are
        # used.
        metabatch_results = []
        avg_loss_lastbatch = np.asarray([0.0, 0.0])
        for task in meta_batch:
            # Train on task for a number of num_inner_training_iterations iterations
            metalearner.task_begin(task)

            ret_info = task.fit_n_iterations(model, tf.constant(FLAGS.num_inner_training_iterations), tf.constant(FLAGS.inner_batch_size))

            if 'last_minibatch_loss' in ret_info:
                avg_loss_lastbatch += ret_info['last_minibatch_loss']

            metabatch_results.append(metalearner.task_end(task))

        # Update the meta-learner after all batch has been computed
        metalearner.update(metabatch_results)


        ## META-TESTING every `test_every_k_iterations' iterations
        if outer_iter % FLAGS.test_every_k_iterations == 0:
            # Evaluate the meta-learner on a set of the validation set
            print("Time: ", time.time()-last_time)

            val_task_loss = []
            val_task_accuracy = []
            for validation_iter in range(FLAGS.num_validation_batches):
                batch_validation = metaval_task_distribution.sample_batch()

                for task in batch_validation:
                    metalearner.task_begin(task)

                    task.fit_n_iterations(model, tf.constant(FLAGS.num_inner_training_iterations), tf.constant(FLAGS.inner_batch_size))
                    out_dict = task.evaluate(model)

                    val_task_loss.append(out_dict['loss'])
                    if 'sparse_categorical_accuracy' in out_dict:
                        val_task_accuracy.append(out_dict['sparse_categorical_accuracy'])

            print('Iter: ', outer_iter,
                  '\n\tavg final loss across validation tasks: ', np.mean(val_task_loss),
                  '\n\taverage test accuracy on validation tasks: ', np.mean(val_task_accuracy)*100.0, '%')
            # print('learning rate: ', model.optimizer.lr)
            last_time = time.time()

        if outer_iter % FLAGS.save_every_k_iterations == 0:
            metalearner.task_begin(meta_batch[0])  # copy back the initial parameters to the model's weights
            tf.saved_model.save(model, FLAGS.model_save_filename)


    if FLAGS.dataset == "sinusoid":
        # For sinusoid, plot the sine wave
        import matplotlib.pyplot as plt

        task = metaval_task_distribution.sample_batch()[0]
        metalearner.task_begin(task)

        test_X, test_y = task.get_test_set()
        preupdate_predicted_y = model.predict(test_X)

        task.fit_n_iterations(model, FLAGS.num_inner_training_iterations, FLAGS.inner_batch_size)

        # Evaluate performance on the test set of the task, without any more parameters updates
        predicted_y = model.predict(test_X)

        plt.plot(task.X, task.y, 'ok')
        plt.plot(task.test_X, task.test_y, 'k')
        plt.plot(task.test_X, predicted_y, 'r')
        plt.plot(task.test_X, preupdate_predicted_y, '--r')
        plt.show()


if __name__ == '__main__':
    app.run(main)
