"""
Training script for split CUB experiment with zero shot transfer.
"""
from __future__ import print_function

import argparse
import os
import sys
import math

import datetime
import numpy as np
import tensorflow as tf
from copy import deepcopy
from six.moves import cPickle as pickle

from utils.data_utils import image_scaling, random_crop_and_pad_image, random_horizontal_flip, construct_split_cub
from utils.utils import get_sample_weights, sample_from_dataset, concatenate_datasets, update_episodic_memory_with_less_data, samples_for_each_class, sample_from_dataset_icarl
from utils.vis_utils import plot_acc_multiple_runs, plot_histogram, snapshot_experiment_meta_data, snapshot_experiment_eval
from model import Model

###############################################################
################ Some definitions #############################
### These will be edited by the command line options ##########
###############################################################

## Training Options
NUM_RUNS = 5           # Number of experiments to average over
TRAIN_ITERS = 2000      # Number of training iterations per task
BATCH_SIZE = 16
LEARNING_RATE = 0.1    
RANDOM_SEED = 1234
VALID_OPTIMS = ['SGD', 'MOMENTUM', 'ADAM']
OPTIM = 'SGD'
OPT_MOMENTUM = 0.9
OPT_POWER = 0.9
VALID_ARCHS = ['CNN', 'VGG', 'RESNET']
ARCH = 'RESNET'
PRETRAIN = True

## Model options
MODELS = ['VAN', 'PI', 'EWC', 'MAS', 'GEM', 'RWALK', 'S-GEM'] # List of valid models 
IMP_METHOD = 'EWC'
SYNAP_STGTH = 75000
FISHER_EMA_DECAY = 0.9      # Exponential moving average decay factor for Fisher computation (online Fisher)
FISHER_UPDATE_AFTER = 50    # Number of training iterations for which the F_{\theta}^t is computed (see Eq. 10 in RWalk paper) 
TOTAL_EPISODIC_MEMORY = 1000    # Total episodic memory size
SAMPLES_PER_CLASS = 25   # Number of samples per task
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
TOTAL_CLASSES = 200          # Total number of classes in the dataset 
EPS_MEM_BATCH_SIZE = 32
DEBUG_EPISODIC_MEMORY = False
HERDING_BASED_SAMPLING = False

## Logging, saving and testing options
LOG_DIR = './split_cub_results'
SNAPSHOT_DIR = './cub_snapshots'
SAVE_MODEL_PARAMS = False
## Evaluation options

## Task split
NUM_TASKS = 20
MULTI_TASK = False

## Dataset specific options
ATTR_DIMS = 312
DATA_DIR='CUB_data/CUB_200_2011/images'
#CUB_TRAIN_LIST = 'dataset_lists/tmp_list.txt'
#CUB_TEST_LIST = 'dataset_lists/tmp_list.txt'
CUB_TRAIN_LIST = 'dataset_lists/CUB_train_list.txt'
CUB_TEST_LIST = 'dataset_lists/CUB_test_list.txt'
CUB_ATTR_LIST = 'dataset_lists/CUB_attr_in_order.pickle'
RESNET18_IMAGENET_CHECKPOINT = './resnet-18-pretrained-imagenet/model.ckpt'

# Define function to load/ store training weights. We will use ImageNet initialization later on
def save(saver, sess, logdir, step):
   '''Save weights.

   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
        saver: TensorFlow Saver object.
        sess: TensorFlow session.
        ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for split CUB hybrid experiment.")
    parser.add_argument("--cross-validate-mode", action="store_true",
            help="If option is chosen then enable the cross validation of the learning rate")
    parser.add_argument("--train-single-epoch", action="store_true", 
            help="If option is chosen then train for single epoch")
    parser.add_argument("--set-hybrid", action="store_true", 
            help="If option is chosen then train using hybrid model")
    parser.add_argument("--eval-single-head", action="store_true",
            help="If option is chosen then evaluate on a single head setting.")
    parser.add_argument("--arch", type=str, default=ARCH,
                        help="Network Architecture for the experiment.\
                                \n \nSupported values: %s"%(VALID_ARCHS))
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS,
                       help="Total runs/ experiments over which accuracy is averaged.")
    parser.add_argument("--train-iters", type=int, default=TRAIN_ITERS,
                       help="Number of training iterations for each task.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Mini-batch size for each task.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                       help="Starting Learning rate for each task.")
    parser.add_argument("--optim", type=str, default=OPTIM,
                        help="Optimizer for the experiment. \
                                \n \nSupported values: %s"%(VALID_OPTIMS))
    parser.add_argument("--imp-method", type=str, default=IMP_METHOD,
                       help="Model to be used for LLL. \
                        \n \nSupported values: %s"%(MODELS))
    parser.add_argument("--synap-stgth", type=float, default=SYNAP_STGTH,
                       help="Synaptic strength for the regularization.")
    parser.add_argument("--fisher-ema-decay", type=float, default=FISHER_EMA_DECAY,
                       help="Exponential moving average decay for Fisher calculation at each step.")
    parser.add_argument("--fisher-update-after", type=int, default=FISHER_UPDATE_AFTER,
                       help="Number of training iterations after which the Fisher will be updated.")
    parser.add_argument("--do-sampling", action="store_true",
                       help="Whether to do sampling")
    parser.add_argument("--mem-size", type=int, default=TOTAL_EPISODIC_MEMORY,
                       help="Number of samples per class from previous tasks.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                       help="Directory from where the CUB data will be read.\
                               NOTE: Provide path till <CUB_DIR>/images")
    parser.add_argument("--init-checkpoint", type=str, default=RESNET18_IMAGENET_CHECKPOINT,
                       help="TF checkpoint file containing initialization for ImageNet.\
                               NOTE: NPZ file for VGG and TF Checkpoint for ResNet")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                       help="Directory where the plots and model accuracies will be stored.")
    return parser.parse_args()

def train_task_sequence(model, sess, saver, datasets, class_attr, num_classes_per_task, task_labels, cross_validate_mode, train_single_epoch, eval_single_head, do_sampling, 
        episodic_mem_size, train_iters, batch_size, num_runs, init_checkpoint):
    """
    Train and evaluate LLL system such that we only see a example once
    Args:
    Returns:
        dict    A dictionary containing mean and stds for the experiment
    """
    # List to store accuracy for each run
    runs = []

    break_training = 0
    # Loop over number of runs to average over
    for runid in range(num_runs):
        print('\t\tRun %d:'%(runid))

        # Initialize all the variables in the model
        sess.run(tf.global_variables_initializer())

        if PRETRAIN:
            # Load the variables from a checkpoint
            if model.network_arch == 'RESNET':
                # Define loader (weights which will be loaded from a checkpoint)
                restore_vars = [v for v in model.trainable_vars if 'fc' not in v.name and 'attr_embed' not in v.name]
                loader = tf.train.Saver(restore_vars)
                load(loader, sess, init_checkpoint)
            elif model.network_arch == 'VGG':
                # Load the pretrained weights from the npz file
                weights = np.load(init_checkpoint)
                keys = sorted(weights.keys())
                for i, key in enumerate(keys[:-2]): # Load everything except the last layer
                    sess.run(model.trainable_vars[i].assign(weights[key]))

        # Run the init ops
        model.init_updates(sess)

        # List to store accuracies for a run
        evals = []

        # List to store the classes that we have so far - used at test time
        test_labels = []

        # Labels for all the tasks that we have seen in the past
        prev_task_labels = []

        if model.imp_method == 'GEM':
            # List to store the episodic memories of the previous tasks
            task_based_memory = []

        if model.imp_method == 'S-GEM':
            # Reserve a space for episodic memory
            episodic_images = np.zeros([episodic_mem_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
            episodic_labels = np.zeros([episodic_mem_size, TOTAL_CLASSES])

        if do_sampling:
            # List to store important samples from the previous tasks
            last_task_x = None
            last_task_y_ = None

        # Mask for the softmax
        logit_mask = np.zeros(TOTAL_CLASSES)

        # Training loop for all the tasks
        for task in range(len(datasets)):
            print('\t\tTask %d:'%(task))

            # If not the first task then restore weights from previous task
            if(task > 0):
                model.restore(sess)

            # If sampling flag is set append the previous datasets
            if(do_sampling and task > 0):
                task_train_images, task_train_labels = concatenate_datasets(datasets[task]['train']['images'], 
                                                                            datasets[task]['train']['labels'],
                                                                            last_task_x, last_task_y_)
            else:
                # Extract training images and labels for the current task
                task_train_images = datasets[task]['train']['images']
                task_train_labels = datasets[task]['train']['labels']

            # If multi_task is set then train using all the datasets of all the tasks
            if MULTI_TASK:
                if task == 0:
                    for t_ in range(1, len(datasets)):
                        task_train_images = np.concatenate((task_train_images, datasets[t_]['train']['images']), axis=0)
                        task_train_labels = np.concatenate((task_train_labels, datasets[t_]['train']['labels']), axis=0)

                else:
                    # Skip training for this task
                    continue

            print('Received {} images, {} labels at task {}'.format(task_train_images.shape[0], task_train_labels.shape[0], task))

            # Test for the tasks that we've seen so far
            test_labels += task_labels[task]

            # Declare variables to store sample importance if sampling flag is set
            if do_sampling:
                # Get the sample weighting
                task_sample_weights = get_sample_weights(task_train_labels, test_labels)
            else:
                # Assign equal weights to all the examples
                task_sample_weights = np.ones([task_train_labels.shape[0]], dtype=np.float32)

            num_train_examples = task_train_images.shape[0]

            # Train a task observing sequence of data
            logit_mask[:] = 0
            if train_single_epoch:
                # Ceiling operation
                num_iters = (num_train_examples + batch_size - 1) // batch_size
                if cross_validate_mode:
                    if do_sampling:
                        logit_mask[test_labels] = 1.0
                    else:
                        logit_mask[task_labels[task]] = 1.0
            else:
                num_iters = train_iters
                if do_sampling:
                    logit_mask[test_labels] = 1.0
                else:
                    logit_mask[task_labels[task]] = 1.0

            # Randomly suffle the training examples
            perm = np.arange(num_train_examples)
            np.random.shuffle(perm)
            train_x = task_train_images[perm]
            train_y = task_train_labels[perm]
            task_sample_weights = task_sample_weights[perm]

            # Array to store accuracies when training for task T
            ftask = []

            if MULTI_TASK:
                logit_mask[:] = 1.0
                masked_class_attrs = class_attr
            else:
                # Attribute mask
                masked_class_attrs = np.zeros_like(class_attr)
                attr_offset = task * num_classes_per_task
                masked_class_attrs[attr_offset:attr_offset+num_classes_per_task] = class_attr[attr_offset:attr_offset+num_classes_per_task]

            # Training loop for task T
            for iters in range(num_iters):

                if train_single_epoch and not cross_validate_mode:
                    #if (iters <= 50 and iters % 5 == 0) or (iters > 50 and iters % 50 == 0):
                    if (iters < 10) or (iters % 5 == 0):
                        # Snapshot the current performance across all tasks after each mini-batch
                        fbatch = test_task_sequence(model, sess, datasets, class_attr, num_classes_per_task, task_labels, 
                                cross_validate_mode, eval_single_head=eval_single_head, test_labels=test_labels)
                        ftask.append(fbatch)
                        # Set the output labels over which the model needs to be trained
                        logit_mask[:] = 0
                        if do_sampling:
                            logit_mask[test_labels] = 1.0
                        else:
                            logit_mask[task_labels[task]] = 1.0

                if train_single_epoch:
                    offset = iters * batch_size
                    if (offset+batch_size <= num_train_examples):
                        residual = batch_size
                    else:
                        residual = num_train_examples - offset

                    feed_dict = {model.x: train_x[offset:offset+residual], model.y_: train_y[offset:offset+residual],
                            model.class_attr: masked_class_attrs,
                            model.sample_weights: task_sample_weights[offset:offset+residual],
                            model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 0.5,
                            model.train_phase: True}
                else:
                    offset = (iters * batch_size) % (num_train_examples - batch_size)
                    feed_dict = {model.x: train_x[offset:offset+batch_size], model.y_: train_y[offset:offset+batch_size],
                            model.class_attr: masked_class_attrs,
                            model.sample_weights: task_sample_weights[offset:offset+batch_size],
                            model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 0.5,
                            model.train_phase: True}

                if model.imp_method == 'VAN':
                    feed_dict[model.output_mask] = logit_mask
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'EWC':
                    feed_dict[model.output_mask] = logit_mask
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                    # Update fisher after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        sess.run(model.set_running_fisher)
                        sess.run(model.reset_tmp_fisher)
                    
                    _, _, loss = sess.run([model.set_tmp_fisher, model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'PI':
                    feed_dict[model.output_mask] = logit_mask
                    _, _, _, loss = sess.run([model.weights_old_ops_grouped, model.train, model.update_small_omega, 
                                              model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'MAS':
                    feed_dict[model.output_mask] = logit_mask
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'GEM':
                    if task == 0:
                        logit_mask[:] = 0
                        logit_mask[task_labels[task]] = 1.0
                        feed_dict[model.output_mask] = logit_mask
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.reg_loss], feed_dict=feed_dict)
                    else:
                        # Compute the gradients on the episodic memory of all the previous tasks
                        for prev_task in range(task):
                            # T-th task gradients
                            # Note that the model.train_phase flag is false to avoid updating the batch norm params while doing forward pass on prev tasks
                            logit_mask[:] = 0
                            logit_mask[task_labels[prev_task]] = 1.0
                            sess.run([model.task_grads, model.store_task_gradients], feed_dict={model.x: task_based_memory[prev_task]['images'], 
                                model.y_: task_based_memory[prev_task]['labels'], model.task_id: prev_task, model.keep_prob: 1.0, 
                                model.output_mask: logit_mask, model.train_phase: False})

                        # Compute the gradient on the mini-batch of the current task
                        logit_mask[:] = 0
                        logit_mask[task_labels[task]] = 1.0
                        feed_dict[model.output_mask] = logit_mask
                        feed_dict[model.task_id] = task
                        _, _,loss = sess.run([model.task_grads, model.store_task_gradients, model.reg_loss], feed_dict=feed_dict)
                        # Store the gradients
                        sess.run([model.gem_gradient_update, model.store_grads], feed_dict={model.task_id: task})
                        # Apply the gradients
                        sess.run(model.train_subseq_tasks)

                elif model.imp_method == 'S-GEM':
                    if task == 0:
                        logit_mask[:] = 0
                        logit_mask[task_labels[task]] = 1.0
                        feed_dict[model.output_mask] = logit_mask
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.reg_loss], feed_dict=feed_dict)
                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        # Set the mask for all the previous tasks so far
                        logit_mask[:] = 0
                        logit_mask[prev_task_labels] = 1.0
                        # Sample a random subset from episodic memory buffer
                        mem_sample_mask = np.random.choice(episodic_mem_size, EPS_MEM_BATCH_SIZE, replace=False) # Sample without replacement so that we don't sample an example more than once
                        # Store the reference gradient
                        sess.run(model.store_ref_grads, feed_dict={model.x: episodic_images[mem_sample_mask], model.y_: episodic_labels[mem_sample_mask],
                            model.class_attr: prev_class_attrs,
                            model.keep_prob: 1.0, model.output_mask: logit_mask, model.train_phase: True})
                        # Compute the gradient for current task and project if need be
                        logit_mask[:] = 0
                        logit_mask[task_labels[task]] = 1.0
                        feed_dict[model.output_mask] = logit_mask
                        _, loss = sess.run([model.train_subseq_tasks, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'RWALK':
                    feed_dict[model.output_mask] = logit_mask
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                        # Store the current value of the weights
                        sess.run(model.weights_delta_old_grouped)
                    # Update fisher and importance score after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        # Update the importance score using distance in riemannian manifold   
                        sess.run(model.update_big_omega_riemann)
                        # Now that the score is updated, compute the new value for running Fisher
                        sess.run(model.set_running_fisher)
                        # Store the current value of the weights
                        sess.run(model.weights_delta_old_grouped)
                        # Reset the delta_L
                        sess.run([model.reset_small_omega])

                    _, _, _, _, loss = sess.run([model.set_tmp_fisher, model.weights_old_ops_grouped, 
                        model.train, model.update_small_omega, model.reg_loss], feed_dict=feed_dict)

                if (iters % 50 == 0):
                    print('Step {:d} {:.3f}'.format(iters, loss))

                if (math.isnan(loss)):
                    print('ERROR: NaNs NaNs NaNs!!!')
                    break_training = 1
                    break

            print('\t\t\t\tTraining for Task%d done!'%(task))

            # Update the previous task labels and attributes
            prev_task_labels += task_labels[task]
            prev_class_attrs = np.zeros_like(class_attr)
            prev_attr_offset = (task + 1) * num_classes_per_task
            prev_class_attrs[:prev_attr_offset] = class_attr[:prev_attr_offset]

            if break_training:
                break

            # Compute the inter-task updates, Fisher/ importance scores etc
            # Don't calculate the task updates for the last task
            if task < len(datasets) - 1:
                # TODO: For MAS, should the gradients be for current task or all the previous tasks
                model.task_updates(sess, task, task_train_images, task_labels[task], num_classes_per_task=num_classes_per_task, class_attr=class_attr) 
                print('\t\t\t\tTask updates after Task%d done!'%(task))

                # If importance method is 'GEM' then store the episodic memory for the task
                if model.imp_method == 'GEM' or model.imp_method == 'S-GEM':
                    # Do the uniform sampling/ only get examples from current task
                    importance_array = np.ones([task_train_images.shape[0]], dtype=np.float32)

                    if model.imp_method == 'GEM':
                        # Get the important samples from the current task
                        imp_images, imp_labels = sample_from_dataset(datasets[task]['train'], importance_array,
                                task_labels[task], SAMPLES_PER_CLASS)
                        task_memory = {
                                'images': deepcopy(imp_images),
                                'labels': deepcopy(imp_labels),
                                }
                        task_based_memory.append(task_memory)

                    elif model.imp_method == 'S-GEM':
                        data_to_sample_from = {
                                'images': task_train_images,
                                'labels': task_train_labels,
                                }
                        if HERDING_BASED_SAMPLING:
                            # Compute the features of training data
                            features = sess.run(model.features, feed_dict={model.x: task_train_images, model.y_: task_train_labels,
                            model.keep_prob: 1.0, model.output_mask: logit_mask, model.train_phase: False})
                            update_episodic_memory_with_less_data(data_to_sample_from, features, episodic_mem_size, task, episodic_images, episodic_labels, task_labels=task_labels[task], is_herding=True)
                        else:
                            update_episodic_memory_with_less_data(data_to_sample_from, importance_array, episodic_mem_size, task, episodic_images, episodic_labels)

                        # Inspect episodic memory
                        if DEBUG_EPISODIC_MEMORY:
                            # Which labels are present in the memory
                            unique_labels = np.unique(np.nonzero(episodic_labels)[-1])
                            print('Unique Labels present in the episodic memory'.format(unique_labels))
                            print('Labels count:')
                            for lbl in unique_labels:
                                print('Label {}: {} samples'.format(lbl, np.where(np.nonzero(episodic_labels)[-1] == lbl)[0].size))

                            # Is there any space which is not filled
                            print('Empty space: {}'.format(np.where(np.sum(episodic_labels, axis=1) == 0)))

                        print('Episodic memory of {} images at task {} saved!'.format(episodic_images.shape[0], task))

                # If sampling flag is set, store few of the samples from previous task
                if do_sampling:
                    # Do the uniform sampling/ only get examples from current task
                    importance_array = np.ones([datasets[task]['train']['images'].shape[0]], dtype=np.float32)
                    # Get the important samples from the current task
                    imp_images, imp_labels = sample_from_dataset(datasets[task]['train'], importance_array, 
                            task_labels[task], SAMPLES_PER_CLASS)

                    if imp_images is not None:
                        if last_task_x is None:
                            last_task_x = imp_images
                            last_task_y_ = imp_labels
                        else:
                            last_task_x = np.concatenate((last_task_x, imp_images), axis=0)
                            last_task_y_ = np.concatenate((last_task_y_, imp_labels), axis=0)

                    # Delete the importance array now that you don't need it in the current run
                    del importance_array

                    print('\t\t\t\tEpisodic memory is saved for Task%d!'%(task))

            if cross_validate_mode:
                if (task == NUM_TASKS - 1) or MULTI_TASK:
                    # List to store accuracy for all the tasks for the current trained model
                    ftask = test_task_sequence(model, sess, datasets, class_attr, num_classes_per_task, task_labels, cross_validate_mode, eval_single_head=eval_single_head, test_labels=test_labels)
            elif train_single_epoch:
                fbatch = test_task_sequence(model, sess, datasets, class_attr, num_classes_per_task, task_labels, cross_validate_mode, eval_single_head=eval_single_head, test_labels=test_labels)
                ftask.append(fbatch)
            else:
                # Multi-epoch training, so compute accuracy at the end
                ftask = test_task_sequence(model, sess, datasets, class_attr, num_classes_per_task, task_labels, cross_validate_mode, eval_single_head=eval_single_head, test_labels=test_labels)

            if SAVE_MODEL_PARAMS:
                save(saver, sess, SNAPSHOT_DIR, iters)

            if not cross_validate_mode:
                # Store the accuracies computed at task T in a list
                evals.append(np.array(ftask))

            # Reset the optimizer
            model.reset_optimizer(sess)

            #-> End for loop task

        if not cross_validate_mode:
            runs.append(np.array(evals))

        if break_training:
            break
        # End for loop runid
    if cross_validate_mode:
        return np.mean(ftask)
    else:
        runs = np.array(runs)
        return runs

def test_task_sequence(model, sess, test_data, class_attr, num_classes_per_task, test_tasks, cross_validate_mode, eval_single_head=True, test_labels=None):
    """
    Snapshot the current performance
    """
    list_acc = []

    if cross_validate_mode:
        test_set = 'test'
    else:
        test_set = 'test'

    logit_mask = np.zeros(TOTAL_CLASSES)
    if eval_single_head:
        # Single-head evaluation setting
        logit_mask[:len(test_labels)] = 1.0
        masked_class_attrs = np.zeros_like(class_attr)
        masked_class_attrs[:len(test_labels)] = class_attr[:len(test_labels)]

    for task, labels in enumerate(test_tasks):
        if not eval_single_head:
            # Multi-head evaluation setting
            logit_mask[:] = 0
            logit_mask[labels] = 1.0
            masked_class_attrs = np.zeros_like(class_attr)
            attr_offset = task * num_classes_per_task
            masked_class_attrs[attr_offset:attr_offset+num_classes_per_task] = class_attr[attr_offset:attr_offset+num_classes_per_task]
    
        task_test_images = test_data[task][test_set]['images']
        task_test_labels = test_data[task][test_set]['labels']
        total_test_samples = task_test_images.shape[0]
        samples_at_a_time = 10
        total_corrects = 0
        for i in range(total_test_samples/ samples_at_a_time):
            offset = i*samples_at_a_time
            feed_dict = {model.x: task_test_images[offset:offset+samples_at_a_time], 
                    model.y_: task_test_labels[offset:offset+samples_at_a_time],
                    model.class_attr: masked_class_attrs,
                    model.keep_prob: 1.0, model.train_phase: False, model.output_mask: logit_mask}
            total_corrects += np.sum(sess.run(model.correct_predictions, feed_dict=feed_dict))
        # Compute the corrects on residuals
        offset = (i+1)*samples_at_a_time
        num_residuals = total_test_samples % samples_at_a_time 
        feed_dict = {model.x: task_test_images[offset:offset+num_residuals], 
                model.y_: task_test_labels[offset:offset+num_residuals], 
                model.class_attr: masked_class_attrs,
                model.keep_prob: 1.0, model.train_phase: False, model.output_mask: logit_mask}
        total_corrects += np.sum(sess.run(model.correct_predictions, feed_dict=feed_dict))

        # Mean accuracy on the task
        acc = total_corrects/ float(total_test_samples)
        list_acc.append(acc)
    
    return list_acc

def main():
    """
    Create the model and start the training
    """

    # Get the CL arguments
    args = get_arguments()

    # Check if the network architecture is valid
    if args.arch not in VALID_ARCHS:
        raise ValueError("Network architecture %s is not supported!"%(args.arch))

    # Check if the method to compute importance is valid
    if args.imp_method not in MODELS:
        raise ValueError("Importance measure %s is undefined!"%(args.imp_method))
    
    # Check if the optimizer is valid
    if args.optim not in VALID_OPTIMS:
        raise ValueError("Optimizer %s is undefined!"%(args.optim))

    # Create log directories to store the results
    if not os.path.exists(args.log_dir):
        print('Log directory %s created!'%(args.log_dir))
        os.makedirs(args.log_dir)

    # Get the task labels from the total number of tasks and full label space
    task_labels = []
    label_array = np.arange(TOTAL_CLASSES)
    num_classes_per_task = TOTAL_CLASSES// NUM_TASKS
    for i in range(NUM_TASKS):
        jmp = num_classes_per_task
        offset = i*jmp
        task_labels.append(list(label_array[offset:offset+jmp]))

    # Load the split CUB dataset
    datasets, CUB_attr = construct_split_cub(task_labels, args.data_dir, CUB_TRAIN_LIST, CUB_TEST_LIST, IMG_HEIGHT, IMG_WIDTH, attr_file=CUB_ATTR_LIST)

    if args.cross_validate_mode:
        models_list = [args.imp_method]
        learning_rate_list = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    else:
        models_list = [args.imp_method]
    for imp_method in models_list:
        if imp_method == 'VAN':
            synap_stgth_list = [0]
            if args.cross_validate_mode:
                pass
            else:
                #learning_rate_list = [0.03] # => cross-validated learning-rate for SGD ZST, HYBRID, Resnet
                learning_rate_list = [0.001] # => cross-validated learning-rate for SGD ZST, HYBRID, VGG
        elif imp_method == 'PI':
            if args.cross_validate_mode:
                #synap_stgth_list = [args.synap_stgth]
                synap_stgth_list = [0.1, 1, 10, 100]
            else:
                #synap_stgth_list = [args.synap_stgth]
                #synap_stgth_list = [0.01] # => cross-validated lambda ZST, Resnet
                #learning_rate_list = [0.03] # => cross-validated learning-rate for SGD ZST, HYBRID, Resnet
                synap_stgth_list = [1] # => cross-validated lambda ZST, VGG
                learning_rate_list = [0.001] # => cross-validated learning-rate for SGD ZST, HYBRID, VGG
        elif imp_method == 'EWC':
            if args.cross_validate_mode:
                synap_stgth_list = [1, 10, 100]
                #synap_stgth_list = [args.synap_stgth]
                #synap_stgth_list = [0.1, 1, 10, 100, 1000]
            else:
                #synap_stgth_list = [args.synap_stgth]
                #synap_stgth_list = [10] # => cross-validated lambda, Resnet
                #learning_rate_list = [0.03] # => cross-validated learning-rate for SGD ZST, HYBRID, Resnet
                synap_stgth_list = [1] # => cross-validated lambda ZST, VGG
                learning_rate_list = [0.001] # => cross-validated learning-rate for SGD ZST, HYBRID, VGG
        elif imp_method == 'MAS':
            if args.cross_validate_mode:
                #synap_stgth_list = [args.synap_stgth]
                synap_stgth_list = [0.1, 1, 10]
            else:
                #synap_stgth_list = [args.synap_stgth]
                #synap_stgth_list = [0.1] # => cross-validated lambda, Resnet
                #learning_rate_list = [0.03] # => cross-validated learning-rate for SGD, Resnet
                synap_stgth_list = [1] # => cross-validated lambda ZST, VGG
                learning_rate_list = [0.001] # => cross-validated learning-rate for SGD ZST, HYBRID, VGG
        elif imp_method == 'RWALK':
            if args.cross_validate_mode:
                #synap_stgth_list = [args.synap_stgth]
                synap_stgth_list = [0.1, 1, 10, 100, 1000]
            else:
                #synap_stgth_list = [args.synap_stgth]
                #synap_stgth_list = [0.1] # => cross-validated lambda ZST, Resnet
                #learning_rate_list = [0.03] # => cross-validated learning-rate for SGD ZST, Resnet
                synap_stgth_list = [10] # => cross-validated lambda ZST, VGG
                learning_rate_list = [0.001] # => cross-validated learning-rate for SGD ZST, HYBRID, VGG
        elif imp_method == 'GEM':
            synap_stgth_list = [0]
            if args.cross_validate_mode:
                pass
            else:
                learning_rate_list = [0.0]
        elif imp_method == 'S-GEM':
            synap_stgth_list = [0]
            if args.cross_validate_mode:
                pass
            else:
                learning_rate_list = [0.03] # => cross-validated learning rate for SGD, Resnet-18
 
        for synap_stgth in synap_stgth_list:
            for lr in learning_rate_list:
                # Generate the experiment key and store the meta data in a file
                exper_meta_data = {'ARCH': args.arch,
                    'DATASET': 'SPLIT_CUB',
                    'HYBRID': args.set_hybrid,
                    'NUM_RUNS': args.num_runs,
                    'EVAL_SINGLE_HEAD': args.eval_single_head, 
                    'TRAIN_SINGLE_EPOCH': args.train_single_epoch, 
                    'IMP_METHOD': imp_method, 
                    'SYNAP_STGTH': synap_stgth,
                    'FISHER_EMA_DECAY': args.fisher_ema_decay,
                    'FISHER_UPDATE_AFTER': args.fisher_update_after,
                    'OPTIM': args.optim, 
                    'LR': lr, 
                    'BATCH_SIZE': args.batch_size, 
                    'EPS_MEMORY': args.do_sampling, 
                    'MEM_SIZE': args.mem_size}
                experiment_id = "SPLIT_CUB_HYB_%r_%s_%r_%r_%s_%s_%s_%r_%s-"%(args.set_hybrid, args.arch, args.eval_single_head, args.train_single_epoch, imp_method, 
                        str(synap_stgth).replace('.', '_'), 
                        str(args.batch_size), args.do_sampling, str(args.mem_size)) + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
                snapshot_experiment_meta_data(args.log_dir, experiment_id, exper_meta_data)

                # Variables to store the accuracies and standard deviations of the experiment
                acc_mean = dict()
                acc_std = dict()

                # Reset the default graph
                tf.reset_default_graph()
                graph  = tf.Graph()
                with graph.as_default():

                    # Set the random seed
                    tf.set_random_seed(RANDOM_SEED)

                    # Define Input and Output of the model
                    x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
                    y_ = tf.placeholder(tf.float32, shape=[None, TOTAL_CLASSES])
                    attr = tf.placeholder(tf.float32, shape=[TOTAL_CLASSES, ATTR_DIMS])

                    if not args.train_single_epoch:
                        # Define ops for data augmentation
                        x_aug = image_scaling(x)
                        x_aug = random_crop_and_pad_image(x_aug, IMG_HEIGHT, IMG_WIDTH)

                    # Define the optimizer
                    if args.optim == 'ADAM':
                        opt = tf.train.AdamOptimizer(learning_rate=lr)

                    elif args.optim == 'SGD':
                        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

                    elif args.optim == 'MOMENTUM':
                        base_lr = tf.constant(lr)
                        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - train_step / training_iters), OPT_POWER))
                        opt = tf.train.MomentumOptimizer(lr, OPT_MOMENTUM)

                    # Create the Model/ contruct the graph
                    if args.train_single_epoch:
                        # When training using a single epoch then there is no need for data augmentation
                        model = Model(x, y_, NUM_TASKS, opt, imp_method, synap_stgth, args.fisher_update_after,
                                args.fisher_ema_decay, network_arch=args.arch, is_ATT_DATASET=True, attr=attr, hybrid=args.set_hybrid)
                    else:
                        model = Model(x_aug, y_, NUM_TASKS, opt, imp_method, synap_stgth, args.fisher_update_after, 
                                args.fisher_ema_decay, network_arch=args.arch, is_ATT_DATASET=True, x_test=x, attr=attr, hybrid=args.set_hybrid)

                    # Set up tf session and initialize variables.
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True

                    with tf.Session(config=config, graph=graph) as sess:
                        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
                        runs = train_task_sequence(model, sess, saver, datasets, CUB_attr, num_classes_per_task, task_labels, args.cross_validate_mode, 
                                args.train_single_epoch, args.eval_single_head, args.do_sampling, args.mem_size, args.train_iters, 
                                args.batch_size, args.num_runs, args.init_checkpoint)
                        # Close the session
                        sess.close()

                # Clean up
                del model

                if args.cross_validate_mode:
                    # If cross-validation flag is enabled, store the stuff in a text file
                    cross_validate_dump_file = args.log_dir + '/' + 'SPLIT_CUB_%s_%s'%(imp_method, args.optim) + '.txt'
                    with open(cross_validate_dump_file, 'a') as f:
                            f.write('ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {}\n'.format(args.arch, lr, synap_stgth, runs))
                else:
                    # Compute the mean and std
                    acc_mean = runs.mean(0)
                    acc_std = runs.std(0)
                    # Store all the results in one dictionary to process later
                    exper_acc = dict(mean=acc_mean, std=acc_std)
                    # Store the experiment output to a file
                    snapshot_experiment_eval(args.log_dir, experiment_id, exper_acc)

if __name__ == '__main__':
    main()
