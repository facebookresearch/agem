# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for split AWA experiment with hybrid learning.
"""
from __future__ import print_function

import argparse
import os
import sys
import math
import random
import time 

import datetime
import numpy as np
import tensorflow as tf
from copy import deepcopy
from six.moves import cPickle as pickle

from utils.data_utils import image_scaling, random_crop_and_pad_image, random_horizontal_flip, construct_split_awa
from utils.utils import get_sample_weights, sample_from_dataset, concatenate_datasets, update_episodic_memory, samples_for_each_class, sample_from_dataset_icarl, load_task_specific_data, load_task_specific_data_in_proportion
from utils.vis_utils import plot_acc_multiple_runs, plot_histogram, snapshot_experiment_meta_data, snapshot_experiment_eval, snapshot_task_labels
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
VALID_ARCHS = ['CNN', 'VGG', 'RESNET-B']
ARCH = 'RESNET-B'
PRETRAIN = False

## Model options
MODELS = ['VAN', 'PI', 'EWC', 'MAS', 'RWALK', 'A-GEM'] #List of valid models
IMP_METHOD = 'VAN'
SYNAP_STGTH = 75000
FISHER_EMA_DECAY = 0.9      # Exponential moving average decay factor for Fisher computation (online Fisher)
FISHER_UPDATE_AFTER = 50    # Number of training iterations for which the F_{\theta}^t is computed (see Eq. 10 in RWalk paper) 
SAMPLES_PER_CLASS = 20   # Number of samples per task
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
TOTAL_CLASSES = 50          # Total number of classes in the dataset 
MEASURE_CONVERGENCE_AFTER = 0.9
EPS_MEM_BATCH_SIZE = 128
DEBUG_EPISODIC_MEMORY = False
KEEP_EPISODIC_MEMORY_FULL = False
K_FOR_CROSS_VAL = 3
CLASSES_PER_TASK = 5

## Logging, saving and testing options
LOG_DIR = './split_awa_results'
SNAPSHOT_DIR = './awa_snapshots'
SAVE_MODEL_PARAMS = False
RESNET18_IMAGENET_CHECKPOINT = './resnet-18-pretrained-imagenet/model.ckpt'

## Evaluation options

## Task split
NUM_TASKS = 20
MULTI_TASK = False

## Dataset specific options
ATTR_DIMS = 85
DATA_DIR= './AWA_data/Animals_with_Attributes2/'
AWA_TRAIN_LIST = './dataset_lists/AWA_train_list.txt'
AWA_VAL_LIST = './dataset_lists/AWA_val_list.txt'
AWA_TEST_LIST = './dataset_lists/AWA_test_list.txt'
#AWA_TRAIN_LIST = './dataset_lists/tmp_list_awa.txt'
#AWA_VAL_LIST = './dataset_lists/tmp_list_awa.txt'
#AWA_TEST_LIST = './dataset_lists/tmp_list_awa.txt'
AWA_ATTR_LIST = 'dataset_lists/AWA_attr_in_order.pickle'

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
    parser = argparse.ArgumentParser(description="Script for split AWA Hybrid experiment.")
    parser.add_argument("--cross-validate-mode", action="store_true",
            help="If option is chosen then snapshoting after each batch is disabled")
    parser.add_argument("--online-cross-val", action="store_true",
            help="If option is chosen then enable the online cross validation of the learning rate")
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
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                       help="Random Seed.")
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
    parser.add_argument("--mem-size", type=int, default=SAMPLES_PER_CLASS,
                       help="Number of samples per class from previous tasks.")
    parser.add_argument("--is-herding", action="store_true",
                        help="Herding based sampling")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                       help="Directory from where the AWA data will be read.\
                               NOTE: Provide path till <AWA_DIR>/Animals_with_Attributes2")
    parser.add_argument("--init-checkpoint", type=str, default=RESNET18_IMAGENET_CHECKPOINT,
                       help="TF checkpoint file containing initialization for ImageNet.\
                               NOTE: NPZ file for VGG and TF Checkpoint for ResNet")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                       help="Directory where the plots and model accuracies will be stored.")
    return parser.parse_args()

def train_task_sequence(model, sess, saver, datasets, class_attr, num_classes_per_task, cross_validate_mode, train_single_epoch, do_sampling, is_herding,  
        episodic_mem_size, train_iters, batch_size, num_runs, init_checkpoint, online_cross_val, random_seed):
    """
    Train and evaluate LLL system such that we only see a example once
    Args:
    Returns:
        dict    A dictionary containing mean and stds for the experiment
    """
    # List to store accuracy for each run
    runs = []
    task_labels_dataset = []

    break_training = 0
    # Loop over number of runs to average over
    for runid in range(num_runs):
        print('\t\tRun %d:'%(runid))

        # Initialize the random seeds
        np.random.seed(random_seed+runid)
        random.seed(random_seed+runid)

        # Get the task labels from the total number of tasks and full label space
        task_labels = []
        classes_per_task = num_classes_per_task
        classes_appearing_in_tasks = dict()
        for cls in range(TOTAL_CLASSES):
            classes_appearing_in_tasks[cls] = 0

        if online_cross_val:
            label_array = np.arange(TOTAL_CLASSES)
            for tt in range(model.num_tasks):
                offset = tt * classes_per_task
                task_labels.append(list(label_array[offset:offset+classes_per_task]))
        else:
            for tt in range(model.num_tasks):
                task_labels.append(random.sample(range(K_FOR_CROSS_VAL*classes_per_task, TOTAL_CLASSES), classes_per_task))
                for lab in task_labels[tt]:
                    classes_appearing_in_tasks[lab] += 1
                print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
            print('Class frequency in Tasks: {}'.format(classes_appearing_in_tasks))

        # Store the task labels
        task_labels_dataset.append(task_labels)

        # Initialize all the variables in the model
        sess.run(tf.global_variables_initializer())

        if PRETRAIN:
            # Load the variables from a checkpoint
            if model.network_arch == 'RESNET-B':
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

        if model.imp_method == 'S-GEM':
            # List to store the episodic memories of the previous tasks
            task_based_memory = []

        if model.imp_method == 'A-GEM':
            # Reserve a space for episodic memory
            episodic_images = np.zeros([episodic_mem_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
            episodic_labels = np.zeros([episodic_mem_size, model.num_tasks*TOTAL_CLASSES])
            episodic_filled_counter = 0
            a_gem_logit_mask = np.zeros([model.num_tasks, model.total_classes])
            # Labels for all the tasks that we have seen in the past
            prev_task_labels = []
            prev_class_attrs = np.zeros([model.total_classes, class_attr.shape[1]])

        if do_sampling:
            # List to store important samples from the previous tasks
            last_task_x = None
            last_task_y_ = None

        # Mask for softmax
        logit_mask = np.zeros(model.total_classes)

        max_batch_dimension = 500

        # Dict to store the number of times a class has already been seen in the training
        class_seen_already = dict()
        for cls in range(TOTAL_CLASSES):
            class_seen_already[cls] = 0

        # Training loop for all the tasks
        for task in range(len(task_labels)):
            print('\t\tTask %d:'%(task))

            # If not the first task then restore weights from previous task
            if(task > 0):
                model.restore(sess)

            # Increment the class seen count
            for cls in task_labels[task]:
                class_seen_already[cls] += 1

            task_train_images, task_train_labels = load_task_specific_data_in_proportion(datasets[0]['train'], task_labels[task], classes_appearing_in_tasks, class_seen_already)
            print('Received {} images, {} labels at task {}'.format(task_train_images.shape[0], task_train_labels.shape[0], task))
            print('Unique labels in the task: {}'.format(np.unique(np.nonzero(task_train_labels)[1])))

            # Assign equal weights to all the examples
            task_sample_weights = np.ones([task_train_labels.shape[0]], dtype=np.float32)

            num_train_examples = task_train_images.shape[0]

            # Train a task observing sequence of data
            logit_mask[:] = 0
            if train_single_epoch:
                # Ceiling operation
                num_iters = (num_train_examples + batch_size - 1) // batch_size
            else:
                num_iters = train_iters

            logit_mask_offset = task * TOTAL_CLASSES
            classes_adjusted_for_head = [cls + logit_mask_offset for cls in task_labels[task]]
            logit_mask[classes_adjusted_for_head] = 1.0

            # Randomly suffle the training examples
            perm = np.arange(num_train_examples)
            np.random.shuffle(perm)
            train_x = task_train_images[perm]
            train_y = task_train_labels[perm]
            task_sample_weights = task_sample_weights[perm]

            # Array to store accuracies when training for task T
            if cross_validate_mode:
                # Because we will evaluate at the end
                ftask = 0
            elif train_single_epoch:
                # Because we will evaluate after every mini-batch of every task
                ftask = np.zeros([max_batch_dimension+1, model.num_tasks])
                batch_dim_count = 0
            else:
                # Multi-epoch because we will evaluate after every task
                ftask = []

            # Attribute mask
            masked_class_attrs = np.zeros([model.total_classes, class_attr.shape[1]])
            masked_class_attrs[classes_adjusted_for_head] = class_attr[task_labels[task]]
            #masked_class_attrs[task_labels[task]] = class_attr[task_labels[task]]

            # Number of iterations after which convergence is checked
            convergence_iters = int(num_iters * MEASURE_CONVERGENCE_AFTER)

            final_train_labels = np.zeros([batch_size, model.total_classes])
            head_offset = task * TOTAL_CLASSES

            # Training loop for task T
            for iters in range(num_iters):

                if train_single_epoch and not cross_validate_mode:
                    if (iters < 11):
                        # Snapshot the current performance across all tasks after each mini-batch
                        fbatch = test_task_sequence(model, sess, datasets[0]['test'], class_attr, num_classes_per_task, task_labels, task, online_cross_val)
                        ftask[batch_dim_count] = fbatch
                        # Increment the batch_dim_count
                        batch_dim_count += 1
                        # Set the output labels over which the model needs to be trained
                        if model.imp_method == 'A-GEM':
                            a_gem_logit_mask[:] = 0
                            a_gem_logit_mask[task][classes_adjusted_for_head] = 1.0
                        else:
                            logit_mask[:] = 0
                            logit_mask[classes_adjusted_for_head] = 1.0
                        
                offset = iters * batch_size
                if (offset+batch_size <= num_train_examples):
                    residual = batch_size
                else:
                    residual = num_train_examples - offset

                final_train_labels[:residual, head_offset:head_offset+TOTAL_CLASSES] = train_y[offset:offset+residual]
                feed_dict = {model.x: train_x[offset:offset+residual], model.y_: final_train_labels[:residual], 
                        model.class_attr: masked_class_attrs,
                        model.sample_weights: task_sample_weights[offset:offset+residual],
                        model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 0.5, 
                        model.train_phase: True}

                if model.imp_method == 'VAN':
                    feed_dict[model.output_mask] = logit_mask
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'EWC' or model.imp_method == 'M-EWC':
                    feed_dict[model.output_mask] = logit_mask
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                    # Update fisher after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        sess.run(model.set_running_fisher)
                        sess.run(model.reset_tmp_fisher)
                    
                    if (iters >= convergence_iters) and (model.imp_method == 'M-EWC'):
                        _, _, _, _, loss = sess.run([model.weights_old_ops_grouped, model.set_tmp_fisher, model.train, model.update_small_omega,
                            model.reg_loss], feed_dict=feed_dict)
                    else:
                        _, _, loss = sess.run([model.set_tmp_fisher, model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'PI':
                    feed_dict[model.output_mask] = logit_mask
                    _, _, _, loss = sess.run([model.weights_old_ops_grouped, model.train, model.update_small_omega, 
                                              model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'MAS':
                    feed_dict[model.output_mask] = logit_mask
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'S-GEM':
                    if task == 0:
                        logit_mask[:] = 0
                        logit_mask[task_labels[task]] = 1.0
                        feed_dict[model.output_mask] = logit_mask
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:
                        # Randomly sample a task from the previous tasks
                        prev_task = np.random.randint(0, task)
                        # Set the logit mask for the randomly sampled task
                        logit_mask[:] = 0
                        logit_mask[task_labels[prev_task]] = 1.0
                        prev_class_attrs = np.zeros_like(class_attr)
                        if online_cross_val:
                            attr_offset = prev_task * num_classes_per_task
                        else:
                            attr_offset = (prev_task + K_FOR_CROSS_VAL) * num_classes_per_task
                        prev_class_attrs[attr_offset:attr_offset+num_classes_per_task] = class_attr[attr_offset:attr_offset+num_classes_per_task]
                        # Store the reference gradient
                        sess.run(model.store_ref_grads, feed_dict={model.x: task_based_memory[prev_task]['images'], model.y_: task_based_memory[prev_task]['labels'],
                            model.class_attr: prev_class_attrs,
                            model.keep_prob: 1.0, model.output_mask: logit_mask, model.train_phase: True})
                        # Compute the gradient for current task and project if need be
                        logit_mask[:] = 0
                        logit_mask[task_labels[task]] = 1.0
                        feed_dict[model.output_mask] = logit_mask
                        _, loss = sess.run([model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)

                elif model.imp_method == 'A-GEM':
                    if task == 0:
                        a_gem_logit_mask[:] = 0
                        a_gem_logit_mask[task][classes_adjusted_for_head] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, a_gem_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        # Set the mask for all the previous tasks so far
                        a_gem_logit_mask[:] = 0
                        for tt in range(task):
                            logit_mask_offset = tt * TOTAL_CLASSES
                            classes_adjusted_for_head = [cls + logit_mask_offset for cls in task_labels[tt]]
                            a_gem_logit_mask[tt][classes_adjusted_for_head] = 1.0

                        if KEEP_EPISODIC_MEMORY_FULL:
                            mem_sample_mask = np.random.choice(episodic_mem_size, EPS_MEM_BATCH_SIZE, replace=False) # Sample without replacement so that we don't sample an example more than once
                        else:
                            if episodic_filled_counter <= EPS_MEM_BATCH_SIZE:
                                mem_sample_mask = np.arange(episodic_filled_counter)
                            else:
                                # Sample a random subset from episodic memory buffer
                                mem_sample_mask = np.random.choice(episodic_filled_counter, EPS_MEM_BATCH_SIZE, replace=False) # Sample without replacement so that we don't sample an example more than once
                        # Store the reference gradient
                        ref_feed_dict = {model.x: episodic_images[mem_sample_mask], model.y_: episodic_labels[mem_sample_mask], model.class_attr: prev_class_attrs,  
                                model.keep_prob: 1.0, model.train_phase: True}
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, a_gem_logit_mask)}
                        ref_feed_dict.update(logit_mask_dict)
                        ref_feed_dict[model.mem_batch_size] = float(len(mem_sample_mask))
                        sess.run(model.store_ref_grads, feed_dict=ref_feed_dict)
                        # Compute the gradient for current task and project if need be
                        a_gem_logit_mask[:] = 0
                        logit_mask_offset = task * TOTAL_CLASSES
                        classes_adjusted_for_head = [cls + logit_mask_offset for cls in task_labels[task]]
                        a_gem_logit_mask[task][classes_adjusted_for_head] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, a_gem_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size
                        _, loss = sess.run([model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)

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

                if (iters % 100 == 0):
                    print('Step {:d} {:.3f}'.format(iters, loss))

                if (math.isnan(loss)):
                    print('ERROR: NaNs NaNs NaNs!!!')
                    break_training = 1
                    break

            print('\t\t\t\tTraining for Task%d done!'%(task))

            if model.imp_method == 'A-GEM':
                # Update the previous task labels
                prev_task_labels.extend(classes_adjusted_for_head)
                prev_class_attrs[classes_adjusted_for_head] = class_attr[task_labels[task]]

            if break_training:
                break

            # Compute the inter-task updates, Fisher/ importance scores etc
            # Don't calculate the task updates for the last task
            if task < (len(task_labels) - 1):
                # TODO: For MAS, should the gradients be for current task or all the previous tasks
                model.task_updates(sess, task, task_train_images, task_labels[task], num_classes_per_task=num_classes_per_task, class_attr=class_attr, online_cross_val=online_cross_val) 
                print('\t\t\t\tTask updates after Task%d done!'%(task))

                # If importance method is '*-GEM' then store the episodic memory for the task
                if 'GEM' in model.imp_method:
                    data_to_sample_from = {
                            'images': task_train_images,
                            'labels': task_train_labels,
                            }
                    if model.imp_method == 'S-GEM':
                        # Get the important samples from the current task
                        if is_herding: # Sampling based on MoF
                            # Compute the features of training data
                            features_dim = model.image_feature_dim
                            features = np.zeros([num_train_examples, features_dim])
                            samples_at_a_time = 32
                            residual = num_train_examples % samples_at_a_time
                            for i in range(num_train_examples// samples_at_a_time):
                                offset = i * samples_at_a_time
                                features[offset:offset+samples_at_a_time] = sess.run(model.features, feed_dict={model.x: task_train_images[offset:offset+samples_at_a_time],
                                    model.y_: task_train_labels[offset:offset+samples_at_a_time], model.keep_prob: 1.0,
                                    model.output_mask: logit_mask, model.train_phase: False})
                            if residual > 0:
                                offset = (i + 1) * samples_at_a_time
                                features[offset:offset+residual] = sess.run(model.features, feed_dict={model.x: task_train_images[offset:offset+residual],
                                    model.y_: task_train_labels[offset:offset+residual], model.keep_prob: 1.0,
                                    model.output_mask: logit_mask, model.train_phase: False})
                            imp_images, imp_labels = sample_from_dataset_icarl(data_to_sample_from, features, task_labels[task], SAMPLES_PER_CLASS)
                        else: # Random sampling
                            # Do the uniform sampling/ only get examples from current task
                            importance_array = np.ones(num_train_examples, dtype=np.float32)
                            imp_images, imp_labels = sample_from_dataset(data_to_sample_from, importance_array, task_labels[task], SAMPLES_PER_CLASS)
                        task_memory = {
                                'images': deepcopy(imp_images),
                                'labels': deepcopy(imp_labels),
                                }
                        task_based_memory.append(task_memory)

                    elif model.imp_method == 'A-GEM':
                        # Do the uniform sampling/ only get examples from current task
                        importance_array = np.ones(num_train_examples, dtype=np.float32)
                        if KEEP_EPISODIC_MEMORY_FULL:
                            update_episodic_memory(data_to_sample_from, importance_array, episodic_mem_size, task, episodic_images, episodic_labels)
                        else:
                            imp_images, imp_labels = sample_from_dataset(data_to_sample_from, importance_array, task_labels[task], SAMPLES_PER_CLASS)
                        if not KEEP_EPISODIC_MEMORY_FULL: # Fill the memory to always keep M/T samples per task
                            total_imp_samples = imp_images.shape[0]
                            eps_offset = task * total_imp_samples
                            episodic_images[eps_offset:eps_offset+total_imp_samples] = imp_images
                            episodic_labels[eps_offset:eps_offset+total_imp_samples, head_offset:head_offset+TOTAL_CLASSES] = imp_labels
                            episodic_filled_counter += total_imp_samples
                        print('Unique labels in the episodic memory: {}'.format(np.unique(np.nonzero(episodic_labels)[1])))
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
                if (task == model.num_tasks - 1) or MULTI_TASK:
                    # List to store accuracy for all the tasks for the current trained model
                    ftask = test_task_sequence(model, sess, datasets[0]['test'], class_attr, num_classes_per_task, task_labels, task, online_cross_val)
            elif train_single_epoch:
                fbatch = test_task_sequence(model, sess, datasets[0]['test'], class_attr, num_classes_per_task, task_labels, task, False)
                ftask[batch_dim_count] = fbatch
                print('Task: {}, {}'.format(task, fbatch))
            else:
                # Multi-epoch training, so compute accuracy at the end
                ftask = test_task_sequence(model, sess, datasets[0]['test'], class_attr, num_classes_per_task, task_labels, task, online_cross_val)

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
        return runs, task_labels_dataset

def test_task_sequence(model, sess, test_data, class_attr, num_classes_per_task, all_task_labels, task, online_cross_val):
    """
    Snapshot the current performance
    """
    final_acc = np.zeros(model.num_tasks)
    test_set = 'test'
    if model.imp_method == 'A-GEM':
        logit_mask = np.zeros([model.num_tasks, model.total_classes])
    else:
        logit_mask = np.zeros(model.total_classes)

    for tt, labels in enumerate(all_task_labels):

        if tt > task:
            return final_acc

        samples_at_a_time = 10
        task_images, task_labels = load_task_specific_data(test_data, labels)
        global_class_indices = np.column_stack(np.nonzero(task_labels))
        logit_mask_offset = tt * TOTAL_CLASSES
        classes_adjusted_for_head = [cls + logit_mask_offset for cls in labels]
        logit_mask[:] = 0
        if model.imp_method == 'A-GEM':
            logit_mask[tt][classes_adjusted_for_head] = 1.0
            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, logit_mask)}
        else:
            logit_mask[classes_adjusted_for_head] = 1.0
        #masked_class_attrs = np.zeros_like(class_attr)
        #masked_class_attrs[labels] = class_attr[labels]
        masked_class_attrs = np.zeros([model.total_classes, class_attr.shape[1]])
        masked_class_attrs[classes_adjusted_for_head] = class_attr[labels]
        final_train_labels = np.zeros([samples_at_a_time, model.total_classes])
        head_offset = tt * TOTAL_CLASSES
        acc = np.zeros(len(labels))
        for cli, cls in enumerate(labels):
            class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == cls][:,np.array([True, False])])
            class_indices = np.sort(class_indices, axis=None)
            task_test_images = task_images[class_indices]
            task_test_labels = task_labels[class_indices]
            total_test_samples = task_test_images.shape[0]
            total_corrects = 0
            if total_test_samples < samples_at_a_time:
                i = -1

            for i in range(total_test_samples/ samples_at_a_time):
                offset = i*samples_at_a_time
                final_train_labels[:, head_offset:head_offset+TOTAL_CLASSES] = task_test_labels[offset:offset+samples_at_a_time]
                feed_dict = {model.x: task_test_images[offset:offset+samples_at_a_time], 
                        model.y_: final_train_labels,
                        model.class_attr: masked_class_attrs,
                        model.keep_prob: 1.0, model.train_phase: False}
                if model.imp_method == 'A-GEM':
                    feed_dict.update(logit_mask_dict)
                    corrects = sess.run(model.correct_predictions[tt], feed_dict=feed_dict)
                else:
                    feed_dict[model.output_mask] = logit_mask
                    corrects = sess.run(model.correct_predictions, feed_dict=feed_dict)
                total_corrects += np.sum(corrects)
            # Compute the corrects on residuals
            offset = (i+1)*samples_at_a_time
            num_residuals = total_test_samples % samples_at_a_time
            final_train_labels[:num_residuals, head_offset:head_offset+TOTAL_CLASSES] = task_test_labels[offset:offset+num_residuals]
            feed_dict = {model.x: task_test_images[offset:offset+num_residuals], 
                    model.y_: final_train_labels[:num_residuals], 
                    model.class_attr: masked_class_attrs,
                    model.keep_prob: 1.0, model.train_phase: False}
            if model.imp_method == 'A-GEM':
                feed_dict.update(logit_mask_dict)
                corrects = sess.run(model.correct_predictions[tt], feed_dict=feed_dict)
            else:
                feed_dict[model.output_mask] = logit_mask
                corrects = sess.run(model.correct_predictions, feed_dict=feed_dict)
            total_corrects += np.sum(corrects)
            if total_test_samples != 0:
                # Mean accuracy on the task
                acc[cli] = total_corrects/ float(total_test_samples)

        final_acc[tt] = np.mean(acc)
    
    return final_acc

def main():
    """
    Create the model and start the training
    """

    # Get the CL arguments
    args = get_arguments()
    
    # Initialize the random seed of numpy
    np.random.seed(args.random_seed)

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

    if args.online_cross_val:
        num_tasks = K_FOR_CROSS_VAL
    else:
        num_tasks = NUM_TASKS - K_FOR_CROSS_VAL

    # Load the split AWA dataset
    data_labs = [np.arange(TOTAL_CLASSES)]
    datasets, AWA_attr = construct_split_awa(data_labs, args.data_dir, AWA_TRAIN_LIST, AWA_VAL_LIST, AWA_TEST_LIST, IMG_HEIGHT, IMG_WIDTH, attr_file=AWA_ATTR_LIST)
    if args.online_cross_val:
        AWA_attr[K_FOR_CROSS_VAL*CLASSES_PER_TASK:] = 0
    else:
        AWA_attr[:K_FOR_CROSS_VAL*CLASSES_PER_TASK] = 0

    print('Attributes: {}'.format(np.sum(AWA_attr, axis=1)))

    if args.cross_validate_mode:
        models_list = MODELS
        learning_rate_list = [0.1, 0.03, 0.01, 0.001, 0.0003]
    else:
        models_list = [args.imp_method]
    for imp_method in models_list:
        if imp_method == 'VAN':
            synap_stgth_list = [0]
            if args.online_cross_val or args.cross_validate_mode:
                pass
            else:
                learning_rate_list = [0.003]
        elif imp_method == 'PI':
            if args.online_cross_val or args.cross_validate_mode:
                synap_stgth_list = [0.1, 1, 10]
            else:
                synap_stgth_list = [10]
                learning_rate_list = [0.003]
        elif imp_method == 'EWC' or imp_method == 'M-EWC':
            if args.online_cross_val or args.cross_validate_mode:
                synap_stgth_list = [0.1, 1, 10, 100]
            else:
                synap_stgth_list = [100]
                learning_rate_list = [0.003]
        elif imp_method == 'MAS':
            if args.online_cross_val or args.cross_validate_mode:
                synap_stgth_list = [0.1, 1, 10, 100]
            else:
                synap_stgth_list = [0.1]
                learning_rate_list = [0.001]
        elif imp_method == 'RWALK':
            if args.online_cross_val or args.cross_validate_mode:
                synap_stgth_list = [0.1, 1, 10, 100]
            else:
                synap_stgth_list = [10]     # Check again!
                learning_rate_list = [0.003]
        elif imp_method == 'S-GEM':
            synap_stgth_list = [0]
            if args.online_cross_val:
                pass
            else:
                learning_rate_list = [args.learning_rate]
        elif imp_method == 'A-GEM':
            synap_stgth_list = [0]
            if args.online_cross_val or args.cross_validate_mode:
                pass
            else:
                learning_rate_list = [0.003]

        for synap_stgth in synap_stgth_list:
            for lr in learning_rate_list:
                # Generate the experiment key and store the meta data in a file
                exper_meta_data = {'ARCH': args.arch,
                    'DATASET': 'SPLIT_AWA',
                    'HYBRID': args.set_hybrid,
                    'NUM_RUNS': args.num_runs,
                    'TRAIN_SINGLE_EPOCH': args.train_single_epoch, 
                    'IMP_METHOD': imp_method, 
                    'SYNAP_STGTH': synap_stgth,
                    'FISHER_EMA_DECAY': args.fisher_ema_decay,
                    'FISHER_UPDATE_AFTER': args.fisher_update_after,
                    'OPTIM': args.optim, 
                    'LR': lr, 
                    'BATCH_SIZE': args.batch_size, 
                    'EPS_MEMORY': args.do_sampling, 
                    'MEM_SIZE': args.mem_size, 
                    'IS_HERDING': args.is_herding}
                experiment_id = "SPLIT_AWA_HERDING_%r_HYB_%r_%s_%r_%s_%s_%s_%r_%s-"%(args.is_herding, args.set_hybrid, args.arch, args.train_single_epoch, imp_method, 
                        str(synap_stgth).replace('.', '_'), 
                        str(args.batch_size), args.do_sampling, str(args.mem_size)) + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
                snapshot_experiment_meta_data(args.log_dir, experiment_id, exper_meta_data)

                # Reset the default graph
                tf.reset_default_graph()
                graph  = tf.Graph()
                with graph.as_default():

                    # Set the random seed
                    tf.set_random_seed(args.random_seed)

                    # Define Input and Output of the model
                    x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
                    y_ = tf.placeholder(tf.float32, shape=[None, num_tasks*TOTAL_CLASSES])
                    attr = tf.placeholder(tf.float32, shape=[num_tasks*TOTAL_CLASSES, ATTR_DIMS])

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
                        model = Model(x, y_, num_tasks, opt, imp_method, synap_stgth, args.fisher_update_after,
                                args.fisher_ema_decay, network_arch=args.arch, is_ATT_DATASET=True, attr=attr)
                    else:
                        model = Model(x_aug, y_, num_tasks, opt, imp_method, synap_stgth, args.fisher_update_after, 
                                args.fisher_ema_decay, network_arch=args.arch, is_ATT_DATASET=True, x_test=x, attr=attr)

                    # Set up tf session and initialize variables.
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True

                    time_start = time.time()
                    with tf.Session(config=config, graph=graph) as sess:
                        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
                        runs, task_labels_dataset = train_task_sequence(model, sess, saver, datasets, AWA_attr, CLASSES_PER_TASK, args.cross_validate_mode, 
                                args.train_single_epoch, args.do_sampling, args.is_herding, args.mem_size*CLASSES_PER_TASK*num_tasks, args.train_iters, 
                                args.batch_size, args.num_runs, args.init_checkpoint, args.online_cross_val, args.random_seed)
                        # Close the session
                        sess.close()
                    time_end = time.time()
                    time_spent = time_end - time_start
                    print('Time spent: {}'.format(time_spent))

                # Clean up
                del model

                if args.cross_validate_mode:
                    # If cross-validation flag is enabled, store the stuff in a text file
                    cross_validate_dump_file = args.log_dir + '/' + 'SPLIT_AWA_HYBRID_%s_%s'%(imp_method, args.optim) + '.txt'
                    with open(cross_validate_dump_file, 'a') as f:
                        f.write('HERDING: {} \t ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {}\n'.format(args.is_herding, args.arch, lr, synap_stgth, runs))
                else:
                    # Store all the results in one dictionary to process later
                    exper_acc = dict(mean=runs)
                    exper_labels = dict(labels=task_labels_dataset)
                    # Store the experiment output to a file
                    snapshot_experiment_eval(args.log_dir, experiment_id, exper_acc)
                    snapshot_task_labels(args.log_dir, experiment_id, exper_labels)

if __name__ == '__main__':
    main()
