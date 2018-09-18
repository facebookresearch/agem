"""
Training script for permute MNIST experiment.
"""
from __future__ import print_function

import argparse
import os
import sys
import math
import time

import datetime
import numpy as np
import tensorflow as tf
from copy import deepcopy
from six.moves import cPickle as pickle

from utils.data_utils import construct_permute_mnist
from utils.utils import get_sample_weights, sample_from_dataset, update_episodic_memory, concatenate_datasets, samples_for_each_class, sample_from_dataset_icarl, compute_fgt
from utils.vis_utils import plot_acc_multiple_runs, plot_histogram, snapshot_experiment_meta_data, snapshot_experiment_eval
from model import Model

###############################################################
################ Some definitions #############################
### These will be edited by the command line options ##########
###############################################################

## Training Options
NUM_RUNS = 10           # Number of experiments to average over
TRAIN_ITERS = 5000      # Number of training iterations per task
BATCH_SIZE = 16
LEARNING_RATE = 1e-3    
RANDOM_SEED = 1234
VALID_OPTIMS = ['SGD', 'MOMENTUM', 'ADAM']
OPTIM = 'SGD'
OPT_POWER = 0.9
OPT_MOMENTUM = 0.9
VALID_ARCHS = ['FC-S', 'FC-B']
ARCH = 'FC-S'

## Model options
MODELS = ['VAN', 'PI', 'EWC', 'MAS', 'GEM', 'RWALK', 'S-GEM', 'M-GEM','FTR_EXT'] #List of valid models 
IMP_METHOD = 'EWC'
SYNAP_STGTH = 75000
FISHER_EMA_DECAY = 0.9      # Exponential moving average decay factor for Fisher computation (online Fisher)
FISHER_UPDATE_AFTER = 10    # Number of training iterations for which the F_{\theta}^t is computed (see Eq. 10 in RWalk paper) 
SAMPLES_PER_CLASS = 25   # Number of samples per task
INPUT_FEATURE_SIZE = 784
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHANNELS = 1
TOTAL_CLASSES = 10          # Total number of classes in the dataset 
EPS_MEM_BATCH_SIZE = 256
KEEP_EPISODIC_MEMORY_FULL = False
DEBUG_EPISODIC_MEMORY = False
USE_GPU = True
K_FOR_CROSS_VAL = 3
TIME_MY_METHOD = False
COUNT_VIOLATIONS = False
MEASURE_PERF_ON_EPS_MEMORY = True

## Logging, saving and testing options
LOG_DIR = './permute_mnist_results'

## Evaluation options

## Num Tasks
NUM_TASKS = 20
MULTI_TASK = False

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for permutted mnist experiment.")
    parser.add_argument("--cross-validate-mode", action="store_true",
                       help="If option is chosen then snapshoting after each batch is disabled")
    parser.add_argument("--online-cross-val", action="store_true",
                       help="If option is chosen then enable the online cross validation of the learning rate")
    parser.add_argument("--train-single-epoch", action="store_true",
                       help="If option is chosen then train for single epoch")
    parser.add_argument("--eval-single-head", action="store_true",
                       help="If option is chosen then evaluate on a single head setting.")
    parser.add_argument("--arch", type=str, default=ARCH, help="Network Architecture for the experiment.\
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
    parser.add_argument("--is-herding", action="store_true",
                        help="Herding based sampling")
    parser.add_argument("--mem-size", type=int, default=SAMPLES_PER_CLASS,
                       help="Number of samples per class from previous tasks.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                       help="Directory where the plots and model accuracies will be stored.")
    return parser.parse_args()

def train_task_sequence(model, sess, datasets, cross_validate_mode, train_single_epoch, eval_single_head, do_sampling, is_herding, 
        episodic_mem_size, train_iters, batch_size, num_runs, online_cross_val):
    """
    Train and evaluate LLL system such that we only see a example once
    Args:
    Returns:
        dict    A dictionary containing mean and stds for the experiment
    """
    # List to store accuracy for each run
    runs = []

    # Loop over number of runs to average over
    for runid in range(num_runs):
        print('\t\tRun %d:'%(runid))

        # Initialize all the variables in the model
        sess.run(tf.global_variables_initializer())

        # Run the init ops
        model.init_updates(sess)

        # List to store accuracies for a run
        evals = []

        # List to store the classes that we have so far - used at test time
        test_labels = []

        if model.imp_method == 'GEM' or model.imp_method == 'M-GEM':
            # List to store the episodic memories of the previous tasks
            task_based_memory = []

        if model.imp_method == 'S-GEM':
            # Reserve a space for episodic memory
            episodic_images = np.zeros([episodic_mem_size, INPUT_FEATURE_SIZE])
            episodic_labels = np.zeros([episodic_mem_size, TOTAL_CLASSES])
            episodic_filled_counter = 0

        if do_sampling:
            # List to store important samples from the previous tasks
            last_task_x = None
            last_task_y_ = None

        # Mask for softmax
        # Since all the classes are present in all the tasks so nothing to mask
        logit_mask = np.ones(TOTAL_CLASSES)
        if COUNT_VIOLATIONS:
            violation_count = np.zeros(model.num_tasks)
            vc = 0

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

            # If multi_task is set the train using datasets of all the tasks
            if MULTI_TASK:
                if task == 0:
                    for t_ in range(1, len(datasets)):
                        task_train_images = np.concatenate((task_train_images, datasets[t_]['train']['images']), axis=0)
                        task_train_labels = np.concatenate((task_train_labels, datasets[t_]['train']['labels']), axis=0)
                else:
                    # Skip training for this task
                    continue
            print('Received {} images, {} labels at task {}'.format(task_train_images.shape[0], task_train_labels.shape[0], task))

            # Declare variables to store sample importance if sampling flag is set
            if do_sampling:
                # Get the sample weighting
                task_sample_weights = get_sample_weights(task_train_labels, test_labels)
            else:
                # Assign equal weights to all the examples
                task_sample_weights = np.ones([task_train_labels.shape[0]], dtype=np.float32)

            num_train_examples = task_train_images.shape[0]

            # Train a task observing sequence of data
            if train_single_epoch:
                num_iters = num_train_examples // batch_size
            else:
                num_iters = train_iters

            # Randomly suffle the training examples
            perm = np.arange(num_train_examples)
            np.random.shuffle(perm)
            train_x = task_train_images[perm]
            train_y = task_train_labels[perm]
            task_sample_weights = task_sample_weights[perm]

            # Array to store accuracies when training for task T
            ftask = []

            # Training loop for task T
            for iters in range(num_iters):

                if train_single_epoch and not cross_validate_mode:
                    if (iters < 10) or (iters < 100 and iters % 10 == 0) or (iters % 100 == 0):
                        # Snapshot the current performance across all tasks after each mini-batch
                        fbatch = test_task_sequence(model, sess, datasets, online_cross_val, eval_single_head=eval_single_head)
                        ftask.append(fbatch)

                offset = (iters * batch_size) % (num_train_examples - batch_size)

                feed_dict = {model.x: train_x[offset:offset+batch_size], model.y_: train_y[offset:offset+batch_size], 
                #feed_dict = {model.x: np.reshape(train_x[offset:offset+batch_size], (-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)), model.y_: train_y[offset:offset+batch_size], 
                        model.sample_weights: task_sample_weights[offset:offset+batch_size],
                        model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0, 
                        model.output_mask: logit_mask, model.train_phase: True}

                if model.imp_method == 'VAN':
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'FTR_EXT':
                    if task == 0:
                        _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
                    else:
                        _, loss = sess.run([model.train_classifier, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'EWC':
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                    # Update fisher after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        sess.run(model.set_running_fisher)
                        sess.run(model.reset_tmp_fisher)
                    
                    _, _, loss = sess.run([model.set_tmp_fisher, model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'PI':
                    _, _, _, loss = sess.run([model.weights_old_ops_grouped, model.train, model.update_small_omega, 
                                              model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'MAS':
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'GEM':
                    if task == 0:
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.reg_loss], feed_dict=feed_dict)
                    else:
                        # Compute the gradients on the episodic memory of all the previous tasks
                        for prev_task in range(task):
                            # T-th task gradients.
                            sess.run(model.store_ref_grads, feed_dict={model.x: task_based_memory[prev_task]['images'],
                                model.y_: task_based_memory[prev_task]['labels'], model.task_id: prev_task, model.keep_prob: 1.0, 
                                model.output_mask: logit_mask, model.train_phase: True})

                        # Compute the gradient on the mini-batch of the current task
                        feed_dict[model.task_id] = task
                        _, loss = sess.run([model.train_subseq_tasks, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'M-GEM':
                    if task == 0:
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.reg_loss], feed_dict=feed_dict)
                    else:
                        # Randomly sample a task from the previous tasks
                        prev_task = np.random.randint(0, task)
                        # Store the reference gradient
                        sess.run(model.store_ref_grads, feed_dict={model.x: task_based_memory[prev_task]['images'], model.y_: task_based_memory[prev_task]['labels'],
                            model.keep_prob: 1.0, model.output_mask: logit_mask, model.train_phase: True})
                        # Compute the gradient for current task and project if need be
                        _, loss = sess.run([model.train_subseq_tasks, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'S-GEM':
                    if task == 0:
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.reg_loss], feed_dict=feed_dict)
                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        if KEEP_EPISODIC_MEMORY_FULL:
                            mem_sample_mask = np.random.choice(episodic_mem_size, EPS_MEM_BATCH_SIZE, replace=False) # Sample without replacement so that we don't sample an example more than once
                        else:
                            if episodic_filled_counter <= EPS_MEM_BATCH_SIZE:
                                mem_sample_mask = np.arange(episodic_filled_counter)
                            else:
                                # Sample a random subset from episodic memory buffer
                                mem_sample_mask = np.random.choice(episodic_filled_counter, EPS_MEM_BATCH_SIZE, replace=False) # Sample without replacement so that we don't sample an example more than once
                        # Store the reference gradient
                        sess.run(model.store_ref_grads, feed_dict={model.x: episodic_images[mem_sample_mask], model.y_: episodic_labels[mem_sample_mask],
                            model.keep_prob: 1.0, model.output_mask: logit_mask, model.train_phase: True})
                        if COUNT_VIOLATIONS:
                            vc, _, loss = sess.run([model.violation_count, model.train_subseq_tasks, model.reg_loss], feed_dict=feed_dict)
                        else:
                            # Compute the gradient for current task and project if need be
                            _, loss = sess.run([model.train_subseq_tasks, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'RWALK':
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

                if (iters % 500 == 0):
                    print('Step {:d} {:.3f}'.format(iters, loss))

                if (math.isnan(loss)):
                    print('ERROR: NaNs NaNs Nans!!!')
                    sys.exit(0)

            print('\t\t\t\tTraining for Task%d done!'%(task))

            if model.imp_method == 'S-GEM' and COUNT_VIOLATIONS:
                violation_count[task] = vc
                print('Task {}: Violation Count: {}'.format(task, violation_count))
                sess.run(model.reset_violation_count, feed_dict=feed_dict)

            # Compute the inter-task updates, Fisher/ importance scores etc
            # Don't calculate the task updates for the last task
            if task < len(datasets) - 1 or MEASURE_PERF_ON_EPS_MEMORY:
                model.task_updates(sess, task, task_train_images, np.arange(TOTAL_CLASSES))
                print('\t\t\t\tTask updates after Task%d done!'%(task))

                # If importance method is '*-GEM' then store the episodic memory for the task
                if 'GEM' in model.imp_method:
                    data_to_sample_from = {
                            'images': task_train_images,
                            'labels': task_train_labels,
                            }
                    if model.imp_method == 'GEM' or model.imp_method == 'M-GEM':
                        # Get the important samples from the current task
                        if is_herding: # Sampling based on MoF
                            # Compute the features of training data
                            features_dim = model.image_feature_dim
                            features = np.zeros([num_train_examples, features_dim])
                            samples_at_a_time = 100
                            for i in range(num_train_examples// samples_at_a_time):
                                offset = i * samples_at_a_time
                                features[offset:offset+samples_at_a_time] = sess.run(model.features, feed_dict={model.x: task_train_images[offset:offset+samples_at_a_time], 
                                    model.y_: task_train_labels[offset:offset+samples_at_a_time], model.keep_prob: 1.0, 
                                    model.output_mask: logit_mask, model.train_phase: False})
                            imp_images, imp_labels = sample_from_dataset_icarl(data_to_sample_from, features, np.arange(TOTAL_CLASSES), SAMPLES_PER_CLASS)
                        else: # Random sampling
                            # Do the uniform sampling
                            importance_array = np.ones(num_train_examples, dtype=np.float32)
                            imp_images, imp_labels = sample_from_dataset(data_to_sample_from, importance_array, np.arange(TOTAL_CLASSES), SAMPLES_PER_CLASS)
                        task_memory = {
                                'images': deepcopy(imp_images),
                                'labels': deepcopy(imp_labels),
                                }
                        task_based_memory.append(task_memory)

                    elif model.imp_method == 'S-GEM':
                        if is_herding: # Sampling based on MoF
                            # Compute the features of training data
                            features_dim = model.image_feature_dim
                            features = np.zeros([num_train_examples, features_dim])
                            samples_at_a_time = 100
                            for i in range(num_train_examples// samples_at_a_time):
                                offset = i * samples_at_a_time
                                features[offset:offset+samples_at_a_time] = sess.run(model.features, feed_dict={model.x: task_train_images[offset:offset+samples_at_a_time], 
                                    model.y_: task_train_labels[offset:offset+samples_at_a_time], model.keep_prob: 1.0, 
                                    model.output_mask: logit_mask, model.train_phase: False})
                            if KEEP_EPISODIC_MEMORY_FULL:
                                update_episodic_memory(data_to_sample_from, features, episodic_mem_size, task, episodic_images, episodic_labels, task_labels=np.arange(TOTAL_CLASSES), is_herding=True)
                            else:
                                imp_images, imp_labels = sample_from_dataset_icarl(data_to_sample_from, features, np.arange(TOTAL_CLASSES), SAMPLES_PER_CLASS)
                        else: # Random sampling
                            # Do the uniform sampling
                            importance_array = np.ones(num_train_examples, dtype=np.float32)
                            if KEEP_EPISODIC_MEMORY_FULL:
                                update_episodic_memory(data_to_sample_from, importance_array, episodic_mem_size, task, episodic_images, episodic_labels)
                            else:
                                imp_images, imp_labels = sample_from_dataset(data_to_sample_from, importance_array, np.arange(TOTAL_CLASSES), SAMPLES_PER_CLASS)
                        if not KEEP_EPISODIC_MEMORY_FULL: # Fill the memory to always keep M/T samples per task
                            total_imp_samples = imp_images.shape[0]
                            eps_offset = task * total_imp_samples
                            episodic_images[eps_offset:eps_offset+total_imp_samples] = imp_images
                            episodic_labels[eps_offset:eps_offset+total_imp_samples] = imp_labels
                            episodic_filled_counter += total_imp_samples
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

            if train_single_epoch and not cross_validate_mode: 
                fbatch = test_task_sequence(model, sess, datasets, False, eval_single_head=eval_single_head)
                ftask.append(fbatch)
                ftask = np.array(ftask)
            else:
                if MEASURE_PERF_ON_EPS_MEMORY:
                    eps_mem = {
                            'images': episodic_images, 
                            'labels': episodic_labels,
                            }
                    # Measure perf on episodic memory
                    ftask = test_task_sequence(model, sess, eps_mem, online_cross_val, eval_single_head=eval_single_head)
                else:
                    # List to store accuracy for all the tasks for the current trained model
                    ftask = test_task_sequence(model, sess, datasets, online_cross_val, eval_single_head=eval_single_head)
            
            # Store the accuracies computed at task T in a list
            evals.append(ftask)

            # Reset the optimizer
            model.reset_optimizer(sess)

            #-> End for loop task

        runs.append(np.array(evals))
        # End for loop runid

    runs = np.array(runs)

    return runs

def test_task_sequence(model, sess, test_data, cross_validate_mode, eval_single_head=True):
    """
    Snapshot the current performance
    """
    if TIME_MY_METHOD:
        # Only compute the training time
        return np.zeros(model.num_tasks)

    list_acc = []
    logit_mask = np.ones(TOTAL_CLASSES)

    if MEASURE_PERF_ON_EPS_MEMORY:
        for task in range(model.num_tasks):
            mem_offset = task*SAMPLES_PER_CLASS*TOTAL_CLASSES
            feed_dict = {model.x: test_data['images'][mem_offset:mem_offset+SAMPLES_PER_CLASS*TOTAL_CLASSES], 
                    model.y_: test_data['labels'][mem_offset:mem_offset+SAMPLES_PER_CLASS*TOTAL_CLASSES], model.keep_prob: 1.0, 
                    model.output_mask: logit_mask, model.train_phase: False}
            acc = model.accuracy.eval(feed_dict = feed_dict)
            list_acc.append(acc)
        print(list_acc)
        return list_acc

    if cross_validate_mode:
        test_set = 'validation'
    else:
        test_set = 'test'
    for task, _ in enumerate(test_data):
        feed_dict = {model.x: test_data[task][test_set]['images'], 
        #feed_dict = {model.x: np.reshape(test_data[task][test_set]['images'], (-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)), 
                model.y_: test_data[task][test_set]['labels'], model.keep_prob: 1.0, 
                model.output_mask: logit_mask, model.train_phase: False}
        acc = model.accuracy.eval(feed_dict = feed_dict)
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

    # Generate the experiment key and store the meta data in a file
    exper_meta_data = {'DATASET': 'PERMUTE_MNIST',
            'NUM_RUNS': args.num_runs,
            'EVAL_SINGLE_HEAD': args.eval_single_head, 
            'TRAIN_SINGLE_EPOCH': args.train_single_epoch, 
            'IMP_METHOD': args.imp_method, 
            'SYNAP_STGTH': args.synap_stgth,
            'FISHER_EMA_DECAY': args.fisher_ema_decay,
            'FISHER_UPDATE_AFTER': args.fisher_update_after,
            'OPTIM': args.optim, 
            'LR': args.learning_rate, 
            'BATCH_SIZE': args.batch_size, 
            'EPS_MEMORY': args.do_sampling, 
            'MEM_SIZE': args.mem_size,
            'IS_HERDING': args.is_herding}
    experiment_id = "PERMUTE_MNIST_HERDING_%r_%s_%r_%r_%s_%s_%s_%r_%s-"%(args.is_herding, args.arch, args.eval_single_head, args.train_single_epoch, args.imp_method, str(args.synap_stgth).replace('.', '_'), 
            str(args.batch_size), args.do_sampling, str(args.mem_size)) + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    snapshot_experiment_meta_data(args.log_dir, experiment_id, exper_meta_data)

    # Load the permute mnist dataset
    datasets = construct_permute_mnist(NUM_TASKS)

    # Get the subset of data depending on training or cross-validation mode
    if args.online_cross_val:
        sub_datasets = datasets[:K_FOR_CROSS_VAL]
        num_tasks = K_FOR_CROSS_VAL
    else:
        sub_datasets = datasets[K_FOR_CROSS_VAL:]
        num_tasks = NUM_TASKS - K_FOR_CROSS_VAL

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
        x = tf.placeholder(tf.float32, shape=[None, INPUT_FEATURE_SIZE])
        #x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        y_ = tf.placeholder(tf.float32, shape=[None, TOTAL_CLASSES])

        # Define the optimizer
        if args.optim == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        elif args.optim == 'SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)

        elif args.optim == 'MOMENTUM':
            base_lr = tf.constant(args.learning_rate)
            learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - train_step / training_iters), OPT_POWER))
            opt = tf.train.MomentumOptimizer(args.learning_rate, OPT_MOMENTUM)

        # Create the Model/ contruct the graph
        model = Model(x, y_, num_tasks, opt, args.imp_method, args.synap_stgth, args.fisher_update_after, 
                args.fisher_ema_decay, network_arch=args.arch)

        # Set up tf session and initialize variables.
        if USE_GPU:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(
                    device_count = {'GPU': 0}
                    )

        time_start = time.time()
        with tf.Session(config=config, graph=graph) as sess:
            runs = train_task_sequence(model, sess, sub_datasets, args.cross_validate_mode, args.train_single_epoch, args.eval_single_head, 
                    args.do_sampling, args.is_herding, args.mem_size*TOTAL_CLASSES*num_tasks, args.train_iters, args.batch_size, args.num_runs, args.online_cross_val)
            # Close the session
            sess.close()
        time_end = time.time()
        time_spent = time_end - time_start

    # Compute the mean and std
    acc_mean = runs.mean(0)
    acc_std = runs.std(0)

    # Store all the results in one dictionary to process later
    exper_acc = dict(mean=acc_mean, std=acc_std)

    # If cross-validation flag is enabled, store the stuff in a text file
    if args.cross_validate_mode:
        cross_validate_dump_file = args.log_dir + '/' + 'PERMUTE_MNIST_%s_%s'%(args.imp_method, args.optim) + '.txt'
        with open(cross_validate_dump_file, 'a') as f:
            if MULTI_TASK:
                f.write('GPU:{} \t ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {}\n'.format(USE_GPU, args.arch, args.learning_rate, 
                    args.synap_stgth, acc_mean[-1, :].mean()))
            else:
                f.write('GPU: {} \t ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {} \t Fgt: {} \t Time: {}\n'.format(USE_GPU, args.arch, args.learning_rate, 
                    args.synap_stgth, acc_mean[-1, :].mean(), compute_fgt(acc_mean), str(time_spent)))

    # Store the experiment output to a file
    snapshot_experiment_eval(args.log_dir, experiment_id, exper_acc)

if __name__ == '__main__':
    main()
