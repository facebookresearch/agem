"""
Training script for split MNIST experiment.
"""
from __future__ import print_function

import argparse
import os
import sys

import datetime
import numpy as np
import tensorflow as tf
from copy import deepcopy
from six.moves import cPickle as pickle

from utils.data_utils import construct_permute_mnist, construct_split_mnist 
from utils.utils import get_sample_weights, sample_from_dataset, concatenate_datasets, samples_for_each_class, sample_from_dataset_icarl
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
OPTIM = 'ADAM'
OPT_POWER = 0.9
OPT_MOMENTUM = 0.9

## Model options
MODELS = ['VAN', 'PI', 'EWC', 'MAS', 'RWALK'] #List of valid models 
IMP_METHOD = 'EWC'
SYNAP_STGTH = 75000
FISHER_EMA_DECAY = 0.9      # Exponential moving average decay factor for Fisher computation (online Fisher)
FISHER_UPDATE_AFTER = 10    # Number of training iterations for which the F_{\theta}^t is computed (see Eq. 10 in RWalk paper) 
MEMORY_SIZE_PER_TASK = 10   # Number of samples per task
INPUT_FEATURE_SIZE = 784
TOTAL_CLASSES = 10          # Total number of classes in the dataset 

## Logging, saving and testing options
LOG_DIR = './split_mnist_results'

## Evaluation options

## Task split
TASK_LABELS= [[0,1],[2,3],[4,5],[6,7],[8,9]]
#TASK_LABELS= [[1,4],[2,0],[3,7],[8,9],[6,5]]
#TASK_LABELS= [[7,5],[9,1],[3,6],[4,2],[8,0]]

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for split mnist experiment.")
    parser.add_argument("--train-single-epoch", action="store_true",
                       help="If option is chosen then train for single epoch")
    parser.add_argument("--eval-single-head", action="store_true",
                       help="If option is chosen then evaluate on a single head setting.")
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
    parser.add_argument("--mem-size", type=int, default=MEMORY_SIZE_PER_TASK,
                       help="Number of samples per class from previous tasks.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                       help="Directory where the plots and model accuracies will be stored.")
    return parser.parse_args()

def train_task_sequence(model, sess, datasets, task_labels, train_single_epoch, eval_single_head, do_sampling, 
        samples_per_class, train_iters, batch_size, num_runs):
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

        # List to store important samples from the previous tasks
        last_task_x = None
        last_task_y_ = None

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
            if train_single_epoch:
                # TODO: Use a fix number of batches for now to avoid complicated logic while averaging accuracies
                #num_iters = num_train_examples// batch_size
                num_iters = 10000 // batch_size
            else:
                num_iters = train_iters
                # Set the mask only once before starting the training for the task
                if do_sampling:
                    model.set_active_outputs(sess, test_labels)
                else:
                    model.set_active_outputs(sess, task_labels[task])

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

                if train_single_epoch:
                    if (iters < 10) or (iters % 50 == 0):
                        # Snapshot the current performance across all tasks after each mini-batch
                        fbatch = test_task_sequence(model, sess, datasets, task_labels, eval_single_head=eval_single_head)
                        ftask.append(fbatch)
                        # Set the output labels over which the model needs to be trained 
                        if do_sampling:
                            model.set_active_outputs(sess, test_labels)
                        else:
                            model.set_active_outputs(sess, task_labels[task])

                offset = (iters * batch_size) % (num_train_examples - batch_size)

                feed_dict = {model.x: train_x[offset:offset+batch_size], model.y_: train_y[offset:offset+batch_size], 
                        model.sample_weights: task_sample_weights[offset:offset+batch_size],
                        model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0}

                if model.imp_method == 'VAN':
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

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


                if (iters % 100 == 0):
                    print('Step {:d} {:.3f}'.format(iters, loss))

            print('\t\t\t\tTraining for Task%d done!'%(task))

            # Compute the inter-task updates, Fisher/ importance scores etc
            # Don't calculate the task updates for the last task
            if task < len(datasets) - 1:
                model.task_updates(sess, task, task_train_images)
                print('\t\t\t\tTask updates after Task%d done!'%(task))

                # If sampling flag is set, store few of the samples from previous task
                if do_sampling:
                    # Do the uniform sampling/ only get examples from current task
                    importance_array = np.ones([datasets[task]['train']['images'].shape[0]], dtype=np.float32)
                    # Get the important samples from the current task
                    imp_images, imp_labels = sample_from_dataset(datasets[task]['train'], importance_array, 
                            task_labels[task], samples_per_class)

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

            if train_single_epoch: 
                fbatch = test_task_sequence(model, sess, datasets, task_labels, eval_single_head=eval_single_head)
                ftask.append(fbatch)
                ftask = np.array(ftask)
            else:
                # List to store accuracy for all the tasks for the current trained model
                ftask = test_task_sequence(model, sess, datasets, task_labels, eval_single_head=eval_single_head)
            
            # Store the accuracies computed at task T in a list
            evals.append(ftask)

            # Reset the optimizer
            model.reset_optimizer(sess)

            #-> End for loop task

        runs.append(np.array(evals))
        # End for loop runid

    runs = np.array(runs)

    return runs

def test_task_sequence(model, sess, test_data, test_tasks, eval_single_head=True):
    """
    Snapshot the current performance
    """
    list_acc = []

    if eval_single_head:
        # Single-head evaluation setting
        classes_to_distinguish = []
        for task in test_tasks:
            classes_to_distinguish += task
        model.set_active_outputs(sess, classes_to_distinguish)

    for task, labels in enumerate(test_tasks):
        if not eval_single_head:
            # Multi-head evaluation setting
            model.set_active_outputs(sess, labels)

        feed_dict = {model.x: test_data[task]['test']['images'], 
                model.y_: test_data[task]['test']['labels'], model.keep_prob: 1.0}
        acc = model.accuracy.eval(feed_dict = feed_dict)
        list_acc.append(acc)

    return list_acc

def main():
    """
    Create the model and start the training
    """

    # Get the CL arguments
    args = get_arguments()

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
    exper_meta_data = {'DATASET': 'SPLIT_MNIST',
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
            'MEM_SIZE': args.mem_size}
    experiment_id = "SPLIT_MNIST_%r_%r_%s_%s_%s_%r_%s-"%(args.eval_single_head, args.train_single_epoch, args.imp_method, str(args.synap_stgth).replace('.', '_'), 
            str(args.batch_size), args.do_sampling, str(args.mem_size)) + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    snapshot_experiment_meta_data(args.log_dir, experiment_id, exper_meta_data)

    # Load the split mnist dataset
    datasets = construct_split_mnist(TASK_LABELS)

    #for i in range(len(datasets)):
    #    print(np.unique(np.column_stack(np.nonzero(datasets[i]['train']['labels']))[:,1])

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
        y_ = tf.placeholder(tf.float32, shape=[None, TOTAL_CLASSES])
        sample_weights = tf.placeholder(tf.float32, shape=[None])
        train_step = tf.placeholder(dtype=tf.float32, shape=())
        keep_prob = tf.placeholder(dtype=tf.float32, shape=())
        train_samples = tf.placeholder(dtype=tf.float32, shape=())
        training_iters = tf.placeholder(dtype=tf.float32, shape=())

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
        model = Model(x, y_, sample_weights, keep_prob, train_samples, training_iters, train_step, 
                opt, args.imp_method, args.synap_stgth, args.fisher_update_after, args.fisher_ema_decay, 
                network_arch='FC')

        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config, graph=graph) as sess:
            runs = train_task_sequence(model, sess, datasets, TASK_LABELS, args.train_single_epoch, args.eval_single_head, 
                    args.do_sampling, args.mem_size, args.train_iters, args.batch_size, args.num_runs)
            # Close the session
            sess.close()

    # Compute the mean and std
    acc_mean = runs.mean(0)
    acc_std = runs.std(0)

    # Store all the results in one dictionary to process later
    exper_acc = dict(mean=acc_mean, std=acc_std)

    if args.train_single_epoch:
        print('A5: {}'.format(acc_mean[-1,-1,:].mean()))
    else:
        print(exper_acc)
        print('A5: {}'.format(acc_mean[-1,:].mean()))

    # Store the experiment output to a file
    snapshot_experiment_eval(args.log_dir, experiment_id, exper_acc)

if __name__ == '__main__':
    main()
