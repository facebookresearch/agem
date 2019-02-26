# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Define some utility functions
"""
import numpy as np
import tensorflow as tf

def clone_variable_list(variable_list):
    """
    Clone the variable list
    """
    return [tf.identity(var) for var in variable_list]

def create_fc_layer(input, w, b, apply_relu=True):
    """
    Construct a Fully Connected layer
    Args:
        w                   Weights
        b                   Biases
        apply_relu          Apply relu (T/F)?

    Returns:
        Output of an FC layer
    """
    with tf.name_scope('fc_layer'):
        output = tf.matmul(input, w) + b
            
        # Apply relu
        if apply_relu:
            output = tf.nn.relu(output)

    return output

def create_conv_layer(input, w, b, stride=1, apply_relu=True):
    """
    Construct a convolutional layer
    Args:
        w                   Weights
        b                   Biases
        pre_activations     List where the pre_activations will be stored
        apply_relu          Apply relu (T/F)?

    Returns:
        Output of a conv layer
    """
    with tf.name_scope('conv_layer'):
        # Do the convolution operation
        output = tf.nn.conv2d(input, w, [1, stride, stride, 1], padding='SAME') + b

        # Apply relu
        if apply_relu:
            output = tf.nn.relu(output)

    return output

def load_task_specific_data_in_proportion(datasets, task_labels, classes_appearing_in_tasks, class_seen_already):
    """
    Loads task specific data from the datasets proportionate to classes appearing in different tasks
    """
    global_class_indices = np.column_stack(np.nonzero(datasets['labels']))
    count = 0
    for cls in task_labels:
        if count == 0:
            class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == cls][:,np.array([True, False])])
            total_class_instances = class_indices.size
            num_instances_to_choose = total_class_instances // classes_appearing_in_tasks[cls]
            offset = (class_seen_already[cls] - 1) * num_instances_to_choose
            final_class_indices = class_indices[offset: offset+num_instances_to_choose]
        else:
            current_class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == cls][:,np.array([True, False])])
            total_class_instances = current_class_indices.size
            num_instances_to_choose = total_class_instances // classes_appearing_in_tasks[cls]
            offset = (class_seen_already[cls] - 1) * num_instances_to_choose
            final_class_indices = np.append(final_class_indices, current_class_indices[offset: offset+num_instances_to_choose])
        count += 1
    final_class_indices = np.sort(final_class_indices, axis=None)
    return datasets['images'][final_class_indices, :], datasets['labels'][final_class_indices, :]


def load_task_specific_data(datasets, task_labels):
    """
    Loads task specific data from the datasets
    """
    global_class_indices = np.column_stack(np.nonzero(datasets['labels']))
    count = 0
    for cls in task_labels:
        if count == 0:
            class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == cls][:,np.array([True, False])])
        else:
            class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:,1] == cls][:,np.array([True, False])]))
        count += 1
    class_indices = np.sort(class_indices, axis=None)
    return datasets['images'][class_indices, :], datasets['labels'][class_indices, :]

def samples_for_each_class(dataset_labels, task):
    """
    Numbers of samples for each class in the task
    Args:
        dataset_labels  Labels to count samples from
        task            Labels with in a task

    Returns
    """
    num_samples = np.zeros([len(task)], dtype=np.float32)
    i = 0
    for label in task:
        global_class_indices = np.column_stack(np.nonzero(dataset_labels))
        class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == label][:,np.array([True, False])])
        class_indices = np.sort(class_indices, axis=None)
        num_samples[i] = len(class_indices)
        i += 1

    return num_samples


def get_sample_weights(labels, tasks):
    weights = np.zeros([labels.shape[0]], dtype=np.float32)
    for label in tasks:
        global_class_indices = np.column_stack(np.nonzero(labels))
        class_indices = np.array(np.squeeze(global_class_indices[global_class_indices[:,1] == label][:,np.array([True, False])]))
        total_class_samples = class_indices.shape[0]
        weights[class_indices] = 1.0/ total_class_samples

    # Rescale the weights such that min is 1. This will make the weights of less observed
    # examples 1. 
    weights /= weights.min()

    return weights

def update_episodic_memory_with_less_data(task_dataset, importance_array, total_mem_size, task, episodic_images, episodic_labels, task_labels=None, is_herding=False):
    """
    Update the episodic memory when the task data is less than the memory size
    Args:

    Returns:
    """
    num_examples_in_task = task_dataset['images'].shape[0]
    # Empty spaces in the episodic memory
    empty_spaces = np.sum(np.sum(episodic_labels, axis=1) == 0)
    if empty_spaces >= num_examples_in_task:
        # Find where the empty spaces are in order
        empty_indices = np.where(np.sum(episodic_labels, axis=1) == 0)[0]
        # Store the whole task data in the episodic memory
        episodic_images[empty_indices[:num_examples_in_task]] = task_dataset['images']
        episodic_labels[empty_indices[:num_examples_in_task]] = task_dataset['labels']
    elif empty_spaces == 0:
        # Compute the amount of space in the episodic memory for the new task
        space_for_new_task = total_mem_size// (task + 1) # task 0, 1, ...
        # Get the indices to update in the episodic memory
        eps_mem_indices = np.random.choice(total_mem_size, space_for_new_task, replace=False) # Sample without replacement
        # Get the indices of important samples from the task dataset
        label_importance = importance_array + 1e-32
        label_importance /= np.sum(label_importance) # Convert to a probability distribution
        task_mem_indices = np.random.choice(num_examples_in_task, space_for_new_task, p=label_importance, replace=False) # Sample without replacement
        # Update the episodic memory
        episodic_images[eps_mem_indices] = task_dataset['images'][task_mem_indices]
        episodic_labels[eps_mem_indices] = task_dataset['labels'][task_mem_indices]
    else:
        # When there is some free space but not enough to store the whole task
        # Find where the empty spaces are in order
        empty_indices = np.where(np.sum(episodic_labels, axis=1) == 0)[0]
        # Store some of the examples from task in the memory
        episodic_images[empty_indices] = task_dataset['images'][:len(empty_indices)]
        episodic_labels[empty_indices] = task_dataset['labels'][:len(empty_indices)]
        # Adjust the remanining samples in the episodic memory
        space_for_new_task = (total_mem_size // (task + 1))  - len(empty_indices) # task 0, 1, ...
        # Get the indices to update in the episodic memory
        eps_mem_indices = np.random.choice((total_mem_size - len(empty_indices)), space_for_new_task, replace=False) # Sample without replacement
        # Get the indices of important samples from the task dataset
        label_importance = importance_array[len(empty_indices):] + 1e-32
        label_importance /= np.sum(label_importance) # Convert to a probability distribution
        updated_num_examples_in_task = num_examples_in_task - len(empty_indices)
        task_mem_indices = np.random.choice(updated_num_examples_in_task, space_for_new_task, p=label_importance, replace=False) # Sample without replacement
        task_mem_indices += len(empty_indices) # Add the offset
        # Update the episodic memory
        episodic_images[eps_mem_indices] = task_dataset['images'][task_mem_indices]
        episodic_labels[eps_mem_indices] = task_dataset['labels'][task_mem_indices]

def update_episodic_memory(task_dataset, importance_array, total_mem_size, task, episodic_images, episodic_labels, task_labels=None, is_herding=False):
    """
    Update the episodic memory with new task data
    Args:

    Reruns:
    """
    num_examples_in_task = task_dataset['images'].shape[0]
    # Compute the amount of space in the episodic memory for the new task
    space_for_new_task = total_mem_size// (task + 1) # task 0, 1, ...
    # Get the indices to update in the episodic memory
    eps_mem_indices = np.random.choice(total_mem_size, space_for_new_task, replace=False) # Sample without replacement
    if is_herding and task_labels is not None:
        # Get the samples based on herding
        imp_images, imp_labels = sample_from_dataset_icarl(task_dataset, importance_array, task_labels, space_for_new_task//len(task_labels))
        episodic_images[eps_mem_indices[np.arange(imp_images.shape[0])]] = imp_images
        episodic_labels[eps_mem_indices[np.arange(imp_images.shape[0])]] = imp_labels
    else:
        # Get the indices of important samples from the task dataset
        label_importance = importance_array + 1e-32
        label_importance /= np.sum(label_importance) # Convert to a probability distribution
        task_mem_indices = np.random.choice(num_examples_in_task, space_for_new_task, p=label_importance, replace=False) # Sample without replacement
        # Update the episodic memory
        episodic_images[eps_mem_indices] = task_dataset['images'][task_mem_indices]
        episodic_labels[eps_mem_indices] = task_dataset['labels'][task_mem_indices]

def sample_from_dataset(dataset, importance_array, task, samples_count, preds=None):
    """
    Samples from a dataset based on a probability distribution
    Args:
        dataset             Dataset to sample from
        importance_array    Importance scores (not necessarily have to be a prob distribution)
        task                Labels with in a task
        samples_count       Number of samples to return

    Return:
        images              Important images
        labels              Important labels
    """
   
    count = 0
    # For each label in the task extract the important samples
    for label in task:
        global_class_indices = np.column_stack(np.nonzero(dataset['labels']))
        class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == label][:,np.array([True, False])])
        class_indices = np.sort(class_indices, axis=None)
      
        if (preds is not None):
            # Find the indices where prediction match the correct label
            pred_indices = np.where(preds == label)[0]

            # Find the correct prediction indices
            correct_pred_indices = np.intersect1d(pred_indices, class_indices)

        else:
            correct_pred_indices = class_indices

        # Extract the importance for the label
        label_importance = importance_array[correct_pred_indices] + 1e-32
        label_importance /= np.sum(label_importance)

        actual_samples_count = min(samples_count, np.count_nonzero(label_importance))
        #print('Storing {} samples from {} class'.format(actual_samples_count, label))

        # If no samples are correctly classified then skip saving the samples
        if (actual_samples_count != 0):

            # Extract the important indices
            imp_indices = np.random.choice(correct_pred_indices, actual_samples_count, p=label_importance, replace=False)

            if count == 0:
                images = dataset['images'][imp_indices]
                labels = dataset['labels'][imp_indices]
            else:
                images = np.vstack((images, dataset['images'][imp_indices]))
                labels = np.vstack((labels, dataset['labels'][imp_indices]))

            count += 1

    if count != 0:
        return images, labels
    else:
        return None, None
  
def concatenate_datasets(current_images, current_labels, prev_images, prev_labels):
    """
    Concatnates current dataset with the previous one. This will be used for
    adding important samples from the previous datasets
    Args:
        current_images      Images of current dataset
        current_labels      Labels of current dataset
        prev_images         List containing images of previous datasets
        prev_labels         List containing labels of previous datasets

    Returns:
        images              Concatenated images
        labels              Concatenated labels
    """
    """
    images = current_images
    labels = current_labels
    for i in range(len(prev_images)):
        images = np.vstack((images, prev_images[i]))
        labels = np.vstack((labels, prev_labels[i]))
    """
    images = np.concatenate((current_images, prev_images), axis=0)
    labels = np.concatenate((current_labels, prev_labels), axis=0)
        
    return images, labels


def sample_from_dataset_icarl(dataset, features, task, samples_count, preds=None):
    """
    Samples from a dataset based on a icarl - mean of features
    Args:
        dataset             Dataset to sample from
        features            Features - activation before the last layer
        task                Labels with in a task
        samples_count       Number of samples to return

    Return:
        images              Important images
        labels              Important labels
    """

    print('Herding based sampling!')
    #samples_count = min(samples_count, dataset['images'].shape[0])
    count = 0
    # For each label in the task extract the important samples
    for label in task:
        global_class_indices = np.column_stack(np.nonzero(dataset['labels']))
        class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == label][:,np.array([True, False])])
        class_indices = np.sort(class_indices, axis=None)

        if (preds is not None):
            # Find the indices where prediction match the correct label
            pred_indices = np.where(preds == label)[0]

            # Find the correct prediction indices
            correct_pred_indices = np.intersect1d(pred_indices, class_indices)

        else:
            correct_pred_indices = class_indices

        mean_feature = np.mean(features[correct_pred_indices, :], axis=0)

        actual_samples_count = min(samples_count, len(correct_pred_indices))

        # If no samples are correctly classified then skip saving the samples
        imp_indices = np.zeros(actual_samples_count, dtype=np.int32)
        sample_sum= np.zeros(mean_feature.shape)
        if (actual_samples_count != 0):
            # Extract the important indices
            for i in range(actual_samples_count):
                sample_mean = (features[correct_pred_indices, :] +
                        np.tile(sample_sum, [len(correct_pred_indices),1]))/ float(i + 1)
                norm_distance = np.linalg.norm((np.tile(mean_feature, [len(correct_pred_indices),1])
                        - sample_mean), ord=2, axis=1)
                imp_indices[i] = correct_pred_indices[np.argmin(norm_distance)]
                sample_sum = sample_sum + features[imp_indices[i], :]

            if count == 0:
                images = dataset['images'][imp_indices]
                labels = dataset['labels'][imp_indices]
            else:
                images = np.vstack((images, dataset['images'][imp_indices]))
                labels = np.vstack((labels, dataset['labels'][imp_indices]))

            count += 1

    if count != 0:
        return images, labels
    else:
        return None, None  

def average_acc_stats_across_runs(data, key):
    """
    Compute the average accuracy statistics (mean and std) across runs
    """
    num_runs = data.shape[0]
    avg_acc = np.zeros(num_runs)
    for i in range(num_runs):
        avg_acc[i] = np.mean(data[i][-1])

    return avg_acc.mean()*100, avg_acc.std()*100

def average_fgt_stats_across_runs(data, key):
    """
    Compute the forgetting statistics (mean and std) across runs
    """
    num_runs = data.shape[0]
    fgt = np.zeros(num_runs)
    wst_fgt = np.zeros(num_runs)
    for i in range(num_runs):
        fgt[i] = compute_fgt(data[i])

    return fgt.mean(), fgt.std()

def compute_fgt(data):
    """
    Given a TxT data matrix, compute average forgetting at T-th task
    """
    num_tasks = data.shape[0]
    T = num_tasks - 1
    fgt = 0.0
    for i in range(T):
        fgt += np.max(data[:T,i]) - data[T, i]

    avg_fgt = fgt/ float(num_tasks - 1)
    return avg_fgt

def update_reservior(current_image, current_label, episodic_images, episodic_labels, M, N):
    """
    Update the episodic memory with current example using the reservior sampling
    """
    if M > N:
        episodic_images[N] = current_image
        episodic_labels[N] = current_label
    else:
        j = np.random.randint(0, N)
        if j < M:
           episodic_images[j] = current_image
           episodic_labels[j] = current_label
