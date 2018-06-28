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

def create_conv_layer(input, w, b, apply_relu=True):
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
        output = tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME') + b

        # Apply relu
        if apply_relu:
            output = tf.nn.relu(output)

    return output

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

    #samples_count = min(samples_count, dataset['images'].shape[0])
    count = 0
    # For each label in the task extract the important samples
    for label in task:
        global_class_indices = np.column_stack(np.nonzero(dataset['labels']))
        class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == label][:,np.array([True, False])])
        class_indices = np.sort(class_indices, axis=None)
        print('Samples in class {}: {}'.format(label, len(class_indices)))

        if (preds is not None):
            # Find the indices where prediction match the correct label
            pred_indices = np.where(preds == label)[0]

            # Find the correct prediction indices
            correct_pred_indices = np.intersect1d(pred_indices, class_indices)

        else:
            correct_pred_indices = class_indices

        mean_feature = np.mean(features[correct_pred_indices, :], axis=0)

        actual_samples_count = min(samples_count, len(correct_pred_indices))
        print(actual_samples_count)

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
