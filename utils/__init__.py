# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .data_utils import construct_permute_mnist, construct_split_mnist, construct_split_cifar, construct_split_cub, construct_split_imagenet 
from .data_utils import image_scaling, random_crop_and_pad_image, random_horizontal_flip 
from .utils import clone_variable_list, create_fc_layer, create_conv_layer, sample_from_dataset, update_episodic_memory, update_episodic_memory_with_less_data, concatenate_datasets 
from .utils import samples_for_each_class, sample_from_dataset_icarl, get_sample_weights, compute_fgt, load_task_specific_data, load_task_specific_data_in_proportion
from .utils import average_acc_stats_across_runs, average_fgt_stats_across_runs, update_reservior
from .vis_utils import plot_acc_multiple_runs, plot_histogram, snapshot_experiment_meta_data, snapshot_experiment_eval, snapshot_task_labels
from .resnet_utils import _conv, _fc, _bn, _residual_block, _residual_block_first
from .vgg_utils import vgg_conv_layer, vgg_fc_layer
