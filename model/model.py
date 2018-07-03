"""
Model defintion
"""                                        

import tensorflow as tf        
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from utils import clone_variable_list, create_fc_layer, create_conv_layer
from utils.resnet_utils import _conv, _fc, _bn, _residual_block, _residual_block_first 

PARAM_XI_STEP = 1e-3
NEG_INF = -1e32
EPSILON = 1e-32

def weight_variable(shape, name='fc', init_type='default'):
    """
    Define weight variables
    Args:
        shape       Shape of the bias variable tensor

    Returns:
        A tensor of size shape initialized from a random normal
    """
    with tf.variable_scope(name):
        if init_type == 'default':
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            #weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weights')
        elif init_type == 'zero':
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.constant_initializer(0.1))
            #weights = tf.Variable(tf.constant(0.1, shape=shape, dtype=np.float32), name='weights')

    return weights

def bias_variable(shape, name='fc'):
    """
    Define bias variables
    Args:
        shape       Shape of the bias variable tensor

    Returns:
        A tensor of size shape initialized from a constant
    """
    with tf.variable_scope(name):
        biases = tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.1))

    return biases
    #return tf.Variable(tf.constant(0.1, shape=shape, dtype=np.float32), name='biases') #TODO: Should we initialize it from 0

class Model:
    """
    A class defining the model
    """

    def __init__(self, x, y_, sample_weights, keep_prob, train_samples, training_iters, train_step, train_phase, 
            opt, imp_method, synap_stgth, fisher_update_after, fisher_ema_decay, network_arch='FC'):
        """
        Instantiate the model
        """
        self.x = x
        self.y_ = y_
        self.total_classes = int(self.y_.get_shape()[1])
        self.sample_weights = sample_weights
        self.keep_prob = keep_prob
        self.train_samples = train_samples
        self.training_iters = training_iters
        self.train_step = train_step
        self.train_phase = train_phase
        self.imp_method = imp_method
        self.fisher_update_after = fisher_update_after
        self.fisher_ema_decay = fisher_ema_decay
        self.network_arch = network_arch

        # Define different variables
        self.weights_old = []
        self.star_vars = []
        self.small_omega_vars = []
        self.big_omega_vars = []
        self.big_omega_riemann_vars = []
        self.fisher_diagonal_at_minima = []
        self.hebbian_score_vars = []
        self.running_fisher_vars = []
        self.tmp_fisher_vars = []
        self.max_fisher_vars = []
        self.min_fisher_vars = []
        self.max_score_vars = []
        self.min_score_vars = []
        self.normalized_score_vars = []
        self.score_vars = []
        self.normalized_fisher_at_minima_vars = []
        self.weights_delta_old_vars = []

        # Define an output mask that sets for which classes we want training
        # and prediction
        self.output_mask = tf.Variable(tf.zeros(self.total_classes), trainable=False)

        # A scalar variable for previous syanpse strength
        self.synap_stgth = tf.constant(synap_stgth, shape=[1], dtype=tf.float32)

        # Define approproate network
        if self.network_arch == 'FC':
            input_feature_dim = int(self.x.get_shape()[1])
            layer_dims = [input_feature_dim, 256, 256, self.total_classes]
            self.fc_variables(layer_dims)
            logits = self.fc_feedforward(self.x, self.weights, self.biases)

        elif self.network_arch == 'CNN':
            num_channels = int(self.x.get_shape()[-1])
            self.image_size = int(self.x.get_shape()[1])
            kernels = [3, 3, 3, 3, 3]
            depth = [num_channels, 32, 32, 64, 64, 512]
            self.conv_variables(kernels, depth)
            logits = self.conv_feedforward(self.x, self.weights, self.biases, apply_dropout=True)
            
        elif self.network_arch == 'RESNET':
            # Same resnet-18 as used in GEM paper 
            kernels = [3, 3, 3, 3, 3]
            filters = [20, 20, 40, 80, 160]
            strides = [1, 0, 2, 2, 2]
            logits = self.resnet18_conv_feedforward(self.x, kernels, filters, strides)

        # Prune the predictions to only include the classes for which
        # the training data is present
        self.pruned_logits = tf.where(tf.tile(tf.equal(self.output_mask[None,:], 1.0), 
            [tf.shape(logits)[0], 1]), logits, NEG_INF*tf.ones_like(logits))

        # Save the optimizer 
        self.opt = opt

        # Create list of variables for storing different measures
        # Note: This method has to be called before calculating fisher 
        # or any other importance measure
        self.init_vars()

        # Different entropy measures/ loss definitions
        self.mse = 2.0*tf.nn.l2_loss(self.pruned_logits) # tf.nn.l2_loss computes sum(T**2)/ 2
        self.weighted_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_, 
            self.pruned_logits, self.sample_weights, reduction=tf.losses.Reduction.NONE))
        self.unweighted_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, 
            logits=self.pruned_logits))
        # TODO
        #self.unweighted_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, 
        #    logits=self.pruned_logits))

        # Create operations for loss and gradient calculation
        self.loss_and_gradients(self.imp_method)

        # Store the current weights before doing a train step
        self.get_current_weights()

        # Define the training operation here as Pathint ops depend on the train ops
        self.train_op()

        # Create operations to compute importance depending on the importance methods
        if self.imp_method == 'EWC':
            self.create_fisher_ops()
        elif self.imp_method == 'PI':
            self.create_pathint_ops()
        elif self.imp_method == 'RWALK':
            self.create_fisher_ops()
            self.create_pathint_ops()
        elif self.imp_method == 'MAS':
            self.create_hebbian_ops()

        # Create weight save and store ops
        self.weights_store_ops()

        # Summary operations for visualization
        tf.summary.scalar("unweighted_entropy", self.unweighted_entropy)
        for v in self.trainable_vars:
            tf.summary.histogram(v.name.replace(":", "_"), v)
        self.merged_summary = tf.summary.merge_all()

        # Accuracy measure
        self.correct_predictions = tf.equal(tf.argmax(self.pruned_logits, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))
   
        # Set the operations to reset the optimier when needed
        self.reset_optimizer_ops()
    
####################################################################################
#### Internal APIs of the class. These should not be called/ exposed externally ####
####################################################################################
    def fc_variables(self, layer_dims):
        """
        Defines variables for a 3-layer fc network
        Args:

        Returns:
        """

        self.weights = []
        self.biases = []
        self.trainable_vars = []

        for i in range(len(layer_dims)-1):
            w = weight_variable([layer_dims[i], layer_dims[i+1]], name='fc_%d'%(i))
            b = bias_variable([layer_dims[i+1]], name='fc_%d'%(i))
            self.weights.append(w)
            self.biases.append(b)
            self.trainable_vars.append(w)
            self.trainable_vars.append(b)

    def fc_feedforward(self, h, weights, biases, apply_dropout=False):
        """
        Forward pass through a fc network
        Args:
            h               Input image (tensor)
            weights         List of weights for a fc network
            biases          List of biases for a fc network
            apply_dropout   Whether to apply droupout (True/ False)

        Returns:
            Logits of a fc network
        """
        if apply_dropout:
            h = tf.nn.dropout(h, 1) # Apply dropout on Input?
        for (w, b) in list(zip(weights, biases))[:-1]:
            h = create_fc_layer(h, w, b)
            if apply_dropout:
                h = tf.nn.dropout(h, 1)  # Apply dropout on hidden layers?

        return create_fc_layer(h, weights[-1], biases[-1], apply_relu=False)

    def conv_variables(self, kernel, depth):
        """
        Defines variables of a 5xconv-1xFC convolutional network
        Args:

        Returns:
        """
        self.weights = []
        self.biases = []
        self.trainable_vars = []
        div_factor = 1

        for i in range(len(kernel)):
            w = weight_variable([kernel[i], kernel[i], depth[i], depth[i+1]], name='conv_%d'%(i))
            b = bias_variable([depth[i+1]], name='conv_%d'%(i))
            self.weights.append(w)
            self.biases.append(b)
            self.trainable_vars.append(w)
            self.trainable_vars.append(b)

            # Since we maxpool after every two conv layers
            if ((i+1) % 2 == 0):
                div_factor *= 2

        flat_units = (self.image_size // div_factor) * (self.image_size // div_factor) * depth[-1]
        w = weight_variable([flat_units, self.total_classes], name='fc_%d'%(i))
        b = bias_variable([self.total_classes], name='fc_%d'%(i))
        self.weights.append(w)
        self.biases.append(b)
        self.trainable_vars.append(w)
        self.trainable_vars.append(b)

    def conv_feedforward(self, h, weights, biases, apply_dropout=True):
        """
        Forward pass through a convolutional network
        Args:
            h               Input image (tensor)
            weights         List of weights for a conv network
            biases          List of biases for a conv network
            apply_dropout   Whether to apply droupout (True/ False)

        Returns:
            Logits of a conv network
        """
        for i, (w, b) in enumerate(list(zip(weights, biases))[:-1]):

            # Apply conv operation till the second last layer, which is a FC layer
            h = create_conv_layer(h, w, b)

            if ((i+1) % 2 == 0):

                # Apply max pool after every two conv layers
                h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                
                # Apply dropout   
                if apply_dropout:
                    h = tf.nn.dropout(h, self.keep_prob)

        # Construct FC layers
        shape = h.get_shape().as_list()
        h = tf.reshape(h, [-1, shape[1] * shape[2] * shape[3]])

        return create_fc_layer(h, weights[-1], biases[-1], apply_relu=False)

    def resnet18_conv_feedforward(self, h, kernels, filters, strides):
        """
        Forward pass through a ResNet-18 network

        Returns:
            Logits of a resnet-18 conv network
        """
        self.trainable_vars = []

        # Conv1
        h = _conv(h, kernels[0], filters[0], strides[0], self.trainable_vars, name='conv_1')
        h = _bn(h, self.trainable_vars, self.train_phase, name='bn_1')
        h = tf.nn.relu(h)

        # Conv2_x
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv2_1')
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv2_2')

        # Conv3_x
        h = _residual_block_first(h, filters[2], strides[2], self.trainable_vars, self.train_phase, name='conv3_1')
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv3_2')

        # Conv4_x
        h = _residual_block_first(h, filters[3], strides[3], self.trainable_vars, self.train_phase, name='conv4_1')
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv4_2')

        # Conv5_x
        h = _residual_block_first(h, filters[4], strides[4], self.trainable_vars, self.train_phase, name='conv5_1')
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv5_2')

        # Apply average pooling
        h = tf.reduce_mean(h, [1, 2])

        logits = _fc(h, self.total_classes, self.trainable_vars, name='fc_1')

        return logits 

    def loss_and_gradients(self, imp_method):
        """
        Defines task based and surrogate losses and their
        gradients
        Args:

        Returns:
        """
        if imp_method == 'VAN':
            reg = 0.0
        elif imp_method == 'EWC':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, 
                f in zip(self.trainable_vars, self.star_vars, self.normalized_fisher_at_minima_vars)])
        elif imp_method == 'PI':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, 
                f in zip(self.trainable_vars, self.star_vars, self.big_omega_vars)])
        elif imp_method == 'MAS':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, 
                f in zip(self.trainable_vars, self.star_vars, self.hebbian_score_vars)])
        elif imp_method == 'RWALK':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * (f + scr)) for w, w_star, 
                f, scr in zip(self.trainable_vars, self.star_vars, self.normalized_fisher_at_minima_vars, 
                    self.normalized_score_vars)])
      
        """
        # ***** DON't USE THIS WITH MULTI-HEAD SETTING SINCE THIS WILL UPDATE ALL THE WEIGHTS *****
        # If CNN arch, then use the weight decay
        if self.network_arch == 'CNN':
            self.unweighted_entropy += tf.add_n([0.0005 * tf.nn.l2_loss(v) for v in self.trainable_vars if 'weights' in v.name])
        """

        # Regularized training loss
        self.reg_loss = tf.squeeze(self.unweighted_entropy + self.synap_stgth * reg)

        # Compute the gradients of the vanilla loss
        self.vanilla_gradients_vars = self.opt.compute_gradients(self.unweighted_entropy, 
                var_list=self.trainable_vars)

        # Compute the gradients of regularized loss
        self.reg_gradients_vars = self.opt.compute_gradients(self.reg_loss, 
                var_list=self.trainable_vars)

    def train_op(self):
        """
        Defines the training operation (a single step during training)
        Args:

        Returns:
        """
        if self.imp_method == 'VAN':
            # Define a training operation
            self.train = self.opt.apply_gradients(self.reg_gradients_vars)
        else:
            # Get the value of old weights first
            with tf.control_dependencies([self.weights_old_ops_grouped]):
                # Define a training operation
                self.train = self.opt.apply_gradients(self.reg_gradients_vars)

    def init_vars(self):
        """
        Defines different variables that will be used for the
        weight consolidation
        Args:

        Returns:
        """
        for v in range(len(self.trainable_vars)):

            # List of variables for weight updates
            self.weights_old.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.weights_delta_old_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.star_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False, 
                                              name=self.trainable_vars[v].name.rsplit(':')[0]+'_star'))

            # List of variables for pathint method
            self.small_omega_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.big_omega_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.big_omega_riemann_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

            # List of variables to store fisher information
            self.fisher_diagonal_at_minima.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

            self.tmp_fisher_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.running_fisher_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            # New variables for conv setting for fisher and score normalization
            self.max_fisher_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.min_fisher_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.max_score_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.min_score_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.normalized_score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.normalized_fisher_at_minima_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False, dtype=tf.float32))
            # List of variables to store hebbian information
            self.hebbian_score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

    def get_current_weights(self):
        """
        Get the values of current weights
        Note: These weights are different from star_vars as those
        store the weights after training for the last task.
        Args:

        Returns:
        """
        weights_old_ops = []
        weights_delta_old_ops = []
        for v in range(len(self.trainable_vars)):
            weights_old_ops.append(tf.assign(self.weights_old[v], self.trainable_vars[v]))
            weights_delta_old_ops.append(tf.assign(self.weights_delta_old_vars[v], self.trainable_vars[v]))

        self.weights_old_ops_grouped = tf.group(*weights_old_ops)
        self.weights_delta_old_grouped = tf.group(*weights_delta_old_ops)


    def weights_store_ops(self):
        """
        Defines weight restoration operations
        Args:

        Returns:
        """
        restore_weights_ops = []
        set_star_vars_ops = []

        for v in range(len(self.trainable_vars)):
            restore_weights_ops.append(tf.assign(self.trainable_vars[v], self.star_vars[v]))

            set_star_vars_ops.append(tf.assign(self.star_vars[v], self.trainable_vars[v]))

        self.restore_weights = tf.group(*restore_weights_ops)
        self.set_star_vars = tf.group(*set_star_vars_ops)

    def reset_optimizer_ops(self):
        """
        Defines operations to reset the optimizer
        Args:

        Returns:
        """
        # Set the operation for resetting the optimizer
        self.optimizer_slots = [self.opt.get_slot(var, name) for name in self.opt.get_slot_names()\
                           for var in tf.global_variables() if self.opt.get_slot(var, name) is not None]
        self.slot_names = self.opt.get_slot_names()
        self.opt_init_op = tf.variables_initializer(self.optimizer_slots)

    def create_pathint_ops(self):
        """
        Defines operations for path integral-based importance
        Args:

        Returns:
        """
        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        update_big_omega_riemann_ops = []

        for v in range(len(self.trainable_vars)):
            # Make sure that the variables are updated before calculating delta(theta)
            with tf.control_dependencies([self.train]):
                update_small_omega_ops.append(tf.assign_add(self.small_omega_vars[v], 
                    -(self.vanilla_gradients_vars[v][0] * (self.trainable_vars[v] - self.weights_old[v]))))

            # Ops to reset the small omega
            reset_small_omega_ops.append(tf.assign(self.small_omega_vars[v], self.small_omega_vars[v]*0.0))

            if self.imp_method == 'PI':
                # Update the big omegas at the end of the task using the Eucldeian distance
                update_big_omega_ops.append(tf.assign_add(self.big_omega_vars[v], 
                    tf.nn.relu(tf.div(self.small_omega_vars[v], (PARAM_XI_STEP + tf.square(self.trainable_vars[v] - self.star_vars[v]))))))
            elif self.imp_method == 'RWALK':
                # Update the big omegas after small intervals using distance in riemannian manifold (KL-divergence)
                update_big_omega_riemann_ops.append(tf.assign_add(self.big_omega_riemann_vars[v], 
                    tf.nn.relu(tf.div(self.small_omega_vars[v], 
                        (PARAM_XI_STEP + self.running_fisher_vars[v] * tf.square(self.trainable_vars[v] - self.weights_delta_old_vars[v]))))))


        self.update_small_omega = tf.group(*update_small_omega_ops)
        self.reset_small_omega = tf.group(*reset_small_omega_ops)
        if self.imp_method == 'PI':
            self.update_big_omega = tf.group(*update_big_omega_ops)
        elif self.imp_method == 'RWALK':
            self.update_big_omega_riemann = tf.group(*update_big_omega_riemann_ops)
            self.big_omega_riemann_reset = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.big_omega_riemann_vars]

        if self.imp_method == 'RWALK':
            # For the first task, scale the scores so that division does not have an effect        
            self.scale_score = [tf.assign(s, s*2.0) for s in self.big_omega_riemann_vars]
            # To reduce the rigidity after each task the importance scores are averaged
            self.update_score = [tf.assign_add(scr, tf.div(tf.add(scr, riemm_omega), 2.0)) 
                    for scr, riemm_omega in zip(self.score_vars, self.big_omega_riemann_vars)]

            # Get the min and max in each layer of the scores
            self.get_max_score_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_max(scr, keepdims=True)), 
                axis=0)) for var, scr in zip(self.max_score_vars, self.score_vars)]
            self.get_min_score_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_min(scr, keepdims=True)), 
                axis=0)) for var, scr in zip(self.min_score_vars, self.score_vars)]
            self.max_score = tf.reduce_max(tf.convert_to_tensor(self.max_score_vars))
            self.min_score = tf.reduce_min(tf.convert_to_tensor(self.min_score_vars))
            with tf.control_dependencies([self.max_score, self.min_score]):
                self.normalize_scores = [tf.assign(tgt, (var - self.min_score)/ (self.max_score - self.min_score + EPSILON)) 
                        for tgt, var in zip(self.normalized_score_vars, self.score_vars)]

            # Sparsify all the layers except last layer
            sparsify_score_ops = []
            for v in range(len(self.normalized_score_vars) - 2):
                sparsify_score_ops.append(tf.assign(self.normalized_score_vars[v], 
                    tf.nn.dropout(self.normalized_score_vars[v], self.keep_prob)))

            self.sparsify_scores = tf.group(*sparsify_score_ops)

    def create_fisher_ops(self):
        """
        Defines the operations to compute online update of Fisher
        Args:

        Returns:
        """
        ders = tf.gradients(self.unweighted_entropy, self.trainable_vars)
        fisher_ema_at_step_ops = []
        fisher_accumulate_at_step_ops = []

        # ops for running fisher
        self.set_tmp_fisher = [tf.assign_add(f, tf.square(d)) for f, d in zip(self.tmp_fisher_vars, ders)]

        # Initialize the running fisher to non-zero value
        self.set_initial_running_fisher = [tf.assign(r_f, s_f) for r_f, s_f in zip(self.running_fisher_vars,
                                                                           self.tmp_fisher_vars)]

        self.set_running_fisher = [tf.assign(f, (1 - self.fisher_ema_decay) * f + (1.0/ self.fisher_update_after) *
                                    self.fisher_ema_decay * tmp) for f, tmp in zip(self.running_fisher_vars, self.tmp_fisher_vars)]

        self.get_fisher_at_minima = [tf.assign(var, f) for var, f in zip(self.fisher_diagonal_at_minima,
                                                                         self.running_fisher_vars)]

        self.reset_tmp_fisher = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.tmp_fisher_vars]

        # Get the min and max in each layer of the Fisher
        self.get_max_fisher_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_max(scr, keepdims=True)), axis=0)) 
                for var, scr in zip(self.max_fisher_vars, self.fisher_diagonal_at_minima)]
        self.get_min_fisher_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_min(scr, keepdims=True)), axis=0)) 
                for var, scr in zip(self.min_fisher_vars, self.fisher_diagonal_at_minima)]
        self.max_fisher = tf.reduce_max(tf.convert_to_tensor(self.max_fisher_vars))
        self.min_fisher = tf.reduce_min(tf.convert_to_tensor(self.min_fisher_vars))
        with tf.control_dependencies([self.max_fisher, self.min_fisher]):
            self.normalize_fisher_at_minima = [tf.assign(tgt, 
                (var - self.min_fisher)/ (self.max_fisher - self.min_fisher + EPSILON)) 
                    for tgt, var in zip(self.normalized_fisher_at_minima_vars, self.fisher_diagonal_at_minima)]

        # Sparsify all the layers except last layer
        sparsify_fisher_ops = []
        for v in range(len(self.normalized_fisher_at_minima_vars) - 2):
            sparsify_fisher_ops.append(tf.assign(self.normalized_fisher_at_minima_vars[v],
                tf.nn.dropout(self.normalized_fisher_at_minima_vars[v], self.keep_prob)))

        self.sparsify_fisher = tf.group(*sparsify_fisher_ops)

    def create_hebbian_ops(self):
        """
        Define operations for hebbian measure of importance (MAS)
        """
        # Compute the gradients of mse loss
        self.mse_gradients = tf.gradients(self.mse, self.trainable_vars)
        #with tf.control_dependencies([self.mse_gradients]):
        # Keep on adding gradients to the omega
        self.accumulate_hebbian_scores = [tf.assign_add(omega, tf.abs(grad)) for omega, grad in zip(self.hebbian_score_vars, self.mse_gradients)]
        # Average across the total images
        self.average_hebbian_scores = [tf.assign(omega, omega*(1.0/self.train_samples)) for omega in self.hebbian_score_vars]
        # Reset the hebbian importance variables
        self.reset_hebbian_scores = [tf.assign(omega, tf.zeros_like(omega)) for omega in self.hebbian_score_vars]

#################################################################################
#### External APIs of the class. These will be called/ exposed externally #######
#################################################################################
    def reset_optimizer(self, sess):
        """
        Resets the optimizer state
        Args:
            sess        TF session

        Returns:
        """

        # Call the reset optimizer op
        sess.run(self.opt_init_op)

    def set_active_outputs(self, sess, labels):
        """
        Set the mask for the labels seen so far
        Args:
            sess        TF session
            labels      Mask labels

        Returns:
        """
        new_mask = np.zeros(self.total_classes)

        for l in labels:
            new_mask[l] = 1.0

        sess.run(self.output_mask.assign(new_mask))

    def set_logits_weights(self, sess, labels, logits_weighting):
        """
        """
        weighting_mask = np.zeros(self.total_classes, dtype=np.float32)
        for l in labels:
            weighting_mask[l] = logits_weighting[l] 

        sess.run(self.logit_weights.assign(weighting_mask))

    def init_updates(self, sess):
        """
        Initialization updates
        Args:
            sess        TF session

        Returns:
        """
        # Set the star values to the initial weights, so that we can calculate
        # big_omegas reliably
        sess.run(self.set_star_vars)

    def task_updates(self, sess, task, train_x):
        """
        Updates different variables when a task is completed
        Args:
            sess                TF session
            task                Task ID
            train_x             Training images for the task 
        Returns:
        """
        if self.imp_method == 'VAN':
            # We'll store the current parameters at the end of this function
            pass
        elif self.imp_method == 'EWC':
            # Get the fisher at the end of a task
            sess.run(self.get_fisher_at_minima)
            # Normalize the fisher
            sess.run([self.get_max_fisher_vars, self.get_min_fisher_vars])
            sess.run([self.min_fisher, self.max_fisher, self.normalize_fisher_at_minima])
            # Reset the tmp fisher vars
            sess.run(self.reset_tmp_fisher)
        elif self.imp_method == 'PI':
            # Update big omega variables
            sess.run(self.update_big_omega)
            # Reset the small_omega_vars because big_omega_vars are updated before it
            sess.run(self.reset_small_omega)
        elif self.imp_method == 'RWALK':
            if task == 0:
                # If first task then scale by a factor of 2, so that subsequent averaging does not hurt
                sess.run(self.scale_score)
            # Get the updated importance score
            sess.run(self.update_score)
            # Normalize the scores 
            sess.run([self.get_max_score_vars, self.get_min_score_vars])
            sess.run([self.min_score, self.max_score, self.normalize_scores])
            # Sparsify scores
            """
            # TODO: Tmp remove this?
            kp = 0.8 + (task*0.5)
            if (kp > 1):
                kp = 1.0
            """
            #sess.run(self.sparsify_scores, feed_dict={self.keep_prob: kp})
            # Get the fisher at the end of a task
            sess.run(self.get_fisher_at_minima)
            # Normalize fisher
            sess.run([self.get_max_fisher_vars, self.get_min_fisher_vars])
            sess.run([self.min_fisher, self.max_fisher, self.normalize_fisher_at_minima])
            # Sparsify fisher
            #sess.run(self.sparsify_fisher, feed_dict={self.keep_prob: kp})
            # Store the weights
            sess.run(self.weights_delta_old_grouped)
            # Reset the small_omega_vars because big_omega_vars are updated before it
            sess.run(self.reset_small_omega)
            # Reset the big_omega_riemann because importance score is stored in the scores array
            sess.run(self.big_omega_riemann_reset)
            # Reset the tmp fisher vars
            sess.run(self.reset_tmp_fisher)
        elif self.imp_method == 'MAS':
            # zero out any previous values
            sess.run(self.reset_hebbian_scores)
            # Loop over the entire training dataset to compute the parameter importance
            batch_size = 100
            num_samples = train_x.shape[0]
            for iters in range(num_samples// batch_size):
                offset = iters * batch_size
                sess.run(self.accumulate_hebbian_scores, feed_dict={self.x: train_x[offset:offset+batch_size], self.keep_prob: 1.0})

            # Average the hebbian scores across the training examples
            sess.run(self.average_hebbian_scores, feed_dict={self.train_samples: num_samples})
            
        # Store current weights
        self.init_updates(sess)

    def restore(self, sess):
        """
        Restore the weights from the star variables
        Args:
            sess        TF session

        Returns:
        """
        sess.run(self.restore_weights)
