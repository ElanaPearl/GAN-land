import tensorflow as tf
import numpy as np
from tools import rev_alphabet_map


class LSTM:
    def __init__(self, data, target, dropout, args):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.attn_length = args.attn_length

        self.max_length = int(self.target.get_shape()[1])
        self.alphabet_len = int(self.target.get_shape()[2])

        self._num_hidden = args.num_hidden
        self._num_layers = args.num_layers
        self.length = self.get_length()

        self.mlp_h1_size = args.mlp_h1_size
        self.mlp_h2_size = args.mlp_h2_size

        with tf.variable_scope('prediction'):
            self.logits, self.prediction = self.get_prediction()
        with tf.variable_scope('calc_err'):
            self.error = self.get_error()
            self.test_error = tf.summary.scalar('test_error',self.error)
            self.train_error = tf.summary.scalar('train_error',self.error)

        # Note this cross entropy gives you the cross ent FOR EACH SEQ in the batch
        self.cross_entropy = self.get_cross_entropy(self.target, self.logits)
        self.entropy = self.get_cross_entropy(self.prediction, self.logits, ignore_end_token=True)

        self.cost, cross_ent_summary, ent_summary = self.get_cost(args.entropy_reg, args.lambda_l2_reg)
        self.optimize, gradient_summary, unnorm_grad_summary = self.get_optimizer(args.gradient_limit,
                                                            args.init_learning_rate,
                                                            args.decay_steps,
                                                            args.decay_rate)

        self.train_summaries = [tf.summary.histogram(v.name, v) for v in tf.trainable_variables()]
        self.train_summaries += [gradient_summary, unnorm_grad_summary, cross_ent_summary, ent_summary]


    def get_length(self):
        with tf.variable_scope('calc_lengths'):
            used = tf.sign(tf.reduce_max(self.data, reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
            return length


    def lstm_cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(
            self._num_hidden, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)

        if self.attn_length:
            cell = tf.contrib.rnn.AttentionCellWrapper(cell, self.attn_length, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - self.dropout)

        return cell


    def get_prediction(self):
        # Recurrent network
        with tf.variable_scope('dynamic_rnn'):
            with tf.variable_scope('LSTM_cell'):
                if self._num_layers > 1:
                    cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self._num_layers)])
                else:
                    cell = self.lstm_cell()

                output, _ = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=self.data,
                    sequence_length=self.length,
                    dtype=tf.float32
                )
        output = tf.reshape(output, [-1, self._num_hidden])

        logits = self.multilayer_perceptron(output, self._num_hidden, self.mlp_h1_size, self.mlp_h2_size, self.alphabet_len)

        with tf.variable_scope('softmax_logits'):

            # Subtract max logit from all logits before softmaxing
            logit_pre_softmax = tf.reshape(logits, [-1, self.max_length, self.alphabet_len])

            logits_max_red = logit_pre_softmax - tf.reduce_max(logit_pre_softmax, reduction_indices=2, keep_dims=True)
            prediction = tf.nn.softmax(logits_max_red)
            prediction = tf.reshape(prediction, [-1, self.max_length, self.alphabet_len])
            logits = tf.reshape(logits, [-1, self.max_length, self.alphabet_len])

        with tf.variable_scope('clip_predictions'):
            # Clip predictions
            prediction = tf.clip_by_value(prediction, 1e-5, 1)

            # Renormalize predictions
            prediction = tf.divide(prediction, tf.reshape(tf.reduce_sum(prediction, 2), [-1, self.max_length, 1]))

            # Recompute logits from predictions
            logits = tf.log(prediction)
        return logits, prediction

    @staticmethod
    def multilayer_perceptron(x, in_size, h1_size, h2_size, out_size):
        #weight = tf.get_variable(name='weight', shape=[in_size, out_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
        #bias = tf.get_variable(name='bias', shape=[out_size], initializer=tf.constant_initializer(0.1))
        #logits = tf.matmul(output, weight) + bias
        #return logits

        weights = {
            'h1': tf.get_variable(name='h1', shape=[in_size, h1_size], initializer=tf.truncated_normal_initializer(stddev=0.01)), 
            'h2': tf.get_variable(name='h2', shape=[h1_size, h2_size], initializer=tf.truncated_normal_initializer(stddev=0.01)), 
            #'out': tf.get_variable(name='out', shape=[h2_size, out_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
            'out': tf.get_variable(name='out', shape=[h1_size, out_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
        }
        biases = {
            'b1': tf.get_variable(name='bias_1', shape=[h1_size], initializer=tf.constant_initializer(0.1)),
            'b2': tf.get_variable(name='bias_2', shape=[h2_size], initializer=tf.constant_initializer(0.1)),
            'out': tf.get_variable(name='bias_out', shape=[out_size], initializer=tf.constant_initializer(0.1))
        }

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.elu(layer_1)
        # Hidden layer with RELU activation
        # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # layer_2 = tf.nn.elu(layer_2)
        # Output layer with linear activation
        #out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer


    def get_cost(self, entropy_reg, lambda_l2_reg):
        # Get the avg entropy for the whole batch
        batch_cross_ent = tf.reduce_mean(self.cross_entropy)
        batch_entropy = tf.reduce_mean(self.entropy)


        loss = batch_cross_ent - batch_entropy * entropy_reg
        l2 = lambda_l2_reg * sum(
            tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables())
        loss += l2

        return loss, \
                tf.summary.scalar("cross_entropy_err", batch_cross_ent), \
                tf.summary.scalar("entropy_err", batch_entropy)


    def get_optimizer(self, gradient_limit, init_learning_rate, decay_steps, decay_rate):
        global_step = tf.Variable(0, trainable=False)

        
        learning_rate = tf.train.exponential_decay(init_learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=decay_steps,
                                                    decay_rate=decay_rate,
                                                    staircase=True)
        

        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.name_scope('minimize_cost'):
            gvs = optimizer.compute_gradients(self.cost)
            grads, gvars = list(zip(*gvs)[0]), list(zip(*gvs)[1])
            
            # THIS IS FOR CLIPPING BY GLOBAL NORM
            clip_norm = gradient_limit
            stable_global_norm = tf.global_norm(grads) + .00001
            #stable_global_norm = tf.global_norm(grads)
            clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm, use_norm=stable_global_norm)
            clipped_gvs = zip(clipped_grads, gvars)
            return optimizer.apply_gradients(clipped_gvs, global_step=global_step), \
                 tf.summary.scalar("GradientNorm", tf.global_norm(clipped_grads)), \
                 tf.summary.scalar("GradientNotNorm", stable_global_norm)
            """
            clipped_grads = [tf.clip_by_value(grad, -gradient_limit, gradient_limit, name='clipped_grads') for grad in grads]
            clipped_gvs = zip(clipped_grads, gvars)
            return optimizer.apply_gradients(clipped_gvs, global_step=global_step), \
                 tf.summary.scalar("GradientNorm", tf.global_norm(clipped_grads)), \
                 tf.summary.scalar("GradientNotNorm", tf.global_norm(grads))
            """

    def get_error(self):
        with tf.variable_scope('compute_all_errors'):
            mistakes = tf.not_equal(
                tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
            mistakes = tf.cast(mistakes, tf.float32, name='mistakes')

        with tf.variable_scope('mask_out_unused_positions'):
            mask = tf.sign(tf.reduce_max(self.target, reduction_indices=2))
            mistakes *= mask
        # Average over actual sequence lengths.
        with tf.variable_scope('avg_over_seq_len'):
            mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
            mistakes /= tf.cast(self.length, tf.float32)
            return tf.reduce_mean(mistakes)


    def get_cross_entropy(self, p, q, ignore_end_token=False):
        # If include_end_token then you will not calculate the entropy of the end token
        if ignore_end_token:
            p = p[:,:,:-1]
            q = q[:,:,:-1]


        q -= tf.reduce_max(q, reduction_indices=2, keep_dims=True)

        cross_entropy = - p * (q - tf.reduce_logsumexp(q, axis=2, keep_dims=True))
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=2)

        # Mask out elements after the end of the target sequence
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask

        # Average across each seq to get one value per seq
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)

        return cross_entropy


    def generate_seq(self, session, seed_weights):
        sequence = np.zeros((1, self.max_length, self.alphabet_len))

        # TODO: change this so that it calculates them
        seed = np.random.choice(np.arange(len(seed_weights)), p=seed_weights)

        sequence[0, 0, seed] = 1

        readable_seq = [rev_alphabet_map[seed]]

        for idx in range(self.max_length-1):
            full_pred_dist = session.run(self.prediction, {self.data:sequence, self.dropout: 0.0})

            next_pos_pred_dist = full_pred_dist[0, idx, :]
            next_pred = np.random.choice(np.arange(self.alphabet_len), p=next_pos_pred_dist)
            sequence[0, idx+1, next_pred] = 1
            readable_seq.append(rev_alphabet_map[next_pred])

        return ''.join(readable_seq)