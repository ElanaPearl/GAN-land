from datetime import datetime
import cPickle as pickle
import tensorflow as tf
from math import ceil
import numpy as np
import argparse
import logging
import os

from model_w_label import MultipleSequenceAlignment
from predict import MutationPrediction
import tools

from tensorflow.python import debug as tf_debug

class LSTM:
    def __init__(self, data, target, dropout, attn_length, entropy_reg, lambda_l2_reg, gradient_limit,
                num_hidden, num_layers, init_learning_rate, decay_steps, decay_rate, mlp_h1_size, mlp_h2_size):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.attn_length = attn_length

        self.max_length = int(self.target.get_shape()[1])
        self.alphabet_len = int(self.target.get_shape()[2])

        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.length = self.get_length()

        self.mlp_h1_size = mlp_h1_size
        self.mlp_h2_size = mlp_h2_size

        with tf.variable_scope('prediction'):
            self.logits, self.prediction = self.get_prediction()
        with tf.variable_scope('calc_err'):
            self.error = self.get_error()
            self.test_error = tf.summary.scalar('test_error',self.error)
            self.train_error = tf.summary.scalar('train_error',self.error)

        # Note this cross entropy gives you the cross ent FOR EACH SEQ in the batch
        self.cross_entropy = self.get_cross_entropy(self.target, self.logits)
        self.entropy = self.get_cross_entropy(self.prediction, self.logits, ignore_end_token=True)

        self.cost, cross_ent_summary, ent_summary = self.get_cost(entropy_reg, lambda_l2_reg)
        self.optimize, gradient_summary, unnorm_grad_summary = self.get_optimizer(gradient_limit,
                                                            init_learning_rate,
                                                            decay_steps,
                                                            decay_rate)

        self.train_summaries = [tf.summary.histogram(v.name, v) for v in tf.trainable_variables()]
        self.train_summaries += [gradient_summary, unnorm_grad_summary, cross_ent_summary, ent_summary]


    def get_length(self):
        with tf.variable_scope('calc_lengths'):
            used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
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
            prediction = tf.nn.softmax(logits_max_red, axis=2)
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
            mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
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

        readable_seq = [tools.rev_alphabet_map[seed]]

        for idx in range(self.max_length-1):
            full_pred_dist = session.run(self.prediction, {data:sequence, dropout: 0.0})

            next_pos_pred_dist = full_pred_dist[0, idx, :]
            next_pred = np.random.choice(np.arange(self.alphabet_len), p=next_pos_pred_dist)
            sequence[0, idx+1, next_pred] = 1
            readable_seq.append(tools.rev_alphabet_map[next_pred])

        return ''.join(readable_seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('-gene_name', help='The name of the gene for the protein family', default='FYN')
    parser.add_argument('-use_full_seqs', help='Use full sequences from online?', action='store_true')
    parser.add_argument('-seq_limit', help='If debugging and you want to only use a limited number '\
                        'of sequences for the sake of time, set this.', type=int, default=0)

    # Training Parameters
    parser.add_argument('-batch_size', help='Number of sequences per batch', type=int, default=128)
    parser.add_argument('-num_epochs', help='Number of epochs of training', type=int, default=100)
    parser.add_argument('-restore_path', help='Path to restore model, should be of the format '\
                        '\'year-month-date_hour-min-sec\'', default='')
    parser.add_argument('-debug', help='Use tf_debug', action='store_true')

    # RNN Architecture Parameters
    parser.add_argument('-num_hidden', help='Number of hidden nodes per layer', type=int, default=150)
    parser.add_argument('-num_layers', help='Number of hidden layers. If = 1, this is a normal LSTM ' \
                        'othersiwe this is the number of stacked LSTM layers', type=int, default=1)
    parser.add_argument('-mlp_h1_size', type=int, default=150)
    parser.add_argument('-mlp_h2_size', type=int, default=75)

    # Regularization Parameters
    parser.add_argument('-dropout_prob', help='Value for dropout when training. If left unset, there will be'\
                        'no dropout aka dropout_prob = 0', type=float, default=0.0)
    parser.add_argument('-attn_length', help='Length of attention, if 0, no attention is used', type=int, default=0)
    parser.add_argument('-gradient_limit', help='Max value for a gradient. Values above this get trimmed down'\
                        ' to this value', type=float, default=5.0)
    parser.add_argument('-entropy_reg', help='How much entropy regularization to use, if 0 none', type=float, default=0.0)
    parser.add_argument('-lambda_l2_reg', help="How much l2 regularization to use on the weights, if 0 none", type=float, default=0.0)
    parser.add_argument('-init_learning_rate', help='Initial learning rate', type=float, default=0.01)
    parser.add_argument('-decay_steps', help='Initial learning rate', type=int, default=100)
    parser.add_argument('-decay_rate', help='Initial learning rate', type=float, default=0.9)

    # NOTES TO SELF:
    parser.add_argument('-note', help='Just 4 u', default='')

    # Create dictionary of the parsed args and convert each arg into a local variable
    locals().update(vars(parser.parse_args()))

    # Set run time (or restore run_time from last run)
    run_time = restore_path if restore_path else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set up logging directories
    log_path = os.path.join('model_logs', gene_name, run_time)
    graph_log_path = os.path.join(log_path, 'graphs')
    checkpoint_log_path = os.path.join(log_path, 'checkpoints')
    mutant_pred_path = os.path.join(log_path, 'mutant_preds')

    if not restore_path:
        os.makedirs(log_path)
        os.makedirs(graph_log_path)
        os.makedirs(checkpoint_log_path)
        os.makedirs(mutant_pred_path)

    # Set up logging file
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_path, 'logfile.txt'),
                    format='%(asctime)-15s %(message)s')

    # Log the parameter values
    for param_name, param_value in vars(parser.parse_args()).iteritems():
        logging.info("{}: {}".format(param_name, param_value))

    print "Getting multiple sequence alignment"
    MSA = MultipleSequenceAlignment(gene_name, run_time=run_time, seq_limit=seq_limit, use_full_seqs=use_full_seqs)

    data = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='data')
    target = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='target')
    dropout = tf.placeholder_with_default(0.0, shape=(), name='dropout')
    corr_tensor = tf.placeholder(tf.float32, name='spear_corr')
    corr_summ_tensor = tf.summary.scalar('spear_corr', corr_tensor)

    predictor = MutationPrediction(MSA, tools.feature_to_predict[gene_name])

    print "Constructing model"
    model = LSTM(data, target, dropout=dropout, attn_length=attn_length,
                 entropy_reg=entropy_reg, gradient_limit=gradient_limit,
                 init_learning_rate=init_learning_rate, decay_steps=decay_steps,
                 decay_rate=decay_rate, num_hidden=num_hidden, num_layers=num_layers,
                 lambda_l2_reg=lambda_l2_reg, mlp_h1_size=mlp_h1_size, mlp_h2_size=mlp_h2_size)

    writer = tf.summary.FileWriter(graph_log_path)

    sess = tf.Session()
    if debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)


    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    if restore_path:
        print "Restoring checkpoint"
        last_ckpt = tf.train.latest_checkpoint(checkpoint_log_path)
        saver = tf.train.Saver()
        saver.restore(sess, last_ckpt)

        pretrained_epochs, pretrained_batches = int(last_ckpt.split('_')[-2]), int(last_ckpt.split('_')[-1])
        print "Picking up with epoch {} and batch {}".format(pretrained_epochs, pretrained_batches)
    else:
        saver = tf.train.Saver()
        pretrained_epochs = 0
        pretrained_batches = 0


    num_batches_per_epoch = int(ceil(float(MSA.train_size) / batch_size))

    LAST_ERROR = 10000.0
    MAX_DELTA_ERROR = 0.0

    print "Starting training"

    # TODO: DELETE THIS!!!
    #batch_data, batch_target = MSA.next_batch(batch_size)


    for epoch in range(pretrained_epochs, num_epochs):
        logging.info("Epoch: {}".format(epoch))

        

        for i in range(pretrained_batches, num_batches_per_epoch):
            if i % batch_size == 0:
                saver.save(sess, os.path.join(checkpoint_log_path,'model_{}_{}'.format(epoch,i)))

                # PRINT MAX ERROR
                logging.info("MAX DELTA ERROR: {}".format(MAX_DELTA_ERROR))
                print "MAX DELTA ERROR: ", MAX_DELTA_ERROR

                # GET TEST ERROR
                test_data, test_target = MSA.next_batch(batch_size, test=True)
                test_err_summary, test_err, test_pred = sess.run([model.test_error, model.error, model.prediction], \
                                                        {data: test_data, target: test_target})

                logging.info("Batch: {}, Error: {}".format(i, test_err))

                writer.add_summary(test_err_summary, epoch*num_batches_per_epoch + i)

                logging.info("target: {}".format(MSA.one_hot_to_str(test_target[0])))
                logging.info("pred:   {}".format(MSA.one_hot_to_str(test_pred[0])))

                # GENERATE RANDOM SAMPLE
                seq_sample = model.generate_seq(sess, MSA.seed_weights)
                logging.info("gen:    {}".format(seq_sample))

                # COMPARE PREDICTIONS TO EXPERIMENTAL RESULTS
                mutant_preds, spear_corr = predictor.corr(sess, model, data, target)
                corr_summary = sess.run(corr_summ_tensor, {corr_tensor: spear_corr})
                writer.add_summary(corr_summary, epoch*num_batches_per_epoch + i)

                with open(os.path.join(mutant_pred_path, '{}.pkl'.format(epoch*num_batches_per_epoch + i)), 'w') as f:
                    pickle.dump(mutant_preds, f)


                # MAKE PLOT OF ALL SINGLE MUTATIONS
                single_mutant_preds = predictor.plot_single_mutants(sess, model, data, target,
                                              os.path.join(mutant_pred_path, '{}.png'.format(epoch*num_batches_per_epoch + i)))


                #sample_summary = sess.run(model.sample_seq_summary, {seq_placeholder: seq_sample})
                #writer.add_summary(sample_summary, epoch*num_batches_per_epoch + i)

            batch_data, batch_target = MSA.next_batch(batch_size

            _, train_summaries, train_err_summary, train_err = sess.run([model.optimize, model.train_summaries, model.train_error, model.error], \
                                                               {data: batch_data, target: batch_target, dropout: dropout_prob})

            for summary in train_summaries:
                writer.add_summary(summary, epoch*num_batches_per_epoch + i)

            writer.add_summary(train_err_summary, epoch*num_batches_per_epoch + i)

            if train_err - LAST_ERROR > MAX_DELTA_ERROR:
                MAX_DELTA_ERROR = train_err - LAST_ERROR

            if train_err >= .1 + LAST_ERROR:
                logging.info("{}: err increaed by {}".format(epoch*num_batches_per_epoch + i, train_err-LAST_ERROR))
                with open(os.path.join(log_path, 'batch_{}.pkl'.format(epoch*num_batches_per_epoch + i)), 'w') as f:
                    pickle.dump(batch_data, f)

            LAST_ERROR = train_err

        test_data, test_target = MSA.next_batch(batch_size, test=True)
        logging.info("Error: {}".format(sess.run(model.error, {data: test_data, target: test_target})))


        # The number of batches of a given epoch that we already trained is only relevant
        # to the first epoch we train after we restore the model
        pretrained_batches = 0
