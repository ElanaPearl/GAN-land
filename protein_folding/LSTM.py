from datetime import datetime
import tensorflow as tf
from math import ceil
import numpy as np
import argparse
import logging
import os

from model_w_label import MultipleSequenceAlignment
from predict import MutationPrediction
import tools


class LSTM:
    def __init__(self, data, target, seed_weights, num_hidden=150, num_layers=2, use_multilayer=True):
        self.data = data
        self.target = target

        self.seed_weights = seed_weights

        self.max_length = int(self.target.get_shape()[1])
        self.alphabet_len = int(self.target.get_shape()[2])

        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.length = self.get_length()

        self.use_multilayer = use_multilayer

        with tf.variable_scope('prediction'):
            self.prediction = self.get_prediction()
        with tf.variable_scope('calc_err'):
            self.error = self.get_error()
            self.test_error = tf.summary.scalar('test_error',self.error)
            self.train_error = tf.summary.scalar('train_error',self.error)

        self.cross_entropy = self.get_cross_entropy()
        self.cost = self.get_cost()
        self.optimize, self.gradient_summary = self.get_optimizer()


    
    def get_length(self):
        with tf.variable_scope('calc_lengths'):
            used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
            return length


    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(
            self._num_hidden, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)


    def get_prediction(self):
        # Recurrent network.
        with tf.variable_scope('dynamic_rnn'):
            with tf.variable_scope('LSTM_cell'):
                if self.use_multilayer:
                    cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self._num_layers)])
                else:
                    cell = self.lstm_cell()

                output, _ = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=self.data,
                    sequence_length=self.length,
                    dtype=tf.float32
                )

        # Softmax layer.
        weight, bias = self._weight_and_bias(self._num_hidden, self.alphabet_len)
        # Flatten to apply same weights to all time steps.
        with tf.variable_scope('output_to_prediction'):
            output = tf.reshape(output, [-1, self._num_hidden])
            prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
            prediction = tf.reshape(prediction, [-1, self.max_length, self.alphabet_len])
            return prediction


    def get_cost(self):
        return tf.reduce_mean(tf.negative(self.cross_entropy))


    def get_optimizer(self):
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.name_scope('minimize_cost'):
            gvs = optimizer.compute_gradients(self.cost)
            grads, gvars = list(zip(*gvs)[0]), list(zip(*gvs)[1])
            clip_norm = 10.0 # THIS IS A HYPERPARAMETER
            clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm)
            clipped_gvs = zip(clipped_grads, gvars)
            return optimizer.apply_gradients(clipped_gvs), tf.summary.scalar("GradientNorm", global_norm) # collections=["train_batch", "train_dense"]


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


    def get_cross_entropy(self):
        # Get the predicted value for the correct target value
        # (self.target is one hot so it will mask out all predicted values
        # except for the correct one)
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=2)
        # This just masks out elements after the end of the target sequence
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        # Only consider the error for the length of the protein
        cross_entropy *= mask
        # This gives you the sum of all errors in each seq
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        # Divide each seq's error by its length
        cross_entropy /= tf.cast(self.length, tf.float32)
        return cross_entropy


    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.get_variable(name='weight',
                                shape=[in_size, out_size],
                          initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias = tf.get_variable(name='bias',
                              shape=[out_size], 
                        initializer=tf.constant_initializer(0.1))
        return weight, bias


    def generate_seq(self, session):
        sequence = np.zeros((1, self.max_length, self.alphabet_len))

        # TODO: change this so that it calculates them
        seed = np.random.choice(np.arange(len(self.seed_weights)), p=self.seed_weights)
        
        sequence[0, 0, seed] = 1

        readable_seq = [tools.rev_alphabet_map[seed]]

        for idx in range(self.max_length-1):
            full_pred_dist = session.run(self.prediction, {data:sequence})

            next_pos_pred_dist = full_pred_dist[0, idx, :]
            next_pred = np.random.choice(np.arange(self.alphabet_len), p=next_pos_pred_dist)
            sequence[0, idx+1, next_pred] = 1
            readable_seq.append(tools.rev_alphabet_map[next_pred])

        return ''.join(readable_seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_name', help='The name of the gene for the protein family', default='FYN')
    parser.add_argument('--batch_size', help='Number of sequences per batch', type=int, default=100)
    parser.add_argument('--num_epochs', help='Number of epochs of training', type=int, default=10)
    parser.add_argument('--multilayer', help='Use multiple LSTM layers', type=bool, default=False)
    parser.add_argument('--restore_path', help='Path to restore model, should be of the format '\
                        '\'year-month-date_hour-min-sec\'', default='')
    parser.add_argument('--seq_limit', help='If debugging and you want to only use a limited number '\
                        'of sequences for the sake of time, set this.', type=int, default=0)
    parser.add_argument('--feature_to_predict', help='Experimental feature to compare mutation'\
                        'predictions to.', default='')


    gene_name = parser.parse_args().gene_name
    batch_size = parser.parse_args().batch_size
    num_epochs = parser.parse_args().num_epochs
    multilayer = parser.parse_args().multilayer
    restore_path = parser.parse_args().restore_path
    seq_limit = parser.parse_args().seq_limit
    feature_to_predict = parser.parse_args().feature_to_predict

    # Set run time (or restore run_time from last run)
    if restore_path:
        run_time = restore_path
    else:
        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    # Set up logging directories
    log_path = os.path.join('model_logs', gene_name, run_time)
    graph_log_path = os.path.join(log_path, 'graphs')
    checkpoint_log_path = os.path.join(log_path, 'checkpoints')

    if not restore_path:
        os.makedirs(log_path)
        os.makedirs(graph_log_path)
        os.makedirs(checkpoint_log_path)

    # Set up logging file
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_path, 'logfile.txt'),
                    format='%(asctime)-15s %(message)s')

    # Log the flag values
    for flag_name, flag_value in vars(parser.parse_args()).iteritems():
        logging.info("{}: {}".format(flag_name, flag_value))

    print "Getting multiple sequence alignment"
    MSA = MultipleSequenceAlignment(gene_name, run_time=run_time, seq_limit=seq_limit)

    
    data = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='data')
    target = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='target')
    corr_tensor = tf.placeholder(tf.float32, [], name='spear_corr')
    corr_summ_tensor = tf.summary.scalar('spear_corr', corr_tensor)

    predictor = MutationPrediction(MSA, feature_to_predict, {'data': data, 'target': target})

    print "Constructing model"
    model = LSTM(data, target, seed_weights=MSA.seed_weights, use_multilayer=multilayer)

    writer = tf.summary.FileWriter(graph_log_path)
    sess = tf.Session()
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

    
    print "Starting training"
    for epoch in range(pretrained_epochs, num_epochs):
        logging.info("Epoch: {}".format(epoch))
        
        for i in range(pretrained_batches, num_batches_per_epoch):
            if i % batch_size == 0:
                saver.save(sess, os.path.join(checkpoint_log_path,'model_{}_{}'.format(epoch,i)))

                # GET TEST ERROR
                test_data, test_target = MSA.next_batch(batch_size, test=True)
                test_err_summary, test_err, test_pred = sess.run([model.test_error, model.error, model.prediction], {data: test_data, target: test_target})

                logging.info("Batch: {}, Error: {}".format(i, test_err))

                writer.add_summary(test_err_summary, epoch*num_batches_per_epoch + i)
                
                logging.info("target: {}".format(MSA.one_hot_to_str(test_target[0])))
                logging.info("pred:   {}".format(MSA.one_hot_to_str(test_pred[0])))

                # GENERATE RANDOM SAMPLE
                seq_sample = model.generate_seq(sess)
                logging.info("gen:    {}".format(seq_sample))
                
                # COMPARE PREDICTIONS TO EXPERIMENTAL RESULTS
                _, spear_corr = predictor.predict(sess, model)
                corr_summary = sess.run(corr_summ_tensor, {corr_tensor: spear_corr})
                writer.add_summary(corr_summary, epoch*num_batches_per_epoch + i)

                #sample_summary = sess.run(model.sample_seq_summary, {seq_placeholder: seq_sample})
                #writer.add_summary(sample_summary, epoch*num_batches_per_epoch + i)


            batch_data, batch_target = MSA.next_batch(batch_size)

            _, gradient_summary, train_err_summary = sess.run([model.optimize, model.gradient_summary, model.train_error], {data: batch_data, target: batch_target})
            writer.add_summary(train_err_summary, epoch*num_batches_per_epoch + i)
            writer.add_summary(gradient_summary, epoch*num_batches_per_epoch + i)

        test_data, test_target = MSA.next_batch(batch_size, test=True)
        logging.info("Error: {}".format(sess.run(model.error, {data: test_data, target: test_target})))


        # The number of batches of a given epoch that we already trained is only relevant 
        # to the first epoch we train after we restore the model
        pretrained_batches = 0

