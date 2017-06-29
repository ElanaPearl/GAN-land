# Much code adopted from: danijar.com/variable-sequence-lengths-in-tensorflow/
import functools
import tensorflow as tf
from tensorflow.contrib import rnn
#from tensorflow.contrib.rnn import dynamic_rnn
#from tensorflow.contrib.rnn import core_rnn_cell

from model_w_label import MultipleSequenceAlignment
from datetime import datetime
import os
import argparse
import cPickle as pickle
import numpy as np
from math import ceil
import logging


# TODO: incorperate this in somewhere so it isnt just a rando global variable
rev_alphabet_map = {i: s for i, s in enumerate('ACDEFGHIKLMNPQRSTVWY*')}

class VariableSequenceLabelling:

    def __init__(self, data, target, num_hidden=200, num_layers=3, use_multilayer=True, end_token=20):
        self.data = data
        self.target = target

        self.max_length = int(self.target.get_shape()[1])
        self.alphabet_len = int(self.target.get_shape()[2])

        self.END_TOKEN = end_token

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
        self.cost = self.get_cost()
        self.optimize = self.get_optimizer()

        #self.seq_placeholder = tf.placeholder(tf.string)
        #self.sample_seq_summary = tf.summary.text('sample_seq', self.seq_placeholder)
    
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
        # Compute cross entropy for each frame.
        with tf.variable_scope('compute_cross_ent'):
            cross_entropy = self.target * tf.log(self.prediction)
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
            mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
            cross_entropy *= mask
        # Average over actual sequence lengths.
        with tf.variable_scope('avg_over_seq_len'):
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
            cross_entropy /= tf.cast(self.length, tf.float32)
            return tf.reduce_mean(cross_entropy)


    def get_optimizer(self):
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.name_scope('minimize_cost'):
            return optimizer.minimize(self.cost)


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

        # TODO: calculate distribution over first letters and use that as sample seed
        seed = np.random.randint(0, self.alphabet_len-2)
        sequence[0, 0, seed] = 1

        readable_seq = [rev_alphabet_map[seed]]

        for idx in range(1, self.max_length):
            logits = sess.run(self.prediction, {data:sequence})
            next_logit = logits[0,idx,:]
            next_pred = np.random.choice(np.arange(self.alphabet_len), p=next_logit)
            sequence[0, idx, next_pred] = 1
            readable_seq.append(rev_alphabet_map[next_pred])

        # FILTER OUT ONLY THE LETTERS BEFORE THE *
        trimmed_seq = []
        for x in readable_seq:
            if x == '*':
                break
            trimmed_seq.append(x)

        return ''.join(trimmed_seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--align_name', help='The name of the alignment file for the protein family', \
                         default='FYN_HUMAN_hmmerbit_plmc_n5_m30_f50_t0.2_r80-145_id100_b33.a2m')
    parser.add_argument('--batch_size', help='Number of sequences per batch', type=int, default=50)
    parser.add_argument('--num_epochs', help='Number of epochs of training', type=int, default=10)
    parser.add_argument('--multilayer', help='Use multiple LSTM layers', type=bool, default=False)
    parser.add_argument('--restore_path', help='Path to restore model, should be of the format '\
                        'year-month-date_hour-min-sec\'', default='')
    
    align_name = os.path.join('./alignments/', parser.parse_args().align_name)
    batch_size = parser.parse_args().batch_size
    num_epochs = parser.parse_args().num_epochs
    multilayer = parser.parse_args().multilayer
    restore_path = parser.parse_args().restore_path

    # Restore the old model
    if restore_path:
        run_time = restore_path
    else:
        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    log_path = os.path.join('./model_logs', 'LSTM_'+run_time)
    logfile_path = os.path.join(log_path, 'logfile.txt')
    graph_log_path = os.path.join(log_path, 'graphs')
    checkpoint_log_path = os.path.join(log_path, 'checkpoints')
    test_ids_path = os.path.join(log_path, 'test_ids.pkl')

    # TODO: ADD A WEIGHT PATH HERE TOO

    if not os.path.exists(graph_log_path):
        os.makedirs(graph_log_path)

    if not os.path.exists(checkpoint_log_path):
        os.makedirs(checkpoint_log_path)

    logging.basicConfig(level=logging.INFO, filename=logfile_path,
                    format='%(asctime)-15s %(message)s')


    for flag_name in ['align_name', 'batch_size', 'num_epochs', 'multilayer', 'restore_path']:
        logging.info("{}: {}".format(flag_name, eval(flag_name)))

    print "Getting multiple sequence alignment"
    MSA = MultipleSequenceAlignment(align_name, test_ids_path=test_ids_path)

    max_length = MSA.max_seq_len
    alphabet_len = MSA.alphabet_len
    num_batches_per_epoch = int(ceil(float(MSA.train_size) / batch_size))
    num_test_batches = MSA.test_size / batch_size
    END_TOKEN = MSA.alphabet_map['*']

    data = tf.placeholder(tf.float32, [None, max_length, alphabet_len], name='data')
    target = tf.placeholder(tf.float32, [None, max_length, alphabet_len], name='target')


    print "Constructing model"
    model = VariableSequenceLabelling(data, target, use_multilayer=multilayer, end_token=END_TOKEN)


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


    print "Starting training"
    for epoch in range(pretrained_epochs, num_epochs):
        logging.info("Epoch: {}".format(epoch))
        
        for i in range(pretrained_batches, num_batches_per_epoch):
            #if i % batch_size == 0:
            if i % batch_size == 0:
                logging.info("Batch: {}".format(i))

                # GET TEST ERROR
                test_data, test_target = MSA.next_batch(batch_size, test=True)
                test_err_summary = sess.run(model.test_error, {data: test_data, target: test_target})

                writer.add_summary(test_err_summary, epoch*num_batches_per_epoch + i)
                saver.save(sess, os.path.join(checkpoint_log_path,'model_{}_{}'.format(epoch,i)))

                # GENERATE RANDOM SAMPLE
                seq_sample = model.generate_seq(sess)
                logging.info("Sample: {}".format(seq_sample))
                #sample_summary = sess.run(model.sample_seq_summary, {seq_placeholder: seq_sample})
                #writer.add_summary(sample_summary, epoch*num_batches_per_epoch + i)

            """
            if i == 10 or i == 1800:
                variables_names =[v.name for v in tf.trainable_variables()]
                values = sess.run(variables_names)
  
                with open('trained_vars_{}_{}.pkl'.format(epoch, i),'w') as f:
                    pickle.dump(dict(zip(variables_names, values)) ,f)
            """

            batch_data, batch_target = MSA.next_batch(batch_size)

            _, train_err_summary = sess.run([model.optimize, model.train_error], {data: batch_data, target: batch_target})
            writer.add_summary(train_err_summary, epoch*num_batches_per_epoch + i)

        test_data, test_target = MSA.next_batch(batch_size, test=True)
        logging.info("Error: {}".format(sess.run(model.error, {data: test_data, target: test_target})))


        # The number of batches of a given epoch that we already trained is only relevant 
        # to the first epoch we train after we restore the model
        pretrained_batches = 0
