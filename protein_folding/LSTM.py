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

def lazy_property(function):
    #This is a wrapper to memoize properties
    #so we don't evaluate until we need them....
    #read this: https://pypi.python.org/pypi/lazy-property
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class VariableSequenceLabelling:

    def __init__(self, data, target, num_hidden=200, num_layers=3, use_multilayer=False):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.use_multilayer = use_multilayer
        with tf.variable_scope('prediction'):
            self.prediction
        with tf.variable_scope('calc_err'):
            self.error
            self.test_error = tf.summary.scalar('test_error',self.error)
            self.train_error = tf.summary.scalar('train_error',self.error)
        self.optimize

    @property
    def length(self):
        with tf.variable_scope('calc_lengths'):
            used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
            return length  
    
    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(
            self._num_hidden, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)

    @lazy_property
    def prediction(self):
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
            ) # You get value error if input is none or an empty list
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        with tf.variable_scope('output_to_prediction'):
            output = tf.reshape(output, [-1, self._num_hidden])
            prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
            prediction = tf.reshape(prediction, [-1, max_length, num_classes])
            return prediction


    @property
    def cost(self):
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

    @property
    def optimize(self):
        learning_rate = 0.0003
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # with tf.variable_scope('minimize_cost'):
        # ^ when I include that I get:
        # ValueError: Variable minimize_cost/prediction/dynamic_rnn/rnn/basic_lstm_cell/weights/Adam_optimizer/
        # already exists, disallowed. Did you mean to set reuse=True in VarScope?
        with tf.name_scope('minimize_cost'):
            return optimizer.minimize(self.cost)

    @property
    def error(self):
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
        new_seq = [self.START_TOKEN
        # while the last element isn't end_token

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--align_name', help='The name of the alignment file for the protein family', \
                         default='FYN_HUMAN_hmmerbit_plmc_n5_m30_f50_t0.2_r80-145_id100_b33.a2m')
    parser.add_argument('--batch_size', help='Number of sequences per batch', type=int, default=50)
    parser.add_argument('--num_epochs', help='Number of epochs of training', type=int, default=10)
    parser.add_argument('--multilayer', help='Use multiple LSTM layers', type=bool, default=False)
    parser.add_argument('--restore_path', help='Path to restore model, should be of the format '\
                        '\'LSTM_year-month-date_hour-min-sec\'', default='')
    
    align_name = './alignments/' + parser.parse_args().align_name
    batch_size = parser.parse_args().batch_size
    num_epochs = parser.parse_args().num_epochs
    multilayer = parser.parse_args().multilayer
    restore_path = parser.parse_args().restore_path

    print "Getting multiple sequence alignment"
    MSA = MultipleSequenceAlignment(align_name)

    length = MSA.max_seq_len
    seq_len = MSA.alphabet_len
    num_batches_per_epoch = MSA.train_size / batch_size

    data = tf.placeholder(tf.float32, [None, length, seq_len], name='data')
    target = tf.placeholder(tf.float32, [None, length, seq_len], name='target')

    print "Constructing model"
    model = VariableSequenceLabelling(data, target, use_multilayer=multilayer)

    # TODO: abstract out the act of making this logging system

    if restore_path:
        log_path = './model_logs/{}/'.format(restore_path)
        with open(log_path + 'test_set_ids.pkl') as f:
            MSA.restore_test_set(pickle.load(f))
    else:
        log_path = './model_logs/LSTM_{}/'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_path)
        with open(log_path + 'test_set_ids.pkl', 'w') as f:
            pickle.dump(MSA.test_seq_ids, f)

    graph_log_path = log_path + '/graphs/'
    checkpoint_log_path = log_path + '/checkpoints/'
    
    if not os.path.exists(graph_log_path):
        os.makedirs(graph_log_path)

    if not os.path.exists(checkpoint_log_path):
        os.makedirs(checkpoint_log_path)

    writer = tf.summary.FileWriter(graph_log_path)
    sess = tf.Session()
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    

    if restore_path:
        print "Restoring checkpoint"
        last_ckpt = tf.train.latest_checkpoint(checkpoint_log_path)
        #saver = tf.train.import_meta_graph(last_ckpt+'.meta')
        saver = tf.train.Saver()
        saver.restore(sess, last_ckpt)

        pretrained_epochs, pretrained_batches = int(last_ckpt.split('_')[-2]), int(last_ckpt.split('_')[-1])
        print "Picking up with epoch {} and batch {}".format(pretrained_epochs, pretrained_batches)
    else:
        saver = tf.train.Saver()
        pretrained_epochs = 0
        pretrained_batches = 0

    test_data, test_target = MSA.test_data

    print "Starting training"
    for epoch in range(pretrained_epochs, num_epochs):
        print "Epoch: ", epoch
        for i in range(pretrained_batches, num_batches_per_epoch):
            if i % batch_size == 0:
                print "Batch: ", i
                test_err_summary = sess.run(model.test_error, {data: test_data, target: test_target})
                writer.add_summary(test_err_summary, epoch*num_batches_per_epoch + i)
                saver.save(sess, checkpoint_log_path+'model_{}_{}'.format(epoch,i))

            batch_data, batch_target = MSA.next_batch(batch_size)

            _, train_err_summary = sess.run([model.optimize, model.train_error], {data: batch_data, target: batch_target})
            writer.add_summary(train_err_summary, epoch*num_batches_per_epoch + i)

        # The number of batches of a given epoch that we already trained is only relevant 
        # to the first epoch we train after we restore the model
        pretrained_batches = 0

        error = sess.run(model.error, {data: test_data, target: test_target})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
