# Much code adopted from: danijar.com/variable-sequence-lengths-in-tensorflow/
import functools
import tensorflow as tf
from tensorflow.contrib import rnn
#from tensorflow.contrib.rnn import dynamic_rnn
#from tensorflow.contrib.rnn import core_rnn_cell

from model_w_label import MultipleSequenceAlignment
from datetime import datetime
import os

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

    def __init__(self, data, target, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.first_cell = True
        #with tf.variable_scope('prediction'):
        self.prediction
        with tf.variable_scope('calc_error'):
            self.error
        self.optimize


    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length  
        
    @lazy_property
    def prediction(self):
        # Recurrent network.

        with tf.variable_scope('dynamic_rnn'):
            with tf.variable_scope('LSTM_cell'):
                cell = rnn.BasicLSTMCell(num_units=self._num_hidden, reuse=tf.get_variable_scope().reuse)

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
        with tf.variable_scope('calc_predictions'):
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
        #weight = tf.truncated_normal([in_size, out_size], stddev=0.01, name='weight')
        #bias = tf.constant(0.1, shape=[out_size], name='bias')
        weight = tf.get_variable(name='weight',
                                shape=[in_size, out_size],
                          initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias = tf.get_variable(name='bias',
                              shape=[out_size], 
                        initializer=tf.constant_initializer(0.1))
        return weight, bias

"""
def get_dataset():
    # Read dataset and flatten images.
    dataset = sets.Ocr()
    dataset = sets.OneHot(dataset.target, depth=2)(dataset, columns=['target'])
    dataset['data'] = dataset.data.reshape(
        dataset.data.shape[:-2] + (-1,)).astype(float)
    train, test = sets.Split(0.66)(dataset)
    return train, test
"""

if __name__ == '__main__':
    align_name = 'FYN_HUMAN_hmmerbit_plmc_n5_m30_f50_t0.2_r80-145_id100_b33.a2m'
    MSA = MultipleSequenceAlignment('./alignments/'+align_name)

    # length of longest sequence
    # output size should be self.alphabet_len (21 including END char)
    length = MSA.max_seq_len

    seq_len = MSA.alphabet_len
    # Add +1 for end of seq maybe
    # output_seq_len = MSA.alphabet_len


    data = tf.placeholder(tf.float32, [None, length, seq_len], name='data')
    target = tf.placeholder(tf.float32, [None, length, seq_len], name='target')
    test_data, test_target = MSA.get_test_data()

    model = VariableSequenceLabelling(data, target)


    # TODO: abstract out the act of making this logging system
    log_dir_name = 'LSTM_graph_logs/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'

    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)

    writer = tf.summary.FileWriter('./'+log_dir_name)
    sess = tf.Session()
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        print "epoch: ", epoch
        for i in range(100):
            if i % 10 == 0:
                print "Batch: ", i
            batch_data, batch_target = MSA.next_batch(10)

            sess.run(model.optimize, {data: batch_data, target: batch_target})
        error = sess.run(model.error, {data: test_data, target: test_target})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))