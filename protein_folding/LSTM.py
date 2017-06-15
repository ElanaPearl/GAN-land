# Much code adopted from: danijar.com/variable-sequence-lengths-in-tensorflow/
import functools
import tensorflow as tf
from tensorflow.contrib import rnn
#from tensorflow.contrib.rnn import dynamic_rnn
#from tensorflow.contrib.rnn import core_rnn_cell

from model_w_label import MultipleSequenceAlignment


def lazy_property(function):
    #This is a wrapper to memoize properties
    #so we don't evaluate until we need them....
    #read this: https://pypi.python.org/pypi/lazy-property
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            try:
                setattr(self, attribute, function(self))
            except:
                import pdb;pdb.set_trace()
        return getattr(self, attribute)
    return wrapper


class VariableSequenceLabelling:

    def __init__(self, data, target, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.first_cell = True
        self.prediction
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

        if self.first_cell:
            print "FIRST CELL"
            cell = rnn.BasicLSTMCell(num_units=self._num_hidden)
        else:
            print "NON-FIRST CELL!!"
            cell = rnn.BasicLSTMCell(num_units=self._num_hidden, reuse=tf.get_variable_scope().reuse)
            self.first_cell = False

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
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction


    @property
    def cost(self):
        # Compute cross entropy for each frame.
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @property
    def optimize(self):
        learning_rate = 0.0003
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

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
    #output_seq_len = MSA.alphabet_len


    data = tf.placeholder(tf.float32, [None, length, seq_len])
    target = tf.placeholder(tf.float32, [None, length, seq_len])
    test_data, test_target = MSA.get_test_data()

    model = VariableSequenceLabelling(data, target)


    log_dir_name = 'LSTM_graph_logs/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    writer = tf.summary.FileWriter('./'+log_dir_name)
    sess = tf.Session()
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for _ in range(100):
            batch_data, batch_target = MSA.next_batch(10)

            sess.run(model.optimize, {data: batch_data, target: batch_target})
        error = sess.run(model.error, {data: test_data, target: test_target})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))