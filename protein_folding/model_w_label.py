import numpy as np
import tensorflow as tf
import random
import cPickle as pickle
import os

TEST_ALIGN_ID = 'alignments/FYN_HUMAN_hmmerbit_plmc_n5_m30_f50_t0.2_r80-145_id100_b33.a2m'
END_TOKEN = '*'
USE_SMALL = False
MED_SIZE = 10000

class MultipleSequenceAlignment:
    def __init__(self, filename, weight_path='SEQ_WEIGHTS.pkl', test_ids_path='test_ids.pkl'):
        self.filename = filename

        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY*'
        
        self.acceptable_seq_alphabet = self.alphabet + '.-'
        self.alphabet_len = len(self.alphabet)
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}
        self.rev_alphabet_map = {i: s for i, s in enumerate(self.alphabet)}

        # READ IN DATA
        self.seqs = self._read_data()

        # GET METADATA ABOUT SEQUENCES
        self.max_seq_len = max(len(seq) for seq in self.seqs.values())
        self.num_seqs = len(self.seqs)
        self.test_size = self.num_seqs/5
        self.train_size = self.num_seqs - self.test_size


        # GET WEIGHTS FOR SEQUENCES
        if os.path.exists(weight_path):
            with open(weight_path) as f:
                self.seq_weights = pickle.load(f)
        else:
            self.seq_weights = self.calc_seq_weights()
            with open(weight_path,'w') as f:
                pickle.dump(self.seq_weights, f)

        # SELECT TEST AND TRAIN SET
        if os.path.exists(test_ids_path):
            with open(test_ids_path) as f:
                self.test_idx = pickle.load(f)
        else:
            self.test_idx = np.random.choice(np.arange(self.num_seqs), \
                                         size=self.test_size, \
                                         replace=False, \
                                         p=self.seq_weights/sum(self.seq_weights))
            with open(test_ids_path, 'w') as f:
                pickle.dump(self.test_idx, f)

        self.train_idx = list(set(np.arange(self.num_seqs)) - set(self.test_idx))


        # DICTIONARIES TO KEEP TRACK OF WHICH INDICES HAVE BEEN USED IN A GIVEN EPOCH
        self.unused_test_idx = dict(zip(self.test_idx, self.seq_weights[self.test_idx]))
        self.unused_train_idx = dict(zip(self.train_idx, self.seq_weights[self.train_idx]))


    def calc_seq_weights(self):
        # Create encoded version of all of the data
        encoded_seqs = self.encode_all()
        cutoff = 0.2

        X = tf.placeholder(tf.float32, [self.num_seqs, self.max_seq_len, self.alphabet_len], name="X_flat")
        X_flat = tf.reshape(X, [self.num_seqs, self.max_seq_len *self.alphabet_len])



        weights = tf.map_fn(lambda x: 1.0/ tf.reduce_sum(tf.cast(tf.reduce_sum(tf.multiply(X_flat, x), axis=1) / tf.reduce_sum(x) > 1 - cutoff, tf.float32)), X_flat)


        #X_norm_factor = tf.reduce_sum(X_flat, axis=1, keep_dims=True)



        #sq_X = tf.matmul(X_flat, X_flat, transpose_b=True)
        #norm_X = sq_X / X_norm_factor

        #weights = 1.0 / tf.reduce_sum(tf.cast(norm_X > 1 - cutoff, tf.float32), axis=1)

        with tf.Session() as sess:
            return sess.run(weights, feed_dict={X: encoded_seqs})

   
    # TODO: combine encode_all and str_to_one_hot into one fxn bc they're basically the same
    def encode_all(self):
        encoded_seqs = np.zeros((self.num_seqs, self.max_seq_len, self.alphabet_len))

        for i, seq in enumerate(self.seqs.values()):
            # Encode the string as an array of indices
            encoded_seq = [self.alphabet_map[aa] for aa in seq]

            # Turn this into one hot encoding
            encoded_seqs[i, np.arange(len(seq)), encoded_seq] = 1

        return encoded_seqs

 
    def str_to_one_hot(self, seq):
        
        # Encode the string as an array of indices
        encoded_seq = [self.alphabet_map[aa] for aa in seq]

        # Turn this into one hot encoding
        one_hot_seq = np.zeros((self.max_seq_len, self.alphabet_len))
        one_hot_seq[np.arange(len(encoded_seq)), encoded_seq] = 1
        return one_hot_seq


    def one_hot_to_str(self, seq):
        """ Converts a one-hot encoded sequence back into an amino acid sequence.

        Given a one-hot encoded seq of shape seq_len x alphabet_len this returns
        a string representing the amino acid sequence the input represents
        """

        # Trim off the end (after the end token)
        trimmed_seq = seq[:np.sum(seq)]

        encoded_seq = np.argmax(trimmed_seq, 1)
        decoded_seq = map(lambda idx: self.rev_alphabet_map[idx], encoded_seq)
        return ''.join(decoded_seq)


    def next_batch(self, batch_size, test=False):
        """ Generate a minibatch of train or test data: inputs and outputs.
        Creates an array of size batch_size x max_seq_length x alphabet_len
        and fills it with the one-hot encoded versions of batch_size number of
        random sequences that haven't been used yet in this epoch. For sequences
        that are less than max_seq_len, they are just padded with zeros. Then the
        training outputs are just these matrices but without the first letter and
        with an additional end token at the end
        """

        mb = np.zeros((batch_size, self.max_seq_len, self.alphabet_len))
        output_mb = np.zeros((batch_size, self.max_seq_len, self.alphabet_len))


        if test:
            unused_seq_info = self.unused_test_idx
        else:
            unused_seq_info = self.unused_train_idx


        for i in range(batch_size):      
            # Check if there are any seqs left in this epoch
            if len(unused_seq_info) > 0:
                idx = np.random.choice(unused_seq_info.keys(), \
                                    p=unused_seq_info.values()/sum(unused_seq_info.values()))

                seq = self.seqs[self.seqs.keys()[idx]]
                mb[i] = self.str_to_one_hot(seq)
                output_mb[i] = self.str_to_one_hot(seq[1:] + END_TOKEN)

                # Pop the seq off so that you don't use it again
                del unused_seq_info[idx]

            # If there aren't enough sequences to fill the minibatch
            else:

                # This is necessary
                mb = mb[:i]
                output_mb = output_mb[:i]

                # Set up the training ids for the next epoch
                if test:
                    self.unused_test_idx = dict(zip(self.test_idx, self.seq_weights[self.test_idx]))
                else:
                    self.unused_train_idx = dict(zip(self.train_idx, self.seq_weights[self.test_idx]))
                break

        return mb, output_mb


    def _add_sequence(self, curr_id, curr_seq):
        """ Adds a given sequence to the sequence dictionary.

        First it adds the 'end token' to the sequence then it checks if it
        contains any characters other than traditional amino acids and gaps.
        If it doesn't, then it adds this sequence to the sequences dictionary
        """

        encoded_seq = ''
        ignore_this_seq = False

        for aa in curr_seq:
            # If it is a capital letter and in the alphabet add it to clean seq
            if aa in self.alphabet_map:
                encoded_seq += aa

            # If the sequence includes non aa letters, ignore the sequence
            elif aa.upper() not in self.acceptable_seq_alphabet:
                ignore_this_seq = True

        if not ignore_this_seq:
            self.seqs[curr_id] = encoded_seq


    def _read_data(self):
        """ Converts data into

        Reads in the a2m alignment file an converts it into a dictionary where
        the keys are species ids and the values the sequences. And it only
        includes sequences made of traditional amino acids.
        """

        self.seqs = {}
        i = 0

        with open(self.filename)as f:
            current_sequence = ""
            current_id = None

            for line in f:
                # Start reading new entry. If we already have
                # seen an entry before, return it first.
                if line.startswith(">"):
                    if current_id is not None:
                        self._add_sequence(current_id, current_sequence)

                        
                        if USE_SMALL and i == MED_SIZE:
                            return self.seqs
                        i += 1
                        

                    current_id = line.rstrip()[1:]
                    current_sequence = ""

                elif not line.startswith(";"):
                    current_sequence += line.rstrip()

            # Also do not forget last entry in file
            self._add_sequence(current_id, current_sequence)
        
        return self.seqs
   
