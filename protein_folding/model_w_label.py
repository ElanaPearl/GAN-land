import numpy as np
import tensorflow as tf
import random
import cPickle as pickle
import os
from sklearn.cluster import KMeans
from functools import wraps

from datetime import datetime

TEST_ALIGN_ID = 'FYN_HUMAN_hmmerbit_plmc_n5_m30_f50_t0.2_r80-145_id100_b33.a2m'
END_TOKEN = '*'
USE_SMALL = True
MED_SIZE = 1000
NUM_GROUPS = 50 # ARBITRARY HYPERPARAMETER UGH

def lazy_calculate(function, path):
    if os.path.exists(path):
        with open(path) as f:
            return pickle.load(f)
    else:
        data = function
        with open(path, 'w') as f:
            pickle.dump(data, f)
        return data


class MultipleSequenceAlignment:
    def __init__(self, filename, log_path):

        self.filename = os.path.join('alignments',filename)

        gene_name = filename.split('_')[0]

        alphabet_no_gaps = 'ACDEFGHIKLMNPQRSTVWY*'
        alphabet_w_gaps = 'ACDEFGHIKLMNPQRSTVWY-'

        self.alphabet = alphabet_w_gaps
        self.alphabet_len = len(self.alphabet)
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}
        self.rev_alphabet_map = {i: s for i, s in enumerate(self.alphabet)}

        # READ IN DATA
        self.seqs = self._read_data()

        # GET METADATA ABOUT SEQUENCES
        self.max_seq_len = max(len(seq) for seq in self.seqs.values())
        self.num_seqs = len(self.seqs)
        
        # TODO: MAKE A FUNCTION THAT CHECK IF A PATH EXISTS AND IF SO RESTORES AND ELSE
        # CALCULATES THEN DUMPS -- note this all still assumes you have a seq_logs folder

        # GET WEIGHTS FOR SEQUENCES

        seq_log_path = os.path.join('seq_logs','{}_seq_weights.pkl'.format(gene_name))
        if os.path.exists(seq_log_path): 
            print "RESTORING WEIGHTS"   
            with open(seq_log_path) as f:
                self.seq_weights = pickle.load(f)
        else:
            print "CALCULATING NEW WEIGHTS"
            self.seq_weights = self.calc_seq_weights()
            with open(seq_log_path,'w') as f:
                pickle.dump(self.seq_weights, f)

        # GET GROUPS FOR SEQUENCES
        cluster_path = os.path.join('seq_logs','{}_{}_clusters.pkl'.format(gene_name,NUM_GROUPS))
        if os.path.exists(cluster_path): 
            print "RESTORING GROUPINGS"   
            with open(cluster_path) as f:
                self.seq_clusters = pickle.load(f)
        else:
            print "CALCULATING NEW GROUPINGS" 
            self.seq_clusters = self.cluster_seqs()
            with open(cluster_path,'w') as f:
                pickle.dump(self.seq_clusters, f)

        # SELECT TEST AND TRAIN SET
        # Check if the test set has already been chosen, if so restore that
        if os.path.exists(os.path.join(log_path,'test_ids.pkl')):
            with open(os.path.join(log_path,'test_ids.pkl')) as f:
                self.test_idx = pickle.load(f)
        else:
            self.test_idx = self.choose_test_set()
            with open(os.path.join(log_path,'test_ids.pkl'), 'w') as f:
                pickle.dump(self.test_idx, f)

        self.train_idx = list(set(np.arange(self.num_seqs)) - set(self.test_idx))

        self.test_size = len(self.test_idx)
        self.train_size = len(self.train_idx)

        # DICTIONARIES TO KEEP TRACK OF WHICH INDICES HAVE BEEN USED IN A GIVEN EPOCH
        self.unused_test_idx = dict(zip(self.test_idx, self.seq_weights[self.test_idx]))
        self.unused_train_idx = dict(zip(self.train_idx, self.seq_weights[self.train_idx]))


        # Remove gaps and re-adjust seq lens
        self.seqs = {k: v.replace('-','') for k,v in self.seqs.iteritems()}
        self.alphabet = alphabet_no_gaps
        self.alphabet_len = len(self.alphabet)
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}
        self.rev_alphabet_map = {i: s for i, s in enumerate(self.alphabet)}
        self.max_seq_len = max(len(seq) for seq in self.seqs.values())

        # Calculate distribution of first elements of seqs (for generation purposes)
        self.seed_weights = self.calc_seed_weights()

    def choose_test_set(self):
        test_size = 0
        all_groups = set(np.arange(NUM_GROUPS))
        test_idx = []
        while test_size < self.num_seqs/6:
            group_to_add = random.choice(list(all_groups))
            idx_to_add = np.where(self.seq_clusters == group_to_add)[0]
            test_idx += list(idx_to_add)
            test_size += len(idx_to_add)
            all_groups.remove(group_to_add)
        return test_idx


    def cluster_seqs(self):
        encoded = self.encode_all()
        encoded = np.reshape(encoded, (encoded.shape[0], encoded.shape[1]*encoded.shape[2]))
        KM = KMeans(n_clusters=NUM_GROUPS)
        KM.fit(encoded)
        return KM.labels_


    def calc_seed_weights(self):
        first_vals = dict(zip(list(self.alphabet),np.zeros(self.alphabet_len)))
        for i, v in enumerate(self.seqs.values()):
            first_vals[v[0]] += self.seq_weights[i]

        norm_const = sum(first_vals.values())
        first_vals = {k: v / norm_const for k, v in first_vals.iteritems()}
        return [first_vals[k] for k in self.alphabet]


    def calc_seq_weights(self):
        # Create encoded version of all of the data
        encoded_seqs = self.encode_all()
        cutoff = 0.2

        X = tf.placeholder(tf.float32, [self.num_seqs, self.max_seq_len, self.alphabet_len], name="X_flat")
        X_flat = tf.reshape(X, [self.num_seqs, self.max_seq_len *self.alphabet_len])

        weights = tf.map_fn(lambda x: 1.0/ tf.reduce_sum(tf.cast(tf.reduce_sum(tf.multiply(X_flat, x), axis=1) / tf.reduce_sum(x) > 1 - cutoff, tf.float32)), X_flat)

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
        #import pdb; pdb.set_trace()
        trimmed_seq = seq[:int(np.sum(seq))]

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
            if aa in self.alphabet:
                encoded_seq += aa

            # If the sequence includes non aa letters, ignore the sequence
            elif aa.upper() not in self.alphabet and aa != '.':
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
   
