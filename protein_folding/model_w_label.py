from sklearn.cluster import KMeans
from datetime import datetime
import tensorflow as tf
import numpy as np
import random
import os

import tools


class MultipleSequenceAlignment:
    def __init__(self, gene_name, run_time, seq_limit=None):

        filename = os.path.join('alignments', tools.get_alignment_filename(gene_name))
        make_path = lambda x: os.path.join('model_logs', gene_name, x)

        # READ IN DATA
        self.seqs = self._read_data(filename)

        # GET METADATA ABOUT SEQUENCES
        self.max_seq_len = max(len(seq) for seq in self.seqs.values())
        self.num_seqs = len(self.seqs)

        # GET WEIGHTS FOR SEQUENCES
        seq_log_path = make_path('seq_weights.pkl')
        self.seq_weights = tools.lazy_calculate(self.calc_seq_weights, seq_log_path)

        # GET CLUSTERS FOR SEQUENCES
        cluster_path = make_path('clusters_{}.pkl'.format(tools.n_clusters))
        self.seq_clusters = tools.lazy_calculate(self.cluster_seqs, cluster_path)

        # SELECT TEST AND TRAIN SET
        test_id_path = make_path(os.path.join(run_time, 'test_ids.pkl'))
        self.test_idx = tools.lazy_calculate(self.choose_test_set, test_id_path)
        self.train_idx = list(set(np.arange(self.num_seqs)) - set(self.test_idx))

        self.test_size = len(self.test_idx)
        self.train_size = len(self.train_idx)

        # DICTIONARIES TO KEEP TRACK OF WHICH INDICES HAVE BEEN USED IN A GIVEN EPOCH
        self.unused_test_idx = dict(zip(self.test_idx, self.seq_weights[self.test_idx]))
        self.unused_train_idx = dict(zip(self.train_idx, self.seq_weights[self.train_idx]))

        # Remove gaps
        self.seqs = tools.remove_gaps(self.seqs)

        # Re-adjust the max sequence length
        self.max_seq_len = max(len(seq) for seq in self.seqs.values())

        # GET DIST OF FIRST ELEMENT OF SEQS (for generation purposes)
        seed_weight_path = make_path('seed_weights.pkl')
        self.seed_weights = tools.lazy_calculate(self.calc_seed_weights, seed_weight_path)


    def choose_test_set(self):
        all_groups = set(np.arange(tools.n_clusters))
        test_idx = []
        while len(test_idx) < self.num_seqs/6:
            group_to_add = random.choice(list(all_groups))
            idx_to_add = np.where(self.seq_clusters == group_to_add)[0]
            test_idx += list(idx_to_add)
            all_groups.remove(group_to_add)
        return test_idx


    def cluster_seqs(self):
        encoded = self.encode_all()
        encoded = np.reshape(encoded, (encoded.shape[0], encoded.shape[1]*encoded.shape[2]))
        KM = KMeans(n_clusters=tools.n_clusters)
        KM.fit(encoded)
        return KM.labels_


    def calc_seed_weights(self):
        first_vals = dict(zip(list(tools.alphabet),np.zeros(tools.alphabet_len)))
        for i, v in enumerate(self.seqs.values()):
            first_vals[v[0]] += self.seq_weights[i]

        norm_const = sum(first_vals.values())
        first_vals = {k: v / norm_const for k, v in first_vals.iteritems()}
        return [first_vals[k] for k in tools.alphabet]


    def calc_seq_weights(self):
        # Create encoded version of all of the data
        encoded_seqs = self.encode_all()

        X = tf.placeholder(tf.float32, [self.num_seqs, self.max_seq_len, tools.alphabet_len], name="X_flat")
        X_flat = tf.reshape(X, [self.num_seqs, self.max_seq_len * tools.alphabet_len])

        weights = tf.map_fn(lambda x: 1.0/ tf.reduce_sum(tf.cast(tf.reduce_sum(tf.multiply(X_flat, x), axis=1) / tf.reduce_sum(x) > 1 - tools.similarity_cutoff, tf.float32)), X_flat)

        with tf.Session() as sess:
            return sess.run(weights, feed_dict={X: encoded_seqs})

   
    # TODO: combine encode_all and str_to_one_hot into one fxn bc they're basically the same
    def encode_all(self):
        encoded_seqs = np.zeros((self.num_seqs, self.max_seq_len, tools.alphabet_len))

        for i, seq in enumerate(self.seqs.values()):
            # Encode the string as an array of indices
            encoded_seq = [tools.alphabet_map[aa] for aa in seq]

            # Turn this into one hot encoding
            encoded_seqs[i, np.arange(len(seq)), encoded_seq] = 1

        return encoded_seqs

 
    def str_to_one_hot(self, seq):
        
        # Encode the string as an array of indices
        encoded_seq = [tools.alphabet_map[aa] for aa in seq]

        # Turn this into one hot encoding
        one_hot_seq = np.zeros((self.max_seq_len, tools.alphabet_len))
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
        decoded_seq = map(lambda idx: tools.rev_alphabet_map[idx], encoded_seq)
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

        mb = np.zeros((batch_size, self.max_seq_len, tools.alphabet_len))
        output_mb = np.zeros((batch_size, self.max_seq_len, tools.alphabet_len))


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
                output_mb[i] = self.str_to_one_hot(seq[1:] + tools.END_TOKEN)

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
            if aa in tools.alphabet or aa == tools.GAP:
                encoded_seq += aa

            # If the sequence includes non aa letters, ignore the sequence
            elif aa.upper() not in tools.alphabet + '.':
                ignore_this_seq = True

        if not ignore_this_seq:
            self.seqs[curr_id] = encoded_seq


    def _read_data(self, filename):
        """ Converts data into

        Reads in the a2m alignment file an converts it into a dictionary where
        the keys are species ids and the values the sequences. And it only
        includes sequences made of traditional amino acids.
        """

        self.seqs = {}

        with open(filename)as f:
            current_sequence = ""
            current_id = None

            for line in f:
                # Start reading new entry. If we already have
                # seen an entry before, return it first.
                if line.startswith(">"):
                    if current_id is not None:
                        self._add_sequence(current_id, current_sequence)

                        if seq_limit and len(self.seqs) == seq_limit:
                            return self.seqs

                    current_id = line.rstrip()[1:]
                    current_sequence = ""

                elif not line.startswith(";"):
                    current_sequence += line.rstrip()

            # Also do not forget last entry in file
            self._add_sequence(current_id, current_sequence)
        
        return self.seqs
   
