import numpy as np
import tensorflow as tf
import random
import cPickle as pickle

TEST_ALIGN_ID = 'alignments/FYN_HUMAN_hmmerbit_plmc_n5_m30_f50_t0.2_r80-145_id100_b33.a2m'

class MultipleSequenceAlignment:
    def __init__(self, filename):
        self.filename = filename

        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY*'
        self.end_token = '*'
        

        self.acceptable_seq_alphabet = self.alphabet + '.-'
        self.alphabet_len = len(self.alphabet)
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}
        self.rev_alphabet_map = {i: s for i, s in enumerate(self.alphabet)}

    
        # READ IN DATA
        print "reading data"
        self.seqs = self._read_data()
        print "done reading"

        # GET METADATA ABOUT SEQUENCES
        self.max_seq_len = max(len(seq) for seq in self.seqs.values())
        self.num_seqs = len(self.seqs)
        test_size = self.num_seqs/5


        # ONE HOT ENCODE DATA
        print "one hot encoding"
        self.encoded_seqs = self.one_hot_encode()
        self.species_ids = self.seqs.keys()
        print "Done with one hot"

        # DELETE INITIAL DICT TO SAVE SPACE
        del self.seqs

        # GET WEIGHTS FOR SEQUENCES
        print "calculating seq weight"
        self.seq_weights = self.calc_seq_weights()
        print "done calculating weights"

        with open('SEQ_WEIGHTS.pkl','w') as f:
            pickle.dump(self.seq_weights, f)

        # SELECT TEST AND TRAIN SET
        # TODO: add in caveat for if 
        test_idx = np.random.choice(np.arange(num_seqs), size=test_size, \
                                     replace=False, p=self.seq_weights/sum(self.seq_weights))
        train_idx = list(set(np.arange(num_seqs)) - set(test_idx))

        # Indices in the full array of the test ones, 
        self.unused_test_idx = zip(test_idx, self.seq_weights[test_idx])
        self.unused_train_idx = zip(train_idx, self.seq_weights[train_idx])

    """
    def next_batch(self, train=True):
        if train:
            unused_idx = self.unused_train_idx
        else:
            unused_idx = self.unused_test_idx

        batch_idx = np.random.choice(unused_seq_info.keys(), size=batch_size, replace=False, p=unused_p/sum(unused_p))



        # unused_seq_info = zip(np.arange(len(seqs)), seq_weights)
        # function to create batches:

            # batch_idx = np.random.choice(unused_seq_info.keys(), size=batch_size, replace=False, p=unused_p/sum(unused_p))
        
            # for idx in batch_idx:
                # del unused_seq_info[idx]

            # create minibatch


        # Fill up self.seqs then also create a set of batches and you just pop em one at a time until its empty
        # so make a function for resetting batches

        # a batch is an array of indices
    """

    def calc_seq_weights(self):
        X_flat = tf.placeholder(tf.float32, [self.num_seqs, self.max_seq_len*self.alphabet_len], name="X_flat")
        cutoff = 0.2

        DOT1 = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(X_flat, x)), X_flat)
        DOT2 = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(x, tf.transpose(x))), X_flat)
        weights = tf.map_fn(lambda x: 1.0 / tf.reduce_sum(tf.cast(DOT1/DOT2 > 1 - cutoff, tf.float32)), X_flat)
        with tf.Session() as sess:
            return sess.run(weights, feed_dict={X_flat: self.encoded_seqs.reshape((self.num_seqs, self.max_seq_len*self.alphabet_len))})

   
    def one_hot_encode(self):
        encoded_seqs = np.zeros((self.num_seqs, self.max_seq_len, self.alphabet_len))

        for i, seq in enumerate(self.seqs.values()):
            # Encode the string as an array of indices
            encoded_seq = [self.alphabet_map[aa] for aa in seq]

            # Turn this into one hot encoding
            encoded_seqs[i, np.arange(len(seq)), encoded_seq] = 1

        return encoded_seqs


    def restore_test_set(self, test_seq_ids):
        print "Restoring test set"
        self.all_test_seq_ids = test_seq_ids
        self.reset_train_set()
        self.reset_test_set()

    def reset_train_set(self):
        print "Resetting train set"
        self.train_seq_ids = list(self.all_train_seq_ids)

    def reset_test_set(self):
        print "Resetting test set"
        self.test_seq_ids = list(self.all_test_seq_ids)


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

        with open(self.filename)as f:
            current_sequence = ""
            current_id = None

            for line in f:
                # Start reading new entry. If we already have
                # seen an entry before, return it first.
                if line.startswith(">"):
                    if current_id is not None:
                        self._add_sequence(current_id, current_sequence)

                    current_id = line.rstrip()[1:]
                    current_sequence = ""

                elif not line.startswith(";"):
                    current_sequence += line.rstrip()

            # Also do not forget last entry in file
            self._add_sequence(current_id, current_sequence)
        
        return self.seqs
   

    def convert_to_output(self, seq):
        return seq[1:] + self.end_token


    """
    def str_to_one_hot(self, seq):
        
        # Encode the string as an array of indices
        encoded_seq = [self.alphabet_map[aa] for aa in seq]

        # Turn this into one hot encoding
        one_hot_seq = np.zeros((self.max_seq_len, self.alphabet_len))
        one_hot_seq[np.arange(len(encoded_seq)), encoded_seq] = 1
        return one_hot_seq
    """

    def one_hot_to_str(self, seq):
        """ Converts a one-hot encoded sequence back into an amino acid sequence.

        Given a one-hot encoded seq of shape seq_len x alphabet_len this returns
        a string representing the amino acid sequence the input represents
        """

        # Trim off the end (after the end token)
        trimmed_seq = seq[:np.sum(seq)]

        encoded_seq = np.argmax(seq, 1)
        decoded_seq = map(lambda idx: self.rev_alphabet_map[idx], encoded_seq)
        return ''.join(decoded_seq)


    def next_batch(self, batch_size, stochastic=True, test=False):
        """ Generate a minibatch of training data: inputs and outputs.
        Creates an array of size batch_size x max_seq_length x alphabet_len
        and fills it with the one-hot encoded versions of batch_size number of
        random sequences that haven't been used yet in this epoch. For sequences
        that are less than max_seq_len, they are just padded with zeros. Then the
        training outputs are just these matrices but without the first letter and
        with an addition 'end_token' at the end
        """

        mb = np.zeros((batch_size, self.max_seq_len, self.alphabet_len))
        output_mb = np.zeros((batch_size, self.max_seq_len, self.alphabet_len))

        for i in range(batch_size): 
            # Check if there are any seqs left in this epoch 
            if len(self.train_seq_ids) > 0:
                # Choose a random sequence
                if stochastic:
                    seq_idx = random.randint(0,len(self.train_seq_ids)-1)
                else:
                    seq_idx = i

                seq = self.seqs[self.train_seq_ids[seq_idx]]
                mb[i] = self.str_to_one_hot(seq)
                output_mb[i] = self.str_to_one_hot(self.convert_to_output(seq))

                # Pop the seq off so that you don't use it again
                del self.train_seq_ids[seq_idx]
            else:

                # This is necessary
                mb = mb[:i]
                output_mb = output_mb[:i]
                # Set up the training ids for the next epoch
                self.reset_train_set()
                break

        return mb, output_mb


    def next_batch_test(self, batch_size):
        """ Generate a minibatch of training data: inputs and outputs.
        Creates an array of size batch_size x max_seq_length x alphabet_len
        and fills it with the one-hot encoded versions of batch_size number of
        random sequences that haven't been used yet in this epoch. For sequences
        that are less than max_seq_len, they are just padded with zeros. Then the
        training outputs are just these matrices but without the first letter and
        with an addition 'end_token' at the end
        """

        mb = np.zeros((batch_size, self.max_seq_len, self.alphabet_len))
        output_mb = np.zeros((batch_size, self.max_seq_len, self.alphabet_len))

        for i in range(batch_size): 
            # Check if there are any seqs left in this epoch 
            if len(self.test_seq_ids) > 0:
                # Choose a random sequence
                seq_idx = random.randint(0,len(self.test_seq_ids)-1)
                seq = self.seqs[self.test_seq_ids[seq_idx]]

                mb[i] = self.str_to_one_hot(seq)
                output_mb[i] = self.str_to_one_hot(self.convert_to_output(seq))

                # Pop the seq off so that you don't use it again
                del self.test_seq_ids[seq_idx]
            else:

                mb = mb[:i]
                output_mb = output_mb[:i]

                # Set up the training ids for the next epoch
                self.reset_test_set()
                break

        return mb, output_mb
