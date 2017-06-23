import numpy as np
import random

class MultipleSequenceAlignment:
    def __init__(self, filename):
        self.filename = filename

        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY*'
        self.end_token = '*'

        self.acceptable_seq_alphabet = self.alphabet + '.-'
        self.alphabet_len = len(self.alphabet)
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}
        self.rev_alphabet_map = {i: s for i, s in enumerate(self.alphabet)}

        # Dictionary: {species_id, sequence string}
        self.seqs = self._read_data()

        self.max_seq_len = max(len(seq) for seq in self.seqs.values())
        self.seq_size = self.max_seq_len*self.alphabet_len

        self.num_seqs = len(self.seqs)
        self.test_size = self.num_seqs/5
        self.train_size = self.num_seqs - self.test_size

        self.all_test_seq_ids = random.sample(self.seqs.keys(), self.test_size)
        self.all_train_seq_ids = list(set(self.seqs.keys()) - set(self.all_test_seq_ids))
        self.reset_train_set()
        self.reset_test_set()

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
        #sequence = []
        ignore_this_seq = False

        for aa in curr_seq:
            # If it is a capital letter and in the alphabet add it to clean seq
            if aa in self.alphabet_map:
                encoded_seq += aa

            # If the sequence includes non aa letters, ignore the sequence
            elif aa.upper() not in self.acceptable_seq_alphabet:
                ignore_this_seq = True

        if not ignore_this_seq:
            self.sequences[curr_id] = encoded_seq


    def _read_data(self):
        """ Converts data into 

        Reads in the a2m alignment file an converts it into a dictionary where
        the keys are species ids and the values the sequences. And it only
        includes sequences made of traditional amino acids.
        """

        with open(self.filename)as f:
            current_sequence = ""
            current_id = None

            self.sequences = {}

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
        
        return self.sequences
   

    def convert_to_output(self, seq):
        return seq[1:] + self.end_token


    def str_to_one_hot(self, seq):
        """ Turns a sequence of amino acids into a one-hot encoded matrix.
        """
        
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

        encoded_seq = np.argmax(seq, 1)
        decoded_seq = map(lambda idx: self.rev_alphabet_map[idx], encoded_seq)
        return ''.join(decoded_seq)

    def next_batch(self, batch_size):
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
                seq_idx = random.randint(0,len(self.train_seq_ids)-1)
                seq = self.seqs[self.train_seq_ids[seq_idx]]

                mb[i] = self.str_to_one_hot(seq)
                output_mb[i] = self.str_to_one_hot(self.convert_to_output(seq))

                # Pop the seq off so that you don't use it again
                del self.train_seq_ids[seq_idx]
            else:
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
                # Set up the training ids for the next epoch
                self.reset_test_set()
                break

        return mb, output_mb
