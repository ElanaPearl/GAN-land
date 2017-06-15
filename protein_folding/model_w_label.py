import numpy as np
import random

class MultipleSequenceAlignment:
    def __init__(self, filename, include_gaps=True):
        self.filename = filename
    
        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        self.acceptable_seq_alphabet = self.alphabet + '.-'
        self.alphabet_len = len(self.alphabet)
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}

        # Dictionary: key = seq_id, value = array of seq chars
        self.data_seqs = self._read_data()

        self.max_seq_len = max(len(seq) for seq in self.data_seqs.values())
        self.seq_size = self.max_seq_len*self.alphabet_len


        # TODO: make this test set getting process into an elegant function
        self.test_size = len(self.data_seqs)/5
        test_seq_ids = random.sample(self.data_seqs.keys(), self.test_size)
        self.test_seqs = dict(zip(test_seq_ids, [self.data_seqs[k] for k in test_seq_ids]))

        for k in test_seq_ids:
            del self.data_seqs[k]


    def _add_clean_seq(self, curr_id, curr_seq):
        clean_seq = []
        ignore_this_seq = False

        for aa in curr_seq:
            try:
                aa_c = self.alphabet_map[aa]
                clean_seq.append(aa_c)
            except:
                if aa.upper() not in self.acceptable_seq_alphabet:
                    ignore_this_seq = True

        if not ignore_this_seq:
            self.sequences[curr_id] = clean_seq


    def _read_data(self):
        """ Reads in data from filename and returns a dict of only seqs without weird characters"""

        with open(self.filename)as f:
            current_sequence = ""
            current_id = None

            self.sequences = {}

            for line in f:
                # Start reading new entry. If we already have
                # seen an entry before, return it first.
                if line.startswith(">"):
                    if current_id is not None:
                        self._add_clean_seq(current_id, current_sequence)

                    current_id = line.rstrip()[1:]
                    current_sequence = ""

                elif not line.startswith(";"):
                    current_sequence += line.rstrip()

            # Also do not forget last entry in file
            self._add_clean_seq(current_id, current_sequence)
        
        return self.sequences
    

    def _idx_to_one_hot(self, seq):
        zero_padding = [0]*(self.max_seq_len - len(seq))
        seq += zero_padding

        # Turn this into one hot encoding
        one_hot_seq = np.zeros((self.max_seq_len, self.alphabet_len))
        one_hot_seq[np.arange(self.max_seq_len), seq] = 1
        return one_hot_seq


    def next_batch(self, batch_size, test=False):
        mb = np.zeros((batch_size, self.max_seq_len, self.alphabet_len))

        for i in range(batch_size):  
            if len(self.data_seqs) > 0:
                seq_id = random.choice(self.data_seqs.keys())
                x = self._idx_to_one_hot(self.data_seqs[seq_id])
                mb[i] = x
                del self.data_seqs[seq_id] # Pop the seq off so that you don't use it again
            else:
                break
        
        # Incase you don't get to the end of the batch size, this will trim off the ones
        # you didn't get to (TODO: check if this is necessary)
        mb = mb[:i]
        
        # First trim the first aa off, then add an extra 0 for 'end character'
        # TOO: Should this character actually have its own character aka be 0
        # one hot encoded or just be a 0 vector? Rn it's just a 0 vector
        output_mb = np.concatenate((mb[:,1:,:],
                                    np.zeros((mb.shape[0],1,mb.shape[2]))), axis=1)

        return mb, output_mb

    def get_test_data(self):
        mb = np.zeros((self.test_size, self.max_seq_len, self.alphabet_len))
        for i, seq in enumerate(self.test_seqs.values()):
            mb[i] = self._idx_to_one_hot(seq)

        output_mb = np.concatenate((mb[:,1:,:],
                                    np.zeros((mb.shape[0],1,mb.shape[2]))), axis=1)

        return mb, output_mb
