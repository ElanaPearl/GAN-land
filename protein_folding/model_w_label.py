import numpy as np
import random

class MultipleSequenceAlignment:
    def __init__(self, filename):
        self.filename = filename

        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        self.acceptable_seq_alphabet = self.alphabet + '.-'
        self.alphabet_len = len(self.alphabet)
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}

        # Dictionary: key = seq_id, value = array of seq chars
        self.seqs = self._read_data()

        self.max_seq_len = max(len(seq) for seq in self.seqs.values())
        self.seq_size = self.max_seq_len*self.alphabet_len

        self.num_seqs = len(self.seqs)
        self.test_size = self.num_seqs/5
        self.train_size = self.num_seqs - self.test_size

        self.test_seq_ids = random.sample(self.seqs.keys(), self.test_size)
        self.test_data = self.get_test_data()
        self.reset_train_set()


    def reset_train_set(self):
        self.train_seq_ids = list(set(self.seqs.keys()) - set(self.test_seq_ids))

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
            if len(self.train_seq_ids) > 0:

                seq_idx = random.randint(0,len(self.train_seq_ids))
                x = self._idx_to_one_hot(self.seqs[self.train_seq_ids[seq_idx]])
                mb[i] = x
                del self.train_seq_ids[seq_idx] # Pop the seq off so that you don't use it again
            else:
                self.reset_train_set()
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
        for i, seq_id in enumerate(self.test_seq_ids):
            mb[i] = self._idx_to_one_hot(self.seqs[seq_id])

        output_mb = np.concatenate((mb[:,1:,:],
                                    np.zeros((mb.shape[0],1,mb.shape[2]))), axis=1)

        return mb, output_mb
