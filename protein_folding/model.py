import numpy as np

class MultipleSequenceAlignment:
    def __init__(self, filename, include_gaps=True):
        self.filename = filename
    
        self.alphabet = '-ACDEFGHIKLMNPQRSTVWY'

        self.alphabet_len = len(self.alphabet)
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}
        self.alphabet_map['.'] = -1

        # Dictionary: key = seq_id, value = array of seq chars
        self.seqs = self._read_data()

        # Create set of seqs to sample from for minibatches
        self.seq_ids = set(self.seqs)        

        # Mask out columns with '.' or lowercase letters
        example_seq = self.seqs[self.seqs.keys()[0]]        
        self.position_mask = [aa in self.alphabet for aa in example_seq]

        self.max_seq_len = sum(self.position_mask)
        self.seq_size = self.max_seq_len*self.alphabet_len

    
    def _read_data(self):
        """ Reads in data from filename and returns a matrix (num seqs x seq len)"""

        with open(self.filename)as f:
            current_sequence = ""
            current_id = None

            sequences = {}

            for line in f:
                # Start reading new entry. If we already have
                # seen an entry before, return it first.
                if line.startswith(">"):
                    if current_id is not None:
                        sequences[current_id] = current_sequence

                    current_id = line.rstrip()[1:]
                    current_sequence = ""

                elif not line.startswith(";"):
                    current_sequence += line.rstrip()

            # Also do not forget last entry in file
            sequences[current_id] = current_sequence
        
        return sequences
    

    def _fasta_to_one_hot(self, seq):
        idx_seq = np.array([self.alphabet_map[x.upper()] for x in seq])

        # Get rid of columns where there is an aa < 50% of the time (aka '.' somewhere)
        idx_seq = idx_seq.compress(self.position_mask)

        one_hot_seq = np.zeros((self.max_seq_len, self.alphabet_len))
        one_hot_seq[np.arange(self.max_seq_len), idx_seq] = 1
        return one_hot_seq.ravel()


    def next_batch(self, batch_size):
        minibatch = np.zeros((batch_size, self.max_seq_len*self.alphabet_len))

        fuckup_found = False

        for i in range(batch_size):  
            # Keep trying until the row is full (aka until you get a good seq)
            # Stop trying once you've run out of sequence ids
            while not minibatch[i].any() and self.seq_ids:
                try:
                    seq_id = self.seq_ids.pop()
                    x = self._fasta_to_one_hot(self.seqs[seq_id])
                    minibatch[i] = x
                except:
                    fuckup_found = True
            
        # Ensure that even though you had to ignore some rows, all of them are full
        if fuckup_found:
            for row in minibatch:
                assert(row.any())
            
        return minibatch

