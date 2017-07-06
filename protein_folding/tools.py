import os
import cPickle as pickle

GAP = '-'
alphabet = 'ACDEFGHIKLMNPQRSTVWY*'
alphabet_len = len(alphabet)

alphabet_map = {s: i for i, s in enumerate(alphabet)}
alphabet_map[GAP] = alphabet_len - 1
rev_alphabet_map = {i: s for i, s in enumerate(alphabet)}


def lazy_calculate(function, path):
    if os.path.exists(path):
        with open(path) as f:
            return pickle.load(f)
    else:
        data = function()
        with open(path, 'w') as f:
            pickle.dump(data, f)
        return data
    return wrapper


def get_alignment_filename(gene):
    for filename in os.listdir('alignments'):
        if filename.startswith(gene):
            return filename
