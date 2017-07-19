import os
import cPickle as pickle

# USEFUL VARIABLES AND THINGS
GAP = '-'
END_TOKEN = '*'
alphabet = 'ACDEFGHIKLMNPQRSTVWY*'
alphabet_len = len(alphabet)

alphabet_map = {s: i for i, s in enumerate(alphabet)}
alphabet_map[GAP] = alphabet_len - 1
rev_alphabet_map = {i: s for i, s in enumerate(alphabet)}

feature_to_predict = {
    'BLAT': '2500',
    'BRCA1': 'e3',
    'FYN': 'Tm',
    'GAL4': 'SEL_C_40h',
    'PABP': 'linear'
}


# HYPER PARAMETERS
similarity_cutoff = 0.2
n_clusters = 50


# HELPER FUNCTIONS
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

def get_results_filename(gene):
    for filename in os.listdir('experimental_results'):
        if filename.startswith(gene):
            return os.path.join('experimental_results',filename)


def remove_gaps(seqs):
    return {k: v.replace(GAP, '') for k,v in seqs.iteritems()}

