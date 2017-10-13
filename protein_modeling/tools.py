import os
import numpy as np
import cPickle as pickle
import matplotlib.colors as colors

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


# Class to set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

