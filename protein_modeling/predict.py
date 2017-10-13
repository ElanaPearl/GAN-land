from scipy.stats import spearmanr
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import tools


class MutationPrediction:
    def __init__(self, MSA):
        self.MSA = MSA
        self.res_fn = tools.get_results_filename(self.MSA.gene_name)
        feature_to_predict = tools.feature_to_predict[self.MSA.gene_name]

        self.wildtype_d = np.reshape(self.MSA.str_to_one_hot(self.MSA.trimmed_ref_seq), (1, self.MSA.max_seq_len, tools.alphabet_len))
        self.wildtype_t = np.reshape(self.MSA.str_to_one_hot(self.MSA.trimmed_ref_seq[1:] + tools.END_TOKEN), (1, self.MSA.max_seq_len, tools.alphabet_len))

        experimental_res = pd.read_csv(self.res_fn, sep=';')
        experimental_res = experimental_res[experimental_res[['effect_prediction_epistatic',feature_to_predict]].notnull().all(axis=1)]
        self.measured = experimental_res[feature_to_predict]
        self.mutants = experimental_res['mutant']

        self.mutated_d, self.mutated_t = self.all_measured_mutations()

        self.single_mutated_d, self.single_mutated_t = self.all_single_mutations()


    @staticmethod
    def extract_mutations(mutation_string, offset=0):
        """
        Turns a string containing mutations of the format I100V into a list of tuples with
        format (100, 'I', 'V') (index, from, to)
        Parameters
        --------
        --
        mutation_string : str
            Comma-separated list of one or more mutations (e.g. "K50R,I100V")
        offset : int, default: 0
            Offset to be added to the index/position of each mutation
        Returns
        -------
        list of tuples
            List of tuples of the form (index+offset, from, to)
        """
        if mutation_string.lower() not in ["wild", "wt", ""]:
            mutations = mutation_string.split(",")
            return list(map(
                lambda x: (int(x[1:-1]) - offset, x[0], x[-1]),
                mutations
            ))
        else:
            return []


    def all_single_mutations(self):
        mutated_d = []
        mutated_t = []
        for i in range(len(self.MSA.trimmed_ref_seq)):
            for j in tools.alphabet:
                mutated_seq = list(self.MSA.trimmed_ref_seq)
                mutated_seq[i] = j
                mutated_d.append(self.MSA.str_to_one_hot(''.join(mutated_seq)))
                mutated_t.append(self.MSA.str_to_one_hot(''.join(mutated_seq[1:]) + tools.END_TOKEN))

        mutated_d = np.reshape(mutated_d,(len(mutated_d), self.MSA.max_seq_len,tools.alphabet_len))
        mutated_t = np.reshape(mutated_t,(len(mutated_t), self.MSA.max_seq_len,tools.alphabet_len))

        return mutated_d, mutated_t


    def all_measured_mutations(self):
        offset = int(self.res_fn.split('/')[-1].split('_r')[-1].split('-')[0])

        used_idx = []
        for i, aa in enumerate(self.MSA.full_ref_seq):
            if aa in tools.alphabet:
                used_idx.append(i)

        mutated_d = []
        mutated_t = []

        for mutation_strs in self.mutants:
            mutant_seq = np.array(list(self.MSA.full_ref_seq))
            for mutant_pos, mutant_from, mutant_to in self.extract_mutations(mutation_strs, offset=offset):
                #if self.MSA.full_ref_seq[mutant_pos].upper() != mutant_from:
                #    import pdb; pdb.set_trace()
                assert(self.MSA.full_ref_seq[mutant_pos] == mutant_from)
                mutant_seq[mutant_pos] = mutant_to
            mutated_d.append(self.MSA.str_to_one_hot(''.join(mutant_seq[used_idx])))
            mutated_t.append(self.MSA.str_to_one_hot(''.join(mutant_seq[used_idx[1:]]) + tools.END_TOKEN))

        mutated_d = np.reshape(mutated_d,(len(mutated_d), self.MSA.max_seq_len,tools.alphabet_len))
        mutated_t = np.reshape(mutated_t,(len(mutated_t), self.MSA.max_seq_len,tools.alphabet_len))

        return mutated_d, mutated_t


    def predict_mutations(self, sess, model, data, target, mutated_d, mutated_t):
        # TODO: maybe just re-implement this in tensorflow (?)
        wildtype_seed_prob = [self.MSA.seed_weights[i] for i in np.argmax(self.wildtype_d[:,0,:],1)]
        mutated_seed_prob = [self.MSA.seed_weights[i] for i in np.argmax(mutated_d[:,0,:],1)]

        wildtype_prob = -sess.run(model.cross_entropy, {data: self.wildtype_d, target: self.wildtype_t})
        mutated_prob = -sess.run(model.cross_entropy, {data: mutated_d, target: mutated_t})

        mutation_preds = (mutated_prob + mutated_seed_prob) - (wildtype_prob + wildtype_seed_prob)

        return mutation_preds


    def corr(self, sess, model, data, target):
        mutation_preds = self.predict_mutations(sess, model, data, target, self.mutated_d, self.mutated_t)
        assert(len(self.measured) == len(mutation_preds))

        return mutation_preds, spearmanr(self.measured, mutation_preds).correlation


    def plot_single_mutants(self, sess, model, data, target, fn):
        mutation_preds_list = self.predict_mutations(sess, model, data, target, self.single_mutated_d, self.single_mutated_t)
        mutation_preds = np.reshape(mutation_preds_list,(len(self.MSA.trimmed_ref_seq), tools.alphabet_len)).T

        # Plot it out
        fig, ax = plt.subplots()
        #heatmap = ax.pcolor(mutation_preds, cmap="Blues_r", vmin=-1.5, vmax=0.0)
        heatmap = ax.pcolor(mutation_preds, cmap="bwr", norm=tools.MidpointNormalize(midpoint=0))

        # Format
        fig = plt.gcf()
        fig.set_size_inches(18,4)

        # turn off the frame
        ax.set_frame_on(False)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(tools.alphabet_len)+0.5, minor=False)
        ax.set_xticks(np.arange(len(self.MSA.trimmed_ref_seq))+0.5, minor=False)

        # Set (0,0) as the top left
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(list(self.MSA.trimmed_ref_seq), minor=False) 
        ax.set_yticklabels(list(tools.alphabet), minor=False)

        # Rotate the xticks
        plt.xticks(rotation=90)

        # Turn off all the ticks
        for t in plt.gca().xaxis.get_major_ticks(): 
            t.tick1On = False 
            t.tick2On = False 
        for t in plt.gca().yaxis.get_major_ticks(): 
            t.tick1On = False 
            t.tick2On = False  
        
        wt_idx = np.array([tools.alphabet_map[x] for x in self.MSA.trimmed_ref_seq])

        fig.colorbar(heatmap)

        ax.scatter(np.arange(mutation_preds.shape[1])+0.5, wt_idx+0.5, color='m', s=3)

        plt.savefig(fn)
        plt.clf()

        return mutation_preds_list



