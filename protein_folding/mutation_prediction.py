import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import random
import os

from model_w_label import MultipleSequenceAlignment
from LSTM import LSTM
import tools


parser = argparse.ArgumentParser()
parser.add_argument('--gene_name', help='The name of the gene for the protein family', default='PABP')
parser.add_argument('--run_time', help='Path to restore model, should be of the format '\
                        '\'year-month-date_hour-min-sec\'', default='')

gene = parser.parse_args().gene_name
run_time = parser.parse_args().run_time

log_path = os.path.join('model_logs', gene, run_time)

print "Getting MSA"
MSA = MultipleSequenceAlignment(gene, run_time, seq_limit=1000)


data = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='data')
target = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='target')

print "Setting up graph"
model = LSTM(data, target, MSA.seed_weights)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

print "Restoring model"
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(log_path,'checkpoints')))



wildtype_d = np.reshape(MSA.str_to_one_hot(MSA.trimmed_ref_seq), (1, MSA.max_seq_len, tools.alphabet_len))
wildtype_t = np.reshape(MSA.str_to_one_hot(MSA.trimmed_ref_seq[1:] + tools.END_TOKEN), (1, MSA.max_seq_len, tools.alphabet_len))

def extract_mutations(mutation_string, offset=0):
    """
    Turns a string containing mutations of the format I100V into a list of tuples with
    format (100, 'I', 'V') (index, from, to)
    Parameters
    ----------
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


def all_single_mutations():
    mutated_d = []
    mutated_t = []
    for i in range(len(MSA.trimmed_ref_seq)):
        for j in tools.alphabet:
            mutated_seq = list(MSA.trimmed_ref_seq)
            mutated_seq[i] = j
            mutated_d.append(MSA.str_to_one_hot(''.join(mutated_seq)))
            mutated_t.append(MSA.str_to_one_hot(''.join(mutated_seq[1:]) + tools.END_TOKEN))
    return mutated_d, mutated_t

def all_measured_mutations():
    fn = tools.get_results_filename(gene)
    offset = fn.split('/')[-1].split('_r')[-1].split('-')[0]
    res = pd.read_csv(fn, sep=';', skiprows=1, index_col='mutant')

    used_idx = []
    for i, aa in enumerate(MSA.full_ref_seq):
        if aa in tools.alphabet:
            used_idx.append(i)

    mutated_d = []
    mutated_t = []

    for mutation_strs in pd.read_csv(fn, sep=';', skiprows=1)['mutant']:
        mutant_seq = np.array(list(MSA.full_ref_seq))
        for mutant_pos, mutant_from, mutant_to in extract_mutations(mutation_strs, offset=offset):
            assert(MSA.full_ref_seq[mutant_pos] == mutant_from)
            mutant_seq[mutant_pos] = mutant_to
        mutated_d.append(MSA.str_to_one_hot(''.join(mutant_seq[used_idx])))
        mutated_t.append(MSA.str_to_one_hot(''.join(mutant_seq[used_idx[1:]]) + tools.END_TOKEN))

    return mutated_d, mutated_t

mutated_d, mutated_t = all_measured_mutations()
mutated_d = np.reshape(mutated_d,(len(mutated_d), MSA.max_seq_len,tools.alphabet_len))
mutated_t = np.reshape(mutated_t,(len(mutated_t), MSA.max_seq_len,tools.alphabet_len))

print "Calculating Mutation Probabilities"
wildtype_seed_prob = [MSA.seed_weights[i] for i in np.argmax(wildtype_d[:,0,:],1)]
mutated_seed_prob = [MSA.seed_weights[i] for i in np.argmax(mutated_d[:,0,:],1)]

wildtype_prob = sess.run(model.cross_entropy, {data: wildtype_d, target: wildtype_t})
mutated_prob = sess.run(model.cross_entropy, {data: mutated_d, target: mutated_t})

mutation_preds = (mutated_prob + mutated_seed_prob) - (wildtype_prob + wildtype_seed_prob)


#mutation_preds = np.reshape(mutation_preds,(MSA.max_seq_len, tools.alphabet_len))
#mutation_preds = pd.DataFrame(mutation_preds, columns=list(tools.alphabet), index=used_idx)



print "Saving!"
np.savetxt(os.path.join(log_path,'mutation_preds.csv'), mutation_preds, delimiter=',')
#mutation_preds.to_csv(os.path.join(log_path,'mutation_preds.csv'), sep=',')

