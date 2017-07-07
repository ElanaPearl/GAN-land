import tensorflow as tf
import numpy as np
import argparse
import random
import os

from model_w_label import MultipleSequenceAlignment
from LSTM import LSTM
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--gene_name', help='The name of the gene for the protein family', default='PABP')
parser.add_argument('--restore_path', help='Path to restore model, should be of the format '\
                        '\'year-month-date_hour-min-sec\'', default='')

gene = parser.parse_args().gene_name
run_time = parser.parse_args().restore_path

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



wildtype_d = np.reshape(MSA.str_to_one_hot(MSA.ref_seq), (1, MSA.max_seq_len, tools.alphabet_len))
wildtype_t = np.reshape(MSA.str_to_one_hot(MSA.ref_seq[1:] + tools.END_TOKEN), (1, MSA.max_seq_len, tools.alphabet_len))


# Create all possible mutations of the reference sequence
mutated_d = []
mutated_t = []
for i in range(len(MSA.ref_seq)):
    for j in tools.alphabet:
        mutated_seq = list(MSA.ref_seq)
        mutated_seq[i] = j
        mutated_d.append(MSA.str_to_one_hot(''.join(mutated_seq)))
        mutated_t.append(MSA.str_to_one_hot(''.join(mutated_seq[1:]) + tools.END_TOKEN))

mutated_d = np.reshape(mutated_d, (MSA.max_seq_len*tools.alphabet_len, MSA.max_seq_len, tools.alphabet_len))

print "Calculating Mutation Probabilities"
wildtype_seed_prob = [MSA.seed_weights[i] for i in np.argmax(wildtype_d[:,0,:],1)]
mutated_seed_prob = [MSA.seed_weights[i] for i in np.argmax(mutated_d[:,0,:],1)]

wildtype_prob = sess.run(model.cross_entropy, {data: wildtype_d, target: wildtype_t})
mutated_prob = sess.run(model.cross_entropy, {data: mutated_d, target: mutated_t})

mutation_preds = (mutated_prob + mutated_seed_prob) - (wildtype_prob + wildtype_seed_prob)

mutation_preds = np.reshape(mutation_preds,(MSA.max_seq_len, tools.alphabet_len))

print "Saving!"
np.savetxt(os.path.join(log_path,'mutation_preds.csv'), mutation_preds, delimiter=',')

