import tensorflow as tf
from model_w_label import MultipleSequenceAlignment, TEST_ALIGN_ID
from LSTM import VariableSequenceLabelling
import os
import random
import numpy as np

log_path = './model_logs/LSTM_2017-07-05_15-01-12'

print "Getting MSA"
MSA = MultipleSequenceAlignment(TEST_ALIGN_ID, test_ids_path=os.path.join(log_path,'test_ids.pkl'))


data = tf.placeholder(tf.float32, [None, MSA.max_seq_len, MSA.alphabet_len], name='data')
target = tf.placeholder(tf.float32, [None, MSA.max_seq_len, MSA.alphabet_len], name='target')

print "Setting up graph"
model = VariableSequenceLabelling(data, target)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

print "Restoring model"
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(log_path,'checkpoints')))

wildtype_d, wildtype_t = MSA.next_batch(5, test=True)


print "Making mutations"
mutated_d = np.copy(wildtype_d)
mutated_t = np.copy(wildtype_t)

mutations = np.zeros((wildtype_d.shape[0],2))

# Add random mutations to each mutation
for seq_idx in range(wildtype_d.shape[0]):
    # Choose random idx to randomly mutate (anything but first/last position)
    idx_to_change = random.choice(range(wildtype_d.shape[1]-1))

    old_val = np.argmax(wildtype_d[seq_idx,idx_to_change,:])
    new_val = random.choice(range(wildtype_d.shape[2]))

    mutated_d[seq_idx,idx_to_change,old_val] = 0
    mutated_d[seq_idx,idx_to_change,new_val] = 1

    # The idx to change is subtracted by 1 because the target is offset by 1
    mutated_t[seq_idx,idx_to_change-1,old_val] = 0
    mutated_t[seq_idx,idx_to_change-1,new_val] = 1

    mutations[seq_idx] = [old_val, new_val]

#mutated_first_aa = np.argmax()

print "Getting mutation predictions"
mutated_prob = sess.run(model.cross_entropy, {data: mutated_d, target: mutated_t})
wildtype_prob = sess.run(model.cross_entropy, {data: wildtype_d, target: wildtype_t})

#mutation_prob = (mutated_prob + MSA.seed_weights[mutated_d[:,0,:]])

import pdb; pdb.set_trace()
