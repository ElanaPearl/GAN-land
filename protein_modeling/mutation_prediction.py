import tensorflow as tf

import numpy as np
import argparse
import os

from data_handler import MultipleSequenceAlignment
from predict import MutationPrediction
from model import LSTM
import tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_name', help='The name of the gene for the protein family', default='PABP')
    parser.add_argument('--feature_to_predict', help='Experimental feature to compare mutation'\
                        'predictions to', default='')
    parser.add_argument('--run_time', help='Path to restore model, should be of the format '\
                            '\'year-month-date_hour-min-sec\'', default='')

    # Get parameters
    gene = parser.parse_args().gene_name
    run_time = parser.parse_args().run_time
    feature_to_predict = parser.parse_args().feature_to_predict

    # Restore MSA
    MSA = MultipleSequenceAlignment(gene, run_time)

    data = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='data'),
    target = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='target')

    # Restore model
    model = LSTM(data, target)

    # Set up predictor
    predictor = MutationPrediction(MSA, feature_to_predict)

    # Restore session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    log_path = os.path.join('model_logs', gene, run_time)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(log_path,'checkpoints')))

    # Predict
    mutation_preds, spear_corr = predictor.predict(sess, model, data, target)
    print "Correlation and pval: ", spear_corr
    np.savetxt(os.path.join(log_path,'mutation_preds.csv'), mutation_preds, delimiter=',')
