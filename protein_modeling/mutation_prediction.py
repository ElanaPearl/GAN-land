import tensorflow as tf

import numpy as np
import argparse
import os

from data_handler import MultipleSequenceAlignment
from predict import MutationPrediction
from model import LSTM
import cPickle as pickle
import tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_name', help='The name of the gene for the protein family', default='PABP')
    parser.add_argument('--run_time', help='Path to restore model, should be of the format '\
                            '\'year-month-date_hour-min-sec\'', default='')

    locals().update(vars(parser.parse_args()))

    with open(os.path.join('model_logs', parser.parse_args().gene_name, parser.parse_args().run_time, 'args.pkl')) as f:
        args = pickle.load(f)

    locals().update(vars(args))

    # Restore MSA
    MSA = MultipleSequenceAlignment(gene_name, run_time)

    data = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='data'),
    target = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='target')
    dropout = tf.placeholder_with_default(0.0, shape=(), name='dropout')

    model = LSTM(data, target, dropout=dropout, attn_length=attn_length,
                 entropy_reg=entropy_reg, gradient_limit=gradient_limit,
                 init_learning_rate=init_learning_rate, decay_steps=decay_steps,
                 decay_rate=decay_rate, num_hidden=num_hidden, num_layers=num_layers,
                 lambda_l2_reg=lambda_l2_reg, mlp_h1_size=mlp_h1_size, mlp_h2_size=mlp_h2_size)

    # Restore model
    """
    model = LSTM(data, target, dropout=args.dropout, attn_length=args.attn_length,
                 entropy_reg=args.entropy_reg, gradient_limit=args.gradient_limit,
                 init_learning_rate=args.init_learning_rate, decay_steps=args.decay_steps,
                 decay_rate=args.decay_rate, num_hidden=args.num_hidden, num_layers=args.num_layers,
                 lambda_l2_reg=args.lambda_l2_reg, mlp_h1_size=args.mlp_h1_size, mlp_h2_size=args.mlp_h2_size)
    """

    # Set up predictor
    predictor = MutationPrediction(MSA)

    # Restore session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    log_path = os.path.join('model_logs', gene_name, run_time)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(log_path,'checkpoints')))

    # Predict
    mutation_preds, spear_corr = predictor.predict(sess, model, data, target)
    print "Correlation and pval: ", spear_corr
    np.savetxt(os.path.join(log_path,'mutation_preds.csv'), mutation_preds, delimiter=',')
