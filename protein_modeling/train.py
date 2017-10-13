from datetime import datetime
import cPickle as pickle
import argparse
import logging
import os

import tensorflow as tf
import numpy as np
from data_handler import MultipleSequenceAlignment
from predict import MutationPrediction
import tools
from tensorflow.python import debug as tf_debug
from model import LSTM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('-gene_name', help='The name of the gene for the protein family', default='FYN')
    parser.add_argument('-use_full_seqs', help='Use full sequences from online?', action='store_true')
    parser.add_argument('-seq_limit', help='If debugging and you want to only use a limited number '\
                        'of sequences for the sake of time, set this.', type=int, default=0)

    # Training Parameters
    parser.add_argument('-batch_size', help='Number of sequences per batch', type=int, default=128)
    parser.add_argument('-num_epochs', help='Number of epochs of training', type=int, default=100)
    parser.add_argument('-restore_path', help='Path to restore model, should be of the format '\
                        '\'year-month-date_hour-min-sec\'', default='')
    parser.add_argument('-debug', help='Use tf_debug', action='store_true')

    # RNN Architecture Parameters
    parser.add_argument('-num_hidden', help='Number of hidden nodes per layer', type=int, default=150)
    parser.add_argument('-num_layers', help='Number of hidden layers. If = 1, this is a normal LSTM ' \
                        'othersiwe this is the number of stacked LSTM layers', type=int, default=1)
    parser.add_argument('-mlp_h1_size', type=int, default=150)
    parser.add_argument('-mlp_h2_size', type=int, default=75)

    # Regularization Parameters
    parser.add_argument('-dropout_prob', help='Value for dropout when training. If left unset, there will be'\
                        'no dropout aka dropout_prob = 0', type=float, default=0.0)
    parser.add_argument('-attn_length', help='Length of attention, if 0, no attention is used', type=int, default=0)
    parser.add_argument('-gradient_limit', help='Max value for a gradient. Values above this get trimmed down'\
                        ' to this value', type=float, default=5.0)
    parser.add_argument('-entropy_reg', help='How much entropy regularization to use, if 0 none', type=float, default=0.0)
    parser.add_argument('-lambda_l2_reg', help="How much l2 regularization to use on the weights, if 0 none", type=float, default=0.0)
    parser.add_argument('-init_learning_rate', help='Initial learning rate', type=float, default=0.01)
    parser.add_argument('-decay_steps', help='Initial learning rate', type=int, default=100)
    parser.add_argument('-decay_rate', help='Initial learning rate', type=float, default=0.9)

    # NOTES TO SELF:
    parser.add_argument('-note', help='Just 4 u', default='')

    # Create dictionary of the parsed args and convert each arg into a local variable
    locals().update(vars(parser.parse_args()))

    # Set run time (or restore run_time from last run)
    run_time = restore_path if restore_path else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set up logging directories
    log_path = os.path.join('model_logs', gene_name, run_time)
    graph_log_path = os.path.join(log_path, 'graphs')
    checkpoint_log_path = os.path.join(log_path, 'checkpoints')
    mutant_pred_path = os.path.join(log_path, 'mutant_preds')

    if not restore_path:
        os.makedirs(log_path)
        os.makedirs(graph_log_path)
        os.makedirs(checkpoint_log_path)
        os.makedirs(mutant_pred_path)

    # Set up logging file
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_path, 'logfile.txt'),
                    format='%(asctime)-15s %(message)s')

    # Log the parameter values
    for param_name, param_value in vars(parser.parse_args()).iteritems():
        logging.info("{}: {}".format(param_name, param_value))

    print "Getting multiple sequence alignment"
    MSA = MultipleSequenceAlignment(gene_name, run_time=run_time, use_full_seqs=use_full_seqs)

    data = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='data')
    target = tf.placeholder(tf.float32, [None, MSA.max_seq_len, tools.alphabet_len], name='target')
    dropout = tf.placeholder_with_default(0.0, shape=(), name='dropout')
    corr_tensor = tf.placeholder(tf.float32, name='spear_corr')
    corr_summ_tensor = tf.summary.scalar('spear_corr', corr_tensor)

    #predictor = MutationPrediction(MSA, tools.feature_to_predict[gene_name])

    print "Constructing model"
    model = LSTM(data, target, dropout=dropout, attn_length=attn_length,
                 entropy_reg=entropy_reg, gradient_limit=gradient_limit,
                 init_learning_rate=init_learning_rate, decay_steps=decay_steps,
                 decay_rate=decay_rate, num_hidden=num_hidden, num_layers=num_layers,
                 lambda_l2_reg=lambda_l2_reg, mlp_h1_size=mlp_h1_size, mlp_h2_size=mlp_h2_size)

    writer = tf.summary.FileWriter(graph_log_path)

    sess = tf.Session()
    if debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)


    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    if restore_path:
        print "Restoring checkpoint"
        last_ckpt = tf.train.latest_checkpoint(checkpoint_log_path)
        saver = tf.train.Saver()
        saver.restore(sess, last_ckpt)

        pretrained_epochs, pretrained_batches = int(last_ckpt.split('_')[-2]), int(last_ckpt.split('_')[-1])
        print "Picking up with epoch {} and batch {}".format(pretrained_epochs, pretrained_batches)
    else:
        saver = tf.train.Saver()
        pretrained_epochs = 0
        pretrained_batches = 0


    num_batches_per_epoch = int(float(MSA.train_size) / batch_size)

    print "Starting training"


    for epoch in range(pretrained_epochs, num_epochs):
        logging.info("Epoch: {}".format(epoch))

        

        for i in range(pretrained_batches, num_batches_per_epoch):
            if i % batch_size == 0:
                saver.save(sess, os.path.join(checkpoint_log_path,'model_{}_{}'.format(epoch,i)))

                # GET TEST ERROR
                test_data, test_target = MSA.next_batch(batch_size, test=True)
                test_err_summary, test_err, test_pred = sess.run([model.test_error, model.error, model.prediction], \
                                                        {data: test_data, target: test_target})

                logging.info("Batch: {}, Error: {}".format(i, test_err))

                writer.add_summary(test_err_summary, epoch*num_batches_per_epoch + i)

                logging.info("target: {}".format(MSA.one_hot_to_str(test_target[0])))
                logging.info("pred:   {}".format(MSA.one_hot_to_str(test_pred[0])))

                # GENERATE RANDOM SAMPLE
                seq_sample = model.generate_seq(sess, MSA.seed_weights)
                logging.info("gen:    {}".format(seq_sample))

                # COMPARE PREDICTIONS TO EXPERIMENTAL RESULTS
                """
                mutant_preds, spear_corr = predictor.corr(sess, model, data, target)
                corr_summary = sess.run(corr_summ_tensor, {corr_tensor: spear_corr})
                writer.add_summary(corr_summary, epoch*num_batches_per_epoch + i)

                with open(os.path.join(mutant_pred_path, '{}.pkl'.format(epoch*num_batches_per_epoch + i)), 'w') as f:
                    pickle.dump(mutant_preds, f)


                # MAKE PLOT OF ALL SINGLE MUTATIONS
                single_mutant_preds = predictor.plot_single_mutants(sess, model, data, target,
                                              os.path.join(mutant_pred_path, '{}.png'.format(epoch*num_batches_per_epoch + i)))
                """

                #sample_summary = sess.run(model.sample_seq_summary, {seq_placeholder: seq_sample})
                #writer.add_summary(sample_summary, epoch*num_batches_per_epoch + i)

            batch_data, batch_target = MSA.next_batch(batch_size)

            _, train_summaries, train_err_summary, train_err = sess.run([model.optimize, model.train_summaries, model.train_error, model.error], \
                                                               {data: batch_data, target: batch_target, dropout: dropout_prob})

            for summary in train_summaries:
                writer.add_summary(summary, epoch*num_batches_per_epoch + i)

            writer.add_summary(train_err_summary, epoch*num_batches_per_epoch + i)


        test_data, test_target = MSA.next_batch(batch_size, test=True)
        logging.info("Error: {}".format(sess.run(model.error, {data: test_data, target: test_target})))


        # The number of batches of a given epoch that we already trained is only relevant
        # to the first epoch we train after we restore the model
        pretrained_batches = 0
