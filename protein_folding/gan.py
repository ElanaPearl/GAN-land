import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import os

from model import MultipleSequenceAlignment
import gumbel_softmax as GS

USE_GUMBEL = True
LOGGING = False

align_name = 'FYN_HUMAN_hmmerbit_plmc_n5_m30_f50_t0.2_r80-145_id100_b33.a2m'

print "Importing multiple sequence alignment..."
MSA = MultipleSequenceAlignment('./alignments/'+align_name)
print


print "Setting up graph..."


tau = tf.Variable(1.0, name="temperature")


batch_size = 150
train_steps = 1000
Z_dim = 100

n_input = MSA.n_input
print "Length of an protein seq: {}".format(n_input)

n_hidden = 183

X = tf.placeholder(tf.float32, shape=[None, n_input], name="X")


with tf.variable_scope('disc_params'):
    D_W1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), \
                            shape=[n_input, n_hidden], \
                            name="W1")
    D_b1 = tf.Variable(tf.zeros(shape=[n_hidden]), name="b1")

    D_W2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), \
                            shape=[n_hidden, 1], \
                            name="W2")

    D_b2 = tf.Variable(tf.zeros(shape=[1]), name="b2")

    theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, Z_dim],name="Z")

with tf.variable_scope('gen_params'):
    G_W1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), \
                            shape=[Z_dim, n_hidden], \
                            name="W1")
    G_b1 = tf.Variable(tf.zeros(shape=[n_hidden], name="b1"))

    G_W2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), \
                            shape=[n_hidden, n_input], \
                            name="W2")
    G_b2 = tf.Variable(tf.zeros(shape=[n_input], name="b2"))

    theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z, temp):
    with tf.variable_scope('layer_1'):
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)

    with tf.variable_scope('layer_2'):
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2

        with tf.variable_scope('gumbel_softmax'):
        #    if USE_GUMBEL:
            G_gumbel = GS.gumbel_softmax(G_log_prob, temp)
            return G_gumbel

        #G_prob = tf.nn.sigmoid(G_log_prob)
        #return G_prob


def discriminator(x):
    with tf.variable_scope('layer_1'):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    with tf.variable_scope('layer_2'):
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


with tf.variable_scope('generator'):
    G_sample = generator(Z, tau)

with tf.variable_scope('discriminator'):
    with tf.variable_scope('disc_real'):
        D_real, D_logit_real = discriminator(X)
    with tf.variable_scope('disc_generated'):
        D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------


with tf.variable_scope('discriminator_loss'):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
                                    logits=D_logit_real, \
                                    labels=tf.ones_like(D_logit_real, name="real_labels")))

    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
                                    logits=D_logit_fake, \
                                    labels=tf.zeros_like(D_logit_fake, name="gen_labels")))

    D_loss = D_loss_real + D_loss_fake

with tf.variable_scope('generator_loss'):
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,labels=tf.ones_like(D_logit_fake)))

D_loss_summary = tf.summary.scalar('D_loss', D_loss)
G_loss_summary = tf.summary.scalar('G_loss', G_loss)


# Only use var list to specifcy to only optimize certain values otherwise will optimize everything
with tf.variable_scope('D_solver'):
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)

with tf.variable_scope('G_solver'):
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

dir_name = 'graph_logs/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

writer = tf.summary.FileWriter('./'+dir_name)

sess = tf.Session()

writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')


ANNEAL_RATE = .00015
MIN_TEMP=0.3 # TODO: why? what's wrong with too small of a temp?

tau_0 = 1.0
temp = tau_0

print "Training..."

start_time = time.time()
for it in xrange(train_steps):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(3, Z_dim), tau: temp})

        #import pdb; pdb.set_trace()

        np.savetxt('out/{}.txt'.format(str(it/1000).zfill(3)), samples, fmt='%.4e', delimiter=',')
            
        temp = np.maximum(tau_0*np.exp(-ANNEAL_RATE*it), MIN_TEMP)
        

    # Get new batch
    X_mb = MSA.next_batch(batch_size)

    _, D_loss_curr, D_summary = sess.run([D_solver, D_loss, D_loss_summary], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
    writer.add_summary(D_summary, it)
    _, G_loss_curr, G_summary = sess.run([G_solver, G_loss, G_loss_summary], feed_dict={Z: sample_Z(batch_size, Z_dim), tau: temp})
    # when I don't pass in tau to this feed dict ^ , it still runs... why?

    writer.add_summary(G_summary, it)
    
    if it % 1000 == 0:
        print('Iter: {}'.format(it / 1000))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print('Time: {}'.format(time.time()-start_time))
        start_time = time.time()
        
        print()