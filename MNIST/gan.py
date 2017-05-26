import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import os

batch_size = 128
train_steps = 5000
Z_dim = 100

n_input = 784
n_hidden = 128

"""
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
"""

X = tf.placeholder(tf.float32, shape=[None, n_input], name="X")

with tf.variable_scope('disc_params'):
    D_W1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), \
                            shape=[n_input, 128], \
                            name="W1")
    D_b1 = tf.Variable(tf.zeros(shape=[128]), name="b1")

    D_W2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), \
                            shape=[128, 1], \
                            name="W2")

    D_b2 = tf.Variable(tf.zeros(shape=[1]), name="b2")

    theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, 100],name="Z")

with tf.variable_scope('gen_params'):
    G_W1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), \
                            shape=[100, 128], \
                            name="W1")
    G_b1 = tf.Variable(tf.zeros(shape=[128], name="b1"))

    G_W2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), \
                            shape=[128, n_input], \
                            name="W2")
    G_b2 = tf.Variable(tf.zeros(shape=[n_input], name="b2"))

    theta_G = [G_W1, G_W2, G_b1, G_b2]

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):
    with tf.variable_scope('layer_1'):
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)

    with tf.variable_scope('layer_2'):
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    with tf.variable_scope('layer_1'):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    with tf.variable_scope('layer_2'):
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

with tf.variable_scope('generator'):
    G_sample = generator(Z)

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
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss', G_loss)


# Only use var list to specifcy to only optimize certain values otherwise will optimize everything
with tf.variable_scope('D_solver'):
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)

with tf.variable_scope('G_solver'):
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

merged = tf.summary.merge_all()

dir_name = 'graph_logs/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

writer = tf.summary.FileWriter('./'+dir_name)


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()

writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')


i = 0
    
for it in xrange(train_steps):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    # Get new batch
    X_mb, _ = mnist.train.next_batch(batch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
    #writer.add_summary(summary_D, it)
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})
    #writer.add_summary(summary_G, it)
    
    if it % 1000 == 0:
        print('Iter: {}'.format(i))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        
        print()