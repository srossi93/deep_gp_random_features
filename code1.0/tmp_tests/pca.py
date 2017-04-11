import numpy as np
import tensorflow as tf


N = 1000
D_in = 2
D_out = 10

# generate data
W_true = np.random.randn(D_in, D_out)
X_true = np.random.randn(N,D_in)
Y_true = np.dot(X_true, W_true)
Y_tf = tf.constant(Y_true.astype(np.float32))


W = tf.Variable(tf.ones([D_in, D_out]), name='weights', trainable=True)
X = tf.Variable(tf.ones([N, D_in]), name='latents', trainable=True) #placeholder(tf.float32, [None, D_in], name='placeholder_latent')

Y_est = tf.matmul(X, W)
loss = tf.reduce_sum((Y_tf-Y_est)**2)


#alatent = tf.Variable(tf.zeros([N, D_in]), name='latent', trainable=True)

train_step = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=tf.trainable_variables())
init_op = tf.global_variables_initializer()

#print(tf.trainable_variables())
#print(Y_tf)


with tf.Session() as sess:
    sess.run(init_op)

    for n in range(15000):
        sess.run(train_step)
        if (n+1) % 1000 == 0:
            print('iter %i, %f' % (n+1, sess.run(loss)))

    print(X.eval())
    print(W.eval())
