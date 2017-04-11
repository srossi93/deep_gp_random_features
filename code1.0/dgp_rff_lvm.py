## Copyright 2016 Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import DataSet
import utils
import likelihoods
import time
from dgp_interface import DGPRFF_Interface

current_milli_time = lambda: int(round(time.time() * 1000))

class DgpRff_LVM(DGPRFF_Interface):


    def __init__(self, likelihood_fun, num_examples, d_in, d_out, n_layers, n_rff, df, kernel_type, kernel_arccosine_degree, is_ard, feed_forward, q_Omega_fixed, theta_fixed, learn_Omega, LVM=True):
        """
        :param likelihood_fun: Likelihood function
        :param num_examples: total number of input samples
        :param d_in: Dimensionality of the latent space
        :param d_out: Dimensionality of the output
        :param n_layers: Number of hidden layers
        :param n_rff: Number of random features for each layer
        :param df: Number of GPs for each layer
        :param kernel_type: Kernel type: currently only random Fourier features for RBF and arccosine kernels are implemented
        :param kernel_arccosine_degree: degree parameter of the arccosine kernel
        :param is_ard: Whether the kernel is ARD or isotropic
        :param feed_forward: Whether the original inputs should be fed forward as input to each layer
        :param Omega_fixed: Whether the Omega weights should be fixed throughout the optimization
        :param theta_fixed: Whether covariance parameters should be fixed throughout the optimization
        :param learn_Omega: How to treat Omega - fixed (from the prior), optimized, or learned variationally
        :param LVM: Flag to perform Non-linear Principal Component Analysis with DGPLVM
        """

        #self.X = tf.Variable(tf.zeros([1, d_in]), name='s', trainable=False)

        self.batchSize = 100

        #self.sing_latents = [tf.Variable(tf.zeros([self.batchSize, d_in], tf.float32), name='singular_latent', trainable=False) for i in range(num_examples/self.batchSize)]

        #self.latents = tf.Variable(tf.concat( [l for l in self.sing_latents], axis=0), name='latents')

        #self.batches_of_latent = tf.Variable(tf.zeros([self.batchSize, d_in], tf.float32), name='latents', trainable=True) #for i in range(num_examples/self.batchSize) ]

        self.latents = tf.Variable(tf.random_normal([num_examples, d_in]), name='latents', trainable=True)
        self.LVM = True
        self.s = tf.Variable(1., name='s_parameter', trainable=True)
        self.gamma = tf.Variable(1e0, name='gamma_parameter', trainable=True)
        super(DgpRff_LVM, self).__init__(likelihood_fun, num_examples, d_in, d_out, n_layers, n_rff, df, kernel_type, kernel_arccosine_degree, is_ard, feed_forward, q_Omega_fixed, theta_fixed, learn_Omega, LVM)

        #self.p, self.q, self.kl = self.compute_affinity_obs()

        self.loss, self.kl, self.ell, self.layer_out = self.get_nelbo()
        self.session = tf.Session()


    ## Returns the expected log-likelihood term in the variational lower bound
    def get_ell(self):

        Din = self.d_in[0]
        MC = self.mc
        N_L = self.nl
        X = self.latents
        Y = self.Y
        batch_size = tf.shape(X)[0] # This is the actual batch size when X is passed to the graph of computations


        ## The representation of the information is based on 3-dimensional tensors (one for each layer)
        ## Each slice [i,:,:] of these tensors is one Monte Carlo realization of the value of the hidden units
        ## At layer zero we simply replicate the input matrix X self.mc times
        #print(X)
        self.layer = []
        self.layer.append(tf.multiply(tf.ones([self.mc, batch_size, Din]), X))

        ## Forward propagate information from the input to the output through hidden layers
        Omega_from_q  = self.sample_from_Omega()
        W_from_q = self.sample_from_W()
        # TODO: basis features should be in a different class
        for i in range(N_L):
            layer_times_Omega = tf.matmul(self.layer[i], Omega_from_q[i])  # X * Omega

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF":
                Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(axis=2, values=[tf.cos(layer_times_Omega), tf.sin(layer_times_Omega)])
            if self.kernel_type == "arccosine":
                if self.arccosine_degree == 0:
                    Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(axis=2, values=[tf.sign(tf.maximum(layer_times_Omega, 0.0))])
                if self.arccosine_degree == 1:
                    Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(axis=2, values=[tf.maximum(layer_times_Omega, 0.0)])
                if self.arccosine_degree == 2:
                    Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(axis=2, values=[tf.square(tf.maximum(layer_times_Omega, 0.0))])

            F = tf.matmul(Phi, W_from_q[i])
            if self.feed_forward and not (i == (N_L-1)): ## In the feed-forward case, no concatenation in the last layer so that F has the same dimensions of Y
                F = tf.concat(axis=2, values=[F, self.layer[0]])

            self.layer.append(F)

        ## Output layer
        layer_out = self.layer[N_L]

        ## Given the output layer, we compute the conditional likelihood across all samples
        ll = self.likelihood.log_cond_prob(Y, layer_out)

        ## Mini-batch estimation of the expected log-likelihood term
        ell = tf.reduce_sum(tf.reduce_mean(ll, 0)) * self.num_examples / tf.cast(batch_size, "float32")
        #print(layer_out)
        return ell, layer_out

    #def compute_exps(self, data, n, m):
    #    return tf.exp(-tf.norm(data[n] - data[m])/(2 * tf.pow(self.s, 2)))

    def compute_distance(self, data):
        expanded_a = tf.stack([data for i in range(self.num_examples)])#tf.expand_dims(data,1)
        #expanded_b = tf.transpose(expended_a)#tf.expand_dims(data,0)

        #print(expanded_b.get_shape())

        #distances = tf.norm(expanded_a - tf.transpose(expanded_a, perm=[1, 0, 2]), axis=2)
        distances = tf.reduce_sum(tf.pow(expanded_a - tf.transpose(expanded_a, perm=[1, 0, 2]), 2), 2)

        return distances

    def compute_affinity_obs(self):

        #affinity_obs = tf.Variable(tf.zeros([self.num_examples, self.num_examples], tf.float32, ''))
        p = tf.exp(tf.div(self.compute_distance(self.Y), tf.multiply(2., tf.pow(self.s, 2))))
        q = tf.pow((1 + self.compute_distance(self.latents)), 2)
        aff = tf.multiply(p, tf.log(tf.div(p, q)))

        #print(kl.get_shape()
        return aff #self.compute_distance(self.latents)



    ## Maximize variational lower bound --> minimize Nelbo
    def get_nelbo(self):
        kl = self.get_kl()
        ell, layer_out = self.get_ell()
        affinity = tf.reduce_sum(self.compute_affinity_obs())

        #print(affinity, ell)

        nelbo  = kl - ell# + affinity

        return nelbo, kl, ell, layer_out

    ## Function that learns the deep GP model with random Fourier feature approximation
    def learn(self, data, learning_rate, mc_train, batch_size, n_iterations, optimizer = None, display_step=100, test = None, mc_test=None, loss_function=None, duration = 1000000, less_prints=False):
        total_train_time = 0


        #Z = tf.Variable(tf.ones([len(data.Y), self.d_in[0]], tf.float32), name='latent_variable', trainable=True)

        #change_shape = tf.assign(self.X, latent_variables, validate_shape=False)
        #self.session.run(change_shape)

        ## Initialize all variables

        batch_size = self.num_examples

        if optimizer is None:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)

        ## Set all_variables to contain the complete set of TF variables to optimize
        #self.X = tf.Variable(tf.zeros(self.d_in[0]), name='latentVariables', trainable=True)
        all_variables = tf.trainable_variables()
        #[print(v) for v in all_variables]
        ## Define the optimizer
        train_step = optimizer.minimize(self.loss, var_list=all_variables)

        init = tf.global_variables_initializer()
        ##init = tf.initialize_all_variables()

        ## Fix any variables that are supposed to be fixed
        train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

        ## Initialize TF session
        self.session.run(init)

        #var_grad = tf.gradients(self.loss, [self.kl])[0]
        #print('grad: ', self.session.run(var_grad, ))
        ## Set the folder where the logs are going to be written
        # summary_writer = tf.train.SummaryWriter('logs/', self.session.graph)
        summary_writer = tf.summary.FileWriter('logs/', self.session.graph)

        if not(less_prints):
            #X = tf.Variable(tf.zeros([mc_train, Din]), trainable=True)
            nelbo, kl, ell, _ =  self.session.run(self.get_nelbo(), feed_dict={self.Y: data.X, self.mc: mc_train})
            print("Initial kl=" + repr(kl) + "  nell=" + repr(-ell) + "  nelbo=" + repr(nelbo), end=" ")
            print("  log-sigma2 =", self.session.run(self.log_theta_sigma2))

        ## Present data to DGP n_iterations times
        ## TODO: modify the code so that the user passes the number of epochs (number of times the whole training set is presented to the DGP)
        for iteration in range(n_iterations):

            #pippo = self.compute_affinity_obs()
            #print(self.session.run(pippo, feed_dict={self.Y:data.X}))

            #print(self.session.run(self.p, feed_dict={self.Y:data.X}), end='\n\n')
            #print(self.session.run(self.q), end='\n\n')
            #print(self.session.run(self.kl))
            #print(self.session.run(tf.reduce_sum(self.q)))

            ## Stop after a given budget of minutes is reached
            if (total_train_time > 1000 * 60 * duration):
                break

            ## Present one batch of data to the DGP
            start_train_time = current_milli_time()
            batch = data.next_batch(batch_size)

            monte_carlo_sample_train = mc_train
            if (current_milli_time() - start_train_time) < (1000 * 60 * duration / 2.0):
                monte_carlo_sample_train = 1

            self.session.run(train_step, feed_dict={self.Y: batch[0], self.mc: mc_train})
            total_train_time += current_milli_time() - start_train_time

            ## After reaching enough iterations with Omega fixed, unfix it
            if self.q_Omega_fixed_flag == True:
                if iteration >= self.q_Omega_fixed:
                    self.q_Omega_fixed_flag = False
                    train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

            if self.theta_fixed_flag == True:
                if iteration >= self.theta_fixed:
                    self.theta_fixed_flag = False
                    train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

            ## Display logs every "FLAGS.display_step" iterations
            if iteration % display_step == 0:
                start_predict_time = current_milli_time()

                if less_prints:
                    print("i=" + repr(iteration), end = " ")

                else:
                    nelbo, kl, ell, _ = self.session.run(self.get_nelbo(),
                                                     feed_dict={self.Y: data.X, self.mc: mc_train})
                    print("i=" + repr(iteration)  + "  kl=" + repr(kl) + "  nell=" + repr(-ell)  + "  nelbo=" + repr(nelbo), end=" ")

                    print(" log-sigma2=", self.session.run(self.log_theta_sigma2), end=" ")
                    # print(" log-lengthscale=", self.session.run(self.log_theta_lengthscale), end=" ")
                    # print(" Omega=", self.session.run(self.mean_Omega[0][0,:]), end=" ")
                    # print(" W=", self.session.run(self.mean_W[0][0,:]), end=" ")

                if loss_function is not None:

                    pred, nll_train = self.predict(data, mc_test)
                    elapsed_time = total_train_time + (current_milli_time() - start_predict_time)
                    print(loss_function.get_name() + "=" + "%.4f" % loss_function.eval(data.X, pred), end = " ")
                    print(" nll_test=" + "%.5f" % (nll_train / len(data.X)), end = " ")

                print(" time=" + repr(elapsed_time), end = " ")
                print("")
        #print(Z)
        nelbo, kl, ell, _ = self.session.run(self.get_nelbo(),
                                         feed_dict={self.Y: data.X, self.mc: mc_train})
        print("")
        print("kl = " + repr(kl))
        print("nell = " + repr(-ell))
        print("nelbo = " + repr(nelbo))
        print("log-sigma2 = ", self.session.run(self.log_theta_sigma2))
        print("log-lengthscale = ", self.session.run(self.log_theta_lengthscale))
        print("Omega = ", self.session.run(self.mean_Omega[0][0,:]))
        print("W = ", self.session.run(self.mean_W[0][0,:]))
        print("")

        model_path = './models/model.ckpt'
        saver = tf.train.Saver()
        save_path = saver.save(self.session, model_path)
        print("Model saved in file: %s" % save_path)

        return

        ## Return predictions on some data
    def predict(self, data, mc_test):
        out = self.likelihood.predict(self.layer_out)

        nll = - tf.reduce_sum(-np.log(mc_test) + utils.logsumexp(self.likelihood.log_cond_prob(self.Y, self.layer_out), 0))
        #nll = - tf.reduce_sum(tf.reduce_mean(self.likelihood.log_cond_prob(self.Y, self.layer_out), 0))
        pred, neg_ll = self.session.run([out, nll], feed_dict={self.Y : data.X, self.mc:mc_test})
        mean_pred = pred#np.mean(pred, 0)
        return mean_pred, neg_ll

    ## Return the list of TF variables that should be "free" to be optimized
    # TODO: move this method to the interface
    def get_vars_fixing_some(self, all_variables):
        if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == True):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega") and not v.name.startswith("log_theta") or v.name.startswith('latent_variables'))]
        if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == False):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega") or v.name.startswith('latent_variables'))]
        if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == True):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("log_theta") or v.name.startswith('latent_variables'))]
        if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == False):
            variational_parameters = all_variables

        #[print(v) for v in tf.trainable_variables()]
        #var = [v for v in tf.global_variables() if v.name == "latents_1:0"][0]
        #print(var)
        #print(tf.get_default_graph().get_tensor_by_name('latents_1:0'))
        return variational_parameters


    def sample_latent_space(self, data):
        s = range(10)
        for i in range(len(data.Y)/2):
            #label = tf.gather(self.Y, i)
            latent_sample = tf.gather(self.latents, i)
            print(self.session.run(self.latents), '-->', data.Y[i])#, feed_dict={self.Y:data.X}), '--->', self.session.run(latent))
        return