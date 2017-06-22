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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import DataSet
import utils
import likelihoods
import time
import dgp_interface


current_milli_time = lambda: int(round(time.time() * 1000))

class DgpRff(dgp_interface.DGPRFF_Interface):


    def __init__(self, likelihood_fun, num_examples, d_in, d_out, n_layers, n_rff, df, kernel_type, kernel_arccosine_degree, is_ard, feed_forward, q_Omega_fixed, theta_fixed, learn_Omega, LVM=False):
        """
        :param likelihood_fun: Likelihood function
        :param num_examples: total number of input samples
        :param d_in: Dimensionality of the input
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
        :param LVM: Model to perform Non-linear Principal Component Analysis with DGP
        """

        self.X = tf.placeholder(tf.float32, [None, d_in])

        self.LVM = False
        super(DgpRff, self).__init__(likelihood_fun, num_examples, d_in, d_out, n_layers, n_rff, df, kernel_type, kernel_arccosine_degree, is_ard, feed_forward, q_Omega_fixed, theta_fixed, learn_Omega, LVM)

                ## Builds whole computational graph with relevant quantities as part of the class
        self.loss, self.kl, self.ell, self.layer_out = self.get_nelbo()
        #
        #        ## Initialize the session
        self.session = tf.Session()


    ## Returns the expected log-likelihood term in the variational lower bound
    def get_ell(self):
        Din = self.d_in[0]
        MC = self.mc
        N_L = self.nl
        X = self.X
        Y = self.Y
        batch_size = tf.shape(X)[0] # This is the actual batch size when X is passed to the graph of computations


        ## The representation of the information is based on 3-dimensional tensors (one for each layer)
        ## Each slice [i,:,:] of these tensors is one Monte Carlo realization of the value of the hidden units
        ## At layer zero we simply replicate the input matrix X self.mc times
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

        return ell, layer_out


    ## Return predictions on some data
    def predict(self, data, mc_test):
        out = self.likelihood.predict(self.layer_out)

        nll = - tf.reduce_sum(-np.log(mc_test) + utils.logsumexp(self.likelihood.log_cond_prob(self.Y, self.layer_out), 0))
        #nll = - tf.reduce_sum(tf.reduce_mean(self.likelihood.log_cond_prob(self.Y, self.layer_out), 0))
        pred, neg_ll = self.session.run([out, nll], feed_dict={self.X:data.X, self.Y: data.Y, self.mc:mc_test})
        mean_pred = np.mean(pred, 0)
        return mean_pred, neg_ll

    ## Return the list of TF variables that should be "free" to be optimized
    def get_vars_fixing_some(self, all_variables):
        if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == True):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega") and not v.name.startswith("log_theta"))]

        if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == False):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega"))]

        if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == True):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("log_theta"))]

        if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == False):
            variational_parameters = all_variables

        return variational_parameters

    ## Function that learns the deep GP model with random Fourier feature approximation
    def learn(self, data, learning_rate, mc_train, batch_size, n_iterations, optimizer = None, display_step=100, test = None, mc_test=None, loss_function=None, duration = 1000000, less_prints=False):
        total_train_time = 0

        if optimizer is None:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)

        ## Set all_variables to contain the complete set of TF variables to optimize
        #self.X = tf.Variable(tf.zeros(self.d_in[0]), name='latentVariables', trainable=True)
        all_variables = tf.trainable_variables()
        #[print(v) for v in all_variables]

        ## Define the optimizer
        train_step = optimizer.minimize(self.loss, var_list=all_variables)

        ## Initialize all variables
        init = tf.global_variables_initializer()
        ##init = tf.initialize_all_variables()

        ## Fix any variables that are supposed to be fixed
        train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

        ## Initialize TF session
        self.session.run(init)

        ## Set the folder where the logs are going to be written
        # summary_writer = tf.train.SummaryWriter('logs/', self.session.graph)
        summary_writer = tf.summary.FileWriter('logs/', self.session.graph)

        if not(less_prints):
            #X = tf.Variable(tf.zeros([mc_train, Din]), trainable=True)
            nelbo, kl, ell, _ =  self.session.run(self.get_nelbo(), feed_dict={self.X: data.X, self.Y: data.Y, self.mc: mc_train})
            print("Initial kl=" + repr(kl) + "  nell=" + repr(-ell) + "  nelbo=" + repr(nelbo), end=" ")
            print("  log-sigma2 =", self.session.run(self.log_theta_sigma2))

        ## Present data to DGP n_iterations times
        ## TODO: modify the code so that the user passes the number of epochs (number of times the whole training set is presented to the DGP)
        for iteration in range(n_iterations):

            ## Stop after a given budget of minutes is reached
            if (total_train_time > 1000 * 60 * duration):
                break

            ## Present one batch of data to the DGP
            start_train_time = current_milli_time()
            batch = data.next_batch(batch_size)

            monte_carlo_sample_train = mc_train
            if (current_milli_time() - start_train_time) < (1000 * 60 * duration / 2.0):
                monte_carlo_sample_train = 1

            self.session.run(train_step, feed_dict={self.X: batch[0], self.Y: batch[1], self.mc: monte_carlo_sample_train})
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
                                                     feed_dict={self.X: data.X, self.Y: data.Y, self.mc: mc_train})
                    print("i=" + repr(iteration)  + "  kl=" + repr(kl) + "  nell=" + repr(-ell)  + "  nelbo=" + repr(nelbo), end=" ")

                    print(" log-sigma2=", self.session.run(self.log_theta_sigma2), end=" ")
                    # print(" log-lengthscale=", self.session.run(self.log_theta_lengthscale), end=" ")
                    # print(" Omega=", self.session.run(self.mean_Omega[0][0,:]), end=" ")
                    # print(" W=", self.session.run(self.mean_W[0][0,:]), end=" ")

                if loss_function is not None:
                    pred, nll_test = self.predict(test, mc_test)
                    elapsed_time = total_train_time + (current_milli_time() - start_predict_time)
                    print(loss_function.get_name() + "=" + "%.4f" % loss_function.eval(test.Y, pred), end = " ")
                    print(" nll_test=" + "%.5f" % (nll_test / len(test.Y)), end = " ")
                print(" time=" + repr(elapsed_time), end = " ")
                print("")
