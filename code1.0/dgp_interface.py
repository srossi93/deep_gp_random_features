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
import abc

current_milli_time = lambda: int(round(time.time() * 1000))

class DGPRFF_Interface(object):
    __metaclass__ = abc.ABCMeta

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
        self.likelihood = likelihood_fun
        self.kernel_type = kernel_type
        self.is_ard = is_ard
        self.feed_forward = feed_forward
        self.q_Omega_fixed = q_Omega_fixed
        self.theta_fixed = theta_fixed
        self.q_Omega_fixed_flag = q_Omega_fixed > 0
        self.theta_fixed_flag = theta_fixed > 0
        self.learn_Omega = learn_Omega
        self.arccosine_degree = kernel_arccosine_degree
        self.LVM = LVM

        ## These are all scalars
        self.num_examples = num_examples
        self.nl = n_layers ## Number of hidden layers
        self.n_Omega = n_layers  ## Number of weigh matrices is "Number of hidden layers"
        self.n_W = n_layers

        ## These are arrays to allow flexibility in the future
        self.n_rff = n_rff * np.ones(n_layers, dtype = np.int64)
        self.df = df * np.ones(n_layers, dtype=np.int64)

        ## Dimensionality of Omega matrices
        if self.feed_forward:
            self.d_in = np.concatenate([[d_in], self.df[:(n_layers - 1)] + d_in])
        else:
            self.d_in = np.concatenate([[d_in], self.df[:(n_layers - 1)]])
        self.d_out = self.n_rff

        ## Dimensionality of W matrices
        if self.kernel_type == "RBF":
            self.dhat_in = self.n_rff * 2
            self.dhat_out = np.concatenate([self.df[:-1], [d_out]])

        if self.kernel_type == "arccosine":
            self.dhat_in = self.n_rff
            self.dhat_out = np.concatenate([self.df[:-1], [d_out]])

        ## When Omega is learned variationally, define the right KL function and the way Omega are constructed
        if self.learn_Omega == "var":
            self.get_kl = self.get_kl_Omega_to_learn
            self.sample_from_Omega = self.sample_from_Omega_to_learn

        ## When Omega is optimized, fix some standard normals throughout the execution that will be used to construct Omega
        if self.learn_Omega == "optim":
            self.get_kl = self.get_kl_Omega_to_learn
            self.sample_from_Omega = self.sample_from_Omega_optim

            self.z_for_Omega_fixed = []
            for i in range(self.n_Omega):
                tmp = utils.get_normal_samples(1, self.d_in[i], self.d_out[i])
                self.z_for_Omega_fixed.append(tf.Variable(tmp[0,:,:], trainable = False))

        ## When Omega is fixed, fix some standard normals throughout the execution that will be used to construct Omega
        if self.learn_Omega == "no":
            self.get_kl = self.get_kl_Omega_fixed
            self.sample_from_Omega = self.sample_from_Omega_fixed

            self.z_for_Omega_fixed = []
            for i in range(self.n_Omega):
                tmp = utils.get_normal_samples(1, self.d_in[i], self.d_out[i])
                self.z_for_Omega_fixed.append(tf.Variable(tmp[0,:,:], trainable = False))

        ## Parameters defining prior over Omega
        self.log_theta_sigma2 = tf.Variable(tf.zeros([n_layers]), name="log_theta_sigma2")

        if self.is_ard:
            self.llscale0 = []
            for i in range(self.nl):
                self.llscale0.append(tf.constant(.1 * np.log(self.d_in[i]), tf.float32))
        else:
            self.llscale0 = tf.constant(.1 * np.log(self.d_in), tf.float32)

        if self.is_ard:
            self.log_theta_lengthscale = []
            for i in range(self.nl):
                self.log_theta_lengthscale.append(tf.Variable(tf.multiply(tf.ones([self.d_in[i]]), self.llscale0[i]), name="log_theta_lengthscale"))
        else:
            self.log_theta_lengthscale = tf.Variable(self.llscale0, name="log_theta_lengthscale")
        self.prior_mean_Omega, self.log_prior_var_Omega = self.get_prior_Omega(self.log_theta_lengthscale)

        ## Set the prior over weights
        self.prior_mean_W, self.log_prior_var_W = self.get_prior_W()

        ## Initialize posterior parameters
        if self.learn_Omega == "var":
            self.mean_Omega, self.log_var_Omega = self.init_posterior_Omega()
        if self.learn_Omega == "optim":
            self.mean_Omega, self.log_var_Omega = self.init_posterior_Omega()

        self.mean_W, self.log_var_W = self.init_posterior_W()

        ## Set the number of Monte Carlo samples as a placeholder so that it can be different for training and test
        self.mc =  tf.placeholder(tf.int32)

        ## Batch data placeholders
        Din = d_in
        Dout = d_out

        self.Y = tf.placeholder(tf.float32, [None, d_out])
        self.X = tf.placeholder(tf.float32, [None, d_in])

        ## Builds whole computational graph with relevant quantities as part of the class
#        self.loss, self.kl, self.ell, self.layer_out = self.get_nelbo()
#
#        ## Initialize the session
#        self.session = tf.Session()


    ## Definition of a prior for Omega - which depends on the lengthscale of the covariance function
    def get_prior_Omega(self, log_lengthscale):
        if self.is_ard:
            prior_mean_Omega = []
            log_prior_var_Omega = []
            for i in range(self.nl):
                prior_mean_Omega.append(tf.zeros([self.d_in[i],1]))
            for i in range(self.nl):
                log_prior_var_Omega.append(-2 * log_lengthscale[i])
        else:
            prior_mean_Omega = tf.zeros(self.nl)
            log_prior_var_Omega = -2 * log_lengthscale
        return prior_mean_Omega, log_prior_var_Omega

    ## Definition of a prior over W - these are standard normals
    def get_prior_W(self):
        prior_mean_W = tf.zeros(self.n_W)
        log_prior_var_W = tf.zeros(self.n_W)
        return prior_mean_W, log_prior_var_W

    ## Function to initialize the posterior over omega
    def init_posterior_Omega(self):
        mu, sigma2 = self.get_prior_Omega(self.llscale0)

        mean_Omega = [tf.Variable(mu[i] * tf.ones([self.d_in[i], self.d_out[i]]), name="q_Omega") for i in range(self.n_Omega)]
        log_var_Omega = [tf.Variable(sigma2[i] * tf.ones([self.d_in[i], self.d_out[i]]), name="q_Omega") for i in range(self.n_Omega)]

        return mean_Omega, log_var_Omega

    ## Function to initialize the posterior over W
    def init_posterior_W(self):
        mean_W = [tf.Variable(tf.zeros([self.dhat_in[i], self.dhat_out[i]]), name="q_W") for i in range(self.n_W)]
        log_var_W = [tf.Variable(tf.zeros([self.dhat_in[i], self.dhat_out[i]]), name="q_W") for i in range(self.n_W)]

        return mean_W, log_var_W

    ## Function to compute the KL divergence between priors and approximate posteriors over model parameters (Omega and W) when q(Omega) is to be learned
    def get_kl_Omega_to_learn(self):
        kl = 0
        for i in range(self.n_Omega):
            kl = kl + utils.DKL_gaussian(self.mean_Omega[i], self.log_var_Omega[i], self.prior_mean_Omega[i], self.log_prior_var_Omega[i])
        for i in range(self.n_W):
            kl = kl + utils.DKL_gaussian(self.mean_W[i], self.log_var_W[i], self.prior_mean_W[i], self.log_prior_var_W[i])
        return kl

    ## Function to compute the KL divergence between priors and approximate posteriors over model parameters (W only) when q(Omega) is not to be learned
    def get_kl_Omega_fixed(self):
        kl = 0
        for i in range(self.n_W):
            kl = kl + utils.DKL_gaussian(self.mean_W[i], self.log_var_W[i], self.prior_mean_W[i], self.log_prior_var_W[i])
        return kl

    ## Returns samples from approximate posterior over Omega
    def sample_from_Omega_to_learn(self):
        Omega_from_q = []
        for i in range(self.n_Omega):
            z = utils.get_normal_samples(self.mc, self.d_in[i], self.d_out[i])
            Omega_from_q.append(tf.add(tf.multiply(z, tf.exp(self.log_var_Omega[i] / 2)), self.mean_Omega[i]))

        return Omega_from_q

    ## Returns Omega values calculated from fixed random variables and mean and variance of q() - the latter are optimized and enter the calculation of the KL so also lengthscale parameters get optimized
    def sample_from_Omega_optim(self):
        Omega_from_q = []
        for i in range(self.n_Omega):
            z = tf.multiply(self.z_for_Omega_fixed[i], tf.ones([self.mc, self.d_in[i], self.d_out[i]]))
            Omega_from_q.append(tf.add(tf.multiply(z, tf.exp(self.log_var_Omega[i] / 2)), self.mean_Omega[i]))

        return Omega_from_q

    ## Returns samples from prior over Omega - in this case, randomness is fixed throughout learning (and Monte Carlo samples)
    def sample_from_Omega_fixed(self):
        Omega_from_q = []
        for i in range(self.n_Omega):
            z = tf.multiply(self.z_for_Omega_fixed[i], tf.ones([self.mc, self.d_in[i], self.d_out[i]]))

            if self.is_ard == True:
                reshaped_log_prior_var_Omega = tf.tile(tf.reshape(self.log_prior_var_Omega[i] / 2, [self.d_in[i],1]), [1,self.d_out[i]])
                Omega_from_q.append(tf.multiply(z, tf.exp(reshaped_log_prior_var_Omega)))
            if self.is_ard == False:
                Omega_from_q.append(tf.add(tf.multiply(z, tf.exp(self.log_prior_var_Omega[i] / 2)), self.prior_mean_Omega[i]))

        return Omega_from_q

    ## Returns samples from approximate posterior over W
    def sample_from_W(self):
        W_from_q = []
        for i in range(self.n_W):
            z = utils.get_normal_samples(self.mc, self.dhat_in[i], self.dhat_out[i])
            W_from_q.append(tf.add(tf.multiply(z, tf.exp(self.log_var_W[i] / 2)), self.mean_W[i]))
        return W_from_q

    @abc.abstractmethod
    ## Returns the expected log-likelihood term in the variational lower bound
    def get_ell(self):
        raise NotImplementedError("Subclass should implement this.")


    ## Maximize variational lower bound --> minimize Nelbo
#    @abc.abstractmethod
    def get_nelbo(self):
        kl = self.get_kl()
        ell, layer_out = self.get_ell()
        nelbo  = kl - ell
        return nelbo, kl, ell, layer_out

    @abc.abstractmethod
    ## Return predictions on some data
    def predict(self, data, mc_test):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    ## Return the list of TF variables that should be "free" to be optimized
    def get_vars_fixing_some(self, all_variables):
        raise NotImplementedError("Subclass should implement this.")


    @abc.abstractmethod
    ## Function that learns the deep GP model with random Fourier feature approximation
    def learn(self, data, learning_rate, mc_train, batch_size, n_iterations, optimizer = None, display_step=100, test = None, mc_test=None, loss_function=None, duration = 1000000, less_prints=False):
        raise NotImplementedError("Subclass should implement this.")
