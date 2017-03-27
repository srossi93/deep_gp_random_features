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

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes


from dataset import DataSet
import utils
import likelihoods
from dgp_rff_lvm import DgpRff_LVM

# import baselines

import losses


def import_oil():
    """
    This import oil dataset and saves the data as an object of our DataSet class
    :return:
    """
    train_data = np.loadtxt('./dataset/oil/DataTrn.txt', delimiter=' ')
    train_labels = np.loadtxt('./dataset/oil/DataTrnLbls.txt', delimiter=' ')
    #print(train_data)
    #print(train_labels)

    test_data = np.loadtxt('./dataset/oil/DataTst.txt', delimiter=' ')
    test_labels = np.loadtxt('./dataset/oil/DataTstLbls.txt', delimiter=' ')
    #print(test_data)
    #print(test_labels)

    validation_data = np.loadtxt('./dataset/oil/DataVdn.txt', delimiter=' ')
    validation_labels = np.loadtxt('./dataset/oil/DataVdnLbls.txt', delimiter=' ')
    #print(validation_data)
    #print(validation_labels)


#
    data = DataSet(train_data, train_labels)
    test = DataSet(test_data, test_labels)
    val = DataSet(validation_data, validation_labels)

    return data, test, val


if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test, _ = import_oil()

    print(len(data.X))

    ## Here we define a custom loss for dgp to show
    error_rate = losses.ZeroOneLoss(data.Dout)

    ## Likelihood
    like = likelihoods.Gaussian()

    ## Optimizer
    optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    ## Main dgp object
    dgp = DgpRff_LVM(like, data.num_examples, 2, data.X.shape[1], FLAGS.nl, \
                 FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree,\
                 FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, \
                 FLAGS.learn_Omega, True)



    ## Learning
    dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
                 FLAGS.display_step, test, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints)



    dgp.sample_latent_space(data)



    #pred, nll_test = dgp.predict(test, 1)
    #print(pred)
    #print(nll_test)
