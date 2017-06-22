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

from dataset import DataSet
import utils
import likelihoods
from dgp_rff import DgpRff
import losses
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from dgp_rff_lvm import DgpRff_LVM
from sklearn.preprocessing import scale

def import_dataset(dataset, fold):

    train_X = np.loadtxt('FOLDS/' + dataset + '_ARD_Xtrain__FOLD_' + fold, delimiter=' ')
    train_Y = np.loadtxt('FOLDS/' + dataset + '_ARD_ytrain__FOLD_' + fold, delimiter=' ')
    test_X = np.loadtxt('FOLDS/' + dataset + '_ARD_Xtest__FOLD_' + fold, delimiter=' ')
    test_Y = np.loadtxt('FOLDS/' + dataset + '_ARD_ytest__FOLD_' + fold, delimiter=' ')

    train_X = scale(train_X, axis=0, with_mean=True, with_std=True, copy=True )

    data = DataSet(train_X, train_Y, shuffle=False)
    test = DataSet(test_X, test_Y, shuffle=False)

    return data, test

if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test = import_dataset(FLAGS.dataset, FLAGS.fold)

    ## Here we define a custom loss for dgp to show
    print('\n\nTraining shape: ' + str(data.X.shape))
    #pprint(data.to_dataframe())


    ## Here we define a custom loss for dgp to show
    error_rate = losses.RootMeanSqError(data.Dout) if not FLAGS.clustering else losses.NMI(data.Dout)

    ## Likelihood
    like = likelihoods.Gaussian()

    ## Optimizer
    optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    ## Main dgp object
    dgp = DgpRff_LVM(like, data.num_examples, FLAGS.latent_dimensions, data.X.shape[1], FLAGS.nl, \
                 FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree,\
                 FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, \
                 FLAGS.learn_Omega, True, FLAGS.clustering)


    print('\nLearning with'+' '+
          '--learning_rate='+ str(FLAGS.learning_rate)+' '+
          '--optimizer='+     str(FLAGS.optimizer)+' '+
          '--mc_test='+       str(FLAGS.mc_test)+' '+
          '--mc_train='+      str(FLAGS.mc_train)+' '+
          '--n_iterations='+  str(FLAGS.n_iterations)+' '+
          '--initializer='+   str(FLAGS.initializer)+' '+
          '--nl='+            str(FLAGS.nl)+' '+
          '--n_rff='+         str(FLAGS.n_rff)+' '+
          '--df='+            str(FLAGS.df)+' '+
          '--kernel_type='+   str(FLAGS.kernel_type)+' '+
          '--is_ard='+        str(FLAGS.is_ard)+' '+
          '--feed_forward='+  str(FLAGS.feed_forward)+' '+
          '--q_Omega_fixed='+ str(FLAGS.q_Omega_fixed)+' '+
          '--theta_fixed='+   str(FLAGS.theta_fixed)+' '+
          '--learn_Omega='+   str(FLAGS.learn_Omega)+' '+
          '--clustering='+    str(FLAGS.clustering)+' '+
          '--latent_dimensions='+str(FLAGS.latent_dimensions)+'\n')

    dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
              FLAGS.display_step, test, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints,
              FLAGS.initializer, save_img=True)

    #cm =
    print(confusion_matrix(np.argmax(data.Y, 1), np.argmax(dgp.p, 1))) if FLAGS.clustering else 0
