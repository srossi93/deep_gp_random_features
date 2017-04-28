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
import pandas as pd
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes


from dataset import DataSet
import utils
import likelihoods
from dgp_rff_lvm import DgpRff_LVM
import matplotlib.pyplot as plt

from pprint import pprint
# import baselines

import losses
import sys
import os

def get_dataframe(data, labels):
    data_df = pd.DataFrame(data)
    labels_df = pd.DataFrame(labels)
    for i in range(len(labels[0])):
        class_name = 'class'+str(i)
        data_df[class_name] = labels_df[i]
    return data_df


def import_oil(reduced_number_classes=False):
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

    train_data_df = pd.DataFrame(train_data)
    train_labels_df = pd.DataFrame(train_labels)
    test_data_df = pd.DataFrame(test_data)
    test_labels_df = pd.DataFrame(test_labels)
    validation_data_df = pd.DataFrame(validation_data)
    validation_labels_df = pd.DataFrame(validation_labels)

    for i in range(len(train_labels[0])):
        class_name = 'class'+str(i)
        train_data_df[class_name]      = train_labels_df[i]
        test_data_df[class_name]       = test_labels_df[i]
        validation_data_df[class_name] = validation_labels_df[i]

    validation_data_df = get_dataframe(validation_data, validation_labels)
    #train_data_df['class0'] = train_labels_df[0]
    #train_data_df['class1'] = train_labels_df[1]
    #train_data_df['class2'] = train_labels_df[2]

    #test_data_df['class0'] = test_labels_df[0]
    #test_data_df['class1'] = test_labels_df[1]
    #test_data_df['class2'] = test_labels_df[2]

    #validation_data_df['class0'] = validation_labels_df[0]
    #validation_data_df['class1'] = validation_labels_df[1]
    #validation_data_df['class2'] = validation_labels_df[2]


    if reduced_number_classes == False:
        data = DataSet(train_data[:1000], train_labels[:1000])
        #print(train_data[0:10])
        test = DataSet(test_data, test_labels)
        val  = DataSet(validation_data, validation_labels)
    else:
        #print(train_data_df.info())
        tmp = train_data_df[(train_data_df['class0'] == 1.) | (train_data_df['class1'] == 1.)]
        #tmp = tmp.drop(tmp.columns[[-3, -2, -1]], axis=1).values.tolist()
        #tmp = tmp[['class0', 'class1']].values.tolist()
        #pprint(tmp)#.values.tolist())
        #pprint(train_data)
        #pprint( train_data_df[(train_data_df['class0'] == 1.) | (train_data_df['class1'] == 1.)]  )
        data = DataSet(np.array(tmp.drop(tmp.columns[[-3, -2, -1]], axis=1).values.tolist()),
                       np.array(tmp[['class0', 'class1']].values.tolist()), )

        tmp = test_data_df[(test_data_df['class0'] == 1.) | (test_data_df['class1'] == 1.)]
        test = DataSet(np.array(tmp.drop(tmp.columns[[-3, -2, -1]], axis=1).values.tolist()),
                       np.array(tmp[['class0', 'class1']].values.tolist()))

        tmp = validation_data_df[(validation_data_df['class0'] == 1.) | (validation_data_df['class1'] == 1.)]
        val = DataSet(np.array(tmp.drop(tmp.columns[[-3, -2, -1]], axis=1).values.tolist()),
                       np.array(tmp[['class0', 'class1']].values.tolist()))

        #data, test, val = [data, test, val]

    return data, test, val


if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test, _ = import_oil(reduced_number_classes=False)

    print(len(data.X))


    ## Here we define a custom loss for dgp to show
    error_rate = losses.RootMeanSqError(data.Dout)

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
    directory = '../'+str(FLAGS.initializer)+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    sys.stdout = open(directory+'log.log', 'w')
    print('Learning with'+' '+
          '--learning_rate='+ str(FLAGS.learning_rate)+' '+
          '--optimizer='+     str(FLAGS.optimizer)+' '+
          '--mc_test='+       str(FLAGS.mc_test)+' '+
          '--mc_train='+      str(FLAGS.mc_train)+' '+
          '--n_iterations='+  str(FLAGS.n_iterations)+' '+
          '--initializer='+   str(FLAGS.initializer)+' '+
          '--nl='+            str(FLAGS.nl)+' '+
          '--n_rff='+         str(FLAGS.n_rff)+' '+
          '--df='+            str(FLAGS.df)+' '+
          '--kernel_type='+   str(FLAGS.kernel_type))

    dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
              FLAGS.display_step, test, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints,
              FLAGS.initializer)



    #pprint(dgp.session.run(dgp.latents))
    # latents = pd.DataFrame(dgp.session.run(dgp.latents), columns=['x', 'y'])
    # labels = pd.DataFrame(data.Y)
    # ##print(latents)
    # ##print(labels)
    # latents['class0'] = labels[0]
    # latents['class1'] = labels[1]
    # latents['class2'] = labels[2]
    #
    # print(len(latents[latents['class0']==1]))
    # print(len(latents[latents['class1']==1]))
    # print(len(latents[latents['class2']==1]))
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)

    #ax.scatter(latents[latents['class0']==1].x, latents[latents['class0']==1].y, s=5, label='class0')
    #ax.scatter(latents[latents['class1']==1].x, latents[latents['class1']==1].y, s=5, label='class1')
    #ax.scatter(latents[latents['class2']==1].x, latents[latents['class2']==1].y, s=5, label='class2')
    #ax.legend()
    #plt.ylabel('latent_dimension[1]')
    #plt.xlabel('latent_dimension[0]')
    #plt.title('Distribution of training samples in the latent space')



    #plt.savefig('latent_space.pdf')
    ##plt.savefig('./oil_omega'+str(FLAGS.q_Omega_fixed)+'_theta'+str(FLAGS.theta_fixed)+'_nrff'+str(FLAGS.n_rff)+'.pdf')

    #pred, nll_test = dgp.predict(test, 1)
    #print(pred)
    #print(nll_test)
