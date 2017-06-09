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

import utils
import likelihoods
import losses
import sys
import os
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes
from sklearn.metrics import confusion_matrix, adjusted_rand_score

from dataset import DataSet
from dgp_rff_lvm import DgpRff_LVM

from pprint import pprint

def get_score(real, predicted):
    num_classes = len(real[0])
    num_samples = len(real)
    matrix = np.array(np.zeros([num_classes, num_classes]))

    new_real = []
    new_predicted = []

    for i in range(num_samples):
        new_real.append(np.argmax(real[i]))
        new_predicted.append(np.argmax(predicted[i]))

    return (adjusted_rand_score(new_real, new_predicted))

def get_confusion_matrix(real, predicted):
    num_classes = len(real[0])
    num_samples = len(real)
    matrix = np.array(np.zeros([num_classes, num_classes]))

    new_real = []
    new_predicted = []

    for i in range(num_samples):
        new_real.append(np.argmax(real[i]))
        new_predicted.append(np.argmax(predicted[i]))
        #print(np.argmax(real[i]), np.argmax(predicted[i]))

    cm = confusion_matrix(new_real, new_predicted)
    return cm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize:
        filename='./img/confusion_matrix_normalized.pdf'
    else:
        filename='./img/confusion_matrix.pdf'
    plt.savefig(filename)
    plt.close()



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


    data = DataSet(train_data[:1000], train_labels[:1000])
    test = DataSet(test_data[:1], test_labels[:1])
    val  = DataSet(validation_data, validation_labels)

    if reduced_number_classes == True:
        train_data_df = data.to_dataframe()
        test_data_df = test.to_dataframe()
        validation_data_df = val.to_dataframe()

        tmp = train_data_df[(train_data_df['class0'] == 1.) | (train_data_df['class2'] == 1.)]
        data = DataSet(np.array(tmp.drop(tmp.columns[[-3, -2, -1]], axis=1).values.tolist()),
                       np.array(tmp[['class0', 'class2']].values.tolist()), )

        tmp = test_data_df[(test_data_df['class0'] == 1.) | (test_data_df['class2'] == 1.)]
        test = DataSet(np.array(tmp.drop(tmp.columns[[-3, -2, -1]], axis=1).values.tolist()),
                       np.array(tmp[['class0', 'class2']].values.tolist()))

        tmp = validation_data_df[(validation_data_df['class0'] == 1.) | (validation_data_df['class2'] == 1.)]
        val = DataSet(np.array(tmp.drop(tmp.columns[[-3, -2, -1]], axis=1).values.tolist()),
                       np.array(tmp[['class0', 'class2']].values.tolist()))


    return data, test, val


if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test, _ = import_oil(reduced_number_classes=False)

    print('Training size: ' + str(len(data.X)))
    print('Test size: ' + str(len(test.X)))
    #pprint(data.to_dataframe())


    ## Here we define a custom loss for dgp to show
    error_rate = losses.ARIscore(data.Dout)

    ## Likelihood
    like = likelihoods.Gaussian()

    ## Optimizer
    optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    ## Main dgp object
    dgp = DgpRff_LVM(like, data.num_examples, FLAGS.df, data.X.shape[1], FLAGS.nl, \
                 FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree,\
                 FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, \
                 FLAGS.learn_Omega, True, FLAGS.clustering)

    ## Learning
    directory = '../'+str(FLAGS.initializer)+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    #sys.stdout = open(directory+'log.log', 'w')
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
              FLAGS.initializer, save_img=False)

    cm = get_confusion_matrix(data.Y, dgp.p)
    plot_confusion_matrix(cm, ['horizontally', 'nested', 'homogeneous'])

    #wrong_assignment = 0
    #for i in range(data.num_examples):
    #    #print(str(np.argmax(dgp.p[i])) + ' --> ' + str(np.argmax(data.Y[i])))
    #    print(str(dgp.p[i]) + ' --> ' + str(data.Y[i]))
    #    if np.argmax(dgp.p[i]) != np.argmax(data.Y[i]):
    #        wrong_assignment += 1

    #print('Wrong assignment:'+str(wrong_assignment))
    #pred, nll_test = dgp.predict(test, FLAGS.mc_test)
    #print(pred.shape)
    #print(nll_test)



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
