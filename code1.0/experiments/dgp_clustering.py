import utils
import likelihoods
import losses
import sys
import os
import warnings
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import make_moons, make_blobs, make_circles

from dataset import DataSet
from dgp_rff_lvm import DgpRff_LVM
from pprint import pprint

warnings.filterwarnings('ignore')

def get_score(real, predicted):
    num_classes = len(real[0])
    num_samples = len(real)
    matrix = np.array(np.zeros([num_classes, num_classes]))

    new_real = []
    new_predicted = []

    for i in range(num_samples):
        new_real.append(np.argmax(real[i]))
        new_predicted.append(np.argmax(predicted[i]))

    return (adjusted_rand_score(new_real, new_predicted), normalized_mutual_info_score(new_real, new_predicted) )

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




if __name__ == '__main__':
    FLAGS = utils.get_flags()
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data0, l0 = make_moons(1500, noise=1e-1)
    data1, l1 = make_circles(1500, noise=1e-1, factor=0.1)
    data2, l2 = make_blobs(1500, n_features=2, centers=2, cluster_std=1e-0)
    name0 = 'moons'
    name1 = 'circles'
    name2 = 'blobs'
    #data_X, l = make_moons(500, noise=1e-1)
    #data_X, l = make_blobs(n_samples=1500, n_features=2, centers=2, cluster_std=7e-1)#, noise=1e-1)#, factor=.1)

    for data_X, l, name in [(data0, l0, name0), (data1, l1, name1), (data2, l2, name2)]:
        labels = np.zeros([len(l), 2])
        for i in range(len(l)):
            labels[i,l[i]] = 1
        data = DataSet(data_X, labels)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.scatter(data_X[:,0], data_X[:,1], s=1, c=l, cmap=plt.cm.jet)
        ax.legend()
        plt.ylabel('observed_dimension[1]')
        plt.xlabel('observed_dimension[0]')
        plt.title('Distribution of training samples in the observed space')

        filename='./img/'+name+'_iter_0_obs.pdf'
        plt.savefig(filename)
        plt.close()
        #plt.show()

        print('Training size: ' + str(len(data.X)))

        error_rate = losses.ARIscore(data.Din)

        like = likelihoods.Gaussian()

        optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

        dgp = DgpRff_LVM(like, data.num_examples, FLAGS.df, data.X.shape[1], FLAGS.nl, \
                     FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree,\
                     FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, \
                     FLAGS.learn_Omega, True, FLAGS.clustering)

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
                  FLAGS.display_step, data, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints,
                  FLAGS.initializer, save_img=False)

        cm = get_confusion_matrix(data.Y, dgp.p)
        plot_confusion_matrix(cm, ['0', '1'])
        #plot_confusion_matrix(cm, ['0', '1'], normalize=True)

        score = get_score(data.Y, dgp.p)
        print('\nDataset: %s' % name)
        print('Adjusted Rand Score.........: %.4f' % score[0])
        print('Normalized Mutual Info Score: %.4f' % score[1])


    #fig = plt.figure(figsize=[50, 50])
    #ax = fig.add_subplot(1,1,1)
    #plt.scatter(data_X[:,0], data_X[:,1], s=15, c=l, cmap=plt.cm.RdYlGn)

    #for i in range(data.num_examples):
    #    ax.annotate("%.2f" % (dgp.p[i][0]), (data.X[i,0],data.X[i,1]),)

    #ax.legend()
    #plt.ylabel('observed_dimension[1]')
    #plt.xlabel('observed_dimension[0]')
    #plt.title('Distribution of training samples in the observed space')

    #filename='./img/assignement_obs.pdf'
    #plt.savefig(filename)
    #plt.close()
