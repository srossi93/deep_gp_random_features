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
#import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score
from sklearn.datasets import make_moons, make_blobs, make_circles

from dataset import DataSet
from dgp_rff_lvm import DgpRff_LVM
from pprint import pprint

warnings.filterwarnings('ignore')

def get_score(dataset, real, predicted):
    num_classes = len(real[0])
    num_samples = len(real)
    matrix = np.array(np.zeros([num_classes, num_classes]))

    new_real = np.argmax(real, axis=1)
    new_predicted = np.argmax(predicted, axis=1)

    #for i in range(num_samples):
    #    new_real.append(np.argmax(real[i]))
    #    new_predicted.append(np.argmax(predicted[i]))


    return (adjusted_rand_score(new_real, new_predicted), normalized_mutual_info_score(new_real, new_predicted), fowlkes_mallows_score(new_real, new_predicted), silhouette_score(dataset, new_predicted))

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


if __name__ == '__main__':
    FLAGS = utils.get_flags()
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data0, l0 = make_moons(1500, noise=1e-1)
    data1, l1 = make_circles(1500, noise=1e-1, factor=0.1)
    data2, l2 = make_blobs(1500, n_features=2, centers=2, cluster_std=1e-0)
    data3, l3 = make_blobs(1500, n_features=10, centers=3, cluster_std=2.5e-0)
    name0 = 'moons'
    name1 = 'circles'
    name2 = 'blobs'
    name3 = 'blobs_noisy'
    #data_X, l = make_moons(500, noise=1e-1)
    #data_X, l = make_blobs(n_samples=1500, n_features=2, centers=2, cluster_std=7e-1)#, noise=1e-1)#, factor=.1)

    for data_X, l, name in [(data0, l0, name0), (data1, l1, name1), (data2, l2, name2), (data3, l3, name3)]:
        labels = np.zeros([len(l), len(set(l))])
        for i in range(len(l)):
            labels[i,l[i]] = 1
        data = DataSet(data_X, labels, shuffle=False)

        print('\n\nTraining size: ' + str(len(data.X)))

        error_rate = losses.ARIscore(data.Din)

        like = likelihoods.Gaussian()

        optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

        dgp = DgpRff_LVM(like, data.num_examples, len(set(l)), data.X.shape[1], FLAGS.nl, \
                     FLAGS.n_rff, len(set(l)), FLAGS.kernel_type, FLAGS.kernel_arccosine_degree,\
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

        ######
        #  PRINT METRICS
        ######

        score = get_score(data.X, data.Y, dgp.p)
        print('\nDataset: %s' % name)
        print('Adjusted Rand Score.....: %.4f' % score[0])
        print('Mutual Information Score: %.4f' % score[1])
        print('Fowlkes-Mallows Index...: %.4f' % score[2])
        print('Silhouette Score........: %.4f' % score[3])

        cm = get_confusion_matrix(data.Y, dgp.p)
	print(pd.DataFrame(cm))
