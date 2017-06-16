import utils
import likelihoods
import losses
import sys
import os
import warnings
import itertools

import numpy as np
import pandas as pd
#import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score
from sklearn.datasets import make_moons, make_blobs, make_circles

from dataset import DataSet
from dgp_rff_lvm import DgpRff_LVM
from pprint import pprint
from tabulate import tabulate
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph

warnings.filterwarnings('ignore')

def get_score(dataset, real, predicted):
    real = real[:,0]

    return (adjusted_rand_score(real, predicted), normalized_mutual_info_score(real, predicted), fowlkes_mallows_score(real, predicted))#, silhouette_score(dataset, predicted))

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

    cm = confusion_matrix(real, predicted)
    return cm



def print_scores(data, pred, d_name, c_name):
    score = get_score(data.X, data.Y, pred)
    print('\nDataset: %s\nAlgorithm: %s' % (d_name, c_name))
    print('-------------------------------------')
    print('Adjusted Rand Score.....: %.4f' % score[0])
    print('Mutual Information Score: %.4f' % score[1])
    print('Fowlkes-Mallows Index...: %.4f' % score[2])
    #print('Silhouette Score........: %.4f' % score[3])

    #cm = confusion_matrix(data.Y, pred)
    #print('Confusion matrix: ')
    #print(tabulate(pd.DataFrame(cm), headers='keys', tablefmt='psql'))
    #plot_confusion_matrix(cm, range(len(list(pred))), title=name+'_confusion_matrix')


if __name__ == '__main__':
    FLAGS = utils.get_flags()
    np.random.seed(FLAGS.seed)

    data0, l0 = make_moons(1500, noise=1e-1)
    data1, l1 = make_circles(1500, noise=1e-1, factor=0.1)
    data2, l2 = make_blobs(1500, n_features=2, centers=2, cluster_std=1e-0)
    data3, l3 = make_blobs(1500, n_features=10, centers=4, cluster_std=2.5e-0)
    name0 = 'moons'
    name1 = 'circles'
    name2 = 'blobs'
    name3 = 'blobs_noisy'
    #data_X, l = make_moons(500, noise=1e-1)
    #data_X, l = make_blobs(n_samples=1500, n_features=2, centers=2, cluster_std=7e-1)#, noise=1e-1)#, factor=.1)

    for data, labels, d_name in [(data0, l0, name0), (data1, l1, name1), (data2, l2, name2), (data3, l3, name3)]:

        n_clusters = len(set(labels))
        labels = labels.reshape(-1, 1)
        data = DataSet(data, labels, shuffle=False)

        bandwidth = cluster.estimate_bandwidth(data.X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(data.X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # create clustering estimators
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)
        ward = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity)
        spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',  affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=.3)
        affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)
        average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=n_clusters,  connectivity=connectivity)
        birch = cluster.Birch(n_clusters=n_clusters)

        clustering_algorithms = [ two_means, affinity_propagation, ms, ward, average_linkage, dbscan, birch]
        clustering_names = [ 'two_means', 'affinity_propagation', 'ms', 'ward', 'average_linkage', 'dbscan', 'birch']

        for c_name, algorithm in zip(clustering_names, clustering_algorithms):
            algorithm.fit(data.X)
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(data.X)

            ######
            #  PRINT ASSIGNMENT IN THE OBSERVED SPACE
            ######
            print_scores(data, y_pred, d_name, c_name)
