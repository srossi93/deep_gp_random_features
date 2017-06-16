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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score
from sklearn.datasets import make_moons, make_blobs, make_circles

from dataset import DataSet
from dgp_rff_lvm import DgpRff_LVM
from pprint import pprint

warnings.filterwarnings('ignore')

# Example of use
# export PYTHONPATH="${PYTHONPATH}:."
# python2.7 experiments/dgp_clustering.py --seed=12345 --theta_fixed=7500 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=2 --batch_size=1000 --mc_train=1 --mc_test=1 --n_iterations=10000 --display_step=1000 --learn_Omega=optim --initializer=PCA --clustering=True

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

def plot_confusion_matrix(cm, classes, normalize=False, title='ConfusionMatrix', cmap=plt.cm.Blues):

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

    print(tabulate(pd.DataFrame(cm), headers='keys', tablefmt='psql'))

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize:
        filename='./img/'+title+'_normalized.pdf'
    else:
        filename='./img/'+title+'.pdf'
    plt.savefig(filename)
    plt.close()


def print_scores(data, dgp):
    score = get_score(data.X, data.Y, dgp.p)
    print('\nDataset: %s' % name)
    print('Adjusted Rand Score.....: %.4f' % score[0])
    print('Mutual Information Score: %.4f' % score[1])
    print('Fowlkes-Mallows Index...: %.4f' % score[2])
    print('Silhouette Score........: %.4f' % score[3])

    cm = get_confusion_matrix(data.Y, dgp.p)
    plot_confusion_matrix(cm, range(len(dgp.p[0])), title=name+'_confusion_matrix')


if __name__ == '__main__':
    FLAGS = utils.get_flags()
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data0, l0 = make_moons(1500, noise=.05)
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

        df = data.to_dataframe()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i in range(len(data.Y[0])):
            class_name = 'class'+str(i)
            ax.scatter(df[df[class_name]==1][0], df[df[class_name]==1][1], s=.5, label=class_name)
        #plt.scatter(data_X[:,0], data_X[:,1], s=1, c=l, cmap=plt.cm.jet_r)
        ax.legend()
        plt.ylabel('observed_dimension[1]')
        plt.xlabel('observed_dimension[0]')
        plt.title('Distribution of training samples in the observed space')

        filename='./img/'+name+'_assignment_dataset.pdf'
        plt.savefig(filename)
        plt.close()
        #plt.show()
        print('\n\nDataset: %s' % name )

        print('\nTraining size: ' + str(len(data.X)))

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
              '--kernel_type='+   str(FLAGS.kernel_type)+' '+
              '--is_ard='+        str(FLAGS.is_ard)+' '+
              '--feed_forward='+  str(FLAGS.feed_forward)+' '+
              '--q_Omega_fixed='+ str(FLAGS.q_Omega_fixed)+' '+
              '--theta_fixed='+   str(FLAGS.theta_fixed)+' '+
              '--learn_Omega='+   str(FLAGS.learn_Omega)+' '+
              '--clustering='+    str(FLAGS.clustering))

        dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
                  FLAGS.display_step, data, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints,
                  FLAGS.initializer, save_img=False)

        ######
        #  PRINT METRICS
        ######

        print_scores(data, dgp)

        ######
        #  PRINT ASSIGNMENT IN THE OBSERVED SPACE
        ######

        data_assignment = DataSet(data.X, np.round(dgp.p), shuffle=False)
        df =  data_assignment.to_dataframe()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i in range(len(data_assignment.Y[0])):
            class_name = 'class'+str(i)
            ax.scatter(df[df[class_name]==1][0], df[df[class_name]==1][1], s=.5, label=class_name)
        #plt.scatter(data_X[:,0], data_X[:,1], s=1, c=l, cmap=plt.cm.jet_r)
        ax.legend()
        plt.ylabel('observed_dimension[1]')
        plt.xlabel('observed_dimension[0]')
        plt.title('Cluster assignment')

        filename='./img/'+name+'_assignment_cluster.pdf'
        plt.savefig(filename)
        plt.close()


        #plot_confusion_matrix(cm, ['0', '1'], normalize=True)




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
