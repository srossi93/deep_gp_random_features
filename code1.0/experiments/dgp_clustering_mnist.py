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
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score
from sklearn.datasets import make_moons, make_blobs, make_circles

from dataset import DataSet
from dgp_rff_lvm import DgpRff_LVM
from pprint import pprint
from tabulate import tabulate

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

def process_mnist(images, dtype = dtypes.float32, reshape=True):
    if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    return images

def get_data_info(images):
    rows, cols = images.shape
    std = np.zeros(cols)
    mean = np.zeros(cols)
    for col in range(cols):
        std[col] = np.std(images[:,col])
        mean[col] = np.mean(images[:,col])
    return mean, std

def standardize_data(images, means, stds):
    data = images.copy()
    rows, cols = data.shape
    for col in range(cols):
        if stds[col] == 0:
            data[:,col] = (data[:,col] - means[col])
        else:
            data[:,col] = (data[:,col] - means[col]) / stds[col]
    return data

def import_mnist():
    """
    This import mnist and saves the data as an object of our DataSet class
    :return:
    """
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 0
    ONE_HOT = True
    TRAIN_DIR = 'MNIST_data'


    local_file = base.maybe_download(TRAIN_IMAGES, TRAIN_DIR,
                                     SOURCE_URL + TRAIN_IMAGES)
    train_images = extract_images(open(local_file))

    local_file = base.maybe_download(TRAIN_LABELS, TRAIN_DIR,
                                     SOURCE_URL + TRAIN_LABELS)
    train_labels = extract_labels(open(local_file), one_hot=ONE_HOT)

    local_file = base.maybe_download(TEST_IMAGES, TRAIN_DIR,
                                     SOURCE_URL + TEST_IMAGES)
    test_images = extract_images(open(local_file))

    local_file = base.maybe_download(TEST_LABELS, TRAIN_DIR,
                                     SOURCE_URL + TEST_LABELS)
    test_labels = extract_labels(open(local_file), one_hot=ONE_HOT)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    ## Process images
    train_images = process_mnist(train_images)
    validation_images = process_mnist(validation_images)
    test_images = process_mnist(test_images)

    ## Standardize data
    train_mean, train_std = get_data_info(train_images)
#    train_images = standardize_data(train_images, train_mean, train_std)
#    validation_images = standardize_data(validation_images, train_mean, train_std)
#    test_images = standardize_data(test_images, train_mean, train_std)

    data = DataSet(train_images[0:], train_labels[0:], shuffle=False)
    test = DataSet(test_images[0:], test_labels[0:], shuffle=False)
    val = DataSet(validation_images, validation_labels, shuffle=False)

    #print(np.array(data.Y).shape)
    train_data_df = data.to_dataframe()

    tmp = train_data_df[(train_data_df['class0'] == 1.) | (train_data_df['class1'] == 1.) | (train_data_df['class2'] == 1.)]
    data = DataSet(np.array(tmp.drop(tmp.columns[[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]], axis=1).values.tolist()),
                   np.array(tmp[['class0', 'class1', 'class2']].values.tolist()))

    return data, test, val


if __name__ == '__main__':
    FLAGS = utils.get_flags()
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test, _ = import_mnist()
    name = 'mnist'

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

    print('\n\nTraining size: ' + str(len(data.X)))

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
    plot_confusion_matrix(cm, range(len(dgp.p[0])), title=name+'_confusion_matrix')

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
