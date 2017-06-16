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
##
## Example of use
# python2.7 experiments/dgp_pca_mnist.py --seed=12345 --theta_fixed=5000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=2 --batch_size=1000 --mc_train=1 --mc_test=1 --n_iterations=10000 --display_step=1000 --learn_Omega=optim --initializer=PCA --clustering=False

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes


from dataset import DataSet
import utils
import likelihoods
from dgp_rff_lvm import DgpRff_LVM
from pprint import pprint
# import baselines

import losses

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

    data = DataSet(train_images[0:], train_labels[0:])
    test = DataSet(test_images[0:1], test_labels[0:1])
    val = DataSet(validation_images, validation_labels)

    #print(np.array(data.Y).shape)
    train_data_df = data.to_dataframe()

    tmp = train_data_df[(train_data_df['class0'] == 1.) | (train_data_df['class1'] == 1.)| (train_data_df['class2'] == 1.)| (train_data_df['class3'] == 1.)| (train_data_df['class4'] == 1.)]
    data = DataSet(np.array(tmp.drop(tmp.columns[[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]], axis=1).values.tolist()),
                   np.array(tmp[['class0', 'class1', 'class2', 'class3', 'class4']].values.tolist()))

    return data, test, val


if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test, _ = import_mnist()
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

    
