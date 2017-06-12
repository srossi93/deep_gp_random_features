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
import pandas as pd
from pprint import pprint

## DataSet class
class DataSet():

    def __init__(self, X, Y, shuffle = True):
        #X = X[:10]
        #Y = Y[:10]
        self._num_examples = X.shape[0]
        perm = np.arange(self._num_examples)
        if (shuffle):
            np.random.shuffle(perm)
        self._X = X[perm,:]
        self._Y = Y[perm,:]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._Din = X.shape[1]
        self._Dout = Y.shape[1]


    def next_batch(self, batch_size, latent_variables=None):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if batch_size == self._num_examples and self._epochs_completed < 0:
            self._epochs_completed += 1
            perm = np.random.permutation(self._num_examples)
            self._X = self._X[perm]
            return self._X, self._Y
        else:
            self._epochs_completed += 1
            return self._X, self._Y


        if (self._index_in_epoch > self._num_examples) and (start != self._num_examples):
            self._index_in_epoch = self._num_examples
        if self._index_in_epoch > self._num_examples:   # Finished epoch
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)                  # Shuffle the data
            self._X = self._X[perm,:]
            self._Y = self._Y[perm,:]
            start = 0                               # Start next epoch
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if latent_variables != None:
            #print(latent_variables)
            Z_sliced = tf.slice(latent_variables, [start, 0], [end-start, self._Dout-1], name='sliced_latent_variables')
            #print(Z_sliced)
            return self._X[start:end,:], self._Y[start:end,:], Z_sliced
        return self._X[start:end,:], self._Y[start:end,:]

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def Din(self):
        return self._Din

    @property
    def Dout(self):
        return self._Dout

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    def to_dataframe(self):
        X_df = pd.DataFrame(self.X)
        Y_df = pd.DataFrame(self.Y)
        for i in range(len(self.Y[0])):
            class_name = 'class'+str(i)
            X_df[class_name] = Y_df[i]
        return X_df
