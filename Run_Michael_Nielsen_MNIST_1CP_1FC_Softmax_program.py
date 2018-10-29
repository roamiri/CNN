# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:36:48 2018

@author: roohollah
"""

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
import time

start = time.time()
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
    filter_shape=(20, 1, 5, 5), # Each conv 1 map is 24x24
    poolsize=(2, 2)),# Each pool 1 map is 12x12
    FullyConnectedLayer(n_in=20*12*12, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data) # λ = 0.

end = time.time()

print('time needed to run program:', end - start)