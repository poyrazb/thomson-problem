# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:44:02 2020

@author: souzam
"""

import tensorflow as tf
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# Uncomment this to hide all the ugly status messages
#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# Uncomment this to run on the CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''


# initial set of points randomly distributed
def initial_set(n, d):
    points = []
    for i in range(n):
        p = np.random.normal(size = d)
        p = p / la.norm(p)
        points.append(p)
    return points


# plot 3d
def show(points):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    for p in points:
        ax.scatter(p[0], p[1], p[2], color='blue')


# optimize positions using TensorFlow/Adam
def optimize_tf_adam(points):
    tf.Variable(r0, name='coordinates')    


# main

d = 3
n = 1000

points = initial_set(n, d)
show(points)


print("TensorFlow / Adam / CUDA")
start = time.time()
opt_points = optimize_tf_adam(points)
end = time.time()
print(end - start)

