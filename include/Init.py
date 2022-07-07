import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

"""
initialization
"""
def uniform(shape, scale=0.05, name=None):
	"""Uniform init."""
	initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def glorot(shape, name=None):
	"""Glorot & Bengio (AISTATS 2010) init."""
	"""
	标准化的Glorot初始化——glorot_uniform
	"""
	init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
	initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
	return tf.Variable(initial, name=name)

def he(shape,name=None):
	"""
	He initialization
	r=sqrt(6.0/input)
	init=uniform(-r,r)
	:param shape:
	:param name:
	:return:
	"""
	he_init = tf.variance_scaling_initializer()
	init_range = np.sqrt(6.0 / (shape[0]))
	initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
	return tf.Variable(initial, name=name)

def zeros(shape, name=None):
	"""All zeros."""
	initial = tf.zeros(shape, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def ones(shape, name=None):
	"""All ones."""
	initial = tf.ones(shape, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def trunc_normal(shape, name=None, normalize=True):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=1.0 / np.sqrt(shape[0])))
    if not normalize: return initial
    return tf.nn.l2_normalize(initial, 1)