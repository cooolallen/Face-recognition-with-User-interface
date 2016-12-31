from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from scipy import misc

def get_ckpt_filenames(model_dir):
	files = os.listdir(model_dir)
	ckpt_files = [s for s in files if 'ckpt' in s]
	if len(ckpt_files)==0:
		raise ValueError('No checkpoint file found in the model directory (%s)' % model_dir)
	elif len(ckpt_files)==1:
		ckpt_file = ckpt_files[0]
	else:
		ckpt_iter = [(s,int(s.split('-')[-1])) for s in ckpt_files if 'ckpt' in s]
		sorted_iter = sorted(ckpt_iter, key=lambda tup: tup[1])
		ckpt_file = sorted_iter[-1][0]

	return ckpt_file

def prewhiten(x):
	mean = np.mean(x)
	std = np.std(x)
	std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
	y = np.multiply(np.subtract(x, mean), 1/std_adj)

	return y  

def denormalize_images(images):
	return np.uint8((images+1)*127.5)
		
def check_path(path):
	'''
	FUNC: convert path (can be realtive path or using ~) to absolute path and check
		  the path exists.
	'''
	path = os.path.abspath(path)
	if not os.path.exists(path):
		raise NameError('Not such path as %s' % path)

	return path

def check_dir(dir_path):
	'''
	FUNC: check directory path, if no such directory, then create one
	'''
	dir_path = os.path.abspath(dir_path)
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	else:
		if os.path.isdir(dir_path):
			return dir_path
		else:
			raise NameError('%s is not a directory' % dir_path)

def save_image(img, filename, verbose=False):
	'''
	FUNC: save image at filename. If it's color image, channel follows r,g,b.
		  Image shape can be (*,*), (*,*,3), (*,*,4)
	'''
	#filename = check_path(filename)
	if verbose:
		print('Save image at %s' % filename)
	misc.imsave(filename, img)

def print_vars(vars_in):
	for i in range(len(vars_in)):
		print(vars_in[i].op.name)

def print_all_vars_on_graph(graph):
	with graph.as_default():
		print_vars(tf.all_variables())

def print_all_on_graph(graph):
	for n in graph.as_graph_def().node:
		print(n.name)