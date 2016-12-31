from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os
import scipy.misc as misc
import cv2

from FaceRecognizer import *
import utils

######################## test FaceDatabase ########################
# fd = FaceDatabase()
# print(fd.identity_list,fd.embs_pool,fd.n_identity)
# fd.remove('b')
# print(fd.identity_list,fd.embs_pool,fd.n_identity)
# fd.remove('c')
# print(fd.identity_list,fd.embs_pool,fd.n_identity)
# fd.add('a',[1])
# print(fd.identity_list,fd.embs_pool,fd.n_identity)
# fd.add('b',[1])
# print(fd.identity_list,fd.embs_pool,fd.n_identity)
# fd.add('c',[1])
# print(fd.identity_list,fd.embs_pool,fd.n_identity)
###################################################################

#################### test FaceRecognizer v1 ####################
# image_size = 160
# facenet_model_dir = './pretrained_models/FaceNet'
# mtcnn_model_dir = './pretrained_models/MTCNN'
# image_filename = './people.jpeg'

# image_filename = os.path.abspath(os.path.expanduser(image_filename))
# image = misc.imread(image_filename)

# print(image.shape)

# FR = FaceRecognizer(facenet_model_dir, mtcnn_model_dir)

# tic = time.clock()
# FR.build()
# toc = time.clock()
# build_time = toc - tic
# print('build time: {}'.format(build_time))

# tic = time.clock()
# embs = FR.inference(image)
# toc = time.clock()
# inference_time = toc - tic
# print('inference_time: {}'.format(inference_time))

# print(embs.shape)
##################################################################

####################### test FaceRecognizer v2 ####################### 
# facenet_model_dir = './pretrained_models/FaceNet/'
# mtcnn_model_dir = './pretrained_models/MTCNN/'
# image_dir = './data/images'
# database_verbose = True
# image_dir = os.path.abspath(os.path.expanduser(image_dir))
# image_list = os.listdir(image_dir)

# FR = FaceRecognizer(facenet_model_dir, mtcnn_model_dir, database_verbose=database_verbose)

# tic = time.clock()
# FR.build()
# toc = time.clock()
# build_time = toc - tic
# print('build time: {}'.format(build_time))

# for i, image_name in enumerate(image_list):
# 	# read image
# 	image = cv2.imread(os.path.join(image_dir,image_name))
# 	# inference
# 	tic = time.clock()
# 	bb, bb_names = FR.inference(image)
# 	toc = time.clock()
# 	print('{}: {}'.format(image_name, toc-tic))

# 	FR.add_identity(bb_names[0],'hi'+str(i%3))
# 	if i%4==0:
# 		FR.remove_identity('hi2')

# 	print(FR.list_database(),FR.n_prototype())

# 	# visualize
# 	cv2.imshow('image',image)
# 	cv2.waitKey(0)

# cv2.destroyAllWindows()
####################################################################

####################### test FaceRecognizer v3 ####################### 
facenet_model_dir = './pretrained_models/FaceNet/'
mtcnn_model_dir = './pretrained_models/MTCNN/'
image_dir = './data/images' # you should make sure you do have images in this directory
database_verbose = False
image_dir = os.path.abspath(os.path.expanduser(image_dir))
image_list = os.listdir(image_dir)

FR = FaceRecognizer(facenet_model_dir, mtcnn_model_dir, database_verbose=database_verbose)

tic = time.clock()
FR.build()
toc = time.clock()
build_time = toc - tic
print('build time: {}'.format(build_time))

for i, image_name in enumerate(image_list*3):
	# read image
	image = cv2.imread(os.path.join(image_dir,image_name))
	# inference
	tic = time.clock()
	bb, bb_names = FR.inference(image)
	toc = time.clock()
	print('{}: {}'.format(image_name, toc-tic))

	print('bb_names',bb_names)

	if i<len(image_list):
		FR.add_identity(bb_names[0],'hi'+str(i))

	# print(FR.list_database(),FR.n_prototype())
	print('\n\n')

	# visualize
	cv2.imshow('image',image)
	cv2.waitKey(0)

cv2.destroyAllWindows()
####################################################################