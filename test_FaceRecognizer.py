from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os, sys
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

# 	FR.add_identity_at_current_inference(bb_names[0],'hi'+str(i%3))
# 	if i%4==0:
# 		FR.remove_identity('hi2')

# 	print(FR.list_database(),FR.n_prototype())

# 	# visualize
# 	cv2.imshow('image',image)
# 	cv2.waitKey(0)

# cv2.destroyAllWindows()
####################################################################

####################### test FaceRecognizer v3 ####################### 
# facenet_model_dir = './pretrained_models/FaceNet/'
# mtcnn_model_dir = './pretrained_models/MTCNN/'
# image_dir = './data/images' # you should make sure you do have images in this directory
# database_verbose = False
# db_load_path = 'test.pkl'
# image_dir = os.path.abspath(os.path.expanduser(image_dir))
# image_list = [f for f in os.listdir(image_dir) if not f.startswith('.')]

# FR = FaceRecognizer(facenet_model_dir, 
# 					mtcnn_model_dir,
# 					db_load_path=db_load_path,
# 					database_verbose=database_verbose)

# tic = time.clock()
# FR.build()
# toc = time.clock()
# build_time = toc - tic
# print('build time: {}'.format(build_time))

# for i, image_name in enumerate(image_list*3):
# 	# read image
# 	image = cv2.imread(os.path.join(image_dir,image_name))
# 	# inference
# 	tic = time.clock()
# 	bb, bb_names = FR.inference(image)
# 	toc = time.clock()
# 	print('{}: {}'.format(image_name, toc-tic))

# 	print('bb_names',bb_names)

# 	if i<len(image_list):
# 		FR.add_identity_at_current_inference(bb_names[0],'hi'+str(i))

# 	print('\n\n')

# 	# visualize
# 	cv2.imshow('image',image)
# 	cv2.waitKey(0)

# # FR.save_database('test.pkl')

# cv2.destroyAllWindows()
####################################################################

####################### test FaceRecognizer v4 ####################### 
# facenet_model_dir = './pretrained_models/FaceNet/'
# mtcnn_model_dir = './pretrained_models/MTCNN/'
# image_dir = './data/people' # you should make sure you do have images in this directory
# database_verbose = False
# db_load_path = None
# database_save_path = 'test_johnson.pkl'
# image_dir = os.path.abspath(os.path.expanduser(image_dir))
# image_name_dict = utils.parse_img_dir(image_dir)
# # print(image_name_dict)
# # print(len(image_name_dict['image']))
# # sys.exit()

# FR = FaceRecognizer(facenet_model_dir, 
# 					mtcnn_model_dir,
# 					db_load_path=db_load_path,
# 					database_verbose=database_verbose)

# tic = time.clock()
# FR.build()
# toc = time.clock()
# build_time = toc - tic
# print('build time: {}'.format(build_time))

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# for i in range(len(image_name_dict['image'])):
# 	# read image
# 	image = cv2.imread(os.path.join(image_dir,image_name_dict['image'][i]))

# 	# inference
# 	tic = time.clock()
# 	bb, bb_names, image = FR.inference(image)
# 	toc = time.clock()
# 	print('{}: {}'.format(image_name_dict['image'][i], toc-tic))

# 	print('bb_names',bb_names)

# 	FR.add_identity_at_current_inference(bb_names[0],image_name_dict['name'][i]) #update database

# 	print('\n\n')

# 	# visualize
# 	cv2.imshow('image',image)
# 	cv2.waitKey(0)

# FR.save_database(database_save_path) #update database

# cv2.destroyAllWindows()
####################################################################

####################### test FaceRecognizer v4 ####################### 
facenet_model_dir = './pretrained_models/FaceNet/'
mtcnn_model_dir = './pretrained_models/MTCNN/'
image_dir = './data/test_people' # you should make sure you do have images in this directory
database_verbose = False
db_load_path = 'test_johnson.pkl'
save_img_dir = './data/result'
save_img_dir = utils.check_dir(save_img_dir)
image_dir = os.path.abspath(os.path.expanduser(image_dir))
image_list = [f for f in os.listdir(image_dir) if not f.startswith('.')]

FR = FaceRecognizer(facenet_model_dir, 
					mtcnn_model_dir,
					db_load_path=db_load_path,
					database_verbose=database_verbose)

tic = time.clock()
FR.build()
toc = time.clock()
build_time = toc - tic
print('build time: {}'.format(build_time))

FR.rename_identity('YC','BurBurBur')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
for i, image_name in enumerate(image_list):
	# read image
	image = cv2.imread(os.path.join(image_dir,image_name))
	# inference
	tic = time.clock()
	bb, bb_names, image = FR.inference(image)
	toc = time.clock()
	print('{}: {}'.format(image_name, toc-tic))

	print('bb_names',bb_names)

	print('\n\n')

	cv2.imwrite(os.path.join(save_img_dir,'result_'+image_name),image)

	# visualize
	cv2.imshow('image',image)
	cv2.waitKey(0)

cv2.destroyAllWindows()
####################################################################