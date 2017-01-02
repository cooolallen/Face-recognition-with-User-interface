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

facenet_model_dir = './pretrained_models/FaceNet/'
mtcnn_model_dir = './pretrained_models/MTCNN/'
image_dir = './data/people' # you should make sure you do have images in this directory
database_verbose = False
db_load_path = None
database_save_path = 'test_johnson.pkl'
image_dir = os.path.abspath(os.path.expanduser(image_dir))
image_name_dict = utils.parse_img_dir(image_dir)
print(image_name_dict)
# print(len(image_name_dict['image']))
# sys.exit()

FR = FaceRecognizer(facenet_model_dir, 
					mtcnn_model_dir,
					db_load_path=db_load_path,
					database_verbose=database_verbose)

tic = time.clock()
FR.build()
toc = time.clock()
build_time = toc - tic
print('build time: {}'.format(build_time))

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
for i in range(len(image_name_dict['image'])):
	# read image
	image = cv2.imread(os.path.join(image_dir,image_name_dict['image'][i]))

	# inference
	tic = time.clock()
	bb, bb_names, image = FR.inference(image)
	toc = time.clock()
	print('{}: {}'.format(image_name_dict['image'][i], toc-tic))

	print('bb_names',bb_names)

	if bb_names is not None:
		if len(bb_names)==1:
			FR.add_identity_at_current_inference(bb_names[0],image_name_dict['name'][i]) #update database
		else:
			print('To build database, number of faces per image should be 1')

	print('\n\n')

	# visualize
	cv2.imshow('image',image)
	cv2.waitKey(0)

FR.save_database(database_save_path) #update database

cv2.destroyAllWindows()