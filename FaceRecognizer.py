from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import importlib
import pickle
import random #debug

import align.detect_face
import utils

MTCNN_PARAMS_DEFAULT = {
	'minsize': 20,
	'threshold': [0.6, 0.7, 0.7],
	'factor': 0.8 #0.709
}

class FaceDatabase(object):
	def __init__(self, verbose=False):
		# number of identities
		self._n_identity = 0
		# a list of string, with each element as the name of an identity
		self._identity_list = []
		# number of prototypes, may be different(>=) from number of identities
		self._n_prototype = 0
		# a numpy array with shape [n_prototype, embedding_dim]
		self._embs_pool = [] # WARNING: make sure all elements have the same dimension
		# an integer list correspoinding name (index of identity_list) of embeddings pool with length n_prototype
		self._embs_pool_name = []
		# verbose
		self._verbose = verbose

	def remove(self, name):
		try:
			rm_name_idx = self._identity_list.index(name)
			rm_pool_idx = [i for i,x in enumerate(self._embs_pool_name) if x==rm_name_idx]
			for i, idx in enumerate(rm_pool_idx):
				# using technique idx-i should make sure that rm_pool_idx is sorted from small to big
				del self._embs_pool[idx-i]
				del self._embs_pool_name[idx-i]
			self._identity_list.remove(name)
			self._n_prototype -= len(rm_pool_idx)
			self._n_identity -= 1
			if self._verbose:
				print('Remove "{}" from database'.format(name))
		except ValueError:
			print('Name to be removed "{}" may not be in database'.format(name))

	def add(self, name, embs):
		if name not in self._identity_list: # name not in the database
			self._embs_pool.append(embs)
			self._identity_list.append(name)
			self._n_identity += 1
			self._n_prototype += 1
			self._embs_pool_name.append(self._identity_list.index(name))
			if self._verbose:
				print('Successfully add a new identity "{}" to database'.format(name))
		else: # name already exists but add more prototypes to it
			self._embs_pool.append(embs)
			self._n_prototype += 1
			self._embs_pool_name.append(self._identity_list.index(name))
			if self._verbose:
				print('Add a new prototype to existing identity "{}"'.format(name))

	def rename(self, old_name, new_name):
		try:
			# edit identity list
			name_idx = self._identity_list.index(old_name)
			self._identity_list[name_idx] = new_name
			# no need to edit embeds_pool_name since embeds_pool_name fetch
			# name according to index over identity list
		except ValueError:
			print('No name {} in database. Cannot edit to {}'.format(old_name, new_name))

	@property
	def embs_pool(self):
		return self._embs_pool
	@property
	def embs_pool_name(self):
		return self._embs_pool_name
	@property
	def n_identity(self):
		return self._n_identity
	@property
	def identity_list(self):
		return self._identity_list
	@property
	def n_prototype(self):
		return self._n_prototype
	
class FaceRecognizer(object):
	def __init__(self, facenet_model_dir, mtcnn_model_dir,
				 resize_factor=0.7,
				 match_thresh=0.7,
				 mtcnn_params=MTCNN_PARAMS_DEFAULT,
				 db_load_path = None,
				 database_verbose=False,
				 gpu_memory_fraction=0.4):
		'''
		Arguments:
			facenet_model_dir: Directory containing the FaceNet model, with meta-file and ckpt-file
			mtcnn_model_dir: Directory containing MTCNN model, det1.npy, det2.npy, det3.npy
			resize_factor: input image will be resize to a smaller size before sent into MTCNN and FaceNet
			match_thresh: the lower bound of a valid match (if distance between 2 embeddings is higher 
						  than this threshold, then the corresponding 2 faces will be viewed as different)
			mtcnn_params: Parameters of MTCNN, a dictionary with 3 keys, 'minsize', 'threshold', 'factor'
			db_load_path: if set, load existed Face database
			database_verbose: if True, print extra messages as database updates, and vice versa
			gpu_memory_fraction: Fraction of GPU memory to be queried, in range [0,1]
		Return:
			an instance of FaceRecognizer object
		'''
		self._facenet_model_dir = os.path.abspath(os.path.expanduser(facenet_model_dir))
		self._mtcnn_model_dir = os.path.abspath(os.path.expanduser(mtcnn_model_dir))
		self._minsize = mtcnn_params['minsize']
		self._threshold = mtcnn_params['threshold']
		self._factor = mtcnn_params['factor']
		self._match_thresh = match_thresh
		self._resize_factor = resize_factor
		
		# define graph to be built on
		self._graph = tf.Graph()
		# self._graph = tf.Graph()
		# define session
		sess_conf = tf.ConfigProto()
		sess_conf.gpu_options.allow_growth = True
		sess_conf.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction		
		self._sess = tf.Session(graph=self._graph, config=sess_conf)
		# I/O of Facenet
		self._images_placeholder = []
		self._embeddings = []
		# database
		if db_load_path is not None: # load existing Face database
			db_load_path = utils.check_path(db_load_path)
			with open(db_load_path, 'rb') as handle:
				self._database = pickle.load(handle)
			print('Load FaceDatabase from {}'.format(db_load_path))
			self._database._verbose = database_verbose
		else: # create new Face database
			self._database = FaceDatabase(database_verbose)
		# MTCNN
		self._pnet = []
		self._rnet = []
		self._onet = []
		# embeddings and corresponding names obtained from current inference
		self._cur_embs = []
		self._cur_embs_name = []

	def build(self):
		'''
		FUNC: build graph of FaceNet and MTCNN
		'''
		# build FaceNet graph and load parameters
		with self._graph.as_default():
			ckpt_file = utils.get_ckpt_filenames(self._facenet_model_dir)
			print('Load FaceNet model checkpoint from {}'.format(ckpt_file))
			self._images_placeholder = tf.placeholder(tf.float32, [None, 160, 160, 3], name='input')
			network = importlib.import_module('models.inception_resnet_v1', 'inference')
			logits, _ = network.inference(self._images_placeholder, phase_train=False)
			self._embeddings = tf.nn.l2_normalize(logits, 1, 1e-10, name='embeddings')

			saver = tf.train.Saver()
			saver.restore(self._sess, os.path.join(self._facenet_model_dir, ckpt_file))

		# build MTCNN
		print('Load MTCNN model from directory {}'.format(self._mtcnn_model_dir))
		with self._graph.as_default():
			self._pnet, self._rnet, self._onet = align.detect_face.create_mtcnn(self._sess, self._mtcnn_model_dir)
	
	def inference(self,image):
		'''
		FUNC: detect faces in the image and recognize those faces according to database
		Arguments:
			image: a full image 
		Returns:
			bounding_boxes: bounding boxes containing detected faces
			self._cur_embs_name: names(identities) of faces specified by bounding_boxes. If no match 
								 found, the name will be 'Unknown_x', x is an integer, indexing all 
								 unknown faces in current inference 
		'''
		############################ face detection ############################
		img_h, img_w = image.shape[0:2]
		# resize to a smaller image and make sure iresized image is not too large
		image_resized = cv2.resize(image, (0,0), fx=self._resize_factor, fy=self._resize_factor)
		# MTCNN
		bounding_boxes, _ = align.detect_face.detect_face(image_resized, self._minsize,
														  self._pnet, self._rnet, self._onet, 
														  self._threshold, self._factor)
		# resize bounding box to original resolution
		bounding_boxes = np.array(bounding_boxes)
		bounding_boxes = (bounding_boxes/self._resize_factor).astype(np.int32)
		bounding_boxes[:,0:2] = np.maximum(bounding_boxes[:,0:2],0)
		bounding_boxes[:,2] = np.minimum(bounding_boxes[:,2], img_w)
		bounding_boxes[:,3] = np.minimum(bounding_boxes[:,3], img_h)
		
		# crop image according to bounding boxes, with each cropped image as a face
		cropped_img = []
		for i in range(len(bounding_boxes)):
			# current bounding box enclosing a face
			rect = bounding_boxes[i,0:4].astype(np.int32)
			# crop image
			cropped_img_cur = image[rect[1]:rect[3], rect[0]:rect[2], :]
			if((cropped_img_cur.shape[0]) == 0 or (cropped_img_cur.shape[1]) == 0):
				continue
			# resize images to 160x160
			cropped_img_cur = cv2.resize(cropped_img_cur, (160,160))
			# prewhiten image before sent to FaceNet
			cropped_img_cur = utils.prewhiten(cropped_img_cur)
			# append to list
			cropped_img.append(cropped_img_cur)
		# convert from list to numpy array
		n_face_found = len(cropped_img)
		print('n_face_found: ',n_face_found) #debug
		if n_face_found==0:
			print('No face is found')
			return None, None, image
		cropped_img = np.stack(cropped_img)

		############################ feed images to FaceNet ############################
		feed_dict = { self._images_placeholder: cropped_img }
		self._cur_embs = self._sess.run(self._embeddings, feed_dict=feed_dict)
		# self._cur_embs = [] #debug
		self._cur_embs_name = []
		for i in range(n_face_found):
			# self._cur_embs.append(np.array([random.random(),random.random(),random.random()])) #debug
			self._cur_embs_name.append('Unknown'+str(i))

		############################ matching ############################
		if self._database.n_identity!=0:
			for i in range(n_face_found):
				# compute distance between current face embeddings and all embeddings in database
				# remember self._database.embs_pool is a list and needed to be convert to ndarray
				rep_cur_embs = np.tile(self._cur_embs[i],(self._database.n_prototype,1))
				dist_mat = np.linalg.norm(rep_cur_embs-np.array(self._database.embs_pool), axis=1) # Euclidean distance
				# find the best match (with the smallest distance)
				min_val = np.amin(dist_mat, axis=0)
				min_idx = np.argmin(dist_mat, axis=0)
				# check if the best match good enough (below a threshold)
				if min_val<=self._match_thresh:
					try:
						# if good enough, match the detected face with corresponding identity in database
						id_idx = self._database.embs_pool_name[min_idx]
						
						match_identity = self._database.identity_list[id_idx]
						# also add this detected but recognized embeddings to database
						#self.add_identity_at_current_inference(self._cur_embs_name[i], match_identity)
						# also update current name list and database_embs used now
						self._cur_embs_name[i] = match_identity
					except:
						pass
		############################ draw for visualization ############################
		font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
		for i in range(len(bounding_boxes)):
			flag_j = not ( all(bounding_boxes[i][2:4]) )#or (rect[0]>=0 and rect[0]<img_w) or (rect[1]>=0 and rect[1]<img_h) )
			if flag_j:
				continue
			# current bounding box enclosing a face
			rect = np.array(bounding_boxes[i,0:4]).astype(np.int32)
			# draw rectangle and put text
			cv2.rectangle(image,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)
			cv2.putText(image,self._cur_embs_name[i],(rect[0],rect[1]),font,2.5,(0,0,255),3,8)

		return bounding_boxes, self._cur_embs_name, image
	
	def get_embeddings(self):
		return self._database.embs_pool, self._database.embs_pool_name
	
	def get_current_embeddings(self):
		'''
		FUNC: return all detected embeddings at current inference
		'''
		return np.array(self._cur_embs)

	def add_identity_at_current_inference(self, unknown_name, specified_name):
		'''
		FUNC: add a new identity at current inference. i.e. save the embedding specified 
			  by unknown_name to the database and give it a name, specified_name, and 
			  afterward, the embedding has a recognized identity.
		Arguments:
			unknown_name: a string, should be an unknown name obtained from 
						  current inference, e.g. 'Unknown_0' 
						  WARNING: unknown name should NOT be a known name
			specified_name: a string, name to be changed to from unknown_name
		'''
		try:
			name_idx = self._cur_embs_name.index(unknown_name)
			self._database.add(specified_name, self._cur_embs[name_idx])
		except:
			print('Embedding of unknown name "{}" to be added is not in current frame'.format(unknown_name))

	def add_identity(self, specified_name, embs):
		'''
		FUNC: add a new identity to database (not restricted to detection at current inference)
		Arguments:
			specified_name: a string, name of the identity to be added to the database
			embs: corresponding embedding of the specified name
		'''
		# DOES NOT check whether the embeddings is valid 
		self._database.add(specified_name, embs)

	def rename_identity(self, old_name, new_name):
		'''
		FUNC: change the identity with old_name to new_name
		'''
		self._database.rename(old_name, new_name)

	def remove_identity(self, name):
		'''
		FUNC: remove an identity (specified by name) from database
		'''
		self._database.remove(name)

	def list_database(self):
		'''
		FUNC: return a list of strings --> all identities in the database
		'''
		return self._database.identity_list

	def save_database(self, save_path):
		'''
		FUNC: save FaceDatabase instance to .npy file with path save_path
		'''
		save_path = os.path.abspath(os.path.expanduser(save_path))
		with open(save_path, 'wb') as output:
			pickle.dump(self._database, output)
		print('Save Face database to {}'.format(save_path))

	#FOR DEBUGGING
	def n_prototype(self):
		return self._database.n_prototype
