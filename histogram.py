import numpy as np 
import torch
import logging
import math
import matplotlib
from matplotlib.ticker import PercentFormatter
matplotlib.use('Agg')
import random
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import string
from glove import Glove
from gensim.models import Word2Vec

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


def load_embs(vecs_path):
	embs = {}
	with open(vecs_path) as txt1_f:
		for line in txt1_f:
			info = line.split()
			### Consider multi-words
			idx = 0
			for i in range(len(info)):
				try: 
					float(info[i])
					x = True
				except:
					x = False	
					idx = i		
			label, vec_str = info[:idx+1], info[idx+1:]
			label = ' '.join(label)

			### Not consider multi-words
			# label, vec_str = info[0], info[1:]
			# print('label: ', label, 'vec_str: ', vec_str[:5])
			vec = np.array([float(v) for v in vec_str])
			embs[label] = vec
	return embs


def load_lmms(npz_vecs_path):
	lmms = {}
	loader = np.load(npz_vecs_path)
	labels = loader['labels'].tolist()
	vectors = loader['vectors']
	for label, vector in list(zip(labels, vectors)):
		lmms[label] = vector
	return lmms


def normalisation(data):
	return data / np.mean(data)


def to_percent(y, position):
	return str(100 * y) + '%'
	

if __name__ == '__main__':
	logging.info("Loading Data........")
	embs = load_embs('data/vectors/lmms-large-no-norm-sense.vectors.txt')
	# glove = Glove.load('data/glove-sense-embeddings.model')
	# word2vec = Word2Vec.load('data/word2vec.sense.model.bin')
	words = []
	# for idx, key in enumerate(word2vec.wv.vocab):
	# 	words.append(key)
	# for idx, key in enumerate(glove.dictionary.keys()):
	# 	words.append(key)
	for idx, key in enumerate(embs.keys()):
		words.append(key)

	partition_function_values = []
	for i in range(1000):
		values = []
		rand_vec = np.random.rand(1024,)
		# rand_vec = np.random.rand(300,)
		_norm = np.linalg.norm(rand_vec)
		vec = rand_vec / _norm
		for w in words:			
			# value = np.dot(glove.word_vectors[glove.dictionary[w]], vec)
			# value = np.dot(word2vec[w], vec)
			value = np.dot(embs[w], vec)
			values.append(value)
		values_arr = np.array(values)
		values_arr = np.exp(values_arr)
		values_sum = np.sum(values_arr)
		partition_function_values.append(values_sum)
	print('len of partition_function_values', len(partition_function_values))
	partition_function_values_arr = np.array(partition_function_values)
	norm_partition_function_values = normalisation(partition_function_values_arr)

	fig = plt.figure(dpi=300)
	plt.hist(norm_partition_function_values, bins=3, weights=np.ones(len(norm_partition_function_values)) / len(norm_partition_function_values))
	plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
	
	plt.xlabel('Partition function value')
	plt.ylabel('Percentage')
	# plt.xlim([0.95, 1.05]) # For GloVe
	# plt.xlim([0.5, 1.5]) # For SGNS
	path = 'LMMSsc-partition-function-histogram-test.png'
	plt.savefig(path, format='png', bbox_inches='tight')
	print('Saved figure to %s ' % path)



