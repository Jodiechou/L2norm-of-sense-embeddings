import numpy as np 
import torch
import logging
import math
import matplotlib
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FuncFormatter
# import matplotlib.ticker as ticker
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
				# print('info_i: ', info[i])
				try: 
					float(info[i])
					x = True
				except:
					x = False	
					idx = i		
			label, vec_str = info[:idx+1], info[idx+1:]
			label = ' '.join(label)
			# print('label: ', label, 'vec_str: ', vec_str[:5])

			### Not consider multi-words
			# label, vec_str = info[0], info[1:]
			# print('label: ', label, 'vec_str: ', vec_str[:5])
			vec = np.array([float(v) for v in vec_str])
			embs[label] = vec
	return embs


def normalisation(data):
	# _range = np.max(data) - np.min(data)
	# return (data - np.mean(data)) / _range
	# std = np.std(data)
	# return (data - np.mean(data)) / std
	return data / np.mean(data)


def to_percent(y, position):
	return str(100 * y) + '%'
	

if __name__ == '__main__':
	logging.info("Loading Data........")
	# embs = load_embs('../LMMS/data/vectors/lmms-large-no-norm-sense-substract-avg.vectors.txt')
	# embs = load_embs('../LMMS/data/vectors/lmms-large-no-norm-sense.vectors.txt')
	# glove = Glove.load('data/glove-sense-embeddings-updated-1905.model')
	word2vec = Word2Vec.load('data/word2vec.sense.model.bin')
	words = []
	# print(words)
	for idx, key in enumerate(word2vec.wv.vocab):
		words.append(key)

	# rand_mat = np.random.rand(1000,300)
	partition_function_values = []
	for i in range(1000):
		values = []
		# _norm = np.linalg.norm(rand_mat[i])
		# vec = rand_mat[i] / _norm
		rand_vec = np.random.rand(300,)
		_norm = np.linalg.norm(rand_vec)
		vec = rand_vec / _norm
		for w in words:			
			# value = np.dot(glove.word_vectors[glove.dictionary[w]], vec)
			value = np.dot(word2vec[w], vec)
			values.append(value)
		values_arr = np.array(values)
		values_arr = np.exp(values_arr)
		values_sum = np.sum(values_arr)
		partition_function_values.append(values_sum)
	# 	partition_function_values.append(values_sum)
	print('len of partition_function_values', len(partition_function_values))
	partition_function_values_arr = np.array(partition_function_values)
	norm_partition_function_values = normalisation(partition_function_values_arr)
	# _mean = np.mean(norm_partition_function_values)
	# _max = np.max(norm_partition_function_values)
	# _min = np.min(norm_partition_function_values)
	# print('mean, max, min: ', _mean, _max, _min)
	
	# print('norm_partition_function_values: ', norm_partition_function_values)
	fig = plt.figure(dpi=300)
	_ = plt.hist(norm_partition_function_values, bins=30, weights=np.ones(len(norm_partition_function_values)) / len(norm_partition_function_values))
	plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
	plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.4f'))

	# plt.hist(norm_partition_function_values, bins=60, weights= [1./ len(norm_partition_function_values)] * len(norm_partition_function_values))
	# formatter = FuncFormatter(to_percent)
	# plt.gca().yaxis.set_major_formatter(formatter)
	
	plt.title("Histogram of the partition function for SGNS")
	plt.xlabel('Partition function value')
	plt.ylabel('Percentage')
	path = 'SGNS-partition-function-histogram-test-bins.png'
	plt.savefig(path, format='png', bbox_inches='tight')
	print('Saved figure to %s ' % path)



