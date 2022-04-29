import numpy as np 
import torch
import logging
import math
import lxml.etree
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import random
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.corpus import stopwords
import string
from glove import Glove 
# from nltk.corpus import reuters, wordnet


logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

def load_instances(train_path, keys_path):	
	"""Parse XML of split set and return list of instances (dict)."""
	train_instances = []
	sense_mapping = get_sense_mapping(keys_path)
	# tree = ET.parse(train_path)
	# for text in tree.getroot():
	text = read_xml_sents(train_path)
	for sent_idx, sentence in enumerate(text):
		inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': [], 'id': []}
		for e in sentence:
			inst['tokens_mw'].append(e.text)
			inst['lemmas'].append(e.get('lemma'))
			inst['id'].append(e.get('id'))
			inst['pos'].append(e.get('pos'))
			if 'id' in e.attrib.keys():
				inst['senses'].append(sense_mapping[e.get('id')])
			else:
				inst['senses'].append(None)

		inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])

		"""handling multi-word expressions, mapping allows matching tokens with mw features"""
		idx_map_abs = []
		idx_map_rel = [(i, list(range(len(t.split()))))
						for i, t in enumerate(inst['tokens_mw'])]
		token_counter = 0
		"""converting relative token positions to absolute"""
		for idx_group, idx_tokens in idx_map_rel:  
			idx_tokens = [i+token_counter for i in idx_tokens]
			token_counter += len(idx_tokens)
			idx_map_abs.append([idx_group, idx_tokens])
		inst['tokenized_sentence'] = ' '.join(inst['tokens'])
		inst['idx_map_abs'] = idx_map_abs
		inst['idx'] = sent_idx
		train_instances.append(inst)
	return train_instances


def get_sense_mapping(keys_path):
	sensekey_mapping = {}
	sense2id = {}
	with open(keys_path) as keys_f:
		for line in keys_f:
			id_ = line.split()[0]
			keys = line.split()[1:]
			sensekey_mapping[id_] = keys
	return sensekey_mapping


def read_xml_sents(xml_path):
	with open(xml_path) as f:
		for line in f:
			line = line.strip()
			if line.startswith('<sentence '):
				sent_elems = [line]
			elif line.startswith('<wf ') or line.startswith('<instance '):
				sent_elems.append(line)
			elif line.startswith('</sentence>'):
				sent_elems.append(line)
				yield lxml.etree.fromstring(''.join(sent_elems))


def load_lmms(npz_vecs_path):
	lmms = {}
	loader = np.load(npz_vecs_path, allow_pickle=True)
	labels = loader['labels'].tolist()
	vectors = loader['vectors']
	for label, vector in list(zip(labels, vectors)):
		lmms[label] = vector
	return lmms

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


def load_ares_txt(path):
	sense_vecs = {}
	with open(path, 'r') as sfile:
		for idx, line in enumerate(sfile):
			if idx == 0:
				continue
			splitLine = line.split(' ')
			label = splitLine[0]
			vec = np.array(splitLine[1:], dtype=float)
			dim = vec.shape[0]
			# print('self.dim', self.dim)
			sense_vecs[label] = vec
	return sense_vecs


def dict_slice(adict, start, end):
	keys = adict.keys()
	dict_slice = {}
	for k in list(keys)[start:end]:
		dict_slice[k] = adict[k]
	return dict_slice


## Get the frequency and L2 norm sense embeddings
def get_x_y_axis(word_mfs_nfs, word_count_dict, emb):
	x_axis, y_mfs_axis, y_nfs_axis = [], [], []
	filter_count = 0
	word_count = 0
	mfs_greater_nfs_count = 0
	for word, mfs, nfs in word_mfs_nfs:
		# print('word, mfs, nfs: ', word, mfs, nfs)
		# l2_emb = np.linalg.norm(glove.word_vectors[glove.dictionary[label]])
		word_count += 1
		mfs_l2_emb = np.linalg.norm(emb[mfs])
		nfs_l2_emb = np.linalg.norm(emb[nfs])
		if mfs_l2_emb > nfs_l2_emb:
			mfs_greater_nfs_count += 1
		count = word_count_dict[word]
		# squared_norm_mfs_emb = pow(mfs_l2_emb, 2)
		# squared_norm_nfs_emb = pow(nfs_l2_emb, 2)
		# if count > 5000:
		# 	filter_count += 1
		# 	print('word: ', word, 'count[word]: ', word_count_dict[word])
		# 	continue
	# 	count = count_dict[label]
		# count = math.log(count)
		x_axis.append(count)
		y_mfs_axis.append(mfs_l2_emb)
		y_nfs_axis.append(nfs_l2_emb)
	print('after filtering count: ', filter_count)
	########
		
		# word_freq = count_dict[label]
		# if word_freq > 400:
		# 	# print("word's freq over 1250: ", label)
		# 	continue
		# # word_freq = math.log(word_freq)
		# distinct_sense_num = distinct_sense_count_dict[label]
		# # distinct_sense_num = math.log(distinct_sense_num)
		# x_axis.append(word_freq)
		# y_axis.append(distinct_sense_num)
	# print('word_count: ', word_count, 'mfs_greater_nfs_count: ', mfs_greater_nfs_count)
	return x_axis, y_mfs_axis, y_nfs_axis


def load_glove_embeddings(fn):
	embeddings = {}
	with open(fn, 'r') as gfile:
		for line in gfile:
			splitLine = line.split(' ')
			word = splitLine[0]
			vec = np.array(splitLine[1:], dtype='float32')
			vec = torch.from_numpy(vec)
			embeddings[word] = vec
	return embeddings

### obtain the number of senses given an ambiguous word
def count_senses(word):
	return len(wn.synsets(word))


def get_sk_lemma(sensekey):
	return sensekey.split('%')[0]


if __name__ == '__main__':
	train_path = 'external/wsd_eval/WSD_Evaluation_Framework/' + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
	keys_path = 'external/wsd_eval/WSD_Evaluation_Framework/' + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'
	logging.info("Loading Data........")
	instances = load_instances(train_path, keys_path)
	instances_len = len(instances)
	logging.info("Done. Loaded %d instances from dataset" % instances_len)
	# embs = load_lmms('../eval_static_emb_bias/data/lmms_2348.bert-large-cased.fasttext-commoncrawl.npz')
	# embs = load_lmms('../bias-sense/data/lmms_2048.bert-large-cased.npz')
	# embs = load_lmms('../senseEmbeddings/external/lmms/lmms_1024.bert-large-cased.npz')
	# embs = load_ares_txt("../senseEmbeddings/external/ares/ares_bert_large.txt")
	# glove_embs = load_glove_embeddings('../senseEmbeddings/external/glove/glove.840B.300d.txt')
	# embs = {}
	# emb_keys = list(glove_embs.keys())
	# print('embeddings keys: ', emb_keys[:20])
	# random.seed(52)
	# random.shuffle(emb_keys)
	# print('embedding keys after shuffle: ', emb_keys[:20])
	# for key in emb_keys:
	# 	embs[key] = glove_embs[key]
	# embs = dict_slice(embs, 0, 1000000)
	# print('Got 100k glove embeddings')

	# glove = Glove.load('data/glove.model')
	# glove.word_vectors[glove.dictionary['long']]

	embs = load_embs('../LMMS/data/vectors/lmms-large-no-norm-sense-substract-avg.vectors.txt')
	# embs = load_embs('../LMMS/data/vectors/bert-large-no-norm-word-substract-avg.vectors.txt')
	consider_words = False
	all_senses = []
	all_word_list = []
	distinct_sense_count_dict = {}
	word_info = []
	punc = '''!()-[]{};:'"\,''<>./?@#$%^&*_~``'''
	
	for sent_instance in instances[:10]:
		idx_map_abs = sent_instance['idx_map_abs']
		# print('idx_map_abs: ', idx_map_abs)
		for mw_idx, tok_idxs in idx_map_abs:
			if sent_instance['senses'][mw_idx] is None:
				continue
			lemma = sent_instance['lemmas'][mw_idx]
			token_word = lemma
			all_word_list.append(token_word)
			senses_num = count_senses(token_word)
			distinct_sense_count_dict[token_word] = senses_num
			
			for sense in sent_instance['senses'][mw_idx]:
				all_senses.append(sense)
	word_count_list, word_mfs_nfs = [], []
	word_count_dict = dict(Counter(all_word_list))
	# word_count_dict = dict(sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True))
	senses_count_dict = dict(Counter(all_senses))
	one_sense_word_num = 0
	ambiguous_word_num = 0
	for word in word_count_dict.keys():
		sense_count_list = []
		for sense_key in senses_count_dict.keys():
			if word == get_sk_lemma(sense_key):
				sense_count_list.append((sense_key, senses_count_dict[sense_key]))
				# matches = list(zip(relevant_sks, sense_scores))
		sense_count_list = sorted(sense_count_list, key=lambda x: x[1], reverse=True)
		# print('sense_count_list: ', sense_count_list)
		### sense_count_list[0][0] and sense_count_list[1][0] are the MFS and NFS, respectively
		if len(sense_count_list) > 1:
			word_mfs_nfs.append((word, sense_count_list[0][0], sense_count_list[1][0]))
			ambiguous_word_num += 1
			mfs = sense_count_list[0][0]
			nfs = sense_count_list[1][0]
            print('1 mfs: ', mfs, 'nfs: ', nfs)
			if mfs or nfs not in embs.keys():
				# print('1 mfs: ', mfs, 'nfs: ', nfs)
				continue
			mfs_l2_emb = np.linalg.norm(embs[mfs])
			nfs_l2_emb = np.linalg.norm(embs[nfs])
			word_info.append((word, distinct_sense_count_dict[word], mfs,
			nfs, mfs_l2_emb, nfs_l2_emb, 1))
			
		else:
			word_mfs_nfs.append((word, sense_count_list[0][0], sense_count_list[0][0]))
			one_sense_word_num += 1
			mfs = sense_count_list[0][0]
			nfs = sense_count_list[0][0]
			if mfs or nfs not in embs.keys():
				# print('0 mfs: ', mfs, 'nfs: ', nfs)
				continue
			mfs_l2_emb = np.linalg.norm(embs[mfs])
			nfs_l2_emb = np.linalg.norm(embs[nfs])
			word_info.append((word, distinct_sense_count_dict[word], mfs, 
			nfs, mfs_l2_emb, nfs_l2_emb, 0))
	print('one_sense_word_num: ', one_sense_word_num)
	print('ambiguous_word_num: ', ambiguous_word_num)
	with open('word_info.txt', 'w') as info_f:
		for word, distinct_sense_num, mfs, nfs, l2_mfs, l2_nfs, ambiguous_label in word_info:
			info_f.write('%s %d %s %s %f %f %d\n' % (word, distinct_sense_num, mfs, nfs, l2_mfs, l2_nfs, ambiguous_label))
	print('Saved word_info.txt')
	"""
	x_axis, y_mfs_axis, y_nfs_axis = get_x_y_axis(word_mfs_nfs, word_count_dict, embs)
	fig = plt.figure(dpi=300)	
	x_axis = np.array(x_axis)
	y_mfs_axis = np.array(y_mfs_axis)
	y_nfs_axis = np.array(y_nfs_axis)
	# plt.scatter(x_axis, y_axis)
	# plt.plot(x_axis, y_mfs_axis, label='Squared l2 norm of MFS embeddings')
	# plt.plot(x_axis, y_nfs_axis, label='Squared l2 norm of NFS embeddings')
	plt.scatter(x_axis, y_mfs_axis, label='Squared l2 norm of MFS embeddings')
	plt.scatter(x_axis, y_nfs_axis, label='Squared l2 norm of NFS embeddings')
	
	# plt.xlabel('Log Word Frequency')
	# plt.ylabel('Squared L2 Norm Word Embeddings')
	plt.xlabel('Frequency of words')
	plt.ylabel('L2 Norm Sense Embeddings')
	plt.legend()
	path = 'mfs-vs-nfs-filter.png'
	# path = 'test.png'
	plt.savefig(path, format='png', bbox_inches='tight')
	print('Saved figure to %s ' % path)
	"""


