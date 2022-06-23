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
from gensim.models import Word2Vec
# from nltk.corpus import reuters, wordnet


logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

def load_instances(train_path, keys_path):	
	"""Parse XML of split set and return list of instances (dict)."""
	train_instances = []
	sense_mapping = get_sense_mapping(keys_path)
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
			sense_vecs[label] = vec
	return sense_vecs


def dict_slice(adict, start, end):
	keys = adict.keys()
	dict_slice = {}
	for k in list(keys)[start:end]:
		dict_slice[k] = adict[k]
	return dict_slice


## Get the frequency and L2 norm sense embeddings
def get_x_y_axis(count_dict, emb):
# def get_x_y_axis(count_dict, glove):
#######
### word_freq vs. distinct_sense_num #####
# def get_x_y_axis(count_dict, distinct_sense_count_dict):
######
	x_axis, y_axis = [], []
	filter_count = 0
	########
	for label in count_dict.keys():
		# if label not in glove.dictionary:
		# 	continue
		if label not in embs.keys():
			continue
		# l2_emb = np.linalg.norm(glove.word_vectors[glove.dictionary[label]])
		l2_emb = np.linalg.norm(emb[label])
		squared_norm_emb = pow(l2_emb, 2)
		#### For GloVe ########
		# if squared_norm_emb > 25:
		# # if squared_norm_emb > 30000:
		# 	filter_count += 1
		# 	print('label: ', label, 'count[label]: ', count_dict[label])
		# 	continue
		############
		count = count_dict[label]
		count = math.log(count)
		x_axis.append(count)
		y_axis.append(squared_norm_emb)
	# print('after filtering count: ', filter_count)
	########

		######### word_freq vs. distinct_sense_num ########
		# word_freq = count_dict[label]
		# # word_freq = math.log(word_freq)
		# distinct_sense_num = distinct_sense_count_dict[label]
		# # distinct_sense_num = math.log(distinct_sense_num)
		# x_axis.append(word_freq)
		# y_axis.append(distinct_sense_num)
		###########
	return x_axis, y_axis


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


def count_senses(word):
	return len(wn.synsets(word))


if __name__ == '__main__':
	train_path = 'external/wsd_eval/WSD_Evaluation_Framework/' + 'Training_Corpora/SemCor/semcor.data.xml'
	keys_path = 'external/wsd_eval/WSD_Evaluation_Framework/' + 'Training_Corpora/SemCor/semcor.gold.key.txt'
	logging.info("Loading Data........")
	instances = load_instances(train_path, keys_path)
	instances_len = len(instances)
	logging.info("Done. Loaded %d instances from dataset" % instances_len)

	# glove = Glove.load('data/glove-sense-embeddings.model')
	# word2vec = Word2Vec.load('data/word2vec.sense.model.bin')
	# glove.word_vectors[glove.dictionary['long']]
	# embs = load_embs('data/vectors/bert-large-no-norm-word.vectors.txt')
	# embs = load_lmms('data/lmms_2048.bert-large-cased.npz')
	embs = load_embs('data/vectors/lmms-large-no-norm-sense.vectors.txt')
	consider_words = False
	all_senses = []
	all_word_list = []
	distinct_sense_count_dict = {}
	punc = '''!()-[]{};:'"\,''<>./?@#$%^&*_~``'''
	
	for sent_instance in instances:
		idx_map_abs = sent_instance['idx_map_abs']
		for mw_idx, tok_idxs in idx_map_abs:
			######### word_freq vs. distinct_sense_num ########
			# if sent_instance['senses'][mw_idx] is None:
			# 	continue
			# # sense = sent_instance['senses'][mw_idx][0]
			# # token_word = get_sk_lemma(sense)
			# lemma = sent_instance['lemmas'][mw_idx]
			# token_word = lemma
			# if token_word in stopwords.words('english'):
			# 	continue
			# if token_word == 'person':
			# 	continue
			# # if token_word in punc: 
			# # 	continue
			# senses_num = count_senses(token_word)
			# distinct_sense_count_dict[token_word] = senses_num
			# all_word_list.append(token_word)
			# s =  wn.synsets(token_word)
			# print('token_word: ', token_word, 'senses_num: ', senses_num, 'synsets:', s)
			########
			##### l2 norm embeddings vs. word/sense frequency #########
			if consider_words:
				for j in tok_idxs:
					token_word = sent_instance['tokens'][j].lower()
					if token_word in stopwords.words('english'):
						continue
					if token_word in punc: 
						continue
					if token_word == "''" or token_word == "'s":
						continue
					all_word_list.append(token_word)
			else:
				if sent_instance['senses'][mw_idx] is None:
					continue
				for sense in sent_instance['senses'][mw_idx]:
					all_senses.append(sense)
			########
	word_count_list, top_word_count_list, top_distinct_sense_count_dict = [], [], []
	if consider_words:
		word_count_dict = dict(Counter(all_word_list))
		word_count_dict = dict(sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True))
		distinct_sense_count_dict = dict(sorted(distinct_sense_count_dict.items(), key=lambda item: item[1], reverse=True))
		top_word_count_dict = dict_slice(word_count_dict, 0, 10)
		# top_distinct_sense_count_dict = dict_slice(distinct_sense_count_dict, 0, 10)
		x_axis, y_axis = get_x_y_axis(word_count_dict, embs)
		# x_axis, y_axis = get_x_y_axis(word_count_dict, glove)
		# x_axis, y_axis = get_x_y_axis(word_count_dict, distinct_sense_count_dict)
	else:
		senses_count_dict = dict(Counter(all_senses))
		senses_count_dict = dict(sorted(senses_count_dict.items(), key=lambda item: item[1], reverse=True))
		top_sense_count_dict = dict_slice(senses_count_dict, 0, 50)

		## Check the frequency and L2 norm sense embeddings
		# x_axis, y_axis = get_x_y_axis(senses_count_dict, glove)
		x_axis, y_axis = get_x_y_axis(senses_count_dict, embs)
	
	fig = plt.figure(dpi=600)
	x_axis = np.array(x_axis)
	y_axis = np.array(y_axis)
	p_correlation = np.corrcoef(x_axis, y_axis)
	print('p_correlation between x and y: ', p_correlation)
	plt.scatter(x_axis, y_axis)
	plt.xlabel('Log Word Frequency')
	plt.ylabel('Squared L2 Norm BERT Word Embeddings')
	# plt.xlabel('frequency of words')
	# plt.ylabel('number of distinct senses')
	# plt.ylabel('L2 Norm Sense Embeddings')  
	path = 'l2_embeddings-frequency-lmms-sc-sense-without-substract-avg.png'
	plt.savefig(path, format='png', bbox_inches='tight')
	print('Saved figure to %s ' % path)



