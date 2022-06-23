import numpy as np 
import torch
import logging
import math
import lxml.etree
from collections import Counter
from collections import defaultdict
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
def get_x_y_axis(word_mfs_nfs, word_count_dict, emb):
	x_axis, y_mfs_axis, y_nfs_axis = [], [], []
	filter_count = 0
	word_count = 0
	mfs_greater_nfs_count = 0
	for word, mfs, nfs in word_mfs_nfs:
		word_count += 1
		mfs_l2_emb = np.linalg.norm(emb[mfs])
		nfs_l2_emb = np.linalg.norm(emb[nfs])
		if mfs_l2_emb > nfs_l2_emb:
			mfs_greater_nfs_count += 1
		count = word_count_dict[word]
		x_axis.append(count)
		y_mfs_axis.append(mfs_l2_emb)
		y_nfs_axis.append(nfs_l2_emb)
	print('after filtering count: ', filter_count)
	########
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

def load_ukwac(fn):
	words_ukwac = []
	sentences = []
	with open(fn, 'r', encoding = "ISO-8859-1") as sfile:
		for line in sfile:
			# print('line: ', line)
			splitLine = line.replace('  ', ' ').strip().strip('\n').split(' ')
			sentences.append(splitLine)
			for word in splitLine: 
				words_ukwac.append(word)
	return words_ukwac


if __name__ == '__main__':
	logging.info("Loading Data........")
	train_path = 'external/wsd_eval/WSD_Evaluation_Framework/' + 'Training_Corpora/SemCor/semcor.data.xml'
	keys_path = 'external/wsd_eval/WSD_Evaluation_Framework/' + 'Training_Corpora/SemCor/semcor.gold.key.txt'
	instances = load_instances(train_path, keys_path)
	instances_len = len(instances)
	logging.info("Done. Loaded %d instances from dataset" % instances_len)
	# embs = load_embs('data/vectors/lmms-large-no-norm-sense-substract-avg.vectors.txt')
	# glove = Glove.load('data/glove-sense-embeddings.model')
	word2vec = Word2Vec.load('data/word2vec.sense.model.bin')
	distinct_sense_count_dict = {}
	word_info = []
	all_word_list = []
	all_senses = []
	ambiguous_count_wn = 0
	word_count_list, word_mfs_nfs = [], []
	word_mfs_nfs_dict = defaultdict(list)
	ambiguous_word_sc_list = []
	ambiguous_word_wn_list = []
	one_sense_word_num = 0
	ambiguous_count_sc = 0
	total_word_count = 0
	alpha = 0
	mfs_not_in_embs = 0
	nfs_not_in_embs = 0
	acc = 0
	correct = 0
	wrong = 0
	sense_missing = 0

	for sent_instance in instances:
		idx_map_abs = sent_instance['idx_map_abs']
		for mw_idx, tok_idxs in idx_map_abs:
			if sent_instance['senses'][mw_idx] is None:
				continue
			"****** For the case of ALL NOUN. Check if part of speech is noun or not *******"
			if sent_instance['pos'][mw_idx] != 'NOUN':
				continue
			"************"
			
			lemma = sent_instance['lemmas'][mw_idx]
			token_word = lemma
			all_word_list.append(token_word)
			senses_num = count_senses(token_word)
			distinct_sense_count_dict[token_word] = senses_num

			for sense in sent_instance['senses'][mw_idx]:
				all_senses.append(sense)
	senses_count_dict = dict(Counter(all_senses))
	word_count_dict = dict(Counter(all_word_list))
	word_count_dict = dict(sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True))
	words = list(word_count_dict.keys())
	print("total number of words: ", len(words))
	
	for word in words:
		total_word_count += 1
		sense_count_sc = []
		sense_l2emb = []
		for sense_key in senses_count_dict.keys():
			if word == get_sk_lemma(sense_key):
				sense_count_sc.append((sense_key, senses_count_dict[sense_key]))
		sense_count_sc = sorted(sense_count_sc, key=lambda x: x[1], reverse=True)

		for sense, freq in sense_count_sc:
			# l2_sense_emb = np.linalg.norm(glove.word_vectors[glove.dictionary[sense]], ord=2)
			# l2_sense_emb = abs(glove.word_vectors[glove.dictionary[sense]]).max()
			l2_sense_emb = np.linalg.norm(word2vec[sense], ord=np.inf)
			# l2_sense_emb = np.linalg.norm(embs[sense], ord=np.inf)
			sense_l2emb.append((sense, l2_sense_emb))
		sense_l2emb = sorted(sense_l2emb, key=lambda x: x[1], reverse=True)
		mfs_pred = sense_l2emb[0][0]
		mfs_gold = sense_count_sc[0][0]
		if mfs_pred == mfs_gold:
			correct += 1
		else:
			wrong += 1
	
	total = correct + wrong
	acc = correct / total * 100
	print('total: ', total, 'total_word_count: ', total_word_count, 'acc: ', acc, 'sense missing: ', sense_missing)
	