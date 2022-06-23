import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import logging
from functools import lru_cache
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import math
from numpy.linalg import norm
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.linear_model import LogisticRegression
import joblib
from glove import Glove
from gensim.models import Word2Vec

wn_lemmatizer = WordNetLemmatizer()

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

def get_args(
		emb_dim = 300,
		diag = False
			 ):
	parser = argparse.ArgumentParser(description='Evaluation of WiC solution using LMMS for sense comparison.')
	parser.add_argument('--emb_dim', default=emb_dim, type=int)
	parser.add_argument('-eval_set', default='train', help='Evaluation set', required=False, choices=['train', 'dev', 'test'])
	parser.add_argument('-sv_path', help='Path to sense vectors', required=False, default='data/lmms_2048.bert-large-cased.npz')
	parser.add_argument('-out_path', help='Path to .pkl classifier generated', default='data/models/wic_binary_lmms2048_glove.pkl', required=False)
	parser.add_argument('-device', default='cuda', type=str)
	args = parser.parse_args()
	return args


@lru_cache()
def wn_sensekey2synset(sensekey):
	"""Convert sensekey to synset."""
	lemma = sensekey.split('%')[0]
	for synset in wn.synsets(lemma):
		for lemma in synset.lemmas():
			if lemma.key() == sensekey:
				return synset
	return None


@lru_cache()
def wn_lemmatize(w, postag=None):
	w = w.lower()
	if postag is not None:
		return wn_lemmatizer.lemmatize(w, pos=postag[0].lower())
	else:
		return wn_lemmatizer.lemmatize(w)

@lru_cache()
def wn_first_sense(lemma, postag=None):
	pos_map = {'VERB': 'v', 'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r'}
	first_synset = wn.synsets(lemma, pos=pos_map[postag])[0]
	found = False
	for lem in first_synset.lemmas():
		key = lem.key()
		if key.startswith('{}%'.format(lemma)):
			found = True
			break
	assert found
	return key


def load_wic(setname='train', wic_path='external/wic'):
	data_entries = []
	pos_map = {'N': 'NOUN', 'V': 'VERB'}
	data_path = '%s/%s/%s.data.txt' % (wic_path, setname, setname)
	for line in open(data_path):
		word, pos, idxs, ex1, ex2 = line.strip().split('\t')
		idx1, idx2 = list(map(int, idxs.split('-')))
		data_entries.append([word, pos_map[pos], idx1, idx2, ex1, ex2])

	if setname == 'test': 
		return [e + [None] for e in data_entries]

	gold_entries = []
	gold_path = '%s/%s/%s.gold.txt' % (wic_path, setname, setname)
	for line in open(gold_path):
		gold = line.strip()
		if gold == 'T':
			gold_entries.append(True)
		elif gold == 'F':
			gold_entries.append(False)

	assert len(data_entries) == len(gold_entries)
	return [e + [gold_entries[i]] for i, e in enumerate(data_entries)]


def load_weight(path):
	logging.info("Loading Model Parameters W ...")
	weight = np.load(path)
	weight = weight.f.arr_0
	logging.info('Loaded Model Parameters W')
	return weight


"""Get embeddings from files"""
def load_glove_embeddings(fn):
	embeddings = {}
	with open(fn, 'r') as gfile:
		for line in gfile:
			splitLine = line.split(' ')
			word = splitLine[0]
			vec = np.array(splitLine[1:], dtype='float32')
			embeddings[word] = vec
	return embeddings


def get_bert_embedding(sent):
	tokenized_text = tokenizer.tokenize("[CLS] {0} [SEP]".format(sent))
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [0 for i in range(len(indexed_tokens))]
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])
	tokens_tensor = tokens_tensor.to(device)
	segments_tensors = segments_tensors.to(device)
	model.to(device)
	with torch.no_grad():
		outputs = model(tokens_tensor, token_type_ids=segments_tensors)
	layers_vecs = np.sum([outputs[2][-1], outputs[2][-2], outputs[2][-3], outputs[2][-4]], axis=0) ### use the last 4 layers
	res = list(zip(tokenized_text[1:-1], layers_vecs.cpu().detach().numpy()[0][1:-1]))
	
	## merge subtokens
	sent_tokens_vecs = []
	for token in sent.split():
		token_vecs = []
		sub = []
		for subtoken in tokenizer.tokenize(token):
			encoded_token, encoded_vec = res.pop(0)
			sub.append(encoded_token)
			token_vecs.append(encoded_vec)
			merged_vec = np.array(token_vecs, dtype='float32').mean(axis=0) 
		sent_tokens_vecs.append((token, merged_vec))

	return sent_tokens_vecs


def load_lmms(npz_vecs_path):
	lmms = {}
	loader = np.load(npz_vecs_path)
	labels = loader['labels'].tolist()
	vectors = loader['vectors']
	for label, vector in list(zip(labels, vectors)):
		lmms[label] = vector
	return lmms


def get_sk_pos(sk, tagtype='long'):
	# merges ADJ with ADJ_SAT

	if tagtype == 'long':
		type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
		return type2pos[get_sk_type(sk)]

	elif tagtype == 'short':
		type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
		return type2pos[get_sk_type(sk)]


def get_sk_lemma(sensekey):
	return sensekey.split('%')[0]


def get_sk_type(sensekey):
	return int(sensekey.split('%')[1].split(':')[0])


def load_gloss_embeddings(path):
	gloss_embeddings = {}
	loader = np.load(path, allow_pickle=True)    # gloss_embeddings is loaded a 0d array
	loader = np.atleast_1d(loader.f.arr_0)       # convert it to a 1d array with 1 element
	embeddings = loader[0]				 # a dictionary, key is sense id and value is embeddings
	for key, emb in embeddings.items():
		gloss_embeddings[key] = torch.from_numpy(emb)
	logging.info("Loaded %d gloss embeddings" % len(gloss_embeddings))
	return gloss_embeddings


def load_ares_txt(path):
		sense_vecs = {}
		with open(path, 'r') as sfile:
			for idx, line in enumerate(sfile):
				if idx == 0:
					continue
				splitLine = line.split(' ')
				label = splitLine[0]
				vec = np.array(splitLine[1:], dtype='float32')
				dim = vec.shape[0]
				sense_vecs[label] = vec
		return sense_vecs


class SensesVSM(object):

	def __init__(self, vecs_path, normalize=False):
		self.vecs_path = vecs_path
		self.labels = []
		self.matrix = []
		self.indices = {}
		self.ndims = 0
		self.sense_embed = {}

		if self.vecs_path.endswith('.txt'):
			self.load_txt(self.vecs_path)

		elif self.vecs_path.endswith('.npz'):
			self.load_npz(self.vecs_path)
		self.load_aux_senses()


	def load_txt(self, txt_vecs_path):
		with open(txt_vecs_path, 'r') as vecs_f:
			for line_idx, line in enumerate(vecs_f):
				if line_idx == 0:
					continue
				elems = line.split(' ')
				self.label = elems[0]
				self.labels.append(self.label)
				self.vector = np.array(elems[1:], dtype='float32')
				self.ndims = self.vector.shape[0]
				self.sense_embed[self.label] = self.vector
		print('The dim of the txt file is',self.ndims)


	def load_npz(self, npz_vecs_path):
		loader = np.load(npz_vecs_path)
		self.labels = loader['labels'].tolist()
		self.vectors = np.array(loader['vectors'],dtype = np.float32)
		self.ndims = self.vectors.shape[1]
		for i in range(len(self.labels)):
			self.sense_embed[self.labels[i]] = self.vectors[i]
		print('The dim of the npz file is',self.ndims)


	def load_aux_senses(self):

		self.sk_lemmas = {sk: get_sk_lemma(sk) for sk in self.labels}
		self.sk_postags = {sk: get_sk_pos(sk) for sk in self.labels}
		self.lemma_sks = defaultdict(list)
		for sk, lemma in self.sk_lemmas.items():
			self.lemma_sks[lemma].append(sk)
		self.known_lemmas = set(self.lemma_sks.keys())
		self.sks_by_pos = defaultdict(list)
		for s in self.labels:
			self.sks_by_pos[self.sk_postags[s]].append(s)
		self.known_postags = set(self.sks_by_pos.keys())


	def lemma2sense_id(self):
		self.inst = defaultdict(list)
		for sk in self.labels:
			self.sk_lemmas[sk]
			self.inst[sk].append()


	def match_senses(self, currVec_c, lemma=None, postag=None, topn=100):
		matches = []
		relevant_sks = []
		distance = []
		sense_scores = []

		for sk in self.labels:
			if (lemma is None) or (self.sk_lemmas[sk] == lemma):
				if (postag is None) or (self.sk_postags[sk] == postag):
					relevant_sks.append(sk)
					sense_vec = self.sense_embed[sk]
					sim = np.dot(currVec_c, sense_vec)
					sense_scores.append(sim)
		
		matches = list(zip(relevant_sks, sense_scores))
		matches = sorted(matches, key=lambda x: x[1], reverse=True)
		return matches[:topn]


if __name__ == '__main__':

	args = get_args()
	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because no GPU !!")
		args.device = 'cpu'
	device = torch.device(args.device)
	
	word2id = dict()
	word2sense = dict()

	glove = Glove.load('data/glove-sense-embeddings.model')
	# word2vec = Word2Vec.load('data/word2vec.sense.model.bin')
	senses_vsm = SensesVSM(args.sv_path, normalize=True)
	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased', output_hidden_states=True)
	model.eval()

	logging.info('Processing sentences ...')
	instances, labels = [], []
	for wic_idx, wic_entry in enumerate(load_wic(args.eval_set, wic_path='external/wic')):
		word, postag, idx1, idx2, ex1, ex2, gold = wic_entry			

		bert_ex1 = get_bert_embedding(ex1)
		bert_ex2 = get_bert_embedding(ex2)

		# example1
		ex1_curr_word, ex1_curr_vector = bert_ex1[idx1]
		ex1_curr_lemma = wn_lemmatize(word, postag)
		ex1_curr_vector = ex1_curr_vector / np.linalg.norm(ex1_curr_vector)
		if senses_vsm.ndims == 0 or senses_vsm.ndims == 2048:
			ex1_curr_vector = np.hstack((ex1_curr_vector, ex1_curr_vector))

		ex1_curr_vector = ex1_curr_vector / np.linalg.norm(ex1_curr_vector)    
		ex1_matches = senses_vsm.match_senses(ex1_curr_vector, lemma=ex1_curr_lemma, postag=postag, topn=None)
		ex1_sense_vec = senses_vsm.sense_embed[ex1_matches[0][0]]
	
		# example2
		ex2_curr_word, ex2_curr_vector = bert_ex2[idx2]
		ex2_curr_lemma = wn_lemmatize(word, postag)
		ex2_curr_vector = ex2_curr_vector / np.linalg.norm(ex2_curr_vector)

		if senses_vsm.ndims == 0 or senses_vsm.ndims == 2048:
			ex2_curr_vector = np.hstack((ex2_curr_vector, ex2_curr_vector))

		ex2_curr_vector = ex2_curr_vector / np.linalg.norm(ex2_curr_vector)
		ex2_matches = senses_vsm.match_senses(ex2_curr_vector, lemma=ex2_curr_lemma, postag=postag, topn=None)
		ex2_sense_vec = senses_vsm.sense_embed[ex2_matches[0][0]]

		ex1_context_vec = ex1_curr_vector
		ex2_context_vec = ex2_curr_vector
		s1_sim = np.dot(ex1_context_vec, ex2_context_vec)
		s2_sim = np.dot(ex1_sense_vec, ex2_sense_vec)
		s3_sim = np.dot(ex1_context_vec, ex1_sense_vec)
		s4_sim = np.dot(ex2_context_vec, ex2_sense_vec) 

		'''Uncomment thses lines for binary classifier with l2 norm GloVe-sense or SGNS-sense'''
		# if ex1_matches[0][0] not in glove.dictionary or ex2_matches[0][0] not in glove.dictionary:
		# # if ex1_matches[0][0] not in word2vec.wv.vocab or ex2_matches[0][0] not in word2vec.wv.vocab:
		# 	continue
		# else: 
		# 	ex1_l2_sense_vec = np.linalg.norm(glove.word_vectors[glove.dictionary[ex1_matches[0][0]]], ord=2)
		# 	ex2_l2_sense_vec = np.linalg.norm(glove.word_vectors[glove.dictionary[ex2_matches[0][0]]], ord=2) 
		# 	# ex1_l2_sense_vec = np.linalg.norm(word2vec[ex1_matches[0][0]], ord=2) 
		# 	# ex2_l2_sense_vec = np.linalg.norm(word2vec[ex2_matches[0][0]], ord=2) 
		# 	instances.append([s1_sim, s2_sim, s3_sim, s4_sim, ex1_l2_sense_vec, ex2_l2_sense_vec])
		# 	labels.append(gold)
		'''********************'''

		'''Uncomment thses lines for binary classifier with l2 norm GloVe-sense or SGNS-sense and set l2norm = 0 for missing senses'''
		# if ex1_matches[0][0] not in glove.dictionary or ex2_matches[0][0] not in glove.dictionary:
		# # if ex1_matches[0][0] not in word2vec.wv.vocab or ex2_matches[0][0] not in word2vec.wv.vocab:
		# 	instances.append([s1_sim, s2_sim, s3_sim, s4_sim, 0, 0])
		# 	labels.append(gold)
		# else: 
		# 	ex1_l2_sense_vec =np.linalg.norm(glove.word_vectors[glove.dictionary[ex1_matches[0][0]]], ord=2)
		# 	ex2_l2_sense_vec =np.linalg.norm(glove.word_vectors[glove.dictionary[ex2_matches[0][0]]], ord=2) 
		# 	# ex1_l2_sense_vec = np.linalg.norm(word2vec[ex1_matches[0][0]], ord=2) 
		# 	# ex2_l2_sense_vec = np.linalg.norm(word2vec[ex2_matches[0][0]], ord=2) 
		# 	instances.append([s1_sim, s2_sim, s3_sim, s4_sim, ex1_l2_sense_vec, ex2_l2_sense_vec])
		# 	labels.append(gold)
		'''********************'''

		'''Uncomment out thses lines for original binary classifier'''
		instances.append([s1_sim, s2_sim, s3_sim, s4_sim])
		labels.append(gold)
		'''********************'''
	
	logging.info('Training Logistic Regression ...')
	clf = LogisticRegression(random_state=42)
	clf.fit(instances, labels)
	logging.info('Saving model to %s' % args.out_path)
	joblib.dump(clf, args.out_path)

	