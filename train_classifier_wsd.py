import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import argparse
from time import time
from datetime import datetime
from collections import defaultdict
from collections import Counter
import xml.etree.ElementTree as ET
from functools import lru_cache
import math
import lxml.etree
from sklearn.linear_model import LogisticRegression

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.corpus import wordnet as wn
import re
from glove import Glove
from gensim.models import Word2Vec
import joblib

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

def get_args(
		emb_dim = 300,
		batch_size = 64,
		diag = False
			 ):

	parser = argparse.ArgumentParser(description='WSD Evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
						default='external/wsd_eval/WSD_Evaluation_Framework/')
	parser.add_argument('-batch_size', type=int, default=batch_size, help='Batch size', required=False)
	parser.add_argument('--dataset', default='semcor', help='Name of dataset', required=False,
						choices=['semcor', 'semcor_omsti'])
	parser.add_argument('-out_path', help='Path to .pkl classifier generated', default='data/models/wsd_binary_lmms.pkl', required=False)
	parser.add_argument('-device', default='cuda', type=str)
	parser.set_defaults(use_lemma=True)
	parser.set_defaults(use_pos=True)
	parser.set_defaults(debug=True)
	args = parser.parse_args()

	return args


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


def get_id2sks(wsd_eval_keys):
	"""Maps ids of split set to sensekeys, just for in-code evaluation."""
	id2sks = {}
	with open(wsd_eval_keys) as keys_f:
		for line in keys_f:
			id_ = line.split()[0]
			keys = line.split()[1:]
			id2sks[id_] = keys
	return id2sks


def chunks(l, n):
	"""Yield successive n-sized chunks from given list."""
	for i in range(0, len(l), n):
		yield l[i:i + n]


def str_scores(scores, n=3, r=5):  ###
	"""Convert scores list to a more readable string."""
	return str([(l, round(s, r)) for l, s in scores[:n]])


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


def load_lmms(npz_vecs_path):
		lmms = {}
		loader = np.load(npz_vecs_path)
		labels = loader['labels'].tolist()
		vectors = loader['vectors']
		dim = vectors[0].shape[0]
		for label, vector in list(zip(labels, vectors)):
			lmms[label] = vector
		return lmms


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


def get_synonyms_sk(sensekey, word):
	synonyms_sk = []
	for synset in wn.synsets(word):
		for lemma in synset.lemmas():
			if lemma.key() == sensekey:
				for lemma2 in synset.lemmas():
					synonyms_sk.append(lemma2.key())
	return synonyms_sk


def get_sk_pos(sk, tagtype='long'):
	# merges ADJ with ADJ_SAT
	if tagtype == 'long':
		type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
		return type2pos[get_sk_type(sk)]

	elif tagtype == 'short':
		type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
		return type2pos[get_sk_type(sk)]


def get_sk_type(sensekey):
	return int(sensekey.split('%')[1].split(':')[0])


def get_sk_lemma(sensekey):
	return sensekey.split('%')[0]


def get_synonyms(sensekey, word):
	for synset in wn.synsets(word):
		for lemma in synset.lemmas():
			# print('lemma.key', lemma.key())
			if lemma.key() == sensekey:
				synonyms_list = synset.lemma_names()
	return synonyms_list


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
			merged_vec = torch.from_numpy(merged_vec).to(device)
		sent_tokens_vecs.append((token, merged_vec))

	return sent_tokens_vecs


class SensesVSM(object):

	def __init__(self):
		self.labels = []
		self.matrix = []
		self.indices = {}
		self.ndims = 0
		self.labels = embs.keys()
		self.load_aux_senses()
		
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


	def match_senses(self, lemma=None, postag=None, topn=100):
		matches = []
		relevant_sks = []

		for sk in self.labels:
			if (lemma is None) or (self.sk_lemmas[sk] == lemma):
				if (postag is None) or (self.sk_postags[sk] == postag):
					relevant_sks.append(sk)
		matches = relevant_sks
		return matches[:topn]


if __name__ == '__main__':

	args = get_args()

	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because no GPU !!")
		args.device = 'cpu'

	device = torch.device(args.device)

	'''
	Load pre-trianed sense embeddings for evaluation.
	Check the dimensions of the sense embeddings to guess that they are composed with static embeddings.
	Load fastText static embeddings if required.
	'''
	# glove = Glove.load('data/glove-sense-embeddings.model')
	# word2vec = Word2Vec.load('data/word2vec.sense.model.bin')
	embs = load_lmms('data/lmms_2048.bert-large-cased.npz')
	# embs = load_ares_txt('external/ares/ares_bert_large.txt')

	if args.dataset == 'semcor':
		train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
		keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
	elif args.dataset == 'semcor_omsti':
		train_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
		keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'

	logging.info("Loading Data........")
	train_instances = load_instances(train_path, keys_path)
	train_instances_len = len(train_instances)
	logging.info("Done. Loaded %d instances from dataset" % train_instances_len)

	senses_vsm = SensesVSM()
	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased', output_hidden_states=True)
	model.eval()
	
	'''Train a binary classifier'''
	instances, labels = [], []
	for batch_idx, batch in enumerate(chunks(train_instances, args.batch_size)):
			for sent_info in batch:
				idx_map_abs = sent_info['idx_map_abs']
				sent_bert = get_bert_embedding(sent_info['tokenized_sentence'])

				for mw_idx, tok_idxs in idx_map_abs:
					if sent_info['senses'][mw_idx] is None:
						continue
					for sense in sent_info['senses'][mw_idx]:
						curr_lemma = sent_info['lemmas'][mw_idx]
						curr_postag = sent_info['pos'][mw_idx]	
						
						vec_c = torch.mean(torch.stack([sent_bert[i][1] for i in tok_idxs]), dim=0)
						vec_c = torch.cat((vec_c, vec_c), dim=0)
						correct_sense_vec = torch.from_numpy(embs[sense]).to(device)
						correct_sim = torch.dot(vec_c, correct_sense_vec) / (vec_c.norm() * correct_sense_vec.norm())
						correct_sim = correct_sim.cpu().detach().numpy()

						instances.append([correct_sim])
						labels.append(True)
						relevant_senses = senses_vsm.match_senses(lemma=curr_lemma, postag=curr_postag, topn=None)
						for rel_sense in relevant_senses:
							if rel_sense != sense:
								incorrect_sense_vec = torch.from_numpy(embs[rel_sense]).to(device)
								incorrect_sim = torch.dot(vec_c, incorrect_sense_vec) / (vec_c.norm() * incorrect_sense_vec.norm())
								incorrect_sim = incorrect_sim.cpu().detach().numpy()

								instances.append([incorrect_sim])
								labels.append(False)

	logging.info('Training Logistic Regression ...')
	clf = LogisticRegression(random_state=42)
	clf.fit(instances, labels)
	logging.info('Saving model to %s' % args.out_path)
	joblib.dump(clf, args.out_path)
