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
	parser.add_argument('-test_set', default='senseval3', help='Name of test set', required=False,
						choices=['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL'])
	parser.add_argument('-batch_size', type=int, default=batch_size, help='Batch size', required=False)
	parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False)
	parser.add_argument('-clf_path', help='Path to .pkl LR classifier', required=False, default='data/models/wsd_binary_ares.pkl')
	parser.add_argument('-device', default='cuda', type=str)
	parser.set_defaults(use_lemma=True)
	parser.set_defaults(use_pos=True)
	parser.set_defaults(debug=True)
	args = parser.parse_args()

	return args


def load_wsd_fw_set(wsd_fw_set_path):
	"""Parse XML of split set and return list of instances (dict)."""
	eval_instances = []
	tree = ET.parse(wsd_fw_set_path)
	for text in tree.getroot():
		for sent_idx, sentence in enumerate(text):
			inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': []}
			for e in sentence:
				inst['tokens_mw'].append(e.text)
				inst['lemmas'].append(e.get('lemma'))
				inst['senses'].append(e.get('id'))
				inst['pos'].append(e.get('pos'))

			inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])

			# handling multi-word expressions, mapping allows matching tokens with mw features
			idx_map_abs = []
			idx_map_rel = [(i, list(range(len(t.split()))))
							for i, t in enumerate(inst['tokens_mw'])]
			token_counter = 0
			for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
				idx_tokens = [i+token_counter for i in idx_tokens]
				token_counter += len(idx_tokens)
				idx_map_abs.append([idx_group, idx_tokens])

			inst['tokenized_sentence'] = ' '.join(inst['tokens'])
			inst['idx_map_abs'] = idx_map_abs
			inst['idx'] = sent_idx

			eval_instances.append(inst)

	return eval_instances


def get_id2sks(wsd_eval_keys):
	"""Maps ids of split set to sensekeys, just for in-code evaluation."""
	id2sks = {}
	with open(wsd_eval_keys) as keys_f:
		for line in keys_f:
			id_ = line.split()[0]
			keys = line.split()[1:]
			id2sks[id_] = keys
	return id2sks


def run_scorer(wsd_fw_path, test_set, results_path):
	"""Runs the official java-based scorer of the WSD Evaluation Framework."""
	cmd = 'cd %s && java Scorer %s %s' % (wsd_fw_path + 'Evaluation_Datasets/',
										  '%s/%s.gold.key.txt' % (test_set, test_set),
										  '../../../../' + results_path)
	print(cmd)
	os.system(cmd)


def chunks(l, n):
	"""Yield successive n-sized chunks from given list."""
	for i in range(0, len(l), n):
		yield l[i:i + n]


def str_scores(scores, n=3, r=5):  ###
	"""Convert scores list to a more readable string."""
	return str([(l, round(s, r)) for l, s in scores[:n]])


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


def load_lmms(npz_vecs_path):
		lmms = {}
		loader = np.load(npz_vecs_path)
		labels = loader['labels'].tolist()
		vectors = loader['vectors']
		dim = vectors[0].shape[0]
		for label, vector in list(zip(labels, vectors)):
			lmms[label] = vector
		return lmms


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
		# print('self.sk_lemmas', self.sk_lemmas)

		self.lemma_sks = defaultdict(list)
		for sk, lemma in self.sk_lemmas.items():
			self.lemma_sks[lemma].append(sk)
		self.known_lemmas = set(self.lemma_sks.keys())
		# print('self.known_lemmas', self.known_lemmas)

		self.sks_by_pos = defaultdict(list)
		for s in self.labels:
			self.sks_by_pos[self.sk_postags[s]].append(s)
		self.known_postags = set(self.sks_by_pos.keys())


	# def match_senses(self, context_vec, token_vec, lemma=None, topn=100):
	def match_senses(self, lemma=None, postag=None, topn=100):
		matches = []
		relevant_sks = []

		for sk in self.labels:
			if (lemma is None) or (self.sk_lemmas[sk] == lemma):
				if (postag is None) or (self.sk_postags[sk] == postag):
					relevant_sks.append(sk)
		matches = relevant_sks
		# print("matches", matches)
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
	# glove = Glove.load('data/glove-sense-embeddings-updated-1905.model')
	# word2vec = Word2Vec.load('data/word2vec.sense.model.bin')
	# embs = load_lmms('../bias-sense/data/lmms_2048.bert-large-cased.npz')
	embs = load_ares_txt('../senseEmbeddings/external/ares/ares_bert_large.txt')

	logging.info('Loading LR Classifer ...')
	clf = joblib.load(args.clf_path)

	'''
	Initialize various counters for calculating supplementary metrics.
	'''
	n_instances, n_correct, n_unk_lemmas, acc_sum = 0, 0, 0, 0
	n_incorrect = 0
	num_options = []
	correct_idxs = []
	failed_by_pos = defaultdict(list)

	pos_confusion = {}
	for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
		pos_confusion[pos] = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0}

	'''
	Load evaluation instances and gold labels.
	Gold labels (sensekeys) only used for reporting accuracy during evaluation.
	'''
	wsd_fw_set_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.data.xml' % (args.test_set, args.test_set)
	wsd_fw_gold_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.gold.key.txt' % (args.test_set, args.test_set)
	id2senses = get_id2sks(wsd_fw_gold_path)
	logging.info('Formating testing data')
	eval_instances = load_wsd_fw_set(wsd_fw_set_path)
	logging.info('Finish formating testing data')

	senses_vsm = SensesVSM()
	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased', output_hidden_states=True)
	model.eval()

	'''
	Iterate over evaluation instances and write predictions in WSD_Evaluation_Framework's format.
	File with predictions is processed by the official scorer after iterating over all instances.
	'''
	count = 0
	results_path = 'data/results/%d.%s.%s.key' % (int(time()), args.test_set, args.merge_strategy)
	with open(results_path, 'w') as results_f:
		for batch_idx, batch in enumerate(chunks(eval_instances, args.batch_size)):

			for sent_info in batch:
				idx_map_abs = sent_info['idx_map_abs']
				sent_bert = get_bert_embedding(sent_info['tokenized_sentence'])

				for mw_idx, tok_idxs in idx_map_abs:
					curr_sense = sent_info['senses'][mw_idx]
					'''check if a word contains sense id'''
					if curr_sense is None:
						continue
					curr_lemma = sent_info['lemmas'][mw_idx]
					curr_postag = sent_info['pos'][mw_idx]
					curr_tokens = [sent_info['tokens'][i] for i in tok_idxs]
					vec_c = torch.mean(torch.stack([sent_bert[i][1] for i in tok_idxs]), dim=0)
					vec_c = torch.cat((vec_c, vec_c), dim=0)
					relevant_senses = senses_vsm.match_senses(lemma=curr_lemma, postag=curr_postag, topn=None)
					probs = []
					for rel_sense in relevant_senses:
						sense_vec = torch.from_numpy(embs[rel_sense]).to(device)
						sim = torch.dot(vec_c, sense_vec) / (vec_c.norm() * sense_vec.norm())
						sim = sim.cpu().detach().numpy()
						pred_prob = clf.predict_proba([[sim]])[:,1][0]
						probs.append((rel_sense, pred_prob))
                        
					prob_sort = sorted(probs, key=lambda x: x[1], reverse=True)
					predict = [prob_sort[0][0]]
					print('predict: ', predict)

					if len(predict) > 0:
						results_f.write('{} {}\n'.format(curr_sense, predict[0]))
						
					'''check if our prediction(s) was correct, register POS of mistakes'''
					n_instances += 1
					wsd_correct = False
					gold_sensekeys = id2senses[curr_sense]
					print('gold_sensekeys', gold_sensekeys)

					if len(set(predict).intersection(set(gold_sensekeys))) > 0:
						n_correct += 1
						wsd_correct = True
					elif len(predict) > 0:
						n_incorrect += 1
					if len(predict) > 0:
						failed_by_pos[curr_postag].append((predict, gold_sensekeys))
					else:
						failed_by_pos[curr_postag].append((None, gold_sensekeys))

					'''register if our prediction belonged to a different POS than gold'''
					if len(predict) > 0:
						pred_sk_pos = get_sk_pos(predict[0])
						gold_sk_pos = get_sk_pos(gold_sensekeys[0])
						pos_confusion[gold_sk_pos][pred_sk_pos] += 1

					# register how far the correct prediction was from the top of our matches
					correct_idx = None
					for idx, matched_sensekey in enumerate(predict):
						if matched_sensekey in gold_sensekeys:
							correct_idx = idx
							correct_idxs.append(idx)
							break

					acc = n_correct / n_instances
					logging.info('ACC: %.3f (%d %d/%d)' % (
						acc, n_instances, sent_info['idx'], len(eval_instances)))

	precision = (n_correct / (n_correct + n_incorrect)) * 100
	recall = (n_correct / len(id2senses)) * 100
	if (precision+recall) == 0.0:
		f_score = 0.0
	else:
		f_score = 2 * precision * recall / (precision + recall)

	if args.debug:
		logging.info('Supplementary Metrics:')
		logging.info('Avg. correct idx: %.6f' % np.mean(np.array(correct_idxs)))
		logging.info('Avg. correct idx (failed): %.6f' % np.mean(np.array([i for i in correct_idxs if i > 0])))
		logging.info('Avg. num options: %.6f' % np.mean(num_options))
		logging.info('Num. unknown lemmas: %d' % n_unk_lemmas)

		logging.info('POS Failures:')
		for pos, fails in failed_by_pos.items():
			logging.info('%s fails: %d' % (pos, len(fails)))

		logging.info('POS Confusion:')
		for pos in pos_confusion:
			logging.info('%s - %s' % (pos, str(pos_confusion[pos])))

		logging.info('precision: %.1f' % precision)
		logging.info('recall: %.1f' % recall)
		logging.info('f_score: %.1f' % f_score)

	logging.info('Running official scorer ...')
	run_scorer(args.wsd_fw_path, args.test_set, results_path)		



