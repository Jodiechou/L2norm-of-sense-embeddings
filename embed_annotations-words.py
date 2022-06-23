import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import argparse
from time import time

import lxml.etree
import numpy as np
import torch
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel, BertForMaskedLM

import sys, os  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from transformers_encoder import TransformersEncoder
from fairseq_encoder import FairSeqEncoder


logging.basicConfig(level=logging.INFO,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


def chunks(l, n):
	"""Yield successive n-sized chunks from given list."""
	for i in range(0, len(l), n):
		yield l[i:i + n]


def get_sense_mapping(eval_path):
	sensekey_mapping = {}
	with open(eval_path) as keys_f:
		for line in keys_f:
			id_ = line.split()[0]
			keys = line.split()[1:]
			sensekey_mapping[id_] = keys
	return sensekey_mapping


def read_xml_sents(xml_path):
	with open(xml_path) as f:
		for line_idx, line in enumerate(f):
			line = line.strip()
			if line.startswith('<sentence ') or line == '<sentence>':
				sent_elems = [line]
			elif line.startswith('<wf ') or line.startswith('<instance '):
				sent_elems.append(line)
			elif line.startswith('</sentence>'):
				sent_elems.append(line)
				try:
					yield lxml.etree.fromstring(''.join(sent_elems))
				except lxml.etree.XMLSyntaxError:
					logging.fatal('XML Parsing Error: %d' % line_idx)
					input('...')


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

def get_bert_embedding(sent):
	"""
	input: a sentence
	output: word embeddigns for the words apprearing in the sentence
	"""
	
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
	"""[1:-1] is used to get rid of CLS] and [SEP]"""
	layers_vecs = np.sum([outputs[2][-1], outputs[2][-2], outputs[2][-3], outputs[2][-4]], axis=0) ### use the last 4 layers
	res = list(zip(tokenized_text[1:-1], layers_vecs.cpu().detach().numpy()[0][1:-1]))
	
	"""merge subtokens"""
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


def gen_vecs(args, encoder, train_path, eval_path):

	token_vecs = {}
	token_embeds = {}
	vec_sum = np.zeros(1024)
	sense_mapping = get_sense_mapping(eval_path)

	logging.info('Preparing docs ...')
	instances = load_instances(train_path, keys_path)

	logging.info('Processing docs ...')
	sent_idx = 0
	n_failed = 0
	for batch_idx, batch in enumerate(chunks(instances, args.batch_size)):
		batch_t0 = time()
		for sent_info in batch:
			idx_map_abs = sent_info['idx_map_abs']
			sent_idx += 1
			sent_bert = get_bert_embedding(sent_info['tokenized_sentence'].lower())
			### Considering multi-word:
			# for mw_idx, tok_idxs in idx_map_abs:
			#     vec = np.array([sent_bert[i][1] for i in tok_idxs], dtype=np.float64).mean(axis=0)
			#     token_word = sent_info['tokens_mw'][mw_idx].lower()
			#     if token_word not in token_vecs:
			#         token_vecs[token_word] = {'n': 1, 'vec': vec}
			#     elif len(token_vecs[token_word]) < args.max_instances:
			#         token_vecs[token_word]['n'] += 1
			#         token_vecs[token_word]['vec'] += vec

			### Not considering multi-word:
			for mw_idx, tok_idxs in idx_map_abs:
				for i in tok_idxs:
					vec = np.array(sent_bert[i][1], dtype=np.float64)
					token_word = sent_info['tokens'][i].lower()
					if token_word not in token_vecs:
						token_vecs[token_word] = {'n': 1, 'vec': vec}
					
					elif len(token_vecs[token_word]) < args.max_instances:
						token_vecs[token_word]['n'] += 1
						token_vecs[token_word]['vec'] += vec    

			batch_tspan = time() - batch_t0
			progress = sent_idx/len(instances) * 100
			logging.info('PROGRESS: %.3f - %.3f sents/sec - %d/%d sents, %d sks' % (progress, args.batch_size/batch_tspan, sent_idx, len(instances), len(token_vecs)))
	logging.info('#sents final: %d' % sent_idx)
	logging.info('#vecs: %d' % len(token_vecs))
	logging.info('#failed batches: %d' % n_failed)

	## Compute the mean vector among all the word embeddings
	for token, vec_info in token_vecs.items():
		vec = vec_info['vec'] / vec_info['n']
		vec_sum += vec
		token_embeds[token] = vec
	words_num = len(token_embeds)
	vec_avg = vec_sum / words_num

	logging.info('Writing sense vecs to %s ...' % args.out_path)
	with open(args.out_path, 'w') as vecs_f:
		for token, embed in token_embeds.items():
			vec = embed - vec_avg
			vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
			vecs_f.write('%s %s\n' % (token, vec_str))


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Create sense embeddings from annotated corpora.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-nlm_id', help='HF Transfomers model name', required=False, default='bert-large-cased')
	parser.add_argument('-sense_level', type=str, default='sensekey', help='Representation Level', required=False, choices=['synset', 'sensekey'])
	parser.add_argument('-weights_path', type=str, default='', help='Path to layer weights', required=False)
	parser.add_argument('-eval_fw_path', help='Path to WSD Evaluation Framework', required=False,
						default='external/wsd_eval/WSD_Evaluation_Framework/')
	parser.add_argument('-dataset', default='semcor', help='Name of dataset', required=True, choices=['semcor', 'semcor_uwa10'])
	parser.add_argument('-batch_size', type=int, default=16, help='Batch size', required=False)
	parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length', required=False)
	parser.add_argument('-subword_op', type=str, default='mean', help='Subword Reconstruction Strategy', required=False, choices=['mean', 'first', 'sum'])
	parser.add_argument('-layers', type=str, default='-1 -2 -3 -4', help='Relevant NLM layers', required=False)
	parser.add_argument('-layer_op', type=str, default='sum', help='Operation to combine layers', required=False,
						choices=['mean', 'max', 'sum', 'concat', 'ws'])
	parser.add_argument('-max_instances', type=float, default=float('inf'), help='Maximum number of examples for each sense', required=False)
	parser.add_argument('-out_path', help='Path to resulting vector set', required=True)
	args = parser.parse_args()

	device = torch.device('cuda')

	if args.layer_op == 'ws' and args.weights_path == '':
		raise(BaseException('Weights path must be given with layer_op \'ws\''))

	if args.layer_op == 'ws':
		args.layers = 'all'  # override

	if args.layers == 'all':
		if '-base' in args.nlm_id or args.nlm_id == 'albert-xxlarge-v2':
			nmax_layers = 12 + 1
		else:
			nmax_layers = 24 + 1
		args.layers = [-n for n in range(1, nmax_layers + 1)]
	else:
		args.layers = [int(n) for n in args.layers.split(' ')]

	if args.dataset == 'semcor':
		train_path = args.eval_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
		keys_path = args.eval_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
	elif args.dataset == 'semcor_uwa10':
		train_path = 'external/wsd_eval/WSD_Evaluation_Framework/Training_Corpora/SemCor+UWA10/semcor+uwa10.data.xml'
		keys_path = 'external/wsd_eval/WSD_Evaluation_Framework/Training_Corpora/SemCor+UWA10/semcor+uwa10.gold.key.txt'

	encoder_cfg = {
		'model_name_or_path': args.nlm_id,
		'min_seq_len': 0,
		'max_seq_len': args.max_seq_len,
		'layers': args.layers,
		'layer_op': args.layer_op,
		'weights_path': args.weights_path,
		'subword_op': args.subword_op
	}

	if encoder_cfg['model_name_or_path'].split('-')[0] in ['roberta', 'xlmr']:
		encoder = FairSeqEncoder(encoder_cfg)
	else:
		encoder = TransformersEncoder(encoder_cfg)

	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased')
	model.eval()


	"""
	Pre-processing mapping between sensekeys and synsets.
	"""
	map_sk2syn = {}
	for synset in wn.all_synsets():
		for lemma in synset.lemmas():
			map_sk2syn[lemma.key()] = synset.name()


	gen_vecs(args, encoder, train_path, keys_path)
