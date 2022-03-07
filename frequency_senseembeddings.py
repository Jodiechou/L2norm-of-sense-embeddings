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
	print('dict_slice: ', dict_slice)
	return dict_slice


## Get the frequency and L2 norm sense embeddings
def get_x_y_axis(count_dict, emb):
	x_axis, y_axis = [], []
	filter_count = 0
	# print('len of count_dict: ', len(count_dict))
	for label in count_dict.keys():
		l2_emb = np.linalg.norm(emb[label])
		squared_norm_emb = pow(l2_emb, 2)
		# if squared_norm_emb > 30000:
		# 	filter_count += 1
		# 	print('label: ', label, 'count[label]: ', count_dict[label])
		# 	continue
		count = count_dict[label]
		count = math.log(count)
		x_axis.append(count)
		y_axis.append(squared_norm_emb)
	# print('after filtering count: ', filter_count)
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


if __name__ == '__main__':
	train_path = 'external/wsd_eval/WSD_Evaluation_Framework/' + 'Training_Corpora/SemCor/semcor.data.xml'
	keys_path = 'external/wsd_eval/WSD_Evaluation_Framework/' + 'Training_Corpora/SemCor/semcor.gold.key.txt'
	logging.info("Loading Data........")
	instances = load_instances(train_path, keys_path)
	instances_len = len(instances)
	logging.info("Done. Loaded %d instances from dataset" % instances_len)
	# embs = load_lmms('../eval_static_emb_bias/data/lmms_2348.bert-large-cased.fasttext-commoncrawl.npz')
	# embs = load_lmms('../bias-sense/data/lmms_2048.bert-large-cased.npz')
	# embs = load_lmms('../senseEmbeddings/external/lmms/lmms_1024.bert-large-cased.npz')
	# embs = load_ares_txt("../senseEmbeddings/external/ares/ares_bert_large.txt")
	glove_embs = load_glove_embeddings('../senseEmbeddings/external/glove/glove.840B.300d.txt')
	embs = {}
	emb_keys = list(glove_embs.keys())
	print('embeddings keys: ', emb_keys[:20])
	random.seed(52)
	random.shuffle(emb_keys)
	print('embedding keys after shuffle: ', emb_keys[:20])
	for key in emb_keys:
		embs[key] = glove_embs[key]
	embs = dict_slice(embs, 0, 1000000)
	print('Got 100k glove embeddings')

	# embs = load_embs('../LMMS/data/vectors/lmms-large-no-norm-sense-substract-avg.vectors.txt')
	# embs = load_embs('../LMMS/data/vectors/bert-large-no-norm-word-substract-avg.vectors.txt')
	consider_words = True
	device = torch.device('cuda')
	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased')
	model.eval()
	all_senses = []
	all_word_list = []
	punc = '''!()-[]{};:'"\,''<>./?@#$%^&*_~``'''
	
	for sent_instance in instances:
		idx_map_abs = sent_instance['idx_map_abs']
		# print('idx_map_abs: ', idx_map_abs)
		for mw_idx, tok_idxs in idx_map_abs:
			if consider_words:
				for j in tok_idxs:
					token_word = sent_instance['tokens'][j].lower()
					# token_word = sent_instance['tokens_mw'][mw_idx].lower()
					# print('stop words: ', stopwords.words('english'))
					if token_word in stopwords.words('english'):
						continue
					if token_word in punc: 
						continue
					# if token_word == "''" or token_word == "'s" or token_word == 'one' or token_word == 'would' or token_word == 'said' or token_word == "n't" or token_word == 'could':
						# continue
					if token_word == "''" or token_word == "'s":
						continue
					if token_word not in embs.keys():
						continue
					all_word_list.append(token_word)
			else:
				if sent_instance['senses'][mw_idx] is None:
					continue
				for sense in sent_instance['senses'][mw_idx]:
					# if sense == 'be%2:42:03::' or sense == 'person%1:03:00::' or sense == 'be%2:42:06::' or sense == 'not%4:02:00::' or sense == 'say%2:32:00::' or sense == 'group%1:03:00::' or sense == 'have%2:40:00::' or sense == 'location%1:03:00::' or  sense == 'be%2:42:05::' or  sense == 'be%2:42:00::' or sense == 'be%2:42:04::':
					# 	continue
					# if sense == 'be%2:42:03::' or sense == 'be%2:42:06::' or sense == 'not%4:02:00::' or  sense == 'be%2:42:05::' or  sense == 'be%2:42:00::' or sense == 'be%2:42:04::':
					# 	continue
					### No filtering senses
					all_senses.append(sense)
					# for j in tok_idxs:
					# 	token_word = sent_instance['tokens'][j]
					# 	all_word_list.append(token_word)
	word_count_list, top_word_count_list = [], []
	if consider_words:
		word_count_dict = dict(Counter(all_word_list))
		word_count_dict = dict(sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True))
		# print('word_count_dict: ', word_count_dict)
		# for w, c in word_count_dict.items():
		# 	word_count_list.append([w, c])
		# top_word_count_list = word_count_list[-50:]	
		top_word_count_dict = dict_slice(word_count_dict, 0, 50)
		print('top_word_count_dict: ', top_word_count_dict)
		x_axis, y_axis = get_x_y_axis(word_count_dict, embs)
	else:
		senses_count_dict = dict(Counter(all_senses))
		senses_count_dict = dict(sorted(senses_count_dict.items(), key=lambda item: item[1], reverse=True))
		top_sense_count_dict = dict_slice(senses_count_dict, 0, 50)
		print('top_sense_count_dict: ', top_sense_count_dict)

		## Check the frequency and L2 norm sense embeddings
		x_axis, y_axis = get_x_y_axis(senses_count_dict, embs)
		
	x_axis = np.array(x_axis)
	y_axis = np.array(y_axis)
	plt.scatter(x_axis, y_axis)
	plt.xlabel('Log Word Frequency')
	plt.ylabel('Squared L2 Norm Word Embeddings')
	# plt.ylabel('L2 Norm Sense Embeddings')  
	# path = 'l2_embeddings-frequency-lmms-word-nofiltering-nosquared-substract-avg.png'
	path = 'l2_embeddings-frequency-glove-word.png'
	# path = 'test.png'
	plt.savefig(path, format='png')
	print('Saved figure to %s ' % path)
	


