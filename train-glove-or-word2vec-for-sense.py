from glove import Corpus, Glove # creating a corpus object
from gensim.models import Word2Vec
import lxml.etree
import logging

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


def get_sense_mapping(keys_path):
	sensekey_mapping = {}
	sense2id = {}
	with open(keys_path) as keys_f:
		for line in keys_f:
			id_ = line.split()[0]
			keys = line.split()[1:]
			sensekey_mapping[id_] = keys

	return sensekey_mapping

if __name__ == '__main__':

	logging.info("Loading Data........")
	wsd_fw_path = 'external/wsd_eval/WSD_Evaluation_Framework/'
	train_path = wsd_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
	keys_path = wsd_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
	instances = load_instances(train_path, keys_path)
	# instantiate the corpus
	corpus = Corpus() 
	# this will create the word co-occurence matrix 
	senses_in_instances = []
	for sent in instances:
		sense_temp = []
		for sense in sent['senses']:
			if sense == None:
				continue
			for sense_ in sense:
				sense_temp.append(sense_)
		senses_in_instances.append(sense_temp)
			
	sentences = senses_in_instances

	" *******Train GloVe sense embeddings******* "
	# corpus.fit(sentences, window=10)
	# # instantiate the model
	# glove = Glove(no_components=300, learning_rate=0.05)
	# # and fit over the corpus matrix
	# glove.fit(corpus.matrix, epochs=30, no_threads=2)
	# # finally we add the vocabulary to the model
	# glove.add_dictionary(corpus.dictionary)
	# glove.save('data/glove-sense-embeddings.model')
	" ********* "

	" *******Load GloVe sense embeddings******* "
	# glove = Glove.load('data/glove-sense-embeddings.model')

	" *******Train word2vec sense embeddings******* "
	# train model
	word2vec_model = Word2Vec(sentences, min_count=1, size=300)
	# summarize the loaded model
	print(word2vec_model)
	# summarize vocabulary
	words = list(word2vec_model.wv.vocab)
	print(words)
	# access vector for one word
	print(word2vec_model['mastermind%2:31:00::'])
	# save model
	word2vec_model.save('data/word2vec.sense.model.bin')
	" *********** "

	" *******Load GloVe sense embeddings******* "
	# # load model
	# word2vec_model = Word2Vec.load('data/word2vec.sense.model.bin')
	# print(word2vec_model)
	" *********** "

