# senseEmbeddings-wordEmbeddings
This repository includes the code related to the "On the Curious Case of l2 norm of Sense Embeddings" paper.

Please install the requirements using:
```bash
pip install -r requirements.txt
```

To train the static sense embeddings using GloVe and Word2Vec, you may use:
```bash
python train-glove-or-word2vec-for-sense.py
```

To train static word embeddings using BERT, you may use:
```bash
python embed_annotations-words.py
```

To train static sense embeddings using LMMS on SemCor only, you may use:
```bash
python embed_annotations-sense.py
```
