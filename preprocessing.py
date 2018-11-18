import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

import os
import re
import pickle as pkl
import collections
import codecs


def load_data(directory):
    sub_dirs = ['fake', 'real']
    label_map = {'fake': 0, 'real': 1}

    contents = []
    labels = []

    for dir in sub_dirs:
        dir_path = os.path.join(directory, dir)
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as f:
                content = f.read()
            contents.append(content)
            labels.append(label_map[dir])

    raw_data = {'contents': contents, 'labels': labels}

    out_path = os.path.join(directory, 'raw.pickle')
    with open(out_path, 'wb') as f:
        pkl.dump(raw_data, f)

def text2words(text, lemma=True):
    char_only_text = re.sub("[^a-zA-Z]", " ", text)

    stop_set = set(stopwords.words('english'))

    # tokenize, removing stop words
    words = [word for word in word_tokenize(char_only_text.lower()) if word not in stop_set]

    # lemmatization
    if lemma:
        lemma = nltk.wordnet.WordNetLemmatizer()
        words = list(map(lemma.lemmatize, words))

    return words

def process_content(directory):
	with open(directory + '/raw.pickle', 'rb') as f:
		data = pkl.load(f)
		contents = data['contents']
		labels = data['labels']
	
	for i, content in enumerate(contents):
		contents[i] = text2words(content)
		out_path = os.path.join(directory, 'processed.pickle')

	with open(out_path, 'wb') as f:
		pkl.dump({'contents': contents, 'labels': labels}, f)







