import pandas as pd
import numpy as np
import re
import string

punctuation = string.punctuation
table = str.maketrans(punctuation, len(punctuation) * " ")

def text2sentences(path, number_lines):
	# feel free to make a better tokenization/pre-processing
	sentences = []
	with open(path, encoding="utf8") as f:
		lines = f.readlines()
	for count, line in enumerate(lines):
		if count == number_lines:
			break
			# print("q")
        # if (count == number_lines):
        #     break
		line = line.lower()
		line = re.sub(r'\d+', '', line)
		line = line.translate(table)
		line = re.sub(r' +', ' ', line)
		sentences.append(line.split())
	return sentences

def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'], data['word2'])
	return pairs

def w2id(words):
	count, dic = 0, {}
	for word in set(words):
		dic[word] = count
		count += 1
	return dic

def sigmoid(t):
	return 1 / (1 + np.exp(-t))