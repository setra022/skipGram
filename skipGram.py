from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
import pickle
import random
import utils
from scipy.special import expit
from sklearn.preprocessing import normalize

# python skipGram.py --text europarl-v7.fr-en.txt --model model.pickle
# python skipGram.py --text train3.txt --model model.pickle
# python skipGram.py --text SimLex-999\SimLex-999.txt --model model.pickle --test


__authors__ = ['author1', 'author2', 'author3']
__emails__  = ['fatherchristmoas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

number_lines = 10000


class SkipGram:
	def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5, learningRate = 1e-2):
		words = [w for sentence in sentences for w in sentence]
		self.w2id = utils.w2id(words) # word to ID mapping
		self.trainset = sentences # set of sentences
		self.vocab = set(words) # list of valid words
		self.winSize = winSize
		self.negativeRate = negativeRate
		self.nEmbed = nEmbed
		self.learningRate = learningRate
		self.W_ = np.random.randn(len(self.vocab), nEmbed)
		self.C_ = np.random.randn(len(self.vocab), nEmbed)

	def sample(self, omit):
		"""samples negative words, ommitting those in set omit"""
		negativeIds = None
		while negativeIds is None or omit.intersection(negativeIds):
			negativeWords = random.sample(self.vocab, self.negativeRate)
			negativeIds = [self.w2id[word] for word in negativeWords]
		return negativeIds
		# raise NotImplementedError('this is easy, might want to do some preprocessing to speed up')

	def train(self):
		for counter, sentence in enumerate(self.trainset):
			if counter % 100 == 0:
				if counter != 0:
					print(' > training %d of %d. Average loss : %f' % (counter, len(self.trainset), self.accLoss / self.trainWords))
				self.trainWords = 0
				self.accLoss = 0.

			sentence = list(filter(lambda word: word in self.vocab, sentence))

			for wpos, word in enumerate(sentence):
				wIdx = self.w2id[word]
				winsize = np.random.randint(self.winSize) + 1
				start = max(0, wpos - winsize)
				end = min(wpos + winsize + 1, len(sentence))

				for context_word in sentence[start:end]:
					ctxtId = self.w2id[context_word]
					if ctxtId == wIdx: continue
					negativeIds = self.sample({wIdx, ctxtId})
					self.accLoss += -np.log(utils.sigmoid(np.dot(self.W_[wIdx], self.C_[ctxtId])))
					for negativeId in negativeIds:
						self.accLoss += -np.log(utils.sigmoid(-np.dot(self.W_[wIdx], self.C_[negativeId])))
					self.trainWord(wIdx, ctxtId, negativeIds)
					self.trainWords += 1


	def trainWord(self, wordId, contextId, negativeIds):
		t = np.dot(self.W_[wordId], self.C_[contextId])
		dW = self.C_[contextId] * utils.sigmoid(-t)
		dC = self.W_[wordId] * utils.sigmoid(-t)
		# print(" + dW : ", dW)
		# print(" + dC : ", dC)
		self.W_[wordId] += dW * self.learningRate
		self.C_[contextId] += dC * self.learningRate
		for negativeId in negativeIds:
			t = np.dot(self.W_[wordId], self.C_[negativeId])
			dW = -self.C_[negativeId] * utils.sigmoid(t)
			dC = -self.W_[wordId] * utils.sigmoid(t)
			self.W_[wordId] += dW * self.learningRate
			self.C_[negativeId] += dC * self.learningRate
			# print(" - dW : ", dW)
			# print(" - dC : ", dC)
		

	def save(self, path):
		with open(path, 'wb') as handle:
			pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def similarity(self, word1, word2):
		"""
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""
		id1 = self.w2id.get(word1)
		id2 = self.w2id.get(word2)
		if id1 is None or id2 is None:
			return 1/2
		vect1, vect2 = self.W_[id1], self.W_[id2]
		score = np.dot(vect1, vect2) / (np.linalg.norm(vect1) * np.linalg.norm(vect2))
		return (1 + score) / 2

	@staticmethod
	def load(path):
		with open(path, 'rb') as handle:
			return pickle.load(handle)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')

	opts = parser.parse_args()

	# import pdb;pdb.set_trace()

	if not opts.test:
		sentences = utils.text2sentences(opts.text, number_lines)
		sg = SkipGram(sentences)
		sg.train()
		sg.save(opts.model)

	else:
		pairs = utils.loadPairs(opts.text)

		sg = SkipGram.load(opts.model)
		for a, b in pairs:
			# make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(a, b, sg.similarity(a, b))

