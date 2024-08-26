#!/usr/bin/env python3
# analysis.py

# Author: Sofia Carpentieri
# Supervisor: Janis Goldzycher
# Department of Computational Linguistics
# University of Zurich



import os
from collections import Counter
import re
import html
import json
from tqdm import tqdm

import tweetnlp

model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")

import spacy
from string import punctuation
from spacy.lang.en import stop_words

stop_words = stop_words.STOP_WORDS

nlp = spacy.load("en_core_web_sm")

from transformers import pipeline
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = torch.device('cuda')


classifier_target = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=0)
classifier_religion = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=0)
zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33", device=0)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




class Preprocessor:
	"""
		Class to preprocess a single post, including escaping HTML-elements, handeling board mentions and Part-of-speech-tagging with spaCy.
	"""

	def __init__(self, post:dict):
		self.post_raw = post
		self.clean = self.manual_replacement(post['com'])
		self.pos_tagged = self.pos_tag(self.clean)
		self.preprocessed = self.preprocess_post(self.clean)
		self.num_tokens = len(self.pos_tagged)


	def manual_replacement(self, post, url_replaced=True, board_mentions=True):
		# escape HTML
		text = post.replace('&#039;', "'")
		text = text.replace('<br>', ' ')
		text = text.replace(' class="quote">&gt;', "'")
		text = text.replace('&quot;', "'")
		text = text.replace('&amp;', '&')
		text = text.replace('<span', '')
		text = text.replace('</span>', "'")
		text = text.replace('<wbr>', '')
		
		# remove or replace URLs
		if url_replaced:
			text = " ".join([token if not token.startswith('http') else 'URL' for token in text.split()])
		elif url_replaced == False:
			text = " ".join([token for token in text.split() if not token.startswith('http')])
		
		# handle board mentions
		if board_mentions:
			pass
		else:
			text = " ".join([token for token in text.split() if not token[0] == '/' and not token[-1] == '/' ])

		return text


	def pos_tag(self, text: str):
		doc = nlp(text)
		tagged = dict()
		for i, token in enumerate(doc):
			tagged[i] = {'token': token.text, 'lemma': token.lemma_, 'dep': token.dep_, 'pos': token.pos_, 'tag': token.tag_}
		return tagged


	def preprocess_post(self, text: str):
		text = self.manual_replacement(text)
		text = self.pos_tag(text)
		return text




class DataAnnotator:

	"""
		Class to annotate the posts in a JSON-file and storing data and metadata in separate files.
	"""

	def __init__(self, filepath:str):

		self.filepath = filepath
		self.path_only = "/".join(self.filepath.split('/')[:-1])
		self.filename = self.filepath.split('/')[-1]

		if not os.path.isdir(f'{self.path_only}/extracted'):
			os.mkdir(os.path.join(self.path_only, 'extracted'))
		if not os.path.isdir(f'{self.path_only}/hatespeech'):
			os.mkdir(os.path.join(self.path_only, 'hatespeech'))

		self.extracted = f'{self.path_only}/extracted/extracted_{self.filename[:-5]}.json'
		self.hatespeech_binary = f'{self.path_only}/hatespeech/hatespeech_{self.filename[:-5]}.txt'
		self.religious_hate = f'{self.path_only}/hatespeech/religious_hatespeech_{self.filename[:-5]}.txt'
		self.religious_specific = f'{self.path_only}/hatespeech/religion_specific_hatespeech_{self.filename[:-5]}.txt'
		self.meta = f'{self.path_only}/meta_{self.filename}'
		
		self.country_count = list()
		self.pos_count = list()
		self.content_words = list()

		self.statistics = {'post_count': 0, 'hatespeech_count': 0, 'religious_target': 0, 'specific_religion': []}


	def extract_all(self):
		""" Extracting country, pos_tags and content words for all posts. """

		out = open(self.extracted, 'w', encoding='utf-8')

		for i, post in tqdm(enumerate(open(self.filepath, 'r', encoding='utf-8'))):
			post = json.loads(post)['posts'][0]
			# count the countries
			if 'country' in post.keys():
				self.country_count.append(post['country'])
			else:
				self.country_count.append('UNK')
			if 'com' in post:
				prepro = Preprocessor(post)
				# count pos-tags
				for i, token in prepro.pos_tagged.items():
					self.pos_count.append((prepro.pos_tagged[i]['lemma'], prepro.pos_tagged[i]['pos']))
					# count content words
					if prepro.pos_tagged[i]['lemma'] not in stop_words and prepro.pos_tagged[i]['lemma'] not in punctuation:
						self.content_words.append((prepro.pos_tagged[i]['lemma'], prepro.pos_tagged[i]['pos']))

		extracted_data = {'countries': self.country_count, 'pos_tags': self.pos_count, 'content_words': self.content_words}
		json.dump(extracted_data, out)		


	def predict_hatespeech(self):
		""" Predicting whether a post is hate speech or not. """

		labels = dict()
		out = open(self.hatespeech_binary, 'w', encoding='utf-8')
		for i, post in tqdm(enumerate(open(self.filepath, 'r', encoding='utf-8'))):
			post = json.loads(post)['posts'][0]
			if 'com' in post:
				prepro = Preprocessor(post)
				if prepro.num_tokens > 3:	# only consider posts with at least three tokens
					self.statistics['post_count'] += 1
					labels[i] = (prepro.clean, model.predict(prepro.clean))
					if model.predict(prepro.clean)['label'] == 'HATE':
						self.statistics['hatespeech_count'] += 1
						out.write(prepro.clean + '\n')

		return labels


	def count_targets(self):
		""" Predicting whether a post is targeted towards a religion. """

		out = open(self.religious_hate, 'w', encoding='utf-8')
		with open(self.hatespeech_binary, 'r', encoding='utf-8') as file:
			for i, line in tqdm(enumerate(file)):
				candidate_labels = ['Class', 'Race', 'Gender', 'Religion']
				output = classifier_target(line, candidate_labels, multi_label=False)
				prediction = output['labels'][0]
				if prediction == 'Religion':
					self.statistics['religious_target'] += 1
					out.write(f'{output["labels"]}\t{output["scores"]}\t{line}\n')


	def count_religion(self):
		""" Predicting against which religion a hate speech post is targeted. """

		out = open(self.religious_specific, 'w', encoding='utf-8')

		hypothesis_template = 'This text is about the religion {}'
		classes_verbalized = ['Atheism', 'Buddhism', 'Christanity', 'Hinduism', 'Islam', 'Judaism', 'Mormonism', 'Other']

		with open(self.religious_hate, 'r', encoding='utf-8') as file:
			for i, line in tqdm(enumerate(file)):
				post = line.split('\t')
				if len(post) < 3:
					continue
				else:
					output = zeroshot_classifier(post[2], classes_verbalized, hypothesis_template=hypothesis_template, multi_label=False)
					prediction = output['labels'][0]
					self.statistics['specific_religion'].append(prediction)
					out.write(f'{output["labels"]}\t{output["scores"]}\t{post}\n')


	def annotate_data(self):
		""" Function combining all functions above and writing the output to a meta-file. """

		self.extract_all()
		self.predict_hatespeech()
		self.count_targets()
		self.count_religion()

		with open(self.meta, 'w', encoding='utf-8') as file:
			json.dump(self.statistics, file)


class MetaAnalyser:

	"""
		Class to analyze the metadata extracted by the DataAnnotator and storing its output into separate files.
	"""

	def __init__(self, filepath:str):

		self.filepath = filepath
		self.path_chunks = "/".join(self.filepath.split('/')[:-1])
		self.path_general = "/".join(self.filepath.split('/')[:-2])
		self.filename = self.filepath.split('/')[-1]


		with open(self.filepath, 'r', encoding='utf-8') as metafile: 
			metadata = json.load(metafile)
			
			self.post_count = metadata['post_count']
			self.hatespeech_count = metadata['hatespeech_count']
			self.religious_target = metadata['religious_target']
			self.specific_religion = metadata['specific_religion']


	def get_statistics(self):
		""" Function to get statistics of the meta-file. """
		
		stats = f'{self.filename}\t{self.post_count}\t{self.hatespeech_count}\t{self.religious_target}\t{self.specific_religion}'
		return stats


	def print_statistics(self):
		""" Function to print statistics of the meta-file. """

		print('-'*30)
		print(f'Total posts: {self.post_count} posts')
		print(f'Percentage of hatespeech: {round(100*self.hatespeech_count/self.post_count)}% ({self.hatespeech_count} posts)')
		print(f'Percentage of religious hatespeech in all posts: {round(100*self.religious_target/self.post_count)}% ({self.religious_target} posts)')
		print(f'Percentage of religious hatespeech in all hatespeech posts: {round(100*self.religious_target/self.hatespeech_count)}%')
		print('Break-down of religious hatespeech:')
		for k, v in Counter(self.specific_religion).items():
			print(f'\t{k}\t{v} posts')
		print('-'*30)



def main():
	"""
		Pipeline which takes the path to the chunked files and runs the analysis over every file in the directory.
	"""

	path = input('Path to directory containing all the chunked files: ')

	outfile = f'{path}/meta_overall.tsv'
	with open(outfile, 'w', encoding='utf-8') as out:

		out.write('\n')

		for filename in os.listdir(path):
			if filename.startswith('data_'):
				print('*'*30)
				print(f'Working on file {filename}:')
				f = os.path.join(path, filename)

				d = DataAnnotator(f)
				d.annotate_data()

				m = MetaAnalyser(f'{path}/meta_{filename}')
				stats = m.get_statistics()
				out.write(f'{stats}\n')

				print('-'*30)
				print(f'Statistics for file {filename}:')
				print(d.statistics)
				m.print_statistics()
				print()
			else:
				continue





if __name__ == '__main__':

	main()




