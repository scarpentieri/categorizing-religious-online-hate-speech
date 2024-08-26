#!/usr/bin/env python3
# prompting.py

# Author: Sofia Carpentieri
# Supervisor: Janis Goldzycher
# Department of Computational Linguistics
# University of Zurich




from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import tweetnlp

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

hatespeech_ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")

#columns of the dataset relevant to this evaluation
desired_columns = ['comment_id', 'text', 'hatespeech', 'target_religion', 'target_religion_atheist', 'target_religion_buddhist',
				   'target_religion_christian', 'target_religion_hindu', 'target_religion_jewish', 'target_religion_mormon',
				   'target_religion_muslim', 'target_religion_other']




def extract_subset_hatespeech():
	""" Creating three subsets the bs used fo the evaluation. """

	sub = hatespeech_ds['train'][:]

	sub_hatespeech = dict()

	for k, v in sub.items():
		if k in desired_columns:
			sub_hatespeech[k] = v

	df = pd.DataFrame.from_dict(sub_hatespeech)

	# drop posts labeled as "unsure"
	df_binary = df.loc[df['hatespeech'] != 1.0]

	df_hate_yes = df_binary.loc[df['hatespeech'] == 2.0]
	df_hate_no = df_binary.loc[df['hatespeech'] == 0.0]

	df_hate = df_hate_yes[:5000].append(df_hate_no[:5000])
	df_hate['hatespeech'] = ['YES' if x == 2.0 else 'NO' for x in df_hate['hatespeech']]

	df_religious_hate = df_hate_yes.loc[(df['hatespeech'] == 2.0) & (df['target_religion'] == True)][:]
	df_nonreligious_hate = df_hate_yes.loc[(df['hatespeech'] == 2.0) & (df['target_religion'] == False)][:]

	df_religious = df_religious_hate[:5000].append(df_nonreligious_hate[:5000])
	df_religious['hatespeech'] = ['YES' if x == 2.0 else 'NO' for x in df_religious['hatespeech']]

	df_hate_religious_large = df_binary.loc[(df['hatespeech'] == 2.0) & (df['target_religion'] == True)][:]
	df_hate_religious_large['hatespeech'] = ['YES' if x == 2.0 else 'NO' for x in df_hate_religious_large['hatespeech']]

	df_hate_religious_large['targeted_group'] = 'Other'


	for i, row in df_hate_religious_large.iterrows():
		if row['target_religion_atheist'] == True:
			df_hate_religious_large['targeted_group'][i] = 'Atheism'
		elif row['target_religion_buddhist'] == True:
			df_hate_religious_large['targeted_group'][i] = 'Buddhism'
		elif row['target_religion_christian'] == True:
			df_hate_religious_large['targeted_group'][i] = 'Christianity'
		elif row['target_religion_hindu'] == True:
			df_hate_religious_large['targeted_group'][i] = 'Hinduism'
		elif row['target_religion_jewish'] == True:
			df_hate_religious_large['targeted_group'][i] = 'Judaism'
		elif row['target_religion_atheist'] == True:
			df_hate_religious_large['targeted_group'][i] = 'Mormonism'
		elif row['target_religion_muslim'] == True:
			df_hate_religious_large['targeted_group'][i] = 'Islam'
		elif row['target_religion_other'] == True:
			df_hate_religious_large['targeted_group'][i] = 'Other'


	return df_hate, df_religious, df_hate_religious_large



def predict_hatespeech(data:dict):
	true = list()
	pred = list()

	model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")

	for i, row in tqdm(data.iterrows()):
		if model.predict(row['text'])['label'] == 'HATE':
			pred.append(1)
		else:
			pred.append(0)
		if row['hatespeech'] == 'YES':
			true.append(1)
		else:
			true.append(0)


	print('Accuracy: ', accuracy_score(true, pred))	 
	print('F1 score: ', f1_score(true, pred, average='macro'))



def hatespeech_type_classification(data: dict):

	data = data.apply(lambda x: x.sample(frac=1).values)
	data = data.reset_index()  # make sure indexes pair with number of rows

	model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(model_name)

	#hypothesis = "This text is hateful against a religion."	
	#hypothesis = "This text targets persons based on a religion."	
	hypothesis = "This text targets a religion."	

	true = list()
	pred = list()

	for i, row in tqdm(data.iterrows()):

		input = tokenizer(row['text'], hypothesis, truncation=True, return_tensors="pt")
		output = model(input["input_ids"].to('cpu'))  # device = "cuda:0" or "cpu"
		prediction = torch.softmax(output["logits"][0], -1).tolist()
		label_names = ["entailment", "contradiction"]
		prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
		predicted_label = max(prediction, key=prediction.get)
		if predicted_label == 'entailment':
			pred.append(1)
		else:
			pred.append(0)
		if row['target_religion'] == True:
			true.append(1)
		else:
			true.append(0)

	print('Accuracy: ', accuracy_score(true, pred))
	print('F1 score: ', f1_score(true, pred, average='macro'))



def count_targets(data:dict):

	true = list()
	pred = list()

	classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=0)
	#candidate_labels = ['Class', 'Disability', 'Ethnicity', 'Gender', 'Race', 'Religion', 'Sexuality']
	candidate_labels = ['Class', 'Gender', 'Race', 'Religion']

	file = open('manual_inspection.txt', 'w', encoding='utf-8')

	for i, row in tqdm(data.iterrows()):
		output = classifier(row['text'], candidate_labels, multi_label=False)
		prediction = output['labels'][0]

		if prediction == 'Religion':
			pred.append(1)
		else:
			pred.append(0)

		if row['target_religion'] == True:
			true.append(1)
		else:
			true.append(0)

	print('Accuracy: ', accuracy_score(true, pred))	 
	print('F1 score: ', f1_score(true, pred, average='macro'))


def count_religion(data:dict):
	true = list()
	pred = list()

	classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=0)
	candidate_labels = ['Atheism', 'Buddhism', 'Christanity', 'Hinduism', 'Islam', 'Judaism', 'Mormonism', 'Other']
	candidate_labels_below100 = ['Atheism', 'Christanity', 'Islam', 'Judaism', 'Other']
	candidate_labels_below1000 = ['Islam', 'Judaism', 'Other']

	for i, row in tqdm(data.iterrows()):
		output = classifier(row['text'], candidate_labels, multi_label=False)
		prediction = output['labels'][0]
		pred.append(prediction)
		true.append(row['targeted_group'])

	print('Accuracy: ', accuracy_score(true, pred))	 
	print('F1 score: ', f1_score(true, pred, average='macro'))	


def count_religion_hypothesis(data:dict):
	true = list()
	pred = list()

	#hypothesis_template = "This example is about {}"
	hypothesis_template = 'This text is about the religion {}'
	#hypothesis_template = '{}'
	#hypothesis_template = 'Religion: {}'

	classes_verbalized = ['Atheism', 'Buddhism', 'Christanity', 'Hinduism', 'Islam', 'Judaism', 'Mormonism', 'Other']
	classes_verbalized_below100 = ['Atheism', 'Christanity', 'Islam', 'Judaism', 'Other']
	zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33", device=0)

	for i, row in tqdm(data.iterrows()):
		output = zeroshot_classifier(row['text'], classes_verbalized_below100, hypothesis_template=hypothesis_template, multi_label=False)
		prediction = output['labels'][0]
		pred.append(prediction)
		true.append(row['targeted_group'])

	print('Accuracy: ', accuracy_score(true, pred))	 
	print('F1 score: ', f1_score(true, pred, average='macro'))	


def random_baseline(data:dict):
	total = Counter(sub_religious_large['target_religion'])[True]
	counted = Counter(sub_religious_large['targeted_group']).most_common(7)
	
	for k, v in counted:
		percentage = f'{k}: {round(100*v/total)}%'
		print(percentage)


def majority_baseline(data:dict):
	total = Counter(sub_religious_large['target_religion'])[True]
	counted = Counter(sub_religious_large['targeted_group'])
	most_common = counted.most_common(1)[0][0]

	true = list()
	pred = list()

	for i, row in data.iterrows():
		pred.append(most_common)
		true.append(row['targeted_group'])

	print('Accuracy: ', accuracy_score(true, pred))	 
	print('F1 score: ', f1_score(true, pred, average='macro'))



if __name__ == '__main__':

	sub_hate, sub_religious, sub_religious_large = extract_subset_hatespeech()

	#predict_hatespeech(sub_hate)
	#hatespeech_type_classification(sub_religious)
	#count_targets(sub_religious)
	#count_religion(sub_religious_large)
	#count_religion_hypothesis(sub_religious_large)
	#random_baseline(sub_religious_large)
	#majority_baseline(sub_religious_large)



