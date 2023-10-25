# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from transformers import BertTokenizerFast
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
import pandas as pd
from pytorch_pretrained_bert import Bert_CRF
from utils import AduDataset, AduInferenceDataset, SeqTaggingPadBatchInference, tagging_scheme_config
from datetime import datetime
import nltk.data
import json
import tqdm
import pickle

LABEL_B, LABEL_I, LABEL_O = 0, 1, 2

def get_adu_spans(labels, mapping):
	assert len(labels) == len(mapping)
	adu_spans = []

	# 如果正在處理一個 adu_span，則它會是 (int, int)
	# 否則，它是 None
	# 以 adu_span 當作 FSM 的 State
	adu_span = None
	for idx in range(len(labels)):
		label = labels[idx]
		
		# 沒有在處理 adu_span：
		if adu_span == None:
			# 遇到 O：跳過
			if label == LABEL_O:
				continue
			# 遇到 I：不合理
			elif label == LABEL_I:
				raise Exception('Invalid label sequence.')
			# 遇到 B：開始處理
			elif label == LABEL_B:
				adu_span = mapping[idx]
			else:
				raise Exception('Invalid label.')
			
		# 正在處理 adu_span：
		else:
			# 遇到 O：結束目前的 adu_span
			if label == LABEL_O:
				adu_spans.append(adu_span)
				adu_span = None
			# 遇到 I：合併兩個 span
			elif label == LABEL_I:
				new_span = mapping[idx]
				assert (adu_span[0] < adu_span[1]) and (adu_span[1] <= new_span[0]) and (new_span[0] < new_span[1])
				adu_span = (adu_span[0], new_span[1])
			# 遇到 B：結束目前的 span，並開始處理新的 adu_span
			elif label == LABEL_B:
				adu_spans.append(adu_span)
				adu_span = mapping[idx]
			else:
				raise Exception('Invalid label.')

	if adu_span is not None:
		adu_spans.append(adu_span)

	return adu_spans

def test(model, iterator, device):
	model.eval()
	results = []

	with torch.no_grad():
		for i, batch in tqdm.tqdm(enumerate(iterator), total=len(iterator)):
			input_ids, masks, crf_masks, mappings = batch
			
			input_ids = input_ids.to(device)
			masks = masks.to(device)
			crf_masks = crf_masks.to(device)

			labels_pred = model(
				input_ids=input_ids,
				attention_mask=masks,
				crf_mask=crf_masks
			)['output']
			
			# Save prediction
			assert len(labels_pred) == len(mappings)
			for data_idx in range(len(mappings)):
				# 取得這筆資料的 tokens
				tokens = tokenizer.convert_ids_to_tokens(input_ids[data_idx])

				# labels 不會為句子前後的 [CLS], [SEP] 標示 (因為 crf_mask 的關係)
				# 因此要把 tokens 的頭、尾去除
				labels = labels_pred[data_idx]
				tokens = tokens[1:len(labels)+1]

				mapping = mappings[data_idx]
				adu_spans = get_adu_spans(labels, mapping)
				results.append(adu_spans)

	return results

if __name__=="__main__":
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	device = torch.device("cuda")

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--inferenceset", type=str, default="../preprocessed_data/test/parsing/jaccard_pairs")
	parser.add_argument("--inference_model", type=str, default="./checkpoints/adam-bio-20230310-acc-86.111-101113.pt")
	parser.add_argument("--tagging_scheme", type=str, default="bio", choices=["bio", "bioe", "bilou"])
	parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
	parser.add_argument("--output", type=str, required=True)

	args = parser.parse_args()
	tokenizer = BertTokenizerFast.from_pretrained(args.bert_model)

	# TODO: 現在 Test 裡只處理 BIO, 所以在這裡確保此事
	assert args.tagging_scheme == 'bio'

	# Inferencing
	sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	inference_dataset = AduInferenceDataset(args.inferenceset, is_uncased=True, tokenizer=tokenizer, sentence_tokenizer=sentence_tokenizer)

	inference_iter = data.DataLoader(dataset=inference_dataset,
										batch_size=args.batch_size,
										shuffle=False,
										num_workers=1,
										collate_fn=lambda batch: SeqTaggingPadBatchInference(batch, tokenizer))
	
	bio_num_labels, constraints = tagging_scheme_config(args.tagging_scheme)

	model = Bert_CRF.from_pretrained(
		args.bert_model, 
		bio_num_labels=bio_num_labels, 
		constraints=constraints, 
		include_start_end_transitions=True).to(device)
	model.load_state_dict(torch.load(args.inference_model))
	print('Model initialization Done.')

	print('Performing inference...')
	predictions = test(model, inference_iter, device)

	# Write inference results in a file.
	# 每一個 y_pred 的 element (y_pred[i]) 是一個 sentence
	# 每一個 sentence，則包含了數個 labels，每個 label 對應一個該句子中的 token
	print('Inference Done. Saving File.')

	# 每一個 inference_dataset 的元素，是一個句子
	# 因此長度應該符合 y_pred
	assert len(inference_dataset) == len(predictions)

	with open(args.output + '.temp', 'w') as f:
		f.write(str(predictions))

	for data_idx, adu_spans in enumerate(predictions):
		post_index, is_threads, pair_index, is_positive, comment_index, paragraph_index, sentence_index = inference_dataset.map_sentence_index(data_idx)

		if is_threads == 0:
			inference_dataset.posts[post_index].root.assign_adu_spans(paragraph_index, sentence_index, adu_spans)
		else:
			inference_dataset.posts[post_index].pairs[pair_index].threads[is_positive].comments[comment_index].assign_adu_spans(paragraph_index, sentence_index, adu_spans)

	with open(args.output, 'wb') as f:
		pickle.dump(inference_dataset.posts, f)
	
	
