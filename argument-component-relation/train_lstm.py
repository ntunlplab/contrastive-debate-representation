# -*- coding: utf-8 -*-
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from transformers import BertTokenizer, RobertaTokenizer, LongformerTokenizer
from transformers import BertModel, RobertaModel, LongformerModel
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
import pandas as pd
from models_lstm import Model_Adu_Relation
from utils_lstm import AduRelationDataset, PadBatch
from datetime import datetime
from statistics import mean 
import nltk.data

def LabelTensor(task):
	if task == 'link': # link -> BCELoss -> FloatTensor
		return torch.FloatTensor
	else:			   # link_type -> CELoss -> LongTensor
		return torch.LongTensor 

def count_predictions(feats, task):
	prediction_count = [0, 0]
	prediction_list = []
	if task == 'link':
		feats_round = feats.round()
		for feat in feats_round:
			item = int(feat.item())
			assert item == 0 or item == 1
			prediction_count[item] += 1
			prediction_list.append(item)
	else:
		batch_size = feats.shape[0]
		feats_argmax = feats.argmax(1)
		assert feats_argmax.shape == (batch_size,)
		for feat in feats_argmax:
			item = int(feat.item())
			assert item == 0 or item == 1
			prediction_count[item] += 1
			prediction_list.append(item)

	return prediction_count, prediction_list

def is_close(a, b):
	return abs(a - b) < 1e-4

def get_accuracy(pred, target):
	scores = [1 if is_close(a, b) else 0 for a, b in zip(pred, target)]
	return sum(scores) / len(scores)

def train(e, args, model, iterator, criterion, optimizer, scheduler, device):
	model.train()
	optimizer.zero_grad()
	log_interval = args.log_interval
	losses = 0
	prediction_count = [0, 0]
	prediction_count_per_log = [0, 0]

	prediction_list = []
	labels_list = []
	prediction_list_per_log = []
	labels_list_per_log = []

	for i, batch in enumerate(iterator):
		post_1_tensors, post_2_tensors, post_1_mask, post_2_mask, adu_1_tensors, adu_2_tensors, adu_orders, labels, is_inners_embed = batch

		adu_1_tensors = adu_1_tensors.to(device)
		adu_2_tensors = adu_2_tensors.to(device)
		post_1_tensors = post_1_tensors.to(device)
		post_1_mask = post_1_mask.to(device)
		post_2_tensors = post_2_tensors.to(device) if post_2_tensors is not None else None
		post_2_mask = post_2_mask.to(device)if post_2_mask is not None else None
		is_inners_embed = is_inners_embed.to(device)
		adu_orders = adu_orders.to(device)
		label_tensors = LabelTensor(model.task)(labels).to(device)

		feats = model(post_1_tensors, post_2_tensors, post_1_mask, post_2_mask, adu_1_tensors, adu_2_tensors, adu_orders, is_inners_embed)

		batch_size = label_tensors.shape[0]
		if model.task == 'link':
			assert feats.shape == (batch_size,)
		else:
			assert feats.shape == (batch_size,2)

		loss = criterion(feats, label_tensors)
		losses += loss.item()
		loss.backward()

		# Count predicted
		p_count, p_list = count_predictions(feats.detach().cpu(), model.task)
		prediction_count = [prediction_count[0] + p_count[0], prediction_count[1] + p_count[1]]
		prediction_count_per_log = [prediction_count_per_log[0] + p_count[0], prediction_count_per_log[1] + p_count[1]]
		prediction_list.extend(p_list)
		prediction_list_per_log.extend(p_list)

		for label in label_tensors:
			item = int(label.item())
			assert is_close(label.item(), 0) or is_close(label.item(), 1)
			assert item == 0 or item == 1
			labels_list.append(item)
			labels_list_per_log.append(item)

		# do log?
		if i % log_interval == 0 and i > 0:
			acc_per_log = get_accuracy(prediction_list_per_log, labels_list_per_log)
			f1_per_log = metrics.f1_score(labels_list_per_log, prediction_list_per_log, average=None)
			f1_neg_per_log, f1_pos_per_log, f1_per_log = f1_per_log[0], f1_per_log[1], (f1_per_log[0]+f1_per_log[1])/2
			assert is_close(f1_per_log, metrics.f1_score(labels_list_per_log, prediction_list_per_log, average='macro'))
			print('| epoch {:3d} | loss {:10.2f} | accuracy {:8.3f} | f1 {:8.3f} ({:8.3f},{:8.3f})'.format(e, losses/log_interval, acc_per_log, f1_per_log, f1_neg_per_log, f1_pos_per_log), [count for count in prediction_count_per_log])
			losses = 0
			# https://meetonfriday.com/posts/18392404/
			# 多個 batch 後才 optimize 一次。
			# nn.utils.clip_grad_norm_(model.parameters(), 0.1)

			optimizer.step()
			optimizer.zero_grad()
			prediction_count_per_log = [0, 0]
			prediction_list_per_log = []
			labels_list_per_log = []
		
	optimizer.step()
	optimizer.zero_grad()
	scheduler.step()

	# Calculate f1 score & acc
	acc_score = get_accuracy(labels_list, prediction_list)
	f1_score = metrics.f1_score(labels_list, prediction_list, labels=[0,1], average=None)
	f1_neg, f1_pos, f1_score = f1_score[0], f1_score[1], (f1_score[0]+f1_score[1])/2
	assert is_close(f1_score, metrics.f1_score(labels_list, prediction_list, average='macro'))
	print(' :: Prediction Count (Train)',  [count for count in prediction_count])
	print('    epoch {:3d} | accuracy {:8.3f} | f1 {:8.3f} ({:8.3f},{:8.3f})'.format(e, acc_score, f1_score, f1_neg, f1_pos))


def validate(e, model, iterator, f1_metrics, device):
	model.eval()
	prediction_count = [0, 0]
	prediction_list = []
	labels_list = []

	with torch.no_grad():
		for i, batch in enumerate(iterator):
			post_1_tensors, post_2_tensors, post_1_mask, post_2_mask, adu_1_tensors, adu_2_tensors, adu_orders, labels, is_inners_embed = batch

			adu_1_tensors = adu_1_tensors.to(device)
			adu_2_tensors = adu_2_tensors.to(device)
			adu_orders = adu_orders.to(device)
			post_1_tensors = post_1_tensors.to(device)
			post_1_mask = post_1_mask.to(device)
			post_2_tensors = post_2_tensors.to(device) if post_2_tensors is not None else None
			post_2_mask = post_2_mask.to(device) if post_2_mask is not None else None
			is_inners_embed = is_inners_embed.to(device)
			label_tensors = LabelTensor(model.task)(labels).to(device)

			feats = model(post_1_tensors, post_2_tensors, post_1_mask, post_2_mask, adu_1_tensors, adu_2_tensors, adu_orders, is_inners_embed)

			batch_size = label_tensors.shape[0]
			if model.task == 'link':
				assert feats.shape == (batch_size,)
			else:
				assert feats.shape == (batch_size,2)

			# Count predicted
			p_count, p_list = count_predictions(feats.detach().cpu(), model.task)
			prediction_count = [prediction_count[0] + p_count[0], prediction_count[1] + p_count[1]]
			prediction_list.extend(p_list)

			for label in label_tensors:
				item = int(label.item())
				assert is_close(label.item(), 0) or is_close(label.item(), 1)
				assert item == 0 or item == 1
				labels_list.append(item)

	acc_score = get_accuracy(labels_list, prediction_list)
	f1_score = metrics.f1_score(labels_list, prediction_list, average=None)
	f1_neg, f1_pos, f1_score = f1_score[0], f1_score[1], (f1_score[0]+f1_score[1])/2
	assert is_close(f1_score, metrics.f1_score(labels_list, prediction_list, average='macro'))
	print('\t', ':: Prediction Count (Valid)',  [count for count in prediction_count])
	print('\t', '   epoch {:3d} | accuracy {:8.3f} | f1 {:8.3f} ({:8.3f},{:8.3f})'.format(e, acc_score, f1_score, f1_neg, f1_pos))
	
	if f1_metrics == 'pos':
		f1_score = f1_pos
	elif f1_metrics == 'neg':
		f1_score = f1_neg

	return model.state_dict(), acc_score, f1_score

def validate_multi(e, model, iterators, f1_metrics, main_valid_dataset, device):
	result_model = None
	acc_scores = []
	f1_scores = []
	for iterator in iterators:
		result_model, acc_score, f1_score = validate(e, model, iterator, f1_metrics, device)
		acc_scores.append(acc_score)
		f1_scores.append(f1_score)
	if main_valid_dataset == -1:
		acc_score, f1_score = mean(acc_scores), mean(f1_scores)
	else:
		acc_score, f1_score = acc_scores[main_valid_dataset], f1_scores[main_valid_dataset]
	return result_model, acc_score, f1_score

if __name__=="__main__":
	print('Training Script.')
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	device = torch.device("cuda")
	
	best_model = None
	_best_val_acc = -np.inf
	_best_val_f1 = -np.inf
	_best_epoch = -1

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--log_interval", type=int, default=15)
	parser.add_argument("--roberta_lr", type=float, default=0.00001)
	parser.add_argument("--lr", type=float, default=0.0004)
	parser.add_argument("--n_epochs", type=int, default=100)
	parser.add_argument("--n_warmup_epochs", type=int, default=0, help='Models during the first n_warmup_epochs epochs will not be recorded.')
	parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
	parser.add_argument("--data", type=str, default="../subtask-seqtagging/LREC_dataset/brat_format")

	parser.add_argument("--training_set", type=str, default="inter", choices=['inter', 'inner', 'both'])
	parser.add_argument("--do_duplicates", type=str, default=['False'], choices=['True', 'False'], nargs='+')
	parser.add_argument("--local_negative_sampling_prob", type=float, default=[0.06], nargs='+')
	parser.add_argument("--global_negative_sampling_ratio", type=float, default=[0.02], nargs='+')

	parser.add_argument("--classification_type", type=str, default="link", choices=['link', 'link_type'])
	parser.add_argument("--do_inners_embed", action='store_true')
	parser.add_argument("--do_adu_order", action='store_true')

	parser.add_argument("--model_name", type=str, default="bert-base-uncased", choices=['bert-base-uncased', 'Jeevesh8/sMLM-RoBERTa', 'Jeevesh8/sMLM-LF'])

	parser.add_argument("--f1_metrics", type=str, default='pos', choices=['pos', 'neg', 'average'])
	parser.add_argument("--main_valid_dataset", type=int, default=-1)
	parser.add_argument("--manual_seed", action='store_true')
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--lstm_hidden_dim", type=int, default=256)
	parser.add_argument("--fine_tune_roberta", action='store_true')
	parser.add_argument("--prefix", type=str, default='')
	parser.add_argument("--identifier", type=str, default='None')

    # todo: roberta 系列的參數未使用。

	args = parser.parse_args()
	args.do_duplicates = [True if e == 'True' else False for e in args.do_duplicates]
	print('Arguments:', args)
	print()

	print('====================')
	print(' Identifier:', args.identifier)
	print('====================\n\n')

	# Manual Seed?
	if args.manual_seed:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)

	# Make tokenizer
	if args.model_name == 'bert-base-uncased':
		tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	elif args.model_name == 'Jeevesh8/sMLM-RoBERTa':
		tokenizer = RobertaTokenizer.from_pretrained("Jeevesh8/sMLM-RoBERTa")
	elif args.model_name == 'Jeevesh8/sMLM-LF':
		tokenizer = LongformerTokenizer.from_pretrained("Jeevesh8/sMLM-LF")
	else:
		raise Exception('Unknown model name.')

	# 製作 training dataset
	print("Loading datasets...")
	if args.training_set == 'inner':
		train_dataset = AduRelationDataset(args.data, True, True, args.local_negative_sampling_prob[0], args.global_negative_sampling_ratio[0], True, tokenizer, args.classification_type, args.do_duplicates[0])
		arg_del_index = 1
	elif args.training_set == 'inter':
		train_dataset = AduRelationDataset(args.data, False, True, args.local_negative_sampling_prob[0], args.global_negative_sampling_ratio[0], True, tokenizer, args.classification_type, args.do_duplicates[0])
		arg_del_index = 1
	elif args.training_set == 'both':
		train_dataset_inner = AduRelationDataset(args.data, True, True, args.local_negative_sampling_prob[0], args.global_negative_sampling_ratio[0], True, tokenizer, args.classification_type, args.do_duplicates[0])
		train_dataset_inter = AduRelationDataset(args.data, False, True, args.local_negative_sampling_prob[1], args.global_negative_sampling_ratio[1], True, tokenizer, args.classification_type, args.do_duplicates[1])
		train_dataset = torch.utils.data.ConcatDataset([train_dataset_inner, train_dataset_inter])
		arg_del_index = 2

	args.local_negative_sampling_prob = args.local_negative_sampling_prob[arg_del_index:]
	args.global_negative_sampling_ratio = args.global_negative_sampling_ratio[arg_del_index:]
	args.do_duplicates = args.do_duplicates[arg_del_index:]

	# 製作 validating dataset
	eval_datasets = [None, None, None]
	eval_datasets[0] = AduRelationDataset(args.data, True, False, args.local_negative_sampling_prob[0], args.global_negative_sampling_ratio[0], True, tokenizer, args.classification_type, args.do_duplicates[0])
	eval_datasets[1] = AduRelationDataset(args.data, False, False, args.local_negative_sampling_prob[1], args.global_negative_sampling_ratio[1], True, tokenizer, args.classification_type, args.do_duplicates[1])
	eval_datasets[2] = torch.utils.data.ConcatDataset([eval_datasets[0], eval_datasets[1]])
	print("train data len is {}".format(len(train_dataset)))
	print("validset data len is [{}, {}, {}]".format(len(eval_datasets[0]), len(eval_datasets[1]), len(eval_datasets[2])))
	print('Done loading data.')

	# Statistics
	train_stats = [0, 0]
	eval_stats = [[0, 0], [0, 0], [0, 0]]
	for entry in iter(train_dataset):
		adu_1, adu_2, relation_type, post_1, post_2, data_type = entry
		train_stats[relation_type] += 1
	for eval_dataset_idx in range(3):
		for entry in iter(eval_datasets[eval_dataset_idx]):
			adu_1, adu_2, relation_type, post_1, post_2, data_type = entry
			eval_stats[eval_dataset_idx][relation_type] += 1

	print("Train Data Stats:", train_stats)
	print("Eval Data Stats:", eval_stats)
	# ---

	# Make Model
	model_class = { 
		'bert-base-uncased': BertModel, 
		'Jeevesh8/sMLM-RoBERTa': RobertaModel,
		'Jeevesh8/sMLM-LF': LongformerModel }[args.model_name]
	model = Model_Adu_Relation(
		model_class=model_class, 
		model_name=args.model_name, 
		lstm_hidden=args.lstm_hidden_dim, 
		do_inners_embed=args.do_inners_embed, 
		do_adu_order=args.do_adu_order,
		task=args.classification_type).to(device)
	print('Model initialization Done.')

	# Make Datalaoders
	train_iter = data.DataLoader(dataset=train_dataset,
									batch_size=args.batch_size,
									shuffle=True,
									num_workers=4,
									collate_fn=lambda batch: PadBatch(batch, tokenizer))
	eval_iters = [None, None, None]
	eval_iters[0] = data.DataLoader(dataset=eval_datasets[0],
									batch_size=args.batch_size,
									shuffle=False,
									num_workers=4,
									collate_fn=lambda batch: PadBatch(batch, tokenizer))
	eval_iters[1] = data.DataLoader(dataset=eval_datasets[1],
									batch_size=args.batch_size,
									shuffle=False,
									num_workers=4,
									collate_fn=lambda batch: PadBatch(batch, tokenizer))
	eval_iters[2] = data.DataLoader(dataset=eval_datasets[2],
									batch_size=args.batch_size,
									shuffle=False,
									num_workers=4,
									collate_fn=lambda batch: PadBatch(batch, tokenizer))

	criterion = torch.nn.BCELoss() if args.classification_type == 'link' else torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.88)

	len_train_dataset = len(train_dataset) 
	batch_size = args.batch_size
	
    # Training
	print('Performing training...')
	validate_multi(0, model, eval_iters, args.f1_metrics, args.main_valid_dataset, device)
	for epoch in range(1, args.n_epochs+1):
		print(':: Starting epoch %d' % epoch)
		train(epoch, args, model, train_iter, criterion, optimizer, scheduler, device)
		if epoch <= args.n_warmup_epochs:
			continue
		candidate_model, acc, f1 = validate_multi(epoch, model, eval_iters, args.f1_metrics, args.main_valid_dataset, device)
		if acc >= _best_val_acc and f1 >= _best_val_f1:
			best_model = candidate_model
			_best_val_acc = acc
			_best_val_f1 = f1
			_best_epoch = epoch

	# y_test, y_pred = test(best_model, test_iter, device)
	# print(metrics.classification_report(y_test, y_pred, labels=labels, digits=3))
	model_file_name = args.checkpoint_dir + datetime.now().strftime(args.prefix + '-%Y%m%d-acc=' + '{:.2f}'.format(_best_val_acc) + '-f1=' + '{:.2f}'.format(_best_val_f1) + '-%H%M%S') + '.pt'
	torch.save(best_model, model_file_name) 
	print("Best Model: Epoch {}, F1 {:.3f}, Val Acc:{:.3f}%".format(_best_epoch, _best_val_f1, _best_val_acc))
	print("Saved as:", model_file_name)
	# test_data = pd.read_csv("model_data/0704_bio_test.csv")
	# y_test_useful = []
	# y_pred_useful = []
	# for a, b in zip(y_test, y_pred):
	# 	if a not in ['[CLS]', '[SEP]']:
	# 			y_test_useful.append(a)
	# 			y_pred_useful.append(b)
	# test_data["labeled"] = y_test_useful
	# test_data["pred"] = y_pred_useful
	# test_data.to_csv("result_files/bio_test_result.csv", index=False)
