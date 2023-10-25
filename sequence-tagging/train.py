# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from transformers import BertTokenizer
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
import pandas as pd
# from models import Bert_CRF
from pytorch_pretrained_bert import Bert_CRF
from utils import AduDataset, AduInferenceDataset, SeqTaggingPadBatch, tagging_scheme_config
from datetime import datetime
import random

def train(e, model, iterator, gradient_steps, optimizer, scheduler, device):
	model.train()
	Y, Y_hat = [], []
	losses = 0.0
	step = 0
	optimizer.zero_grad()
	for idx, batch in enumerate(iterator):
		step += 1
		tokens, labels, input_masks, crf_masks = batch
		tokens = tokens.to(device)
		labels = labels.to(device)
		input_masks = input_masks.to(device)
		crf_masks = crf_masks.to(device)
		
		loss, bio_scores, labels_pred = model(
			input_ids=tokens,
			attention_mask=input_masks,
			crf_mask=crf_masks,
			bio_labels=labels,
			check=True
		)

		losses += loss.item()
		loss.backward()

		if idx % gradient_steps and idx > 0:
			optimizer.step()
			optimizer.zero_grad()

		# Save prediction
		for j in labels_pred:
			Y_hat.extend(j)

		# print("PRED:", labels_pred[0])
		# print("TRUE:", labels[0])
		# print("MASK:", crf_masks[0])
		# print("======================\n")

		# Save labels
		mask = (crf_masks==1)
		labels_gold = torch.masked_select(labels, mask)
		Y.append(labels_gold.cpu())
	scheduler.step()
	Y = torch.cat(Y, dim=0).numpy()
	Y_hat = np.array(Y_hat)
	acc = (Y_hat == Y).mean()*100
	print("Epoch: {}, Loss:{:.4f} Acc:{:.3f}".format(e, losses/step, acc))

def validate(e, model, iterator, device):
	model.eval()
	Y, Y_hat = [], []
	losses = 0
	step = 0
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			step += 1

			tokens, labels, input_masks, crf_masks = batch
			tokens = tokens.to(device)
			labels = labels.to(device)
			input_masks = input_masks.to(device)
			crf_masks = crf_masks.to(device)

			loss, bio_scores, labels_pred = model(
				input_ids=tokens,
				attention_mask=input_masks,
				crf_mask=crf_masks,
				bio_labels=labels,
				check=True
			)

			losses += loss.item()

			# Save prediction
			for j in labels_pred:
				Y_hat.extend(j)

			# Save labels
			mask = (crf_masks==1)
			labels_gold = torch.masked_select(labels, mask)
			Y.append(labels_gold.cpu())

	Y = torch.cat(Y, dim=0).numpy()
	Y_hat = np.array(Y_hat)
	acc = (Y_hat == Y).mean()*100
	print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(e, losses/step, acc))
	return model.state_dict(), losses/step, acc

if __name__=="__main__":
	# TODO: Use GPU?
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	device = torch.device("cuda")
	best_model = None
	_best_val_loss = np.inf
	_best_val_acc = -np.inf
	_best_epoch = -1

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=0.001)
	parser.add_argument("--n_epochs", type=int, default=40)
	parser.add_argument("--manual_seed", action='store_true')
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
	parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
	parser.add_argument("--is_cased", action='store_false')
	parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
	parser.add_argument("--tagging_scheme", type=str, default="bio", choices=["bio", "bioe", "bilou"])
	parser.add_argument("--gradient_steps", type=int, default=4)
	parser.add_argument("--trainset", type=str, default="./LREC_dataset/preprocessed/train.dat")
	parser.add_argument("--validset", type=str, default="./LREC_dataset/preprocessed/valid.dat")
	parser.add_argument("--save_prefix", type=str, default="sgd-bio")

	args = parser.parse_args()
	tokenizer = BertTokenizer.from_pretrained(args.bert_model)
	
	# Manual Seed?
	if args.manual_seed:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)

	# 這邊使用 'BIO' / 'BIOE 的設定
	bio_num_labels, constraints = tagging_scheme_config(args.tagging_scheme)

	model = Bert_CRF.from_pretrained(
		args.bert_model, 
		bio_num_labels=bio_num_labels, 
		constraints=constraints, 
		include_start_end_transitions=True).to(device)
	print('Model initialization Done.')

	# Training
	print('Performing training...')

	train_dataset = AduDataset(args.trainset, is_uncased=not args.is_cased, tokenizer=tokenizer, tagging_scheme=args.tagging_scheme)
	eval_dataset = AduDataset(args.validset, is_uncased=not args.is_cased, tokenizer=tokenizer, tagging_scheme=args.tagging_scheme)
	print("train data len is {}".format(len(train_dataset)))
	print("validset data len is {}".format(len(eval_dataset)))
	print('Load Data Done.')

	train_iter = data.DataLoader(dataset=train_dataset,
									batch_size=args.batch_size,
									shuffle=True,
									num_workers=4,
									collate_fn=lambda batch: SeqTaggingPadBatch(batch, train_dataset))

	eval_iter = data.DataLoader(dataset=eval_dataset,
									batch_size=args.batch_size,
									shuffle=False,
									num_workers=4,
									collate_fn=lambda batch: SeqTaggingPadBatch(batch, eval_dataset))

	if args.optimizer == "sgd":			
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
	elif args.optimizer == "adam":
		optimizer = optim.Adam(model.parameters(), lr=args.lr)
	elif args.optimizer == "adamw":
		optimizer = optim.AdamW(model.parameters(), lr=args.lr)
	len_train_dataset = len(train_dataset) 
	epoch = args.n_epochs
	batch_size = args.batch_size
	total_steps = (len_train_dataset // batch_size) * epoch if len_train_dataset % batch_size == 0 else (len_train_dataset // batch_size + 1) * epoch
	scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=3)

	validate(0, model, eval_iter, device)
	for epoch in range(1, args.n_epochs+1):
		train(epoch, model, train_iter, args.gradient_steps, optimizer, scheduler, device)
		candidate_model, loss, acc = validate(epoch, model, eval_iter, device)
		if loss < _best_val_loss and acc > _best_val_acc:
			best_model = candidate_model
			_best_val_loss = loss
			_best_val_acc = acc
			_best_epoch = epoch

	# y_test, y_pred = test(best_model, test_iter, device)
	# print(metrics.classification_report(y_test, y_pred, labels=labels, digits=3))
	torch.save(best_model, args.checkpoint_dir + datetime.now().strftime(args.save_prefix + '-%Y%m%d-acc-' + ('{:.3f}'.format(_best_val_acc)) + '-%H%M%S') + '.pt') 
	print("Best Model: Epoch {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(_best_epoch, _best_val_loss, _best_val_acc))
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
	
