# -*- coding: utf-8 -*-
import pickle
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
from utils_lstm_inference import PotentialAduPairInferenceDataset, PadBatchLink
from datetime import datetime
from statistics import mean 
import nltk.data
from tqdm import tqdm
import code

def is_identical_spans(span_1, span_2):
	return span_1[0] == span_2[0] and span_1[1] == span_2[1]


if __name__=="__main__":
	print('Inference Script (Link).')
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	device = torch.device("cuda")

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--model_path", type=str, default="./both-post/checkpoints/lf-link-20230408-acc=0.80-f1=0.80-233441.pt")
	parser.add_argument("--data_dirs", type=str, nargs='+', default=["../subtask-seqtagging/cmv-train.data.v2.data", "../subtask-seqtagging/cmv-test.data.v2.data"])
	parser.add_argument("--output_dirs", type=str, nargs='+', default=["./dataset-link-marked/train/", "./dataset-link-marked/test/"])
	parser.add_argument("--manual_seed", action='store_true')
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--start_froms", type=int, nargs='+', default=[0, 0])
	parser.add_argument("--end_ats", type=int, nargs='+', default=[99999, 99999])
	
	args = parser.parse_args()
	print('Arguments:', args)
	print()

	# Manual Seed?
	if args.manual_seed:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)

	print("Loading datasets...")
	datasets = [PotentialAduPairInferenceDataset(path) for path in args.data_dirs]
	print("Dataset loading done.")

	print('\t:: Number of posts:', [len(dataset.posts) for dataset in datasets])
	print('\t:: Number of pairs (inner):', [len(dataset.inner_pairs) for dataset in datasets])
	print('\t:: Number of pairs (inter):', [len(dataset.inter_pairs) for dataset in datasets])

	print("Loading Model...")
	tokenizer = LongformerTokenizer.from_pretrained('Jeevesh8/sMLM-LF')
	model = Model_Adu_Relation(
		model_class=LongformerModel, 
		model_name='Jeevesh8/sMLM-LF', 
		lstm_hidden=128, 
		do_inners_embed=False, 
		do_adu_order=False,
		task='link').to(device)
	model.load_state_dict(torch.load(args.model_path))
	model.eval()
	print("Model loading done.")

	label_count = [0, 0]
	validity_rate = []

	# 對於每一個 dataset...
	for index in range(len(datasets)):
		dataset = datasets[index]
		start_from = args.start_froms[index]
		end_at = args.end_ats[index]
		post_indices = range(0, len(dataset.posts))

		# 對於每一個 post...
		print('[Dataset #%d]' % index)
		for post_index in tqdm(post_indices, total=len(list(post_indices)), desc='#Posts'):

			if post_index < start_from or post_index > end_at:
				continue

			post_validity_rate = [0, 0]
			post_link_rate = [0, 0]
			lookup = { }
			inner_pairs = dataset.inner_pairs_of_post(post_index)
			inter_pairs = dataset.inter_pairs_of_post(post_index)
			

			with tqdm(total=(len(inner_pairs)+len(inter_pairs)), leave=False, desc='  #Pairs') as pbar:
				
				inner_iter = data.DataLoader(dataset=inner_pairs,
											batch_size=args.batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=lambda batch: PadBatchLink(batch, tokenizer, lookup))
				inter_iter = data.DataLoader(dataset=inter_pairs,
											batch_size=args.batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=lambda batch: PadBatchLink(batch, tokenizer, lookup))
				
				# Inner-Post
				with torch.no_grad():
					for batch in inner_iter:
						comment_tensors, comment_masks, adu_1_tensors, adu_2_tensors, is_inner_embeds, comment_infos, validities = batch
						
						comment_tensors = comment_tensors.to(device)
						comment_masks = comment_masks.to(device)
						adu_1_tensors = adu_1_tensors.to(device)
						adu_2_tensors = adu_2_tensors.to(device)
						is_inner_embeds = is_inner_embeds.to(device)
						batch_size = comment_tensors.shape[0]

						# print('================')
						# print(torch.max(comment_tensors))
						# print(torch.min(comment_tensors))
						# print(comment_tensors.shape)
						# print(torch.max(adu_1_tensors))
						# print(torch.min(adu_1_tensors))
						# print(torch.max(adu_2_tensors))
						# print(torch.min(adu_2_tensors))
						# print('================\n')
						# print('==============')
						# print(comment_tensors.shape)
						# print(adu_1_tensors)
						# print(adu_2_tensors)

						# for batch_idx in range(batch_size):
						# 	comment_tensor = comment_tensors[batch_idx].detach().cpu()
						# 	adu_1_tensor = adu_1_tensors[batch_idx].detach().cpu()
						# 	adu_2_tensor = adu_2_tensors[batch_idx].detach().cpu()
						# 	adu_1 = list(comment_tensor[ adu_1_tensor[0] : adu_1_tensor[1] + 1 ])
						# 	adu_2 = list(comment_tensor[ adu_2_tensor[0] : adu_2_tensor[1] + 1 ])
						# 	adu_1 = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(adu_1))
						# 	adu_2 = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(adu_2))

						# 	comment = original_comments[batch_idx]
						# 	idx_in_comment = comment_indices[batch_idx]
						# 	span_1, span_2, _ = comment.get_potential_inner_adu_pairs(idx_in_comment)
						# 	comment_text = comment.get_comment()
						# 	orig_adu_1 = comment_text[ span_1[0] : span_1[1] ]
						# 	orig_adu_2 = comment_text[ span_2[0] : span_2[1] ]

						# 	print('---------------')
						# 	print('ADU_1:\t\"%s\"' % adu_1)
						# 	print('O_ADU_1:\t\"%s\"' % orig_adu_1)
						# 	print()
						# 	print('ADU_2:\t\"%s\"' % adu_2)
						# 	print('O_ADU_2:\t\"%s\"' % orig_adu_2)
						# 	print()

						# 	print('---------------')
						# print('==============')
						feats = model(comment_tensors, None, comment_masks, None, adu_1_tensors, adu_2_tensors, None, None)
						assert feats.shape == (batch_size,)

						feats_round = feats.detach().cpu().round()
						results = []
						for feat in feats_round:
							item = int(feat.item())
							assert item == 0 or item == 1
							label_count[item] += 1
							results.append(item)

						for data_idx in range(batch_size):
							# 從 batch 得到原本的 comment 和 index
							_post_index, _is_thread, _pair_index, _is_positive, _comment_index = comment_infos[data_idx]
							assert _post_index == post_index
							batch_adu_1 = adu_1_tensors[data_idx].detach().cpu()
							batch_adu_2 = adu_2_tensors[data_idx].detach().cpu()
							batch_adu_1 = (batch_adu_1[0].item(), batch_adu_1[1].item())
							batch_adu_2 = (batch_adu_2[0].item(), batch_adu_2[1].item())
							validity = validities[data_idx]
							post_validity_rate[0 if validity else 1] += 1
							if results[data_idx] == 1 and validity and (not is_identical_spans(batch_adu_1, batch_adu_2)):
								datasets[index].posts[_post_index].get_comment(_is_thread, _pair_index, _is_positive, _comment_index).real_inner_adu_pairs.append( (batch_adu_1, batch_adu_2) )
								post_link_rate[0] += 1
							post_link_rate[1] += 1
						pbar.update(batch_size)
				"Inner-Post Done."

				# Inter-Post
				with torch.no_grad():
					for batch in inter_iter:
						comment_tensors, comment_masks, adu_1_tensors, adu_2_tensors, is_inner_embeds, comment_infos, validities = batch
						
						comment_tensors = comment_tensors.to(device)
						comment_masks = comment_masks.to(device)
						adu_1_tensors = adu_1_tensors.to(device)
						adu_2_tensors = adu_2_tensors.to(device)
						is_inner_embeds = is_inner_embeds.to(device)
						batch_size = comment_tensors.shape[0]

						feats = model(comment_tensors, None, comment_masks, None, adu_1_tensors, adu_2_tensors, None, None)
						assert feats.shape == (batch_size,)

						feats_round = feats.detach().cpu().round()
						results = []
						for feat in feats_round:
							item = int(feat.item())
							assert item == 0 or item == 1
							label_count[item] += 1
							results.append(item)

						for data_idx in range(batch_size):
							# 從 batch 得到原本的 comment 和 index
							_post_index, _is_thread, _pair_index, _is_positive, _comment_index = comment_infos[data_idx]
							assert _post_index == post_index
							batch_adu_1 = adu_1_tensors[data_idx].detach().cpu()
							batch_adu_2 = adu_2_tensors[data_idx].detach().cpu()
							batch_adu_1 = (batch_adu_1[0].item(), batch_adu_1[1].item())
							batch_adu_2 = (batch_adu_2[0].item(), batch_adu_2[1].item())
							validity = validities[data_idx]
							post_validity_rate[0 if validity else 1] += 1
							if results[data_idx] == 1 and validity and (not is_identical_spans(batch_adu_1, batch_adu_2)):
								datasets[index].posts[_post_index].get_comment(_is_thread, _pair_index, _is_positive, _comment_index).real_inter_adu_pairs.append( (batch_adu_1, batch_adu_2) )
								post_link_rate[0] += 1
							post_link_rate[1] += 1
						pbar.update(batch_size)
				"Inter-Post Done."

			# This post is done. Write into file.
			validity_rate.append('%d: %.2f%% (%d/%d)' % (
				post_index, 
				post_validity_rate[0] / sum(post_validity_rate) * 100,
				post_validity_rate[0],
				sum(post_validity_rate)))
			with open(args.output_dirs[index] + ('%07d.data' % post_index), 'wb') as f:
				pickle.dump(datasets[index].posts[post_index], f)
				# 確保該 post 不會再被改到，清除其資料
				datasets[index].posts[post_index].author = None
				datasets[index].posts[post_index].title = None
				datasets[index].posts[post_index].root = None
				datasets[index].posts[post_index].pairs = None
			with open(args.output_dirs[index] + ('%07d-stats.txt' % post_index), 'w') as f:
				f.write('Validity Rate: %d/%d\nLink Rate: %d/%d\n' % (
					post_validity_rate[0],
					sum(post_validity_rate),
					post_link_rate[0],
					post_link_rate[1]
				))

	print("Inference done.")
	print('Prediction Count:', label_count)
	print('Link Rate: %.2f%' % ( label_count[1] / (label_count[0] + label_count[1]) * 100 ))
	print('Validity Rates:', ''.join(validity_rate))

	with open('inference_link_report.txt', 'w') as f:
		f.write('Prediction Count: ')
		f.write(str(label_count))
		f.write('\n')
		f.write('Link Rate: %.2f%\n' % ( label_count[1] / (label_count[0] + label_count[1]) * 100 ))
		f.write('Validity Rates:', ''.join(validity_rate))
							

