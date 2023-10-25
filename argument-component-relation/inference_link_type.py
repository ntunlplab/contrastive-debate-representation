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
from utils_lstm_inference import AduPairLinkTypeDataset, PadBatchLinkType
from datetime import datetime
from statistics import mean 
import nltk.data
from tqdm import tqdm
import code

CLASSES = ['Support', 'Attack']

if __name__=="__main__":
	print('Inference Script (Type).')
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	device = torch.device("cuda")

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--model_path", type=str, default="./both-post/checkpoints/lf-type-sep-20230409-acc=0.89-f1=0.86-061002.pt")
	parser.add_argument("--parent_paths", type=str, nargs='+', default=["./dataset-link-marked/train/", "./dataset-link-marked/test/"])
	parser.add_argument("--output_dirs", type=str, nargs='+', default=["./dataset-type-marked/train/", "./dataset-type-marked/test/"])
	parser.add_argument("--manual_seed", action='store_true')
	parser.add_argument("--seed", type=int, default=0)
	
	args = parser.parse_args()
	print('Arguments:', args)
	print()

	# Manual Seed?
	if args.manual_seed:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)

	print("Loading datasets...")
	datasets = [AduPairLinkTypeDataset(in_path, out_path) for in_path, out_path in zip(args.parent_paths, args.output_dirs)]

	for dataset_idx in range(len(datasets)):
		invalid_post_indices = [(key, value) for key, value in datasets[dataset_idx].validity_rates.items() if value is not None]
		print('[Dataset #%d]' % dataset_idx, 'List of Invalid Posts:', str(invalid_post_indices))

	print("Dataset loading done.")

	print('\t:: Number of posts:', [len(dataset.posts) for dataset in datasets])
	print('\t:: Number of pairs (inner):', [dataset.inner.num_real_pairs for dataset in datasets])
	print('\t:: Number of pairs (inter):', [dataset.inter.num_real_pairs for dataset in datasets])

	print("Loading Model...")
	tokenizer = LongformerTokenizer.from_pretrained('Jeevesh8/sMLM-LF')
	model = Model_Adu_Relation(
		model_class=LongformerModel, 
		model_name='Jeevesh8/sMLM-LF', 
		lstm_hidden=128, 
		do_inners_embed=True, 
		do_adu_order=False,
		task='link_type').to(device)
	model.load_state_dict(torch.load(args.model_path))
	model.eval()
	print("Model loading done.")

	label_count = [0, 0]
    
	# 對於每一個 dataset...
	for index in range(len(datasets)):
		dataset = datasets[index]
		post_indices = range(0, len(dataset.posts))

		# 對於每一個 post...
		print('[Dataset #%d]' % index)
		for post_index in tqdm(post_indices, total=len(list(post_indices)), desc='#Posts'):
			post_label_count = [0, 0]
			lookup = { }
			inner_pairs = dataset.inner_pairs_of_post(post_index)
			inter_pairs = dataset.inter_pairs_of_post(post_index)
			

			with tqdm(total=(len(inner_pairs)+len(inter_pairs)), leave=False, desc='  #Pairs') as pbar:
				
				inner_iter = data.DataLoader(dataset=inner_pairs,
											batch_size=args.batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=lambda batch: PadBatchLinkType(batch, tokenizer, lookup))
				inter_iter = data.DataLoader(dataset=inter_pairs,
											batch_size=args.batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=lambda batch: PadBatchLinkType(batch, tokenizer, lookup))
				
				# Inner-Post
				with torch.no_grad():
					for batch in inner_iter:
						comment_tensors, comment_masks, adu_1_tensors, adu_2_tensors, is_inner_embeds, comment_infos = batch
						
						comment_tensors = comment_tensors.to(device)
						comment_masks = comment_masks.to(device)
						adu_1_tensors = adu_1_tensors.to(device)
						adu_2_tensors = adu_2_tensors.to(device)
						is_inner_embeds = is_inner_embeds.to(device)
						batch_size = comment_tensors.shape[0]
						
						feats = model(comment_tensors, None, comment_masks, None, adu_1_tensors, adu_2_tensors, None, is_inner_embeds)
						assert feats.shape == (batch_size, 2)

						feats_class = feats.detach().cpu().argmax(dim=1)
						results = []
						for feat in feats_class:
							item = int(feat.item())
							assert item == 0 or item == 1
							label_count[item] += 1
							post_label_count[item] += 1
							results.append(item)

						for data_idx in range(batch_size):
							# 從 batch 得到原本的 comment 和 index
							_post_index, _is_thread, _pair_index, _is_positive, _comment_index = comment_infos[data_idx]
							assert _post_index == post_index
							batch_adu_1 = adu_1_tensors[data_idx].detach().cpu()
							batch_adu_2 = adu_2_tensors[data_idx].detach().cpu()
							batch_adu_1 = (batch_adu_1[0].item(), batch_adu_1[1].item())
							batch_adu_2 = (batch_adu_2[0].item(), batch_adu_2[1].item())
							datasets[index].posts[_post_index].get_comment(_is_thread, _pair_index, _is_positive, _comment_index) \
								.real_inner_adu_pairs.append( (batch_adu_1, batch_adu_2, CLASSES[ results[data_idx] ]) )
						pbar.update(batch_size)
				"Inner-Post Done."

				# Inter-Post
				with torch.no_grad():
					for batch in inter_iter:
						comment_tensors, comment_masks, adu_1_tensors, adu_2_tensors, is_inner_embeds, comment_infos = batch
						
						comment_tensors = comment_tensors.to(device)
						comment_masks = comment_masks.to(device)
						adu_1_tensors = adu_1_tensors.to(device)
						adu_2_tensors = adu_2_tensors.to(device)
						is_inner_embeds = is_inner_embeds.to(device)
						batch_size = comment_tensors.shape[0]

						feats = model(comment_tensors, None, comment_masks, None, adu_1_tensors, adu_2_tensors, None, is_inner_embeds)
						assert feats.shape == (batch_size, 2)

						feats_class = feats.detach().cpu().argmax(dim=1)
						results = []
						for feat in feats_class:
							item = int(feat.item())
							assert item == 0 or item == 1
							label_count[item] += 1
							post_label_count[item] += 1
							results.append(item)

						for data_idx in range(batch_size):
							# 從 batch 得到原本的 comment 和 index
							_post_index, _is_thread, _pair_index, _is_positive, _comment_index = comment_infos[data_idx]
							assert _post_index == post_index
							batch_adu_1 = adu_1_tensors[data_idx].detach().cpu()
							batch_adu_2 = adu_2_tensors[data_idx].detach().cpu()
							batch_adu_1 = (batch_adu_1[0].item(), batch_adu_1[1].item())
							batch_adu_2 = (batch_adu_2[0].item(), batch_adu_2[1].item())
							datasets[index].posts[_post_index].get_comment(_is_thread, _pair_index, _is_positive, _comment_index) \
								.real_inter_adu_pairs.append( (batch_adu_1, batch_adu_2, CLASSES[ results[data_idx] ]) )
						pbar.update(batch_size)
				"Inter-Post Done."

			# 因為在每個 comment 的 real_inter_adu_pairs 和 real_inner_adu_pairs
			# 存在標記 Attack/Support 前後的兩種資料
			# 因此這裡把舊資料刪除
			for (is_thread, pair_index, is_positive, comment_index, comment) in datasets[index].posts[post_index].get_comments():
				orig_num_inner_pairs, orig_num_inter_pairs = len(comment.real_inner_adu_pairs), len(comment.real_inter_adu_pairs)
				comment.real_inner_adu_pairs = [pair for pair in comment.real_inner_adu_pairs if len(pair) == 3]
				comment.real_inter_adu_pairs = [pair for pair in comment.real_inter_adu_pairs if len(pair) == 3]
				assert len(comment.real_inner_adu_pairs) == orig_num_inner_pairs // 2
				assert len(comment.real_inter_adu_pairs) == orig_num_inter_pairs // 2

			# This post is done. Write into file.
			filename = datasets[index].post_filenames[post_index]
			with open(args.output_dirs[index] + ('%s.data' % filename), 'wb') as f:
				pickle.dump(datasets[index].posts[post_index], f)
				# 確保該 post 不會再被改到，清除其資料
				datasets[index].posts[post_index].author = None
				datasets[index].posts[post_index].title = None
				datasets[index].posts[post_index].root = None
				datasets[index].posts[post_index].pairs = None
			with open(args.output_dirs[index] + ('%s-stats.txt' % filename), 'w') as f:
				f.write('Label Count: \n\tSupport: %d\n\tAttack: %d' % (
					post_label_count[0],
					post_label_count[1]
				))

	print("Inference done.")
	print('Prediction Count:\n\tSupport: %d\n\tAttack: %d' % (label_count[0], label_count[1]))

	with open('inference_link_report.txt', 'w') as f:
		f.write('Prediction Count:\n\tSupport: %d\n\tAttack: %d' % (label_count[0], label_count[1]))
							

