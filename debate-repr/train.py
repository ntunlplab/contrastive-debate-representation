import code
import dgl
import numpy as np
import torch
import argparse
import utils
import torch.optim as optim
from transformers import LongformerTokenizer, LongformerModel
from models import DebateModel
import os
import random
import datetime
import time
import tqdm
import math
from torch import nn

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=1)
	
	# dataset paths
	parser.add_argument("--train_data", type=str, default="./dataset/train/")
	parser.add_argument("--test_data", type=str, default="./dataset/test/")
	parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
	parser.add_argument("--num_epochs", type=int, default=30)

	parser.add_argument("--lr", type=float, default=0.15)
	parser.add_argument("--han_lr", type=float, default=0.05)
	parser.add_argument("--optimization_steps", type=int, default=9)
	parser.add_argument("--gradient_clip_norm", type=float, default=0.1)

	parser.add_argument("--contrastive_margins", type=float, nargs='+', default=[40, 16])
	parser.add_argument("--contrastive_weights", type=float, nargs='+', default=[0.072, 0.028])

	parser.add_argument("--from_checkpoint", action='store_true')
	parser.add_argument("--from_epoch", type=int, default=1)
	parser.add_argument("--checkpoint_path", type=str, default=None)

	parser.add_argument("--num_train_samples", type=int, default=15000)
	parser.add_argument("--num_test_samples", type=int, default=7500)
	parser.add_argument("--manual_seed", action='store_true')
	parser.add_argument("--seed", type=int, default=9945)

	# learning rate
	args = parser.parse_args()
	return args

def contrastive_loss(x1, x2, label, margin: float = 1.0):
    """
    Computes Contrastive Loss
    """

    dist = torch.nn.functional.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss

# 回傳：Loss List
def run_batch(batch, device, model, args, back_prop=True):
	category, c1, c2 = batch
	assert category in [0, 1, 2]

	cr1 = run_data(c1, device, model)
	cr2 = run_data(c2, device, model)

	# 計算 Contrastive Loss
	labels = {
		# Different
		0: [1, 1],
		# Same Post, Different Author
		1: [0, 1],
		# Same Post, Same Author
		2: [0, 0]
	}
	loss_comment = contrastive_loss(cr1, cr2, labels[category][0], args.contrastive_margins[0])
	loss_author = contrastive_loss(cr1, cr2, labels[category][1], args.contrastive_margins[1])
	loss = args.contrastive_weights[0] * loss_comment + args.contrastive_weights[1] * loss_author

	if back_prop:
		loss.backward()

	loss_list = [0, 0, 0]
	loss_list[category] = loss.item()

	return loss_list

# 回傳：最後一層的 embedding
def run_data(data, device, model):

	# Batch Data
	case = data['case']
	comment_ids, comment_mask, comment_spans = data['input_ids']
	is_winner = data['is_winner']
	adu_attacked = data['adu_attacked']
	adu_spans, adu_masks, local_adu_in_use_masks, title_adu_span_index = data['adu_span']
	inner_pairs = data['inner_pairs']
	inter_pairs = data['inter_pairs']
	info_scores = data['info_scores']
	global_attention_mask = data['global_attention_mask']

	comment_ids = comment_ids.to(device)
	comment_mask = comment_mask.to(device)
	adu_attacked = adu_attacked.to(device)
	adu_spans = adu_spans.to(device)
	adu_masks = adu_masks.to(device)
	local_adu_in_use_masks = local_adu_in_use_masks.to(device)
	global_attention_mask = global_attention_mask.to(device)

	do_winner_pred = (case == utils.TAIL_OF_THREAD)

	return model(
		do_winner_pred,
		comment_ids, comment_mask, comment_spans, info_scores,
		global_attention_mask,
		adu_spans, adu_masks, local_adu_in_use_masks, title_adu_span_index,
		len(adu_attacked), inner_pairs, inter_pairs, device)

def train(e, model, device, iterator, optimizer, args):
	steps = 0
	start_time = datetime.datetime.now()
	losses, step_losses = [0, 0, 0], [0, 0, 0]
	category_counts = [0, 0, 0]
	maximum_samples = args.num_train_samples
	model.train()
	optimizer.zero_grad()

	for batch in iterator:
		category = batch[0]
		if category_counts[category] >= 3:
			continue
		if steps > maximum_samples:
			break

		steps += 1
		category_counts[category] += 1

		loss_list = run_batch(batch, device, model, args, back_prop=True)
		for i in range(3):
			step_losses[i] += loss_list[i]
			losses[i] += loss_list[i]

		# 每固定幾個 steps 就 optimize & log 一次
		if steps % args.optimization_steps == 0 and steps != 0:
			nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
			optimizer.step()
			optimizer.zero_grad()

			elapsed = datetime.datetime.now() - start_time
			data_per_sec = steps / elapsed.total_seconds()
			print('| training epoch {:3d} ({:7d}/{:7d}) | loss {:6.2f} | ({:6.2f}, {:6.2f}, {:6.2f}) | elapsed {} | etr {} |'.format(
				e, steps, maximum_samples, sum(step_losses),
				step_losses[0], step_losses[1], step_losses[2], 
				str(elapsed).split('.')[0],
				str(datetime.timedelta(seconds=(maximum_samples - steps) / data_per_sec)).split('.')[0]
			))
			step_losses = [0, 0, 0]
			category_counts = [0, 0, 0]
		
	elapsed = datetime.datetime.now() - start_time
	print('| training epoch {:3d} Done. | loss {:6.2f} | ({:6.2f}, {:6.2f}, {:6.2f}) | elapsed {} |'.format(
		e, sum(losses), 
		losses[0], losses[1], losses[2], 
		str(elapsed).split('.')[0]
	))

def valid(e, model, device, iterator, args):
	start_time = datetime.datetime.now()
	losses = [0, 0, 0]
	model.eval()

	steps = 0
	maximum_samples = args.num_test_samples

	for batch in iterator:
		if steps > maximum_samples:
			break
		with torch.no_grad():
			loss_list = run_batch(batch, device, model, args, back_prop=False)
			for i in range(3):
				losses[i] += loss_list[i]
		steps += 1

	print('| validating epoch {:3d} Done. | elapsed {} |'.format(
		e,
		str(datetime.datetime.now() - start_time)
	))
	print('\t| Loss: {:6.2f} | ({:6.2f}, {:6.2f}, {:6.2f})'.format(
		sum(losses),
		losses[0], losses[1], losses[2]
	))

	return model.state_dict(), losses


def main():
	os.environ["CUDA_VISIBLE_DEVICES"] = "1" # TODO: 0
	device = torch.device("cuda")
	args = parse_args()

	print(args)

	# Manual Seed?
	if args.manual_seed:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)

	# Non-existing path?
	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)

	# Tokenizer
	tokenizer = LongformerTokenizer.from_pretrained('Jeevesh8/sMLM-LF')

	# Datasets
	print(':: Loading datasets...')
	train_datasets_list = utils.gather_debate_datasets(args.train_data, tokenizer)
	test_datasets_list = utils.gather_debate_datasets(args.test_data, tokenizer)
	train_dataset = utils.DebateDatasets(train_datasets_list)
	test_dataset = utils.DebateDatasets(test_datasets_list)
	print('  :: Done.')

	train_dataloader = torch.utils.data.DataLoader(
		train_dataset, 
		batch_size=args.batch_size, 
		collate_fn=lambda batch: utils.PadBatchContrastive(batch, tokenizer, args.batch_size))
	test_dataloader = torch.utils.data.DataLoader(
		test_dataset, 
		batch_size=args.batch_size, 
		collate_fn=lambda batch: utils.PadBatchContrastive(batch, tokenizer, args.batch_size))

	# Model
	print(':: Loading model...')
	meta_paths = [
		['Attack'], ['Support'],
		['Attack', 'Support'],
		['Support', 'Attack'],

		['AttackedBy'], ['SupportedBy'],
		['AttackedBy', 'SupportedBy'],
		['SupportedBy', 'AttackedBy'],
	]
	model = DebateModel(meta_paths=meta_paths, info_scores_dim=len(utils.info_all_keys)).to(device)
	prev_params = utils.get_params(model)
	cur_params = utils.get_params(model)
	utils.save_report(prev_params, cur_params, args.checkpoint_dir + 'e-0-report.txt')
	print('  :: Done.')

	optimizer = torch.optim.SGD([
		{ 'lr': args.lr, 'params': model.span_encoder.parameters() },
		{ 'lr': args.lr, 'params': model.comment_compressor.parameters() },
		{ 'lr': args.lr, 'params': model.inner_adu_relation_attention.parameters() },
		{ 'lr': args.lr, 'params': model.inter_adu_relation_attention.parameters() },
		{ 'lr': args.lr, 'params': model.adu_attention.parameters() },
		{ 'lr': args.han_lr, 'params': model.han.parameters() }
	], momentum=0.9)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.93) # TODO: gamma=3, 0.9

	# Checkpoint?
	if args.from_checkpoint:
		print('[Checkpoint]', args.checkpoint_path)
		print('[From Epoch]', args.from_epoch)
		for e in range(1, args.from_epoch):
			# Skipping Sampling
			for i, batch in enumerate(train_dataloader):
				if i == args.num_train_samples - 1:
					break
			for i, batch in enumerate(test_dataloader):
				if i == args.num_test_samples - 1:
					break
			# Skipping Scheduler
			optimizer.zero_grad()
			optimizer.step()
			scheduler.step()
		model.load_state_dict(torch.load(args.checkpoint_path), strict=False)

	# Training & Validation
	best_losses = 1_000_000_000
	for e in range(args.from_epoch, args.num_epochs + 1):
		train(e, model, device, train_dataloader, optimizer, args)
		cur_model_state, losses = valid(e, model, device, test_dataloader, args)

		# Save checkpoint
		model_state_keys = list(cur_model_state.keys())
		for key in model_state_keys:
			if ('lf.' in key): # and ('pooler' not in key):
				del cur_model_state[key]

		# Produce reports
		cur_params = utils.get_params(model)
		utils.save_report(prev_params, cur_params, args.checkpoint_dir + ('e-%d-report.txt' % e))
		prev_params = utils.get_params(model)

		if sum(losses) < best_losses:
			best_losses = sum(losses)
			model_file_name = args.checkpoint_dir + datetime.datetime.now().strftime('%Y%m%d-e=' + str(e) + '-%H%M%S') + '.pt'
			torch.save(cur_model_state, model_file_name)
		
		scheduler.step()

	
if __name__ == '__main__':
	main()
