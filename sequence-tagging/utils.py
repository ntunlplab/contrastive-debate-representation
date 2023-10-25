# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import json
import sys

sys.path.append('/nfs/nas-7.1/ybdiau/master-thesis/')
from utils import *

sys.path.append('')

MAX_LEN = 512
MIN_SENTENCE_LEN = 3

# 給定 tagging scheme，回傳 num_labels 和 constraints
def tagging_scheme_config(name):
	assert name in ['bio', 'bioe', 'bilou']
	if name == 'bio':
		bio_num_labels = 3
		constraints = [
			# B->B, B->I, B->O, B->{e}
			(0, 0), (0, 1), (0, 2), (0, 4),
			# I->B, I->I, I->O, I->{e}
			(1, 0), (1, 1), (1, 2), (1, 4),
			# O->B, O->O, O->{e}
			(2, 0), (2, 2), (2, 4),
			# {s}->B, {s}->O
			(3, 0), (3, 2), 
			# cannot transition from {e}
		]
	elif name == 'bioe':
		bio_num_labels = 4
		constraints = [
			# B->I
			(0, 1),
			# I->I, I->E
			(1, 1), (1, 3),
			# O->B, O->O, O->{e}
			(2, 0), (2, 2), (2, 5),
			# E->B, E->O, E->{e}
			(3, 0), (3, 2), (3, 5),
			# {s}->B, {s}->O
			(4, 0), (4, 2)
			# cannot transition from {e}
		]
	elif name == 'bilou':
		# (B: 0)  (I: 1)  (L: 2)  (O: 3)  (U: 4)  ({s}: 5)  ({e}: 6)
		bio_num_labels = 5
		constraints = [
			# B->I, B->L
			(0, 1), (0, 2),
			# I->I, I->L
			(1, 1), (1, 2),
			# L->B, L->O, L->U, L->{e}
			(2, 0), (2, 3), (2, 4), (2, 6),
			# O->B, O->O, O->U, O->{e}
			(3, 0), (3, 3), (3, 4), (3, 6),
			# U->B, U->O, U->U, U->{e}
			(4, 0), (4, 3), (4, 4), (4, 6),
			# {s}->B, {s}->O, {s}->U
			(5, 0), (5, 3), (5, 4)
			# cannot transition from {e}
		]
	
	return bio_num_labels, constraints

# 現在不能區別不同種類的 adu (testimony, value, ...)
class AduDataset(Dataset):
	'''  
	Generate our dataset.

	f_path: a dataset of sentences and tags. Its format is expected to be as follows:
		file content:
			-DOCSTART-
			<SENTENCE1>
			-DOCSTART-
			<SENTENCE2>
			...
		<SENTENCE>:
			word	_	_	<TAG>
		<TAG>:
			'B-'something or 'I-'something or 'O'
	'''
	def __init__(self, f_path, is_uncased, tokenizer, tagging_scheme):
		self.sents = [] # Each element of self.sents is a list, of the form [word1, word2, ...] from one sentence.
		self.tags = [] # Similar to self.sents, except for that this contains tags.
		self.tokenizer = tokenizer
		self.tagging_scheme = tagging_scheme

		assert tagging_scheme in ['bio', 'bioe', 'bilou']
		self.VOCAB = list(tagging_scheme.upper())
		self.tag2idx = {tag: idx for idx, tag in enumerate(self.VOCAB)}
		self.idx2tag = {idx: tag for idx, tag in enumerate(self.VOCAB)}
		print("VOCAB:", self.VOCAB)

		# > Read f_path file
		with open(f_path) as f:
			current_sentence_words = []
			current_sentence_tags = []

			for line in f.readlines():
				# Upon encountering -DOCSTART- every time, close the previous sentence
				if line.startswith('-DOCSTART'):
					assert len(current_sentence_words) == len(current_sentence_tags)
					if len(current_sentence_words) >= MIN_SENTENCE_LEN:
						self.sents.append(current_sentence_words)
						self.tags.append(current_sentence_tags)
						current_sentence_words = []
						current_sentence_tags = []

				# Otherwise: add one word & tag into current sentence
				# Word is at position 0, and Tag is at position 3 (see the format above)
				else:
					feats = line.rstrip('\n').split()
					assert (feats[3].startswith('B') or feats[3].startswith('I') or feats[3].startswith('O')) 
					feats[0] = feats[0].lower() if is_uncased else feats[0]
					current_sentence_words.append(feats[0])
					current_sentence_tags.append(feats[3][0])

			# Last sentence does not have a succeeding -DOCSTART- to end it. We manually add it here.
			assert len(current_sentence_words) == len(current_sentence_tags)
			if len(current_sentence_words) >= MIN_SENTENCE_LEN:
				self.sents.append(current_sentence_words)
				self.tags.append(current_sentence_tags)
				current_sentence_words = []
				current_sentence_tags = []
		
		# 對於 BIOE，要去除含 B-B 和 B-O 的資料
		if self.tagging_scheme == 'bioe':
			data_len = len(self.sents)
			for idx in range(data_len - 1, -1, -1):
				words, tags = self.sents[idx], self.tags[idx]
				is_valid = True
				for tag_idx in range(len(tags) - 1):
					if (tags[tag_idx] == 'B' and tags[tag_idx+1] == 'B') or \
					(tags[tag_idx] == 'B' and tags[tag_idx+1] == 'O'):
						print("Invalid data: ", words)
						is_valid = False
						break
				if not is_valid:
					del self.sents[idx]
					del self.tags[idx]
		
		# < Read f_path file
		print("f_path is {} ".format(f_path))

	def __getitem__(self, idx):
		words, tags = self.sents[idx], self.tags[idx]
		#token_ids = self.tokenizer.convert_tokens_to_ids(words)
		#label_ids = [tag2idx[tag] for tag in tags]

		# words 是一個 list of words (str)。
		# 因為每一個 words 可能還包含了 wordpiece (比如說，'wordpiece' 是 'word' + '##piece')
		# 也需要視情況修改 tags 的 list。
		# 比如說，" wordpiece is nice . Right ? " 原本是 "B I I O O O"
		# 但是因為實際可以拆成 " word ##piece is nice . Right ? "
		# tags 需要修改成 "B 'I' I I O O O"
		# 具體來說，
		# B-word -> B I I ...
		# I-word -> I I I ...
		# O-word -> O O O ...
		token_ids = []
		label_ids = []

		for word, tag in zip(words, tags):
			token_id_list = self.tokenizer.tokenize(word)
			subsequent_tag = { 'B': 'I', 'I': 'I', 'O': 'O' }
			token_ids.extend(token_id_list)
			label_ids.append(tag)
			
			# 如果一個字被拆成至少 n >= 2 個部分：額外插入 (n-1) 個 subsequent tags
			if len(token_id_list) >= 2:
				label_ids.extend([subsequent_tag[tag] for _ in range(len(token_id_list) - 1)])
				
		# token_ids 如果太長，則需要 clip（這個狀況下，一併 clip label_ids）。並且加入 [cls] 和 [sep]
		assert len(token_ids) == len(label_ids)
		cls = self.tokenizer.cls_token
		sep = self.tokenizer.sep_token
		label_ids = label_ids[:MAX_LEN-2] if len(token_ids) > MAX_LEN-2 else label_ids
		token_ids = token_ids[:MAX_LEN-2] if len(token_ids) > MAX_LEN-2 else token_ids
		token_ids = [cls] + token_ids + [sep]

		# label_ids 此時的長度，應是 (len(token_ids) - 2)
		token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

		# 並且，如果 tagging scheme 是 bioe 或是 bilou，label sequence 還需要特別處理。
		if self.tagging_scheme == 'bioe':
			# B->B、B->O、B->{e}: 不能出現
			# I->B、I->O、I->{e}: 第一個 I 要轉成 {e}
			for i in range(len(label_ids) - 1):
				if (label_ids[i] == 'B' and label_ids[i+1] == 'B') or \
				   (label_ids[i] == 'B' and label_ids[i+1] == 'O'):
					print(label_ids)
					raise Exception("Invalid tag sequence."+str(label_ids)+str(words))
				if (label_ids[i] == 'I' and label_ids[i+1] == 'B') or \
				   (label_ids[i] == 'I' and label_ids[i+1] == 'O'):
					label_ids[i] = 'E'

			if label_ids[-1] == 'B':
				raise Exception("Invalid tag sequence.")
			if label_ids[-1] == 'I':
				label_ids[-1] = 'E'

		elif self.tagging_scheme == 'bilou':
			# B->B、B->O、B->{e}: 第一個 B 要改成 U
			# I->B、I->O、I->{e}: 第一個 I 要改成 L
			for i in range(len(label_ids) - 1):
				if (label_ids[i] == 'B' and label_ids[i+1] == 'B') or \
				   (label_ids[i] == 'B' and label_ids[i+1] == 'O'):
					label_ids[i] = 'U'
				if (label_ids[i] == 'I' and label_ids[i+1] == 'B') or \
				   (label_ids[i] == 'I' and label_ids[i+1] == 'O'):
					label_ids[i] = 'L'
			pass
		label_ids = [self.tag2idx[tag] for tag in label_ids]
		
		seqlen = len(token_ids)
		return token_ids, label_ids, seqlen

	def __len__(self):
		return len(self.sents)

# 提供給 JaccardPairs 使用的 Dataset，其 inference 結果將進一步提供給下游的 persuasiveness 任務
class JaccardPairsDataset(Dataset):
	''' 
	Generate our dataset.

	f_path: the parsed jaccard_pair file
	'''
	def __init__(self, f_path, is_uncased, tokenizer, sentence_tokenizer):
		# 此處的格式參見對 Jaccard Pairs 檔案格式的說明。
		# 因為原檔案格式複雜，因此會將每則貼文、討論串、留言中的文字，拆成句子，並提供 index mapping
		# sentence index <--> post index, is_threads, pair index, is_positive, paragraph index, sentence index
		self.posts : list[Post] = []
		self.tokenizer = tokenizer
		self.sentence_tokenizer = sentence_tokenizer
		# 用來做 mapping 的 prefix arrays，總長 (#posts)
		self.prefix_posts : list[int] = []
		# =============

		# > Read f_path file
		with open(f_path) as f:
			for line in f:
				line = line.rstrip()
				post = eval(line)
				self.__extract_post(post)

	def __getitem__(self, idx):
		post_index, is_threads, pair_index, is_positive, comment_index, paragraph_index, sentence_index = self.map_sentence_index(idx)

		if is_threads == 0:
			paragraph_text = self.posts[post_index].root.paragraphs[paragraph_index]
			sentence_span = self.posts[post_index].root.sentence_spans[paragraph_index][sentence_index]
			sentence_text = paragraph_text[sentence_span[0]:sentence_span[1]]
		else:
			paragraph_text = self.posts[post_index].pairs[pair_index].threads[is_positive].comments[comment_index].paragraphs[paragraph_index]
			sentence_span = self.posts[post_index].pairs[pair_index].threads[is_positive].comments[comment_index].sentence_spans[paragraph_index][sentence_index]
			sentence_text = paragraph_text[sentence_span[0]:sentence_span[1]]
		
		return sentence_text

	# 長度以句子為單位
	def __len__(self):
		return self.prefix_posts[-1]

	def map_sentence_index(self, index):
		# 這裡可以替換成 binary search
		for i in range(len(self.prefix_posts)):
			# 找到了！
			if index < self.prefix_posts[i]:
				post_index = i
				break
		else:
			raise Exception('index not found')

		# 開始 lookup
		lookup_index = self.prefix_posts[post_index - 1] if post_index > 0 else 0
		is_threads = 0
		pair_index = -1 
		is_positive = -1
		comment_index = -1
		paragraph_index = 0
		sentence_index = 0

		while lookup_index < index:
			lookup_index += 1
			# 查找 root
			if is_threads == 0:
				num_current_paragraph_sentences = self.posts[post_index].root.num_sentences_in_paragraph(paragraph_index)

				sentence_index += 1
				if sentence_index >= num_current_paragraph_sentences:
					sentence_index = 0
					paragraph_index += 1
				if paragraph_index >= self.posts[post_index].root.num_paragraphs():
					is_threads = 1
					pair_index = 0
					is_positive = 0
					comment_index = 0
					paragraph_index = 0
			# ==================================
			# 查找 threads
			else:
				num_current_paragraph_sentences = self.posts[post_index].pairs[pair_index].threads[is_positive].comments[comment_index].num_sentences_in_paragraph(paragraph_index)

				sentence_index += 1
				if sentence_index >= num_current_paragraph_sentences:
					sentence_index = 0
					paragraph_index += 1
				if paragraph_index >= self.posts[post_index].pairs[pair_index].threads[is_positive].comments[comment_index].num_paragraphs():
					paragraph_index = 0
					comment_index += 1
				if comment_index >= self.posts[post_index].pairs[pair_index].threads[is_positive].num_comments():
					comment_index = 0
					is_positive += 1
				if is_positive == 2:
					is_positive = 0
					pair_index += 1
				if pair_index >= self.posts[post_index].num_pairs():
					pair_index = 0
					raise Exception('searched through the entire post...')

		return post_index, is_threads, pair_index, is_positive, comment_index, paragraph_index, sentence_index

	def __extract_post(self, raw_post):
		pc_author = raw_post['op_info']['author'],
		pc_title = raw_post['op_info']['title']
		num_sentences = 0

		for paragraph_idx in range(len(raw_post['op_info']['body'])):
			raw_post['op_info']['body'][paragraph_idx] += '\n'
		root = Comment(pc_author, raw_post['op_info']['body'], self.sentence_tokenizer, None)
		num_sentences += root.num_sentences

		# for each pair...
		pairs = []
		for raw_pair in raw_post['pairs']:
			assert len(raw_pair) == 2
			threads = []
			# 每個 pair 有 2 個 threads
			for raw_thread in raw_pair:
				# 每個 thread 是由數個 Comments 組成
				prev_comment = root
				thread_comments = []
				for raw_comment in raw_thread:
					comment_author = raw_comment['author']
					comment_paragraphs = raw_comment['body']
					comment = Comment(comment_author, comment_paragraphs, self.sentence_tokenizer, prev_comment)
					prev_comment = comment
					thread_comments.append(comment)
					num_sentences += comment.num_sentences
				thread = Thread(thread_comments)
				threads.append(thread)
			pair = Pair(threads)
			pairs.append(pair)

		post = Post(pc_author, pc_title, root, pairs)

		# 用 prefix_posts 記錄每個 post 的 prefix 句數
		try:
			self.prefix_posts.append(self.prefix_posts[-1] + num_sentences)
		except:
			self.prefix_posts = [num_sentences]

		self.posts.append(post)

# 提供給 Original Posts 使用的 Dataset，其 inference 結果將進一步提供給下游的 contrastive 任務
class OriginalPostsDataset(Dataset):
	''' 
	Generate our dataset.

	f_path: the parsed jaccard_pair file
	'''
	def __init__(self, f_path, is_uncased, tokenizer, sentence_tokenizer):
		# 此處的格式參見對 Jaccard Pairs 檔案格式的說明。
		# 因為原檔案格式複雜，因此會將每則貼文、討論串、留言中的文字，拆成句子，並提供 index mapping
		# sentence index <--> post index, is_threads, pair index, is_positive, paragraph index, sentence index
		self.posts : list[Post] = []
		self.tokenizer = tokenizer
		self.sentence_tokenizer = sentence_tokenizer
		# 用來做 mapping 的 prefix arrays，總長 (#posts)
		self.prefix_posts : list[int] = []
		# =============

		# > Read f_path file
		with open(f_path) as f:
			for line in f:
				line = line.rstrip()
				post = eval(line)
				self.__extract_post(post)

	def __getitem__(self, idx):
		post_index, is_threads, pair_index, is_positive, comment_index, paragraph_index, sentence_index = self.map_sentence_index(idx)

		if is_threads == 0:
			paragraph_text = self.posts[post_index].root.paragraphs[paragraph_index]
			sentence_span = self.posts[post_index].root.sentence_spans[paragraph_index][sentence_index]
			sentence_text = paragraph_text[sentence_span[0]:sentence_span[1]]
		else:
			paragraph_text = self.posts[post_index].pairs[pair_index].threads[is_positive].comments[comment_index].paragraphs[paragraph_index]
			sentence_span = self.posts[post_index].pairs[pair_index].threads[is_positive].comments[comment_index].sentence_spans[paragraph_index][sentence_index]
			sentence_text = paragraph_text[sentence_span[0]:sentence_span[1]]
		
		return sentence_text

	# 長度以句子為單位
	def __len__(self):
		return self.prefix_posts[-1]

	def map_sentence_index(self, index):
		# 這裡可以替換成 binary search
		for i in range(len(self.prefix_posts)):
			# 找到了！
			if index < self.prefix_posts[i]:
				post_index = i
				break
		else:
			raise Exception('index not found')

		# 開始 lookup
		lookup_index = self.prefix_posts[post_index - 1] if post_index > 0 else 0
		is_threads = 0
		pair_index = -1 
		is_positive = -1
		comment_index = -1
		paragraph_index = 0
		sentence_index = 0

		while lookup_index < index:
			lookup_index += 1
			# 查找 root
			if is_threads == 0:
				num_current_paragraph_sentences = self.posts[post_index].root.num_sentences_in_paragraph(paragraph_index)

				sentence_index += 1
				if sentence_index >= num_current_paragraph_sentences:
					sentence_index = 0
					paragraph_index += 1
				if paragraph_index >= self.posts[post_index].root.num_paragraphs():
					is_threads = 1
					pair_index = 0
					is_positive = 0
					comment_index = 0
					paragraph_index = 0
			# ==================================
			# 查找 threads
			else:
				num_current_paragraph_sentences = self.posts[post_index].pairs[pair_index].threads[is_positive].comments[comment_index].num_sentences_in_paragraph(paragraph_index)

				sentence_index += 1
				if sentence_index >= num_current_paragraph_sentences:
					sentence_index = 0
					paragraph_index += 1
				if paragraph_index >= self.posts[post_index].pairs[pair_index].threads[is_positive].comments[comment_index].num_paragraphs():
					paragraph_index = 0
					comment_index += 1
				if comment_index >= self.posts[post_index].pairs[pair_index].threads[is_positive].num_comments():
					comment_index = 0
					is_positive += 1
				if is_positive == 2:
					is_positive = 0
					pair_index += 1
				if pair_index >= self.posts[post_index].num_pairs():
					pair_index = 0
					raise Exception('searched through the entire post...')

		return post_index, is_threads, pair_index, is_positive, comment_index, paragraph_index, sentence_index

	def __extract_post(self, raw_post):
		pc_author = raw_post['op_info']['author'],
		pc_title = raw_post['op_info']['title']
		num_sentences = 0

		for paragraph_idx in range(len(raw_post['op_info']['body'])):
			raw_post['op_info']['body'][paragraph_idx] += '\n'
		root = Comment(pc_author, raw_post['op_info']['body'], self.sentence_tokenizer, None)
		num_sentences += root.num_sentences

		# for each pair...
		pairs = []
		for raw_pair in raw_post['pairs']:
			assert len(raw_pair) == 2
			threads = []
			# 每個 pair 有 2 個 threads
			for raw_thread in raw_pair:
				# 每個 thread 是由數個 Comments 組成
				prev_comment = root
				thread_comments = []
				for raw_comment in raw_thread:
					comment_author = raw_comment['author']
					comment_paragraphs = raw_comment['body']
					comment = Comment(comment_author, comment_paragraphs, self.sentence_tokenizer, prev_comment)
					prev_comment = comment
					thread_comments.append(comment)
					num_sentences += comment.num_sentences
				thread = Thread(thread_comments)
				threads.append(thread)
			pair = Pair(threads)
			pairs.append(pair)

		post = Post(pc_author, pc_title, root, pairs)

		# 用 prefix_posts 記錄每個 post 的 prefix 句數
		try:
			self.prefix_posts.append(self.prefix_posts[-1] + num_sentences)
		except:
			self.prefix_posts = [num_sentences]

		self.posts.append(post)

def SeqTaggingPadBatch(batch, dataset):
	# batch= [(tokens, labels, seqlen), ...]
	pad = dataset.tokenizer.convert_tokens_to_ids([dataset.tokenizer.pad_token])[0]
	maxlen = max([i[2] for i in batch])
	token_tensors = torch.LongTensor([i[0] + [pad] * (maxlen - len(i[0])) for i in batch])
	label_tensors = torch.LongTensor([i[1] + [dataset.tag2idx['O']] * (maxlen - len(i[1])) for i in batch])

	crf_mask = torch.LongTensor([([1] * len(i[1])) + ([0] * (maxlen - len(i[1]))) for i in batch])
	input_mask = (token_tensors != pad)

	return token_tensors, label_tensors, input_mask, crf_mask

def SeqTaggingPadBatchInference(batch, tokenizer):
	cls_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0]
	sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]
	pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
	sentences = [i for i in batch]
	sentences = [tokenizer(sentence, return_offsets_mapping=True) for sentence in sentences]

	input_ids = [sentence['input_ids'] for sentence in sentences]
	mappings = [sentence['offset_mapping'] for sentence in sentences]
	assert all([mapping[0] == (0, 0) and mapping[-1] == (0, 0) for mapping in mappings])
	mappings = [mapping[1:-1] for mapping in mappings]

	maxlen = max([len(input_id) for input_id in input_ids])

	token_tensors = torch.LongTensor([input_id + ([0] * (maxlen - len(input_id))) for input_id in input_ids ])
	mask = (token_tensors != pad_token_id)
	crf_mask = (token_tensors != pad_token_id) & (token_tensors != cls_token_id) & (token_tensors != sep_token_id) 

	return token_tensors, mask, crf_mask, mappings


