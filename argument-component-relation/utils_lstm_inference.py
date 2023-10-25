import pickle
import nltk
from utils import *
import torch
import os

# ===========================

# 用作潛在的 adu pair link type dataset (link type task)

class AduPairLinkTypeDataset:
	def __init__(self, input_parent_path, output_parent_path):
		# parent path 底下包含多個 *.data 的檔案，每個是一個 Post
		input_files = os.listdir(input_parent_path)
		input_files = [file for file in input_files if file.endswith('.data')]
		output_files = os.listdir(output_parent_path)
		output_files = [file for file in output_files if file.endswith('.data')]

		# 讀取所有的 Post，以及計算它們的 link rate 和 validity rate
		# validity rate 如果是全 valid，則以 None 表示 (避免 floating point issues)
		# 否則應該是一個 0~1 的數字
		# 要跑一次 setup_real_adu_pairs 統計該 post 有多少 real adu pairs
		# 如果 post stats 不存在，就當場計算
		self.posts = []
		self.post_filenames = []
		self.validity_rates = { }
		self.link_rates = { }
		for file in input_files:
			with open(input_parent_path + file[:-5] + '-stats.txt', 'r') as f:
				stats = f.read()
				stats = stats.split('\n')
				validity_rate = stats[0].split(':')[1].lstrip().split('/')
				link_rate = stats[1].split(':')[1].lstrip().split('/')
				validity_rate = None if validity_rate[0] == validity_rate[1] else (int(validity_rate[0]) / int(validity_rate[1]))
				link_rate = (int(link_rate[0]) / int(link_rate[1]))
				self.validity_rates[file[:-5]] = validity_rate
				self.link_rates[file[-5]] = link_rate

			# 如果 file 已經被處理過 (在 output_parent_path 存在)，要跳過
			if file not in output_files:
				with open(input_parent_path + file, 'rb') as f:
					post = pickle.load(f)
					post.setup_real_adu_pairs()
					self.posts.append(post)
					self.post_filenames.append(file[:-5])

		self.inner = AduPairLinkTypeDataset.Pairs(self, 'inner')
		self.inter = AduPairLinkTypeDataset.Pairs(self, 'inter')

	# 回傳 (Comment, idx_in_comment, PAIR, 'inner')
	def inner_pairs_of_post(self, post_idx):
		return self.inner.pairs_of_post(post_idx)
	
	# 回傳 (Comment, idx_in_comment, PAIR, 'inter')
	def inter_pairs_of_post(self, post_idx):
		return self.inter.pairs_of_post(post_idx)

	class Pairs:
		def __init__(self, dataset, type):
			assert type in ['inner', 'inter']
			self.num_real_pairs = 0
			self.dataset = dataset
			self.type = type

			if type == 'inner':
				for post in dataset.posts:
					self.num_real_pairs += post.num_real_inner_adu_pairs
			elif type == 'inter':
				for post in dataset.posts:
					self.num_real_pairs += post.num_real_inter_adu_pairs

		# 回傳 ((post_index + Comment 資訊), idx_in_comment, PAIR, 'inner')
		# 或是 ((post_index + Comment 資訊), idx_in_comment, PAIR, 'inter')
		def pairs_of_post(self, post_idx):
			pairs = []
			post = self.dataset.posts[post_idx]

			if self.type == 'inner':
				for (is_thread, pair_index, is_positive, comment_index, comment) in post.get_comments():
					for pair_idx in range(comment.num_real_inner_adu_pairs()):
						# 形式是 (span, span, idx)
						span_1, span_2, idx_in_comment = comment.get_real_inner_adu_pairs(pair_idx)
						pairs.append( ([post_idx, is_thread, pair_index, is_positive, comment_index, comment], idx_in_comment, (span_1, span_2), 'inner') )
			elif self.type == 'inter':
				for (is_thread, pair_index, is_positive, comment_index, comment) in post.get_comments():
					for pair_idx in range(comment.num_real_inter_adu_pairs()):
						# 形式是 (span, span, idx)
						span_1, span_2, idx_in_comment = comment.get_real_inter_adu_pairs(pair_idx)
						pairs.append( ([post_idx, is_thread, pair_index, is_positive, comment_index, comment], idx_in_comment, (span_1, span_2), 'inter') )

			return pairs

# lookup 是 dict[post_str] -> ids
def PadBatchLinkType(batch, tokenizer, lookup):
	MAX_SEQ_LEN = 4096
	IS_INNER_POST = +0.5
	IS_INTER_POST = -0.5
	cls = tokenizer.convert_tokens_to_ids([ tokenizer.cls_token ])
	sep = tokenizer.convert_tokens_to_ids([ tokenizer.sep_token ])
	unk = tokenizer.convert_tokens_to_ids([ tokenizer.unk_token ])
	user1 = tokenizer.convert_tokens_to_ids(['[USER1]'])
	user2 = tokenizer.convert_tokens_to_ids(['[USER2]'])
	pad_id = tokenizer.convert_tokens_to_ids([ tokenizer.pad_token ])[0]

	batch_size = len(batch)
	comment_tensors = []
	adu_1_tensors = []
	adu_2_tensors = []
	is_inner_embeds = []

	comment_infos = []

	for data in batch:
		comment_info, idx_in_comment, pair, type = data
		post_index, is_thread, pair_index, is_positive, comment_index, comment_1 = comment_info
		comment_2 = comment_1 if type == 'inner' else comment_1.prev_comment
		span_1, span_2 = pair

		# ===============================
		if type == 'inner':
			comment_1_text = comment_1.get_comment()
			if comment_1_text not in lookup:
				lookup[comment_1_text] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment_1_text))
			comment_1_ids = lookup[comment_1_text]

			comment_tensor = (user1 + cls + comment_1_ids + sep)

			# Span 的尾巴應該不能超出 MAX_SEQ_LEN。
			assert span_1[1] < MAX_SEQ_LEN and span_2[1] < MAX_SEQ_LEN

			# 如果 Comment 太長，則截斷
			if len(comment_tensor) > MAX_SEQ_LEN:
				comment_tensor = comment_tensor[:MAX_SEQ_LEN-1] + sep

			comment_tensors.append( comment_tensor )
			adu_1_tensors.append([span_1[0], span_1[1]])
			adu_2_tensors.append([span_2[0], span_2[1]])
			is_inner_embeds.append(IS_INNER_POST)
		
		# -------------------------------
		elif type == 'inter':
			comment_1_text = comment_1.get_comment()
			comment_2_text = comment_2.get_comment()
			if comment_1_text not in lookup:
				lookup[comment_1_text] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment_1_text))
			if comment_2_text not in lookup:
				lookup[comment_2_text] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment_2_text))
			
			comment_1_ids = lookup[comment_1_text]
			comment_2_ids = lookup[comment_2_text]

			post_1 = user1 + cls + comment_1_ids + sep
			post_2 = user2 + cls + comment_2_ids + sep
			comment_tensor = post_1 + post_2

			# Span 的尾巴應該不能超出 MAX_SEQ_LEN。
			assert span_1[1] < MAX_SEQ_LEN and span_2[1] < MAX_SEQ_LEN

			# 如果 Comment 太長，則截斷
			if len(comment_tensor) > MAX_SEQ_LEN:
				comment_tensor = comment_tensor[:MAX_SEQ_LEN-1] + sep

			comment_tensors.append( comment_tensor )
			adu_1_tensors.append([span_1[0], span_1[1]])
			adu_2_tensors.append([span_2[0], span_2[1]])
			is_inner_embeds.append(IS_INTER_POST)

		comment_infos.append( (post_index, is_thread, pair_index, is_positive, comment_index) )

	max_len = max([len(comment_tensor) for comment_tensor in comment_tensors])
	comment_tensors = [comment_tensor + ([pad_id] * (max_len - len(comment_tensor))) for comment_tensor in comment_tensors]
	comment_tensors = torch.LongTensor(comment_tensors).view(batch_size, max_len)
	comment_masks = (comment_tensors != pad_id)
	adu_1_tensors = torch.LongTensor(adu_1_tensors).view(batch_size, 2)
	adu_2_tensors = torch.LongTensor(adu_2_tensors).view(batch_size, 2)
	is_inner_embeds = torch.FloatTensor(is_inner_embeds).view(batch_size, 1)

	return comment_tensors, comment_masks, adu_1_tensors, adu_2_tensors, is_inner_embeds, comment_infos

# ===========================

# 用作潛在的 adu pair link dataset (link task)
class PotentialAduPairInferenceDataset:
	def __init__(self, f_path):

		with open(f_path, 'rb') as f:
			self.posts = pickle.load(f)

		sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		# 所有的 post.root，要再加入 title，做為它的第一個段落和 ADU
		for post in self.posts:
			paragraphs = [post.title + '\n'] + post.root.paragraphs
			adu_spans = [(-len(paragraphs[0]), 0)] + post.root.adu_spans
			adu_spans = [(span[0] + len(paragraphs[0]), span[1] + len(paragraphs[0])) for span in adu_spans]
			post.root = Comment(post.author, paragraphs, sentence_tokenizer, None)
			post.root.adu_spans = adu_spans
			post.root.prev_comment = None

		# 需要指定每個 Comment 的 prev_comment
		# 尤其，每個 thread 的第一個 comment 的 prev_comment 會指向尚未修改的 Comment...
		for post in self.posts:
			for pair_idx, pair in enumerate(post.pairs):
				for thread_idx, thread in enumerate(pair.threads):
					prev_comment = post.root
					for comment_idx, comment in enumerate(thread.comments):
						assert comment == post.get_comment(1, pair_idx, thread_idx, comment_idx)
						comment.prev_comment = prev_comment
						prev_comment = comment

		# post 要跑過 setup
		[post.setup_potential_adu_pairs() for post in self.posts]

		self.inner_pairs = PotentialAduPairInferenceDataset.Pairs(self, 'inner')
		self.inter_pairs = PotentialAduPairInferenceDataset.Pairs(self, 'inter')

		# 計算 inner_pair_prefixes 和 inter_pair_prefixes，
		# 使得可以得知每個 Post 個別有哪些 pair。
		# pair_prefixes: [post #1 的總 pair 數量, post #1~2 的總 pair 數量, ...]
		inner_pair_prefixes = [(0,0)]
		inter_pair_prefixes = [(0,0)]
		for post in self.posts:
			inner_pair_prefixes.append( 
				(inner_pair_prefixes[-1][1], 
				 inner_pair_prefixes[-1][1] + post.num_potential_inner_adu_pairs) )
			inter_pair_prefixes.append( 
				(inter_pair_prefixes[-1][1], 
				 inter_pair_prefixes[-1][1] + post.num_potential_inter_adu_pairs) )
		
		self.inner_pair_prefixes = inner_pair_prefixes[1:]
		self.inter_pair_prefixes = inter_pair_prefixes[1:]
		assert self.inner_pair_prefixes[-1][1] == len(self.inner_pairs)
		assert self.inter_pair_prefixes[-1][1] == len(self.inter_pairs)

	def inner_pairs_of_post(self, post_idx):
		pair_index = self.inner_pair_prefixes[post_idx]
		pairs = []
		for idx in range(pair_index[0], pair_index[1]):
			pairs.append(self.inner_pairs[idx])
		return pairs
		
	def inter_pairs_of_post(self, post_idx):
		pair_index = self.inter_pair_prefixes[post_idx]
		pairs = []
		for idx in range(pair_index[0], pair_index[1]):
			pairs.append(self.inter_pairs[idx])
		return pairs

	class Pairs:
		def __init__(self, dataset, type):
			assert type in ['inner', 'inter']
			self.num_potential_pairs = 0
			self.dataset = dataset
			self.type = type

			if type == 'inner':
				for post in dataset.posts:
					self.num_potential_pairs += post.num_potential_inner_adu_pairs
			elif type == 'inter':
				for post in dataset.posts:
					self.num_potential_pairs += post.num_potential_inter_adu_pairs

		# inner: 回傳 ((post_index + Comment 資訊), idx_in_comment, PAIR, 'inner')
		# inter: 回傳 ((post_index + Comment 資訊), idx_in_comment, PAIR, 'inter')
		def __getitem__(self, idx):
			idx_copy = idx
			for post_index in range(len(self.dataset.posts)):
				post = self.dataset.posts[post_index]
				if self.type == 'inner':
					num_pairs = post.num_potential_inner_adu_pairs
					if idx_copy < num_pairs:
						comment_info, idx_in_comment, pair, type = post.get_potential_inner_adu_pair(idx_copy)
						comment_info = ([post_index] + list(comment_info))
						return (comment_info, idx_in_comment, pair, type)
					idx_copy -= num_pairs

				elif self.type == 'inter':
					num_pairs = post.num_potential_inter_adu_pairs
					if idx_copy < num_pairs:
						comment_info, idx_in_comment, pair, type = post.get_potential_inter_adu_pair(idx_copy)
						comment_info = ([post_index] + list(comment_info))
						return (comment_info, idx_in_comment, pair, type)
					idx_copy -= num_pairs

		def __len__(self):
			return self.num_potential_pairs

def ids_to_string(tokenizer, ids):
	return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))

# 給定 post_ids (來自 tokenizer)，以及 adu (text)
# 尋找是否有一條 post_ids 的 slice，可以完全符合 adu_ids
# 有的話回傳 inclusive interval (i, j)
# 否則回傳 None
def get_adu_span_index(tokenizer, post_ids, adu):
	# 尤其使用在如 Roberta 等 sub-word tokenization，
	# substring 的 token list 不一定會在 string 的 token list 中，
	# 因此需要特殊的方式搜尋。
	# 假定 post_ids 轉回文字後，adu 是在 文字內的
	adu_in_post = adu in ids_to_string(tokenizer, post_ids)
	assert adu_in_post
	
	# 搜尋 start & end 區間
	start = 0
	end = len(post_ids)

	for start in range(0, len(post_ids)):
		post_slice = post_ids[start:end]
		adu_in_slice = adu in ids_to_string(tokenizer, post_slice)
		if not adu_in_slice:
			start -= 1
			break

	for end in range(len(post_ids), start, -1):
		post_slice = post_ids[start:end]
		adu_in_slice = adu in ids_to_string(tokenizer, post_slice)
		if not adu_in_slice:
			break
		
	return (start, end)

# comment: Comment
# adu: (int, int)
def get_span_from_lookup(tokenizer, comment, adu, lookup):
	comment_text = comment.get_comment()
	adu_text = comment_text[adu[0]:adu[1]]
	if not comment_text in lookup:
		comment_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment_text))
		lookup[comment_text] = { 'ids': comment_ids }
	if not adu in lookup[comment_text]:
		comment_ids = lookup[comment_text]['ids']
		span = get_adu_span_index(tokenizer, comment_ids, adu_text)
		lookup[comment_text][adu] = span
	return lookup[comment_text][adu]

# lookup 是 dict[comment_str] -> dict[adu_str] -> (pair_index)
# 	lookup 可以用 lookup[comment_text][adu] 來查詢對應的 adu span
#	lookup 也可用 lookup[comment_text]['ids'] 來查詢它的 tokenizer ids
# lookup 內的 pair_index，是 inclusive 的，且還沒有考慮任何的 special token (cls, USER1, ...)
def PadBatchLink(batch, tokenizer, lookup):
	MAX_SEQ_LEN = 4096
	IS_INNER_POST = +0.5
	IS_INTER_POST = -0.5
	cls = tokenizer.convert_tokens_to_ids([ tokenizer.cls_token ])
	sep = tokenizer.convert_tokens_to_ids([ tokenizer.sep_token ])
	unk = tokenizer.convert_tokens_to_ids([ tokenizer.unk_token ])
	user1 = tokenizer.convert_tokens_to_ids(['[USER1]'])
	user2 = tokenizer.convert_tokens_to_ids(['[USER2]'])
	pad_id = tokenizer.convert_tokens_to_ids([ tokenizer.pad_token ])[0]

	batch_size = len(batch)
	comment_tensors = []
	adu_1_tensors = []
	adu_2_tensors = []
	is_inner_embeds = []
	validities = []

	comment_infos = []

	for data in batch:
		comment_info, idx_in_comment, pair, type = data
		post_index, is_thread, pair_index, is_positive, comment_index, comment_1 = comment_info
		comment_2 = comment_1 if type == 'inner' else comment_1.prev_comment
		adu_1, adu_2 = pair
		span_1 = get_span_from_lookup(tokenizer, comment_1, adu_1, lookup)
		span_2 = get_span_from_lookup(tokenizer, comment_2, adu_2, lookup)

		# ===============================
		if type == 'inner':
			comment_tensor = (user1 + cls + lookup[ comment_1.get_comment() ]['ids'] + sep)

			validity = True
			# 超出長度的 comment
			if len(comment_tensor) > MAX_SEQ_LEN:
				# 如果兩個 ADU 的尾巴都不是停在 sep 前一個
				if (span_1[1] + 2) < (len(comment_tensor) - 2) and \
				   (span_2[1] + 2) < (len(comment_tensor) - 2):
					comment_tensor = comment_tensor[:MAX_SEQ_LEN-1] + sep
					span_1, span_2 = list(span_1), list(span_2)
					span_1[1] = min(span_1[1], MAX_SEQ_LEN - 1 - 1 - 2)
					span_2[1] = min(span_2[1], MAX_SEQ_LEN - 1 - 1 - 2)
					span_1[0] = min(span_1[0], span_1[1])
					span_2[0] = min(span_2[0], span_2[1])
				else:
					# print(" :: Invalid N-post (inner) pair of length %d" % len(comment_tensors))
					comment_tensor = user1 + cls + unk + sep + user2 + cls + unk + sep
					span_1, span_2 = [0, 0], [0, 0]
					validity = False

			comment_tensors.append( comment_tensor )
			adu_1_tensors.append([span_1[0] + 2, span_1[1] + 2])
			adu_2_tensors.append([span_2[0] + 2, span_2[1] + 2])
			is_inner_embeds.append(IS_INNER_POST)
			validities.append(validity)
		
		# -------------------------------
		elif type == 'inter':
			post_1 = user1 + cls + lookup[ comment_1.get_comment() ]['ids'] + sep
			post_2 = user2 + cls + lookup[ comment_2.get_comment() ]['ids'] + sep
			post_1_len = len(post_1)
			comment_tensor = post_1 + post_2

			# 超出長度的 comment
			if len(comment_tensor) > MAX_SEQ_LEN:
				# 如果兩個 ADU 的尾巴都不是停在 sep 前一個，截斷 comment 並調整 spans
				if (span_1[1] + 2) < (len(comment_tensor) - 2) and \
				   (span_2[1] + 2 + post_1_len) < (len(comment_tensor) - 2):
					comment_tensor = comment_tensor[:MAX_SEQ_LEN-1] + sep
					span_1 = [span_1[0] + 2, span_1[1] + 2]
					span_2 = [span_2[0] + 2 + post_1_len, span_2[1] + 2 + post_1_len]
					span_1[1] = min(span_1[1], MAX_SEQ_LEN - 1 - 1)
					span_2[1] = min(span_2[1], MAX_SEQ_LEN - 1 - 1)
					span_1[0] = min(span_1[0], span_1[1])
					span_2[0] = min(span_2[0], span_2[1])

					comment_tensors.append( comment_tensor )
					adu_1_tensors.append(span_1)
					adu_2_tensors.append(span_2)
					is_inner_embeds.append(IS_INTER_POST)
					validities.append(True)
				else:
					# print(" :: Invalid T-post (inter) pair of length %d" % len(comment_tensors))
					comment_tensor = user1 + cls + unk + sep + user2 + cls + unk + sep
					span_1, span_2 = [2, 2], [6, 6]
					comment_tensors.append( comment_tensor )
					adu_1_tensors.append(span_1)
					adu_2_tensors.append(span_2)
					is_inner_embeds.append(IS_INTER_POST)
					validities.append(False)
			else:
				comment_tensors.append( comment_tensor )
				adu_1_tensors.append([span_1[0] + 2, span_1[1] + 2])
				adu_2_tensors.append([span_2[0] + 2 + post_1_len, span_2[1] + 2 + post_1_len])
				is_inner_embeds.append(IS_INTER_POST)
				validities.append(True)

		comment_infos.append( (post_index, is_thread, pair_index, is_positive, comment_index) )

	max_len = max([len(comment_tensor) for comment_tensor in comment_tensors])
	comment_tensors = [comment_tensor + ([pad_id] * (max_len - len(comment_tensor))) for comment_tensor in comment_tensors]
	comment_tensors = torch.LongTensor(comment_tensors).view(batch_size, max_len)
	comment_masks = (comment_tensors != pad_id)
	adu_1_tensors = torch.LongTensor(adu_1_tensors).view(batch_size, 2)
	adu_2_tensors = torch.LongTensor(adu_2_tensors).view(batch_size, 2)
	is_inner_embeds = torch.FloatTensor(is_inner_embeds).view(batch_size, 1)

	return comment_tensors, comment_masks, adu_1_tensors, adu_2_tensors, is_inner_embeds, comment_infos, validities
