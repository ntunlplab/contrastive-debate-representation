# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import json
import nltk.data
import os
from itertools import permutations, combinations
import random
import code
from functools import reduce

validate_ratio = 0.15

# 用來處理 ADU Relation 的資料集
class AduRelationDataset(Dataset):
	train_post_indices = None
	validate_post_indices = None
	order_not_applicable = 0
	order_adu_1_first = +1
	order_adu_2_first = -1

	def _get_adu_order(self, adu1, adu2):
		if not self.is_inner_post:
			return AduRelationDataset.order_not_applicable
		if adu1[0] < adu2[0]:
			return AduRelationDataset.order_adu_1_first
		elif adu1[0] > adu2[0]:
			return AduRelationDataset.order_adu_2_first
		# 兩個 ADU 的開頭一樣，比結尾
		else:
			if adu1[1] < adu2[1]:
				return AduRelationDataset.order_adu_1_first
			elif adu1[1] > adu2[1]:
				return AduRelationDataset.order_adu_2_first
			else:
				# TODO 不應該發生。
				return AduRelationDataset.order_not_applicable

	def _to_ids(self, text):
		return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

	def _ids_to_string(self, ids):
		return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(ids))

	def _get_indices(path, is_train):
		if AduRelationDataset.train_post_indices is None:
			post_indices = os.listdir(path)
			random.shuffle(post_indices)

			num_validate_samples = round(validate_ratio * len(post_indices))
			AduRelationDataset.validate_post_indices = post_indices[:num_validate_samples]
			AduRelationDataset.train_post_indices = post_indices[num_validate_samples:]
		if is_train:
			return AduRelationDataset.train_post_indices
		else:
			return AduRelationDataset.validate_post_indices

	def __init__(self, f_path, is_inner_post, is_train, local_negative_sampling_prob, global_negative_sampling_ratio, is_uncased, tokenizer, classification_type, do_duplicates):
		self.is_uncased = is_uncased
		self.tokenizer = tokenizer
		self.is_inner_post = is_inner_post
		self.classification_type = classification_type
		self.local_negative_sampling_prob = local_negative_sampling_prob
		self.global_negative_sampling_ratio = global_negative_sampling_ratio
		self.is_train = is_train
		self.do_duplicates = do_duplicates
		self.f_path = f_path
		self.max_len = tokenizer.model_max_length

		assert classification_type in ['link', 'link_type']
		assert tokenizer.name_or_path in ['bert-base-uncased', 'Jeevesh8/sMLM-RoBERTa', 'Jeevesh8/sMLM-LF']

		self.search_in_id_space = {
			'bert-base-uncased': True, 'Jeevesh8/sMLM-RoBERTa': False, 'Jeevesh8/sMLM-LF': False
		}[tokenizer.name_or_path]
		
		# cache 的參數依據是：is_inner_post, is_train, is_uncased, tokenizer, classification_type
		# inner_post: N, inter_post: T
		# is_train: T, is_valid: V
		# is_uncased: U, is_cased: C
		# tokenizer:
		# 	(bert-base-uncased): B
		# 	(Jeevesh8/sMLM-RoBERTa): R
		#	(Jeevesh8/sMLM-LF): L
		# link: L, link_type: T
		# ------
		# 如果是 link：{ adu_pairs: [...], adus: {...} }
		#		可以再使用 adu_pairs 計算 global neg sampling，以及用 adus 計算 local neg sampling
		# 如果是 link type:{ adu_pairs: [...], adus: {...} }
		# ====================
		cache_key = '%s%s%s%s%s' % ( # 這裡不小心組了一個 NTU 耶 (下面三行的行頭)
			'N' if is_inner_post else 'T',
			'T' if is_train else 'V',
			'U' if is_uncased else 'C',
			{ 'bert-base-uncased': 'B', 'Jeevesh8/sMLM-RoBERTa': 'R', 'Jeevesh8/sMLM-LF': 'L' }[tokenizer.name_or_path],
			'L' if classification_type == 'link' else 'T'
		)
		cache_path = f_path + '/' + cache_key + '.data'
		is_cache_exist = os.path.isfile(cache_path)
		# ====================

		# 文章的原文查詢。
		# key 比如說是 001_op, 009_negative, ..., value 則是該 post 的原文 (尚未 tokenize)
		self.posts = { } 

		# 記錄所有的 ADU Pairs。
		# key 是 post_key (inner) 或 (post_key, op_key) (inter)
		# value 是一個 array:
		# 	每個 element 是 [arg1, arg2, relation, post_key] 或 [arg1, arg2, relation, [post_key_1, post_key_2]]
		self.adu_pairs = { }

		# 記錄所有的 post key -> adus
		# inner: post_key -> {adus}
		# inter: (post_key, op_key) -> {adus}
		self.adus = { }

		# 以 LREC 來說是 001~121 的資料夾
		# ===============================
		# Posts
		# ===============================
		self.post_indices = AduRelationDataset._get_indices(f'{f_path}/inner_post/', is_train)
		post_types = ['positive', 'negative', 'op']
		for post_index in self.post_indices:
			for post_type in post_types:
				self.__evaluate_post(post_index, post_type, f'{f_path}/inner_post/{post_index}/{post_type}.txt')
				
		# Cached Data?
		# 如果存在的話讀取，不存在的話從頭建置，並儲存
		if is_cache_exist:
			self.recover_from_cache(cache_path)
		else:
			self.build_from_scratch()
			# Save to cache file
			with open(cache_path, 'w') as f:
				f.write( str({ 'adu_pairs': self.adu_pairs, 'adus': self.adus }) )
		
		# 在 inner-post link 的資料裡，adu_1 和 adu_2 先出現，或是後出現的比例是差不多的。

		# Do Local Negative Sampling (if link prediction)
		# Do negative sampling 'None' relation: nCr(adus, 2) for link prediction
		local_negative_adu_pairs = []
		if self.classification_type == 'link' and local_negative_sampling_prob > 0:
			for post_key in self.adus:
				# 這個 post_key 的所有 adu
				adus = self.adus[post_key]
				adus_keys = list(adus.keys())
				# 這個 post_key 的所有 adu_pairs
				adu_pairs = self.adu_pairs[post_key]

				for adu1_key, adu2_key in permutations(adus_keys, 2):
					adu1, adu2 = adus[adu1_key], adus[adu2_key]
					# 沒有 self-relations。
					# 也跳過 span 完全一樣的狀況（開頭和結尾各自的 XOR 會是 0）
					if adu1_key == adu2_key or ( (adu1['span'][0] ^ adu2['span'][0]) | (adu1['span'][1] ^ adu2['span'][1]) == 0 ): 
						continue
					# 如果是確有存在的，則忽略該 pair。
					# TODO: 這裡可以用 set 優化
					is_real_pair = False
					for adu_pair in adu_pairs:
						if adu1['adu'] == adu_pair[0]['adu'] and adu2['adu'] == adu_pair[1]['adu']:
							is_real_pair = True
							break
					if is_real_pair:
						continue
					# 並且，如果是 inter-post，還需要另外檢查：adu1 要來自 post，adu2 要來自 op
					if not is_inner_post:
						assert len(post_key) == 2 # (post_key, op_key)
						if adu1['source'] != post_key[0] or adu2['source'] != post_key[1]:
							continue
					""" 檢查完成 """

					if random.random() > local_negative_sampling_prob:
						continue

					# Pos/Neg argument relations point to arguments in the same post
					if is_inner_post:
						local_negative_adu_pairs.append([adu1, adu2, 'None', post_key])
					# Pos/Neg argument relations point to arguments in the op
					else:
						local_negative_adu_pairs.append([adu1, adu2, 'None', [post_key[0], post_key[1]]])

		# 把 self.adu_pairs 攤平成 list
		self.adu_pairs = [self.adu_pairs[key] for key in self.adu_pairs]
		self.adu_pairs = reduce(lambda x, y: x + y, self.adu_pairs)

		# Do Global Negative Sampling (if inter_post & link prediction)
		# 此用意在挑選極可能完全無關的句子，會從兩篇不一樣的編號文章中抽取。
		global_negative_adu_pairs = []
		if (not is_inner_post) and classification_type == 'link':
			num_global_negative_samples = round(global_negative_sampling_ratio * len(self.adu_pairs))
			for _ in range(num_global_negative_samples):
				# 不斷隨機挑選，直到選到合理的兩個 adus。
				while True:
					sample_1, sample_2 = random.choice(self.adu_pairs), random.choice(self.adu_pairs)
					index_1, index_2 = random.choice([0, 1]), random.choice([0, 1])
					post_key_1, post_key_2 = sample_1[3][index_1], sample_2[3][index_2]
					adu_1, adu_2 = sample_1[index_1], sample_2[index_2]
					if post_key_1[:3] == post_key_2[:3]: continue
					global_negative_adu_pairs.append([adu_1, adu_2, 'None', [post_key_1, post_key_2]])
					break

		# 加入 negative sampling 的結果
		self.adu_pairs.extend(local_negative_adu_pairs)
		self.adu_pairs.extend(global_negative_adu_pairs)

		# 濾掉不能使用的資料
		self.__filter_items()

		# 重複較少的 class 的資料
		if do_duplicates:
			self.__duplicate_items()


	# 給定的設定已經有 cache 時，讀取 cache
	def recover_from_cache(self, cache_path):
		with open(cache_path, 'r') as f:
			cached_data = eval(f.read())
		self.adu_pairs = cached_data['adu_pairs']
		self.adus = cached_data['adus']

	# 給定的設定沒有 cache，需要從頭建
	# 目標是建置出 self.adu_pairs 和 self.adus
	def build_from_scratch(self):
		# ===============================
		# Relations
		# ===============================
		inner_inter_path = 'inner_post' if self.is_inner_post else 'inter_post'
		for post_index in self.post_indices:
			# ** Positive & Negative
			self.__evaluate_relations(
				self.is_inner_post, 
				post_index + '_positive', post_index + '_op', 
				f'{self.f_path}/{inner_inter_path}/{post_index}/positive.ann',
				f'{self.f_path}/inner_post/{post_index}/op.ann')
			self.__evaluate_relations(
				self.is_inner_post, 
			    post_index + '_negative', post_index + '_op', 
				f'{self.f_path}/{inner_inter_path}/{post_index}/negative.ann',
				f'{self.f_path}/inner_post/{post_index}/op.ann')
				
			# ** OP: Only when doing inner_post.
			if self.is_inner_post:
				self.__evaluate_relations(
					self.is_inner_post, 
					post_index + '_op', None, 
					f'{self.f_path}/{inner_inter_path}/{post_index}/op.ann', 
					None)

	# 得到 per class stats
	def get_dataset_stats(self):
		class_counts = [0, 0]
		for adu_pair in self.adu_pairs:
			_, _, relation_type, _ = adu_pair
			adu_class = self.get_class(relation_type)
			class_counts[adu_class] += 1
		return class_counts

	# 給定 post_path 的 post 原文，儲存其內的文字。
	def __evaluate_post(self, post_index, post_type, post_path):
		with open(post_path, 'r') as f:
			post = ''
			for line in f:
				if line.startswith('op_id: ') and post == '':
					continue
				elif line.startswith('op_title: ') and post == '' and post_type == 'op':
					post += line[10:]
				elif line.startswith('op_title: ') and post == '' and post_type != 'op':
					continue
				else:
					post += line
			self.posts[post_index + '_' + post_type] = post.lower() if self.is_uncased else post

	# 給定 post_path 路徑的 post，解讀其內的 ADUs、relations，並加入 local negative sampling
	# ----------------------------------
	# is_inner_post: 是否為 inner-post，控制資料儲存的格式（如果是 inter-post 則會牽涉兩篇文章）
	# post_key: 所關注的文章，其 post 的 indexing key。可以透過 self.posts 的存取對應的文章。
	# op_key: 如果是回篇其他文章的話，op_key 是 post_key 所回覆的文章；否則留空。
	# local_negative_sampling_prob: 做 local negative sampling 的機率。這裡的 local 指的是只在給定的 post_key, op_key 內。
	# post_ann_path: post_key 代表的 ann 文件路徑。該文件會列出所有的 adus in interest，以及他們的 relations。
	# op_ann_path: op_key 代表的 ann 文件路徑，如上
	def __evaluate_relations(self, is_inner_post, post_key, op_key, post_ann_path, op_ann_path):
		# 讀取 post & op 的文章，並且將他們 identifierize
		# 如果 op_key 是 None，他的 ids list 最後會成為 []
		post_key_post = self.posts[post_key]
		op_key_post = self.posts[op_key] if op_key is not None else ''

		post_key_post_ids = self._to_ids(post_key_post)
		op_key_post_ids = self._to_ids(op_key_post)
		
		# adus = { adu_key: { adu 原文, source post, span (i, j) }, ... }
		# adu_pairs = [ [adu_1, adu_2, relation_type, post_keys], ... ] where adu_1 & 2 is { adu, source }
		adus = { }
		adu_pairs = []

		# ===================================
		# Inter-Post 的處理
		# ===================================
		if not is_inner_post:
			adus_in_op = []

			# 收集 op 內的所有 adus
			with open(op_ann_path, 'r') as f:
				for line in f:
					components = line.rstrip().split('\t')
					if not components[0].startswith('T'): 
						continue
					adu_key, _, adu = components
					adu = adu.lower() if self.is_uncased else adu
					adus_in_op.append(adu)

			# 收集 post 文件內的所有 adus。
			# post_ann 中，前幾個 adu 是來自 op，後幾個 adu 來自 post
			op_test_index = 0
			with open(post_ann_path, 'r') as f:
				for line in f:
					components = line.rstrip().split('\t')

					# 1. Arguments
					if components[0].startswith('T'): 
						assert len(components) == 3
						adu_key, _, adu = components
						
						assert adu_key not in adus
						adu = adu.lower() if self.is_uncased else adu

						# 儲存資料
						# adu: 原文
						# source: 該 adu 來自的文章的 key
						# source 暫時留空，待下面分辨是來自 post 還是 op
						# span 暫時留空，下面再給值
						adus[adu_key] = { 'adu': adu, 'source': None, 'span': None }

						# 因為 post_ann 的前面幾個 adu 是來自 op，
						# 所以用一個 op_test_index 來遍歷已收集的 op adus
						if op_test_index < len(adus_in_op):
							assert adus_in_op[op_test_index] == adu
							op_test_index += 1
							adus[adu_key]['source'] = op_key
							# OP 內的 adu，應該要出現在 OP 文章內
							expected_post_ids = op_key_post_ids
						else:
							adus[adu_key]['source'] = post_key
							# POST 內的 adu，應該要出現在 POST 文章內
							expected_post_ids = post_key_post_ids

						# 找到的這條 adu，應該要出現在對應的文章內 (POST / OP)
						adu_ids = self._to_ids(adu)
						span = self.__get_adu_span_index(expected_post_ids, adu, self.search_in_id_space)
						assert span is not None
						adus[adu_key]['span'] = span

					# 2. Relations
					elif components[0].startswith('R'):
						assert len(components) == 2
						if not (components[1].startswith('Attack') or components[1].startswith('Support')):
							print(':: [Unknown Relation Type]:', line.rstrip())
							continue
						relation_type, arg1, arg2 = components[1].split()
						arg1, arg2 = arg1.split(':')[1], arg2.split(':')[1]

						# arg1 & 2 都應該要存在
						assert arg1 in adus and arg2 in adus

						# Inter-Post 時，arg1 應該要在 post，arg2 應該要在 op
						assert adus[arg1]['source'] == post_key and adus[arg2]['source'] == op_key
						adu_pairs.append([adus[arg1], adus[arg2], relation_type, [post_key, op_key]])
					else:
						assert False
			
			self.adu_pairs[(post_key, op_key)] = adu_pairs
			self.adus[(post_key, op_key)] = adus
			""" Inter-Post 擷取完成。 """
	
		# ===================================
		# Inner-Post 的處理
		# ===================================
		else:
			with open(post_ann_path, 'r') as f:
				for line in f:
					components = line.rstrip().split('\t')
					# 1. Argument
					if components[0].startswith('T'): 
						# 每列的資訊包含：[adu_key], [不需用到的資訊], [adu 原文]
						assert len(components) == 3
						adu_key, _, adu = components

						# 不能出現兩個一樣的 adu_key
						assert adu_key not in adus
						adu = adu.lower() if self.is_uncased else adu

						# 確認 adu 的來源：必須是來自 post
						adu_ids = self._to_ids(adu)
						span = self.__get_adu_span_index(post_key_post_ids, adu, self.search_in_id_space)
						assert span is not None

						# 儲存資料
						# adu: 原文
						# source: 該 adu 來自的文章的 key
						source = post_key
						adus[adu_key] = { 'adu': adu, 'source': source, 'span': span }

					# 2. Relation
					elif components[0].startswith('R'): 
						assert len(components) == 2
						if not (components[1].startswith('Attack') or components[1].startswith('Support')):
							print(':: [Unknown Relation Type]:', line.rstrip())
							continue
						relation_type, arg1, arg2 = components[1].split()
						arg1, arg2 = arg1.split(':')[1], arg2.split(':')[1]
						assert arg1 in adus and arg2 in adus

						# Pos/Neg argument relations point to arguments in the same post
						adu_pairs.append([adus[arg1], adus[arg2], relation_type, post_key])

					# 3. Non-Argument & Non-Relation... should not even be there.
					else:
						assert False

			self.adu_pairs[post_key] = adu_pairs
			self.adus[post_key] = adus
			""" Inner-Post 擷取完成 """

	# 把不合法的物品過濾掉。
	# 確切來說，如果 ADU 的元件不存在 post 內，或是範圍會超過的話，就不行。
	# 並且，如果是要做 link type 的話，還要去除 None。
	# Returns: (原物品數, 過濾後物品數)
	def __filter_items(self):
		num_before_filter = len(self.adu_pairs)
		is_longformer = self.tokenizer.name_or_path == 'Jeevesh8/sMLM-LF'
		reserved = 3 if is_longformer else 2

		if self.classification_type == 'link_type':
			for idx in range(len(self.adu_pairs)-1, -1, -1):
				adu_1, adu_2, relation_type, post = self.adu_pairs[idx]
				if relation_type == 'None':
					del self.adu_pairs[idx]

		for idx in range(len(self.adu_pairs)-1, -1, -1):
			adu_1, adu_2, relation_type, post = self.adu_pairs[idx]
			adu_1, adu_2 = adu_1['span'], adu_2['span']
			assert adu_1 is not None and adu_2 is not None

			# 1. INNER-POST
			if self.is_inner_post:
				post_1 = self.posts[post]
				post_1 = self._to_ids(post_1)
				post_1 = post_1[:self.max_len-reserved] if len(post_1) > self.max_len-reserved else post_1

				# ADU span index 如果比文章長，就不算數。
				if adu_1[0] >= len(post_1) or adu_1[1] >= len(post_1):
					adu_1 = None
				if adu_2[0] >= len(post_1) or adu_2[1] >= len(post_1):
					adu_2 = None
			# 2. INTER-POST: Bert / Roberta
			# Post 1、2 分開處理
			elif not is_longformer:
				post_1, post_2 = self.posts[post[0]], self.posts[post[1]]
				post_1, post_2 = self._to_ids(post_1), self._to_ids(post_1)
				post_1 = post_1[:self.max_len-reserved] if len(post_1) > self.max_len-reserved else post_1
				post_2 = post_2[:self.max_len-reserved] if len(post_2) > self.max_len-reserved else post_2

				# ADU span index 如果比文章長，就不算數。
				if adu_1[0] >= len(post_1) or adu_1[1] >= len(post_1):
					adu_1 = None
				if adu_2[0] >= len(post_2) or adu_2[1] >= len(post_2):
					adu_2 = None
			# 3. INTER-POST: Longformer
			# Post 1、2 要接在一起。
			else:
				post_1, post_2 = self.posts[post[0]], self.posts[post[1]]
				post_1, post_2 = self._to_ids(post_1), self._to_ids(post_1)

				# 最好的狀況：兩個 post 相連，再加上 [cls]*2, [sep]*2, [user1], [user2] 後
				# 也不會超過最長長度限制
				if len(post_1) + len(post_2) + 6 <= self.max_len:
					pass
				# TODO: 先看看直接不理會如何。
				else:
					adu_1, adu_2 = None, None

			if adu_1 is None or adu_2 is None:
				del self.adu_pairs[idx]
			else:
				self.adu_pairs[idx] = [adu_1, adu_2, relation_type, post]
		
		num_after_filter = len(self.adu_pairs)
		print("[Dataset] %d (out of %d) data entries are valid." % (num_after_filter, num_before_filter))

	# 給定 post_ids (來自 tokenizer)，以及 adu (text)
	# 尋找是否有一條 post_ids 的 slice，可以完全符合 adu_ids
	# 有的話回傳 inclusive interval (i, j)
	# 否則回傳 None
	def __get_adu_span_index(self, post_ids, adu, search_in_id_space):
		# 把 adu 也轉成 ids，並且搜尋 adu_ids 是否是 post_ids 的 subarray
		if search_in_id_space:
			adu_ids = self._to_ids(adu)
			adu_len = len(adu_ids)
			for start in range(0, len(post_ids) - adu_len + 1):
				post_slice = post_ids[start:start+adu_len]
				identity = [a==p for a,p in zip(adu_ids, post_slice)]
				if all(identity):
					return (start, start+adu_len-1)

			return None
		# 尤其使用在如 Roberta 等 sub-word tokenization，
		# substring 的 token list 不一定會在 string 的 token list 中，
		# 因此需要特殊的方式搜尋。
		else:
			# 假定 post_ids 轉回文字後，adu 是在 文字內的
			adu_in_post = adu in self._ids_to_string(post_ids)
			assert adu_in_post
			
			# 搜尋 start & end 區間
			start = 0
			end = len(post_ids)

			for start in range(0, len(post_ids)):
				post_slice = post_ids[start:end]
				adu_in_slice = adu in self._ids_to_string(post_slice)
				if not adu_in_slice:
					start -= 1
					break

			for end in range(len(post_ids), start, -1):
				post_slice = post_ids[start:end]
				adu_in_slice = adu in self._ids_to_string(post_slice)
				if not adu_in_slice:
					break
				
			return (start, end)

	# 給定 relation_type，回傳其 class
	def get_class(self, relation_type):
		if relation_type == 'None' or (self.classification_type == 'link_type' and relation_type == 'Support'):
			return 0
		else:
			return 1

	# 為了解決 class imbalance，可以把較少的 class 重複多次。
	def __duplicate_items(self):
		num_class_0 = 0
		num_class_1 = 0
		
		for (adu_1, adu_2, relation_type, post) in self.adu_pairs:
			if self.get_class(relation_type) == 0:
				num_class_0 += 1
			else:
				num_class_1 += 1

		if num_class_0 > num_class_1:
			target_class = 1
		elif num_class_1 > num_class_0:
			target_class = 0

		num_diff = abs(num_class_1 - num_class_0)
		added_samples = []
		for _ in range(num_diff):
			while True:
				random_sample = random.choice(self.adu_pairs)
				_, _, relation_type, _ = random_sample
				if self.get_class(relation_type) == target_class:
					added_samples.append(random_sample)
					break
		self.adu_pairs.extend(added_samples)

		

	# 為了得到 sentence embedding，要在 adu 的前後加入 [CLS] 和 [SEP]
	def __getitem__(self, idx):
		adu_1, adu_2, relation_type, post = self.adu_pairs[idx]

		cls = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token]) # [cls_id]
		sep = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token]) # [sep_id]

		is_longformer = self.tokenizer.name_or_path == 'Jeevesh8/sMLM-LF'
		reserved = 3 if is_longformer else 2

		# label_id:
		# 0: None, 1: Support/Attack (LINK)
		# 0: Support, 1: Attack (LINK_TYPE)
		if self.classification_type == 'link':
			label_id = 0 if relation_type == 'None' else 1
		elif self.classification_type == 'link_type':
			label_id = 0 if relation_type == 'Support' else 1

		# ================================================================
		# 第一大類：如果不是 Longformer 的話
		# 用正常的方式給資料：
		# adu_1, adu_2 是區間 (both inclusive)
		# post 是 ids list，前後各補上一個 cls / sep
		# 需要給兩個 post，代表 adu_1 和 adu_2 各自的來源
		# inner_post 的情況，兩個 adu 都來自同一個 post
		if not is_longformer:

			# ADU 的 index 做一些 offset，以：考慮 CLS, 所以 (i, j) 要各 +1
			adu_1 = (adu_1[0] + 1, adu_1[1] + 1)
			adu_2 = (adu_2[0] + 1, adu_2[1] + 1)

			if self.is_inner_post:
				post = self.posts[post]
				post = self._to_ids(post)
				post = post[:self.max_len-reserved] if len(post) > self.max_len-reserved else post
				post = cls + post + sep
				
				return adu_1, adu_2, label_id, post, post, 'inner'
			# Inter-Post Relation，需要給定兩個 posts
			else:
				post_1, post_2 = self.posts[post[0]], self.posts[post[1]]
				post_1, post_2 = self._to_ids(post_1), self._to_ids(post_2)
				post_1 = post_1[:self.max_len-reserved] if len(post_1) > self.max_len-reserved else post_1
				post_2 = post_2[:self.max_len-reserved] if len(post_2) > self.max_len-reserved else post_2
				post_1, post_2 = cls + post_1 + sep, cls + post_2 + sep
				
				return adu_1, adu_2, label_id, post_1, post_2, 'inter'
		# ================================================================
		# 第二大類：如果是 Longformer 的話
		# adu_1, adu_2 是區間 (both inclusive)
		# post 是 ids list，但只需要給一個（不同於第一大類）
		#	如果是 inner_post，會在前面加上 [USER1] cls、後面加上 sep
		#	如果是 inter_post，會把兩個 post 拼接在一起，形如 [USER1] cls ... sep [USER2] cls ... sep
		# 當然，ADU 的 index 也會需要調整。
		# 因為是 Longformer，這裡不特別裁切 post。
		else:
			user_1 = self.tokenizer.convert_tokens_to_ids(['[USER1]'])
			user_2 = self.tokenizer.convert_tokens_to_ids(['[USER2]'])
			adu_1 = (adu_1[0] + 2, adu_1[1] + 2)
			adu_2 = (adu_2[0] + 2, adu_2[1] + 2)
			
			if self.is_inner_post:
				post = self.posts[post]
				post = self._to_ids(post)
				post = user_1 + cls + post + sep
				
				return adu_1, adu_2, label_id, post, None, 'inner'
			# Inter-Post Relation，需要把 adu_2 往後位移，位移量相當於 post_1 的長度
			else:
				post_1, post_2 = self.posts[post[0]], self.posts[post[1]]
				post_1, post_2 = self._to_ids(post_1), self._to_ids(post_2)
				post_1, post_2 = (user_1 + cls + post_1 + sep), (user_2 + cls + post_2 + sep)
				adu_2 = (adu_2[0] + len(post_1), adu_2[1] + len(post_1))
				post = post_1 + post_2
				
				return adu_1, adu_2, label_id, post, None, 'inter'

	def __len__(self):
		return len(self.adu_pairs)

ADU_1_FIRST = +0.5
ADU_2_FIRST = -0.5
ADU_ORDER_NON_APPLICABLE = 0

IS_INNER_POST = +0.5
IS_INTER_POST = -0.5

def PadBatch(batch, tokenizer):
	is_longformer = tokenizer.name_or_path == 'Jeevesh8/sMLM-LF'
	pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
	batch_size = len(batch)

	# ADU tensors
	# batch = [adu_1, adu_2, label, post_1, post_2 or None, 'inner' or 'inter']
	# post_1 有可能是 post_1 或 post_2
	adu_1_tensors = torch.LongTensor([i[0] for i in batch])
	adu_2_tensors = torch.LongTensor([i[1] for i in batch])

	# 製作一個 feature，指示 adu_1 和 adu_2 的前後順序
	adu_orders = []
	for data in batch:
		# inter-post 的狀況，通常無法決定 ADU ORDER
		# 但是 Longformer 因為已經把 post 拼接，所以可以
		if data[5] == 'inter' and not is_longformer:
			adu_orders.append(ADU_ORDER_NON_APPLICABLE)
			continue
		if data[0][0] < data[1][0]:
			adu_orders.append(ADU_1_FIRST)
		elif data[0][0] > data[1][0]:
			adu_orders.append(ADU_2_FIRST)
		# 兩個 ADU 的開頭一樣，比結尾
		else:
			if data[0][1] < data[1][1]:
				adu_orders.append(ADU_1_FIRST)
			elif data[0][1] > data[1][1]:
				adu_orders.append(ADU_2_FIRST)
			else:
				# TODO 不應該發生。
				adu_orders.append(ADU_ORDER_NON_APPLICABLE)
	adu_orders = torch.FloatTensor(adu_orders).view(batch_size, -1)

	# Labels
	labels = [i[2] for i in batch]

	# 是否 Longformer 對於資料的處理方式有差異
	if not is_longformer:
		# Post Tensors
		post_1_maxlen = max([len(i[3]) for i in batch])
		post_2_maxlen = max([len(i[4]) for i in batch])
		post_maxlen = max([post_1_maxlen, post_2_maxlen])

		post_1_tensors = torch.LongTensor([i[3] + [pad_id] * (post_maxlen - len(i[3])) for i in batch])
		post_2_tensors = torch.LongTensor([i[4] + [pad_id] * (post_maxlen - len(i[4])) for i in batch])
		post_1_mask = (post_1_tensors != pad_id)
		post_2_mask = (post_2_tensors != pad_id)

		is_inner = [(i[5] == 'inner') for i in batch]
		is_inner_embed = torch.FloatTensor([[IS_INNER_POST] if i else [IS_INTER_POST] for i in is_inner]).view(batch_size, -1)

		return post_1_tensors, post_2_tensors, post_1_mask, post_2_mask, adu_1_tensors, adu_2_tensors, adu_orders, labels, is_inner_embed
	# Longformer 因為已確定只有一個 post
	# 因此不用再浪費空間回傳 post_2
	else:
		# Post Tensors
		post_maxlen = max([len(i[3]) for i in batch])

		post_tensors = torch.LongTensor([i[3] + [pad_id] * (post_maxlen - len(i[3])) for i in batch])
		post_mask = (post_tensors != pad_id)

		is_inner = [(i[5] == 'inner') for i in batch]
		is_inner_embed = torch.FloatTensor([[IS_INNER_POST] if i else [IS_INTER_POST] for i in is_inner]).view(batch_size, -1)

		return post_tensors, None, post_mask, None, adu_1_tensors, adu_2_tensors, adu_orders, labels, is_inner_embed
