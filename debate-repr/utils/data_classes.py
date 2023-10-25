
# ===========================

class Comment:
	# paragraphs: list[str]
	def __init__(self, author: str, paragraphs, sentence_tokenizer, prev_comment=None):
		self.author = author
		
		# 一個 Comment 包含多個段落。每個元素是一個段落。
		self.paragraphs = paragraphs

		# [0, 段落 1 長度, 段落 1~2 長度，段落 1~3 長度, ..., 段落 (n-1) 長度]
		# 注意每個段落的長度會是原長度 +1，因為要考慮尾巴有一個分隔符號 (\n)
		self.length_prefix = []

		# [ [ 段落一的 spans ], [ 段落二的 spans ], ... ]
		self.sentence_spans : list[list[(int, int)]] = []
		self.num_sentences = 0
		
		# 所有的 adu_spans
		self.adu_spans = []

		# 真正的 adu_pairs (中間存在 link 的)
		self.real_inner_adu_pairs = []
		self.real_inter_adu_pairs = []

		# 本 comment 的前者 (用作計算 inter-post link)
		self.prev_comment = prev_comment

		# 本 comment 的下游
		self.replies = []

		for paragraph in self.paragraphs:
			assert isinstance(paragraph, str)
			self.sentence_spans.append(list(sentence_tokenizer.span_tokenize(paragraph)))
			self.num_sentences += len(self.sentence_spans[-1])
			self.length_prefix.append(len(paragraph))

		self.length_prefix = [0] + self.length_prefix[:-1]
		for i in range(len(self.length_prefix)):
			if i == 0:
				self.length_prefix[i] = 0
			else:
				self.length_prefix[i] += self.length_prefix[i-1]

		# 額外的 information，以 dict 儲存
		self.info = { }

	# ADU SPAN 的格式：尾巴為 exclusive
	# character_level_spans: list[tuple(int, int)]
	def assign_adu_spans(self, paragraph_index: int, sentence_index: int, character_level_spans):
		# 這個 paragraph 第一個字元，在這個 comment 的 index
		length_prefix = self.length_prefix[paragraph_index]
		# 這個 sentence 第一個字元，在這個 paragraph 的 index
		span_of_sentence_in_paragraph = self.sentence_spans[paragraph_index][sentence_index]
		# 這個 sentence 第一個字元，在這個 comment 的 index
		sentence_start_in_comment = length_prefix + span_of_sentence_in_paragraph[0]
		# 這個 adu 的頭尾
		for character_level_span in character_level_spans:
			adu_span_in_comment = (
				character_level_span[0] + sentence_start_in_comment, 
				character_level_span[1] + sentence_start_in_comment)
			assert adu_span_in_comment not in self.adu_spans
			self.adu_spans.append(adu_span_in_comment)
		
	# 回傳這個 comment 的完整原文
	def get_comment(self):
		return ''.join(self.paragraphs)
	
	# 回傳這個 comment 的段落數量
	def num_paragraphs(self):
		return len(self.paragraphs)
	
	# 回傳這個 comment 的句子數量
	def num_sentences_in_paragraph(self, paragraph_index):
		return len(self.sentence_spans[paragraph_index])

	# --------------------------
	# 潛在的 inner-adu-pair 數量
	def num_potential_inner_adu_pairs(self):
		return len(self.adu_spans) * len(self.adu_spans)
	
	# 得到潛在的單個 inner-adu-pair
	# 回傳 (span, span, idx)
	# span 為 exclusive
	def get_potential_inner_adu_pairs(self, idx):
		assert 0 <= idx and idx < self.num_potential_inner_adu_pairs()
		n = len(self.adu_spans)
		return (self.adu_spans[idx // n], self.adu_spans[idx % n], idx)
	
	# 潛在的 inter-adu-pair 數量
	def num_potential_inter_adu_pairs(self):
		if self.prev_comment is None:
			return 0
		return len(self.adu_spans) * len(self.prev_comment.adu_spans)
	
	# 得到潛在的單個 inter-adu-pair
	# ADU_1 來自本篇，ADU_2 來自上一篇。
	# 回傳 (span, span, idx)
	# span 為 exclusive
	def get_potential_inter_adu_pairs(self, idx):
		assert 0 <= idx and idx < self.num_potential_inter_adu_pairs()
		m = len(self.prev_comment.adu_spans)
		return (self.adu_spans[idx // m], self.prev_comment.adu_spans[idx % m], idx)
	
	# --------------------------
	# 真正的 inner-adu-pair 數量
	def num_real_inner_adu_pairs(self):
		return len(self.real_inner_adu_pairs)
	
	# 得到真正的 inner-adu-pair
	# 回傳 (span, span, idx)
	# 注意這裡的 span 是 inclusive，並且是經過 tokenize 後的區間
	# (unnecessarily confusing? anyways.)
	def get_real_inner_adu_pairs(self, idx):
		assert 0 <= idx and idx < self.num_real_inner_adu_pairs()
		span_1, span_2 = self.real_inner_adu_pairs[idx]
		return (span_1, span_2, idx)
	
	# 真正的 inter-adu-pair 數量
	def num_real_inter_adu_pairs(self):
		return len(self.real_inter_adu_pairs)
	
	# 得到真正的 inter-adu-pair
	# 回傳 (span, span, idx)
	# 注意這裡的 span 是 inclusive，並且是經過 tokenize 後的區間
	# (unnecessarily confusing? anyways.)
	def get_real_inter_adu_pairs(self, idx):
		assert 0 <= idx and idx < self.num_real_inter_adu_pairs()
		span_1, span_2 = self.real_inter_adu_pairs[idx]
		return (span_1, span_2, idx)


# ---------------------------

class Thread:
	# comments: list[Comment]
	def __init__(self, comments):
		self.comments = comments

	def num_comments(self):
		return len(self.comments)

# ---------------------------

class Pair:
	# threads: list[Threads]
	def __init__(self, threads):
		assert len(threads) == 2
		self.threads = threads

# ---------------------------

class Post:
	# pairs: list[Pair]
	def __init__(self, author: str, title: str, root: Comment, pairs=[], is_pairs=True) -> None:
		self.author = author
		self.title = title
		self.root = root
		self.pairs = pairs
		self.is_pairs = is_pairs
		
		self.num_potential_inner_adu_pairs = None
		self.num_potential_inter_adu_pairs = None
		self.num_real_inner_adu_pairs = None
		self.num_real_inter_adu_pairs = None

	# 設置潛在的 ADU Pairs Link。
	# 這件事發生在 Post 中的每個 Comment 已經各自標記好 ADU Span 了
	# 將會統計共有多少 potential adu pairs
	def setup_potential_adu_pairs(self):
		self.num_potential_inner_adu_pairs = 0
		self.num_potential_inter_adu_pairs = 0
		for (_, _, _, _, comment) in self.get_comments():
			self.num_potential_inner_adu_pairs += comment.num_potential_inner_adu_pairs()
			self.num_potential_inter_adu_pairs += comment.num_potential_inter_adu_pairs()
		
	# 設置真正的 ADU Pairs Link。
	# 這件事發生在 Post 中的每個 ADU Pair 已經透過 Link Model，標記好哪些 link 確切存在了
	# 將會統計共有多少 real adu pairs
	def setup_real_adu_pairs(self):
		self.num_real_inner_adu_pairs = 0
		self.num_real_inter_adu_pairs = 0
		for (_, _, _, _, comment) in self.get_comments():
			self.num_real_inner_adu_pairs += comment.num_real_inner_adu_pairs()
			self.num_real_inter_adu_pairs += comment.num_real_inter_adu_pairs()

	# 迭代所有的 Comments 的 generator
	# 回傳 (is_thread, pair_index, is_positive, comment_index, Comment)
	def get_comments(self):
		yield (0, -1, -1, -1, self.root)
		for pair_index in range(len(self.pairs)):
			pair = self.pairs[pair_index]
			for is_positive in range(len(pair.threads)):
				thread = pair.threads[is_positive]
				for comment_index in range(len(thread.comments)):
					comment = thread.comments[comment_index]
					yield (1, pair_index, is_positive, comment_index, comment)

	# 透過指定的參數，取回此 Post 內的特定一個 Comment
	def get_comment(self, is_thread, pair_index, is_positive, comment_index):
		if is_thread == 0:
			return self.root
		else:
			return self.pairs[pair_index].threads[is_positive].comments[comment_index]

	# 回傳這個 Post 共有多少 pairs
	def num_pairs(self):
		return len(self.pairs)
	
	# 獲取潛在的 inner_adu_pair
	# 回傳 ((Comment 資訊), idx_in_comment, PAIR, 'inner')
	def get_potential_inner_adu_pair(self, idx):
		for (is_thread, pair_index, is_positive, comment_index, comment) in self.get_comments():
			num_pairs = comment.num_potential_inner_adu_pairs()
			if idx < num_pairs:
				pair = comment.get_potential_inner_adu_pairs(idx)
				pair, idx_in_comment = (pair[0], pair[1]), pair[2]
				return ((is_thread, pair_index, is_positive, comment_index, comment), idx_in_comment, pair, 'inner')
			else:
				idx -= num_pairs

	# 獲取潛在的 inter_adu_pair
	# 回傳 ((Comment 資訊), idx_in_comment, PAIR, 'inter')
	def get_potential_inter_adu_pair(self, idx):
		for (is_thread, pair_index, is_positive, comment_index, comment) in self.get_comments():
			num_pairs = comment.num_potential_inter_adu_pairs()
			if idx < num_pairs:
				pair = comment.get_potential_inter_adu_pairs(idx)
				pair, idx_in_comment = (pair[0], pair[1]), pair[2]
				return ((is_thread, pair_index, is_positive, comment_index, comment), idx_in_comment, pair, 'inter')
			else:
				idx -= num_pairs

