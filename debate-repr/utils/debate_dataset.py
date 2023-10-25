from .data_classes import *
import pickle
import torch
import os
import code
from .info_keys import *
import random
import itertools

NEG_THREAD = 0
POS_THREAD = 1

NO_WINNER_PRED = 0
DO_WINNER_PRED = 1

NO_ATTACKED = 0
DO_ATTACKED = 1

ROOT_OF_POST = 0
MIDDLE_OF_THREAD = 1
TAIL_OF_THREAD = 2

# Index
DATA_PARENT, CASE, COMMENT_IDS, INFO_SCORES, ADU_SPANS, ADU_ATTACKED, LOCAL_ADU_IN_USE_MASK, TITLE_ADU_SPAN_INDEX, \
INNER_PAIRS, INTER_PAIRS, IS_WINNER, AUTHOR, COMMENT_TEXT = list(range(13))

def gather_debate_datasets(path, tokenizer, ignore_tail=False, data_type='normal'):
    assert data_type in ['normal', 'baseline']
    files = [path + file for file in os.listdir(path) if file.endswith('.data')]
    datasets = []
    for file in files:
        if data_type == 'normal':
            datasets.append( DebateDataset(file, tokenizer, ignore_tail) )
        else:
            datasets.append( DebateDatasetBaseline(file, tokenizer, ignore_tail) )
    return datasets

def gather_debate_report_datasets(path, tokenizer, is_baseline=False):
    files = [path + file for file in os.listdir(path) if file.endswith('.data')]
    datasets = [DebateValidationReportDataset(file, tokenizer, is_baseline) for file in files]
    return datasets

# DebateDataset 的集合，可以得知每次給出的資料種類
# 用作 Contrastive Learning。
class DebateDatasets:
    def __init__(self, datasets):
        self.datasets = datasets
        self.indices = list(range(len(self.datasets)))

        # Same Author 顯然比較困難，用到時會需要先建表
        # (data_id_1, data_id_2)
        self.same_author_data = [None] * len(self.datasets)

    def __len__(self):
        return 9999999

    # 目前有三種種類：不同 comment、同 comment 不同作者、同 comment 同作者
    def __getitem__(self, _):
        DIFFERENT, SAME_COMMENT, SAME_AUTHOR = 0, 1, 2
        category = random.choice([DIFFERENT, SAME_COMMENT, SAME_AUTHOR])

        if category == DIFFERENT:
            c1, c2 = -1, -1
            while c1 == c2:
                c1, c2 = random.choices(self.indices, k=2)
            d1, d2 = random.randrange(0, len(self.datasets[c1])), random.randrange(0, len(self.datasets[c2]))
            return (
                category,
                self.datasets[c1][d1],
                self.datasets[c2][d2]
            )
        if category == SAME_COMMENT:
            while True:
                c = random.choice(self.indices)
                d1, d2 = random.randrange(0, len(self.datasets[c])), random.randrange(0, len(self.datasets[c]))
                a1, a2 = (self.datasets[c][d1])[-1][AUTHOR], (self.datasets[c][d2])[-1][AUTHOR]
                t1, t2 = (self.datasets[c][d1])[-1][COMMENT_IDS], (self.datasets[c][d2])[-1][COMMENT_IDS]
                if d1 == d2 or a1 == a2 or t1 == t2:
                    continue
                return (
                    category,
                    self.datasets[c][d1],
                    self.datasets[c][d2]
                )
        if category == SAME_AUTHOR:
            while True:
                c = random.choice(self.indices)
                # 這個 dataset 尚未建表：那就建表
                if self.same_author_data[c] == None:
                    self.same_author_data[c] = []
                    for d1, d2 in itertools.combinations(list(range(len(self.datasets[c]))), 2):
                        a1, a2 = (self.datasets[c][d1])[-1][AUTHOR], (self.datasets[c][d2])[-1][AUTHOR]
                        t1, t2 = (self.datasets[c][d1])[-1][COMMENT_IDS], (self.datasets[c][d2])[-1][COMMENT_IDS]
                        if a1 == a2 and t1 != t2:
                            self.same_author_data[c].append( (d1, d2) )

                if len(self.same_author_data[c]) == 0:
                    continue
                else:
                    d1, d2 = random.choice(self.same_author_data[c])
                    return (
                        category,
                        self.datasets[c][d1],
                        self.datasets[c][d2]
                    )


# Debate 資料的 Dataset。
# 以每一個 Post 為單位，製作一個 DebateDataset
class DebateDataset:
    def __init__(self, post_data_path, tokenizer, ignore_tail=False):
        
        # read the Post in the file
        with open(post_data_path, 'rb') as f:
            post = pickle.load(f)

        self.post_data_path = post_data_path

        # Build indexing.
        # 0-th is the root of the post.
        # For each thread with k comments, k entries of data are present.
        # self.construct_data_from_comment_in_thread(tokenizer, ROOT_OF_POST, post.root, is_winner=None, comment_parent=None)
        self.data = [ self.construct_data_from_comment_in_thread(tokenizer, ROOT_OF_POST, post.root) ]
        for pair_idx, pair in enumerate(post.pairs):
            for thread_idx, thread in enumerate(pair.threads):
                num_comments_in_thread = len(thread.comments)
                parent_data_idx = 0
                comment_parent = post.root
                for comment_idx, comment in enumerate(thread.comments):
                    case = TAIL_OF_THREAD if comment_idx == num_comments_in_thread - 1 else MIDDLE_OF_THREAD
                    is_winner = (thread_idx == POS_THREAD) if case == TAIL_OF_THREAD else None
                    data_entry = self.construct_data_from_comment_in_thread(
                        tokenizer, case, comment, is_winner, parent_data_idx, comment_parent
                    )
                    if case == TAIL_OF_THREAD and ignore_tail:
                        pass
                    else:
                        self.data.append(data_entry)
                    parent_data_idx = len(self.data) - 1
                    comment_parent = comment

    def __len__(self):
        return len(self.data)

    # 不斷把 PARENT 的資料往前疊，因此最後一個才是真正關注的 Comment。
    # 第一個往後則是依次向下的路徑 (root-to-leaf)
    def __getitem__(self, idx):
        thread = []
        data_idx = idx
        while data_idx != -1:
            thread = [self.data[data_idx]] + thread
            data_idx = thread[0][0]
        return thread

    def _comment_to_ids(self, tokenizer, comment):
        if comment is None:
            return None
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment.get_comment()))

    def _offset_span(self, span, offset):
        return (span[0] + offset, span[1] + offset)
    
    def construct_data_from_comment_in_thread(self, tokenizer, case, comment, is_winner=None, data_parent=-1 , comment_parent=None):
        # Case 1. root of the post (comment_parent: None)
        #           NO_WINNER_PRED, DO_ATTACKED
        # Case 2. in the middle of a thread
        #           NO_WINNER_PRED, DO_ATTACKED
        # Case 3. the tail of a thread
        #           DO_WINNER_PRED, NO_ATTACKED
        assert case in [ROOT_OF_POST, MIDDLE_OF_THREAD, TAIL_OF_THREAD]
        
        # TODO: Cannot yet distinguish authors...

        # Pass comment text to tokenizers
        cls = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])
        sep = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])
        user1 = tokenizer.convert_tokens_to_ids(['[USER1]'])
        user2 = tokenizer.convert_tokens_to_ids(['[USER2]'])
        MASK_ON = True
        MASK_OFF = False

        # 把 Current Comment 和 Parent Comment 轉成 ids
        comment_parent_ids = self._comment_to_ids(tokenizer, comment_parent)
        comment_ids = self._comment_to_ids(tokenizer, comment)

        # TODO: intros
        integrated_comment_ids = []
        adu_attacked = []
        adu_spans = []
        inner_post_adu_pairs = []
        inter_post_adu_pairs = []
        local_adu_in_use_mask = []
        info_scores = [comment.info['scores'][key] for key in info_all_keys]
        title_adu_span_index = None
        author = comment.author

        # Root: 
        if case == ROOT_OF_POST:
            integrated_comment_ids = user1 + cls + comment_ids + sep
            adu_attacked = [adu_span['attacked'] for adu_span in comment.adu_spans]
            adu_spans = [self._offset_span(adu_span['span'], 2) for adu_span in comment.adu_spans]
            inner_post_adu_pairs = comment.real_inner_adu_pairs
            
            title_adu_span_index = [adu_span_index for adu_span_index, adu_span in enumerate(comment.adu_spans) if adu_span['span'][0] == 0]
            assert len(title_adu_span_index) == 1
            title_adu_span_index = title_adu_span_index[0]

            # Force topic to be used
            comment.adu_spans[title_adu_span_index]['in-use'] = True
            local_adu_in_use_mask = [adu_span['in-use'] for adu_span in comment.adu_spans]
        
        # Middle of thread / Tail of thread:
        else:
            post_1_ids = (user1 + cls + comment_ids + sep)
            post_2_ids = (user2 + cls + comment_parent_ids + sep)
            integrated_comment_ids = post_1_ids + post_2_ids
            
            # ADU Spans of the Current Comment
            current_adu_spans = [self._offset_span(adu_span['span'], 2) for adu_span in comment.adu_spans]
            # ADU Spans of the Parent Comment (因為接在 Current Comment 後，所以要調整 index)
            post_1_ids_len = len(post_1_ids)
            parent_adu_spans = [self._offset_span(adu_span['span'], post_1_ids_len + 2) for adu_span in comment_parent.adu_spans]
            # 合併兩個 adu_spans 列表：Current Comment 在前，Parent Comment 在後
            adu_spans = current_adu_spans + parent_adu_spans

            # 只需要在意 Current Comment 的 ADU 是否有被 Attacked
            adu_attacked = [adu_span['attacked'] for adu_span in comment.adu_spans]

            local_adu_in_use_mask = [adu_span['in-use'] for adu_span in comment.adu_spans]
            
            # Inner-Pairs 仍會維持一樣的 indices (因為 adu_spans 列表 Current Comment 在前)
            inner_post_adu_pairs = comment.real_inner_adu_pairs

            # 對 Inter Pairs 來說，其 Targeted ADU 必定在 Parent Comment。
            # 因為 adu_spans 列表 Parent Comment 在後，其 index 也需要位移
            inter_post_adu_pairs = comment.real_inter_adu_pairs
            inter_post_adu_pairs = [ (span_1_index, span_2_index + len(current_adu_spans), relation_type) 
                                    for span_1_index, span_2_index, relation_type in inter_post_adu_pairs]

        # 針對 Middle of thread / Tail of thread，
        # 雖然大部分資料都一樣，但前者不需要做 is_winner，後者不需要做 ADU_Attacked
        # 因此再對資料做微調
        if case == MIDDLE_OF_THREAD:
            assert is_winner == None
        # Tail of thread:
        elif case == TAIL_OF_THREAD:
            assert is_winner is not None
            adu_attacked = []

        return (data_parent,
                case, integrated_comment_ids, info_scores, 
                adu_spans, adu_attacked, local_adu_in_use_mask, title_adu_span_index,
                inner_post_adu_pairs,
                inter_post_adu_pairs,
                is_winner, author, comment.get_comment())

# Debate 資料的 Dataset，專供 Baseline 使用。
# 以每一個 Post 為單位，製作一個 DebateDataset
class DebateDatasetBaseline:
    def __init__(self, post_data_path, tokenizer, ignore_tail=False):
        
        # read the Post in the file
        with open(post_data_path, 'rb') as f:
            post = pickle.load(f)

        self.post_data_path = post_data_path

        # Build indexing.
        # 0-th is the root of the post.
        # For each thread with k comments, k entries of data are present.
        # self.construct_data_from_comment_in_thread(tokenizer, ROOT_OF_POST, post.root, is_winner=None, comment_parent=None)
        self.data = [ self.construct_data_from_comment_in_thread(tokenizer, ROOT_OF_POST, post.root) ]
        for pair_idx, pair in enumerate(post.pairs):
            for thread_idx, thread in enumerate(pair.threads):
                num_comments_in_thread = len(thread.comments)
                parent_data_idx = 0
                comment_parent = post.root
                for comment_idx, comment in enumerate(thread.comments):
                    case = TAIL_OF_THREAD if comment_idx == num_comments_in_thread - 1 else MIDDLE_OF_THREAD
                    is_winner = (thread_idx == POS_THREAD) if case == TAIL_OF_THREAD else None
                    data_entry = self.construct_data_from_comment_in_thread(
                        tokenizer, case, comment, is_winner, parent_data_idx, comment_parent
                    )
                    if case == TAIL_OF_THREAD and ignore_tail:
                        pass
                    else:
                        self.data.append(data_entry)
                    parent_data_idx = len(self.data) - 1
                    comment_parent = comment

    def __len__(self):
        return len(self.data)

    # 不斷把 PARENT 的資料往前疊，因此最後一個才是真正關注的 Comment。
    # 第一個往後則是依次向下的路徑 (root-to-leaf)
    def __getitem__(self, idx):
        thread = []
        data_idx = idx
        while data_idx != -1:
            thread = [self.data[data_idx]] + thread
            data_idx = thread[0][0]
        return thread

    def _comment_to_ids(self, tokenizer, comment):
        if comment is None:
            return None
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment.get_comment()))

    def _offset_span(self, span, offset):
        return (span[0] + offset, span[1] + offset)
    
    def construct_data_from_comment_in_thread(self, tokenizer, case, comment, is_winner=None, data_parent=-1 , comment_parent=None):
        assert case in [ROOT_OF_POST, MIDDLE_OF_THREAD, TAIL_OF_THREAD]

        # Pass comment text to tokenizers
        cls = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])
        sep = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])

        # 把 Current Comment 和 Parent Comment 轉成 ids
        comment_parent_ids = self._comment_to_ids(tokenizer, comment_parent)
        comment_ids = self._comment_to_ids(tokenizer, comment)

        # TODO: intros
        integrated_comment_ids = []
        adu_attacked = []
        adu_spans = []
        inner_post_adu_pairs = []
        inter_post_adu_pairs = []
        local_adu_in_use_mask = []
        info_scores = []
        title_adu_span_index = None
        author = comment.author

        integrated_comment_ids = cls + comment_ids + sep

        return (data_parent,
                case, integrated_comment_ids, info_scores, 
                adu_spans, adu_attacked, local_adu_in_use_mask, title_adu_span_index,
                inner_post_adu_pairs,
                inter_post_adu_pairs,
                is_winner, author, comment.get_comment())            

# Debate 資料的 Dataset，但是只收集 pairs of threads。
class DebateValidationReportDataset:
    def __init__(self, post_data_path, tokenizer, is_baseline=False):
        self.is_baseline = is_baseline
        # read the Post in the file
        with open(post_data_path, 'rb') as f:
            post = pickle.load(f)

        # Build indexing.
        # 0-th is the root of the post.
        # For each thread with k comments, k entries of data are present.
        # self.construct_data_from_comment_in_thread(tokenizer, ROOT_OF_POST, post.root, is_winner=None, comment_parent=None)
        self.data = [ self.construct_data_from_comment_in_thread(tokenizer, ROOT_OF_POST, post.root) ]
        self.pairs = []
        for pair_idx, pair in enumerate(post.pairs):
            threads_data = [None, None]
            for thread_idx, thread in enumerate(pair.threads):
                num_comments_in_thread = len(thread.comments)
                parent_data_idx = 0
                comment_parent = post.root
                for comment_idx, comment in enumerate(thread.comments):
                    case = TAIL_OF_THREAD if comment_idx == num_comments_in_thread - 1 else MIDDLE_OF_THREAD
                    is_winner = (thread_idx == POS_THREAD) if case == TAIL_OF_THREAD else None
                    self.data.append(self.construct_data_from_comment_in_thread(
                        tokenizer, case, comment, is_winner, parent_data_idx, comment_parent
                    ))
                    parent_data_idx = len(self.data) - 1
                    comment_parent = comment

                    if comment_idx == num_comments_in_thread - 1:
                        threads_data[thread_idx] = self.data[-1]
            self.pairs.append(threads_data)

    def __len__(self):
        return len(self.pairs)

    # 不斷把 PARENT 的資料往前疊，因此最後一個才是真正關注的 Commet。
    # 第一個往後則是依次向下的路徑 (root-to-leaf)
    def __getitem__(self, idx):
        tail_1, tail_2 = self.pairs[idx]
        thread_1, thread_2 = [tail_1], [tail_2]
        data_idx_1, data_idx_2 = tail_1[0], tail_2[0]

        while data_idx_1 != -1:
            thread_1 = [self.data[data_idx_1]] + thread_1
            data_idx_1 = thread_1[0][0]
        while data_idx_2 != -1:
            thread_2 = [self.data[data_idx_2]] + thread_2
            data_idx_2 = thread_2[0][0]
        return thread_1, thread_2

    def _comment_to_ids(self, tokenizer, comment):
        if comment is None:
            return None
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment.get_comment()))

    def _offset_span(self, span, offset):
        return (span[0] + offset, span[1] + offset)
    
    def construct_data_from_comment_in_thread(self, tokenizer, case, comment, is_winner=None, data_parent=-1 , comment_parent=None):
        # Case 1. root of the post (comment_parent: None)
        #           NO_WINNER_PRED, DO_ATTACKED
        # Case 2. in the middle of a thread
        #           NO_WINNER_PRED, DO_ATTACKED
        # Case 3. the tail of a thread
        #           DO_WINNER_PRED, NO_ATTACKED
        assert case in [ROOT_OF_POST, MIDDLE_OF_THREAD, TAIL_OF_THREAD]

        # Pass comment text to tokenizers
        cls = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])
        sep = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])
        user1 = tokenizer.convert_tokens_to_ids(['[USER1]'])
        user2 = tokenizer.convert_tokens_to_ids(['[USER2]'])
        MASK_ON = True
        MASK_OFF = False

        # 把 Current Comment 和 Parent Comment 轉成 ids
        comment_parent_ids = self._comment_to_ids(tokenizer, comment_parent)
        comment_ids = self._comment_to_ids(tokenizer, comment)

        # TODO: intros
        integrated_comment_ids = []
        adu_attacked = []
        adu_spans = []
        inner_post_adu_pairs = []
        inter_post_adu_pairs = []
        local_adu_in_use_mask = []
        info_scores = []
        title_adu_span_index = None
        author = comment.author

        if self.is_baseline:
            integrated_comment_ids = cls + comment_ids + sep
            return (data_parent,
                    case, integrated_comment_ids, info_scores, 
                    adu_spans, adu_attacked, local_adu_in_use_mask, title_adu_span_index,
                    inner_post_adu_pairs,
                    inter_post_adu_pairs,
                    is_winner, author)

        # Root: 
        if case == ROOT_OF_POST:
            integrated_comment_ids = user1 + cls + comment_ids + sep
            adu_attacked = [adu_span['attacked'] for adu_span in comment.adu_spans]
            adu_spans = [self._offset_span(adu_span['span'], 2) for adu_span in comment.adu_spans]
            inner_post_adu_pairs = comment.real_inner_adu_pairs
            
            title_adu_span_index = [adu_span_index for adu_span_index, adu_span in enumerate(comment.adu_spans) if adu_span['span'][0] == 0]
            assert len(title_adu_span_index) == 1
            title_adu_span_index = title_adu_span_index[0]

            # Force topic to be in-use
            comment.adu_spans[title_adu_span_index]['in-use'] = True
            local_adu_in_use_mask = [adu_span['in-use'] for adu_span in comment.adu_spans]
        
        # Middle of thread / Tail of thread:
        else:
            post_1_ids = (user1 + cls + comment_ids + sep)
            post_2_ids = (user2 + cls + comment_parent_ids + sep)
            integrated_comment_ids = post_1_ids + post_2_ids
            
            # ADU Spans of the Current Comment
            current_adu_spans = [self._offset_span(adu_span['span'], 2) for adu_span in comment.adu_spans]
            # ADU Spans of the Parent Comment (因為接在 Current Comment 後，所以要調整 index)
            post_1_ids_len = len(post_1_ids)
            parent_adu_spans = [self._offset_span(adu_span['span'], post_1_ids_len + 2) for adu_span in comment_parent.adu_spans]
            # 合併兩個 adu_spans 列表：Current Comment 在前，Parent Comment 在後
            adu_spans = current_adu_spans + parent_adu_spans

            local_adu_in_use_mask = [adu_span['in-use'] for adu_span in comment.adu_spans]

            # 只需要在意 Current Comment 的 ADU 是否有被 Attacked
            adu_attacked = [adu_span['attacked'] for adu_span in comment.adu_spans]
            
            # Inner-Pairs 仍會維持一樣的 indices (因為 adu_spans 列表 Current Comment 在前)
            inner_post_adu_pairs = comment.real_inner_adu_pairs

            # 對 Inter Pairs 來說，其 Targeted ADU 必定在 Parent Comment。
            # 因為 adu_spans 列表 Parent Comment 在後，其 index 也需要位移
            inter_post_adu_pairs = comment.real_inter_adu_pairs
            inter_post_adu_pairs = [ (span_1_index, span_2_index + len(current_adu_spans), relation_type) 
                                    for span_1_index, span_2_index, relation_type in inter_post_adu_pairs]

        # 針對 Middle of thread / Tail of thread，
        # 雖然大部分資料都一樣，但前者不需要做 is_winner，後者不需要做 ADU_Attacked
        # 因此再對資料做微調
        if case == MIDDLE_OF_THREAD:
            assert is_winner == None
        # Tail of thread:
        elif case == TAIL_OF_THREAD:
            assert is_winner is not None
            adu_attacked = []

        return (data_parent,
                case, integrated_comment_ids, info_scores, 
                adu_spans, adu_attacked, local_adu_in_use_mask, title_adu_span_index,
                inner_post_adu_pairs,
                inter_post_adu_pairs,
                is_winner, author)

# 針對 Contrastive Loss 推出的 Batch Padding
# 處理來自 DebateDatasets 的資料
# (Category, Comment1, Comment2)
def PadBatchContrastive(batch, tokenizer, batch_size):
    assert batch_size == 1
    assert len(batch) == 1
    assert len(batch[0]) == 3

    category, comment1, comment2 = batch[0]
    return category, PadBatch([comment1], tokenizer, 1), PadBatch([comment2], tokenizer, 1)

# 讓 batch_size = 1，因為單一一個 thread 已經有夠大的資料要處理
def PadBatch(batch, tokenizer, batch_size):
    assert batch_size == 1
    assert len(batch) == 1
    posts = batch[0]
    num_posts = len(posts)

    # Tokenizer Ids
    pad = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    user_1 = tokenizer.convert_tokens_to_ids(['[USER1]'])[0]
    user_2 = tokenizer.convert_tokens_to_ids(['[USER2]'])[0]

    # Case 以 posts 的最後一個 Comment 為主
    case = posts[-1][CASE]

    # is_winner 只考慮最後一個 Post (且只在 case == TAIL_OF_THREAD 時才不為 None)
    is_winner = posts[-1][IS_WINNER]

    # info scores
    info_scores = [post[INFO_SCORES] for post in posts]

    # title adu span
    title_adu_span_index = posts[0][TITLE_ADU_SPAN_INDEX]

    # Authors
    authors = [post[AUTHOR] for post in posts]

    # spans 是用來建立 Arg Graph，因此每個 Comment 都需要
    # adu_spans: 每個 element 來自一個 Comment，是一個 list。會被 padded (-1, -1) 到最長的長度。
    # mask: 標示哪些 adu span 是真的，其他則是 padded
    adu_spans = [post[ADU_SPANS] for post in posts]
    local_adu_in_use_masks = [post[LOCAL_ADU_IN_USE_MASK] for post in posts]
    adu_spans_max_len = max([ len(el) for el in adu_spans ])
    adu_masks = [ ([[True, True]] * len(el)) + ([[False, False]] * (adu_spans_max_len - len(el))) for el in adu_spans ]
    local_adu_in_use_masks = [ masks + ([False] * (adu_spans_max_len - len(masks))) \
                              for masks in local_adu_in_use_masks ]
    adu_spans = [el + ([(-1, -1)] * (adu_spans_max_len - len(el))) for el in adu_spans]

    adu_masks = torch.BoolTensor(adu_masks).view(num_posts, adu_spans_max_len, 2)
    local_adu_in_use_masks = torch.BoolTensor(local_adu_in_use_masks).view(num_posts, adu_spans_max_len)
    adu_spans = torch.IntTensor(adu_spans).view(num_posts, adu_spans_max_len, 2)

    # adu_attacked 只考慮最後一個 Post，其長度相當於最後一個 Post 的 adu_spans 裡，屬於同一個 Post 的 ADU 的數量
    adu_attacked = torch.LongTensor(posts[-1][ADU_ATTACKED])

    # pairs 是用來建立 Arg Graph，因此每個 Comment 都需要
    # 不需製作成 Tensor。
    # TODO: 這裡應該可以先做成更方便的形式？
    inner_pairs = [post[INNER_PAIRS] for post in posts]
    inter_pairs = [post[INTER_PAIRS] for post in posts]

    # 拼接全部的 comments
    comment_ids = [post[COMMENT_IDS] for post in posts]
    comment_spans = [(1, len(el)-2) for el in comment_ids]
    max_comment_len = max([len(el) for el in comment_ids])
    comment_ids = [el + ([pad] * (max_comment_len - len(el))) for el in comment_ids]
    comment_ids = torch.LongTensor(comment_ids).view(num_posts, max_comment_len)
    comment_mask = (comment_ids != pad)
    global_attention_mask = torch.logical_or(comment_ids == user_1, comment_ids == user_2)

    return {
        'case': case,
        'input_ids': (comment_ids, comment_mask, comment_spans),
        'info_scores': info_scores,
        'global_attention_mask': global_attention_mask,
        'is_winner': is_winner,
        'adu_attacked': adu_attacked,
        'adu_span': (adu_spans, adu_masks, local_adu_in_use_masks, title_adu_span_index),
        'inner_pairs': inner_pairs,
        'inter_pairs': inter_pairs,
        'authors': authors
    }

# 讓 batch_size = 1，因為單一一個 thread 已經有夠大的資料要處理
def PadBatchValidationReport(batch, tokenizer, batch_size):
    assert batch_size == 1
    assert len(batch) == 1
    batch = batch[0]
    assert len(batch) == 2
    post_1, post_2 = batch[0], batch[1]

    return PadBatch([post_1], tokenizer, batch_size), PadBatch([post_2], tokenizer, batch_size)
