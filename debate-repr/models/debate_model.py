from torch import nn
from transformers import LongformerModel
import dgl
import models
import torch

EMBEDDING_DIM=768

class DebateModel(nn.Module):
    def __init__(self, 
                 meta_paths,
                 info_scores_dim,
                 lstm_hidden=80, 
                 han_hidden_size=256,
                 han_out_size=256,
                 han_num_heads=[3],
                 han_dropout=0.3,
                 share_span_and_comment_encoder=True, ##
                 compressed_comment_size=200,
                 revert_inter_post_relations=True):
        self.INFO_SCORES_DIM = info_scores_dim
        # Span Representation 的維度：是 BiLSTM 的 outputs 拼接
        self.SPAN_EMBEDDING_DIM = lstm_hidden * 4
        # ADU Representation 的維度，是 HAN 的 output 
        self.ADU_EMBEDDING_DIM = han_out_size
        # Compressed Comment (Context) 的維度，是透過 comment_compressor，由 integrated_comment 壓縮
        self.CONTEXT_EMBEDDING_DIM = compressed_comment_size
        # Comment Embedding 是 Comment Span Embedding
        self.COMMENT_EMBEDDING_DIM = self.SPAN_EMBEDDING_DIM
        # Integrated Comment 的維度，組成為：
        #   1. 上一段的 compressed comments (Context)
        #   2. Attended ADU
        #   3. Attended Inner ADU Pair
        #   4. Attended Inter ADU Pair
        #   5. Current Comment Embedding
        #   6. Info Scores (emotions, toxicity, fallacy)
        self.INTEGRATED_COMMENT_EMBEDDING_DIM = \
            self.CONTEXT_EMBEDDING_DIM + self.COMMENT_EMBEDDING_DIM + (self.ADU_EMBEDDING_DIM * 5 + 4) + info_scores_dim

        self.INTEGRATED_COMMENT_EMBEDDING_DIM_WITHOUT_CONTEXT = \
            self.INTEGRATED_COMMENT_EMBEDDING_DIM - self.CONTEXT_EMBEDDING_DIM
        
        super(DebateModel, self).__init__()
        self.revert_inter_post_relations = revert_inter_post_relations
        self.lstm_hidden = lstm_hidden
        
        # Longformer model for embeddings
        self.lf = LongformerModel.from_pretrained('Jeevesh8/sMLM-LF')
        
        self.span_encoder = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=lstm_hidden, bidirectional=True, batch_first=True)
        
        self.han = models.HAN(
            meta_paths=meta_paths, 
            in_size=self.SPAN_EMBEDDING_DIM, 
            hidden_size=han_hidden_size, 
            out_size=self.ADU_EMBEDDING_DIM, 
            num_heads=han_num_heads, 
            dropout=han_dropout)
        
        self.comment_compressor = \
            nn.LSTM(input_size=self.INTEGRATED_COMMENT_EMBEDDING_DIM_WITHOUT_CONTEXT, hidden_size=self.CONTEXT_EMBEDDING_DIM, bidirectional=False)
        
        self.inner_adu_relation_attention = nn.Sequential(
            nn.Linear(self.COMMENT_EMBEDDING_DIM + (self.ADU_EMBEDDING_DIM * 2) + 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softmax(0)
        )
        self.inter_adu_relation_attention = nn.Sequential(
            nn.Linear(self.COMMENT_EMBEDDING_DIM + (self.ADU_EMBEDDING_DIM * 2) + 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softmax(0)
        )
        self.adu_attention = nn.Sequential(
            nn.Linear(self.COMMENT_EMBEDDING_DIM + self.ADU_EMBEDDING_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softmax(0)
        )

        self._weight_init()

    def _weight_init(self, lower=-0.5, upper=0.5):
        for name, parameter in self.lf.named_parameters():
            parameter.requires_grad = False

        # Span Encoder
        lstm_named_parameters_list = [
            self.span_encoder.named_parameters(),
            self.comment_compressor.named_parameters()
        ]
        for named_parameters in lstm_named_parameters_list:
            for _, parameter in named_parameters:
                parameter.data.uniform_(lower, upper)

        # Comment Compressor, ADU attentions, vulnerability & winner pred
        linear_named_parameters_list = [
            self.inner_adu_relation_attention.named_parameters(),
            self.inter_adu_relation_attention.named_parameters(),
            self.adu_attention.named_parameters()]
        for named_parameters in linear_named_parameters_list:
            for name, parameter in named_parameters:
                if 'weight' in name or 'bias' in name:
                    parameter.data.uniform_(lower, upper)
                else:
                    raise Exception('unknown parameters')

    def _span_representation(self, sequence, index):
        # ADUs are in the range (i, j) inclusive.
        # both representations have size [seqlen, 2 * hidden_size]
        forward_sequence = sequence[:,:self.lstm_hidden]
        backward_sequence = sequence[:,self.lstm_hidden:]

        forward_j =  forward_sequence[index[1],:].view(self.lstm_hidden)
        forward_1i = forward_sequence[index[0]-1,:].view(self.lstm_hidden)
        forward_diff = forward_j - forward_1i

        backward_i =  backward_sequence[index[0],:].view(self.lstm_hidden)
        backward_j1 = backward_sequence[index[1]+1,:].view(self.lstm_hidden)
        backward_diff = backward_i - backward_j1

        output = torch.cat([forward_diff, backward_diff, forward_1i, backward_j1])
        return output

    # 一次只處理一個 thread
    # comment_ids: (num_comments, seq_len)
    def forward(self, 
                do_winner_pred, 
                comment_ids, 
                comment_mask, 
                comment_spans, 
                info_scores,
                global_attention_mask, 
                adu_spans, 
                adu_masks, 
                local_adu_in_use_masks, 
                title_adu_span_index,
                len_adu_attacked, 
                inner_pairs, 
                inter_pairs, 
                device,
                return_all=False):
        # token_embed: (num_comment, seq_len, EMBEDDING_DIM=768)
        with torch.no_grad():
            lf_output = self.lf(input_ids=comment_ids, attention_mask=comment_mask, global_attention_mask=global_attention_mask)
        token_embed = lf_output.last_hidden_state
        comment_embed = lf_output.pooler_output

        # span & comment encoded: (num_comment, seq_len, lstm_hidden=128)
        span_encoded, _ = self.span_encoder(token_embed)

        # for each comment...
        adu_vulnerability = None
        title_embed = None
        comments_without_context = []
        num_comments = comment_ids.shape[0]

        # for comment_idx in range(num_comments):
        #     comment_span = comment_spans[comment_idx]
        #     span_orig = span_encoded[comment_idx]
        #     current_comment_embed = self._span_representation(span_orig, comment_span)
        #     info_scores_tensor = info_scores[comment_idx]
        #     info_scores_tensor = torch.FloatTensor(info_scores_tensor).to(device)
        #     ic_components = [
        #         info_scores_tensor,
        #         current_comment_embed
        #     ]
        #     comment_withou_context = torch.cat(ic_components).view(self.INFO_SCORES_DIM + self.COMMENT_EMBEDDING_DIM)
        #     comments_without_context.append(comment_withou_context)

        for comment_idx in range(num_comments):
            # 第 comment_idx 個段落，抓出其 ADU Span Representations
            comment_span = comment_spans[comment_idx]
            #comment_orig = comment_encoded[comment_idx]
            span_orig = span_encoded[comment_idx]

            # 得到 current_comment_embed
            # current_comment_embed = comment_embed[comment_idx]
            current_comment_embed = self._span_representation(span_orig, comment_span)

            # Comment 有可能完全沒有 adus，此時就不需要計算 attended_adu, attended_inner 和 attended_inter
            real_adu_spans = torch.masked_select(adu_spans[comment_idx], adu_masks[comment_idx]).view(-1, 2)

            # 得到 title embedding
            if title_embed is None and comment_idx == 0:
                # The title adu span must start with 2 (after [user1] and [cls])
                assert real_adu_spans[title_adu_span_index][0] == 2
                title_embed = self._span_representation(span_orig, real_adu_spans[title_adu_span_index])

            if len(real_adu_spans) == 0:
                attended_adu = torch.zeros(size=(self.ADU_EMBEDDING_DIM,)).to(device)
                attended_inner = torch.zeros(size=(self.ADU_EMBEDDING_DIM * 2 + 2,)).to(device)
                attended_inter = torch.zeros(size=(self.ADU_EMBEDDING_DIM * 2 + 2,)).to(device)
            else:
                adus = torch.stack([self._span_representation(span_orig, index) for index in real_adu_spans])
                # assert adus.shape == (len(real_adu_spans), self.SPAN_EMBEDDING_DIM)

                # 針對此 Comment 製作 graph
                # 考慮到只有一步 aggregation step，一次製作多個段落的 graph 也不會有意義
                inner_pairs_comment = inner_pairs[comment_idx]
                inter_pairs_comment = inter_pairs[comment_idx]

                graph_data = {
                    ('adu', 'Attack', 'adu'): 
                        [(pair[0], pair[1]) for pair in inner_pairs_comment if pair[-1] == 'Attack'],
                    ('adu', 'Support', 'adu'): 
                        [(pair[0], pair[1]) for pair in inner_pairs_comment if pair[-1] == 'Support'],

                    ('adu', 'AttackedBy', 'adu') if self.revert_inter_post_relations else ('adu', 'Attack', 'adu'):
                        [(pair[1], pair[0]) if self.revert_inter_post_relations else (pair[0], pair[1]) \
                            for pair in inter_pairs_comment if pair[-1] == 'Attack'],
                    ('adu', 'SupportedBy', 'adu') if self.revert_inter_post_relations else ('adu', 'Support', 'adu'):
                        [(pair[1], pair[0]) if self.revert_inter_post_relations else (pair[0], pair[1]) \
                            for pair in inter_pairs_comment if pair[-1] == 'Support'],
                }

                for key in graph_data:
                    graph_data[key] = ([data[0] for data in graph_data[key]], [data[1] for data in graph_data[key]])
                
                graph = dgl.heterograph(graph_data, num_nodes_dict={ 'adu': len(adus) }).to(device)

                # 得到 ADU Embeddings，以及 pair embeddings
                adu_embeds = self.han(graph, adus)

                relation_type_lut = { 'Attack': [0, 1], 'Support': [1, 0], }

                inner_pairs_embed = \
                    [torch.cat( [ adu_embeds[pair[0]], adu_embeds[pair[1]], torch.FloatTensor(relation_type_lut[pair[2]]).to(device) ] ) \
                        for pair in inner_pairs_comment]
                if self.revert_inter_post_relations:
                    inter_pairs_embed = \
                        [torch.cat( [ adu_embeds[pair[1]], adu_embeds[pair[0]], torch.FloatTensor(relation_type_lut[pair[2]]).to(device) ] ) \
                            for pair in inter_pairs_comment]
                else:
                    inter_pairs_embed = \
                        [torch.cat( [ adu_embeds[pair[0]], adu_embeds[pair[1]], torch.FloatTensor(relation_type_lut[pair[2]]).to(device) ] ) \
                            for pair in inter_pairs_comment]
                
                # assert adu_embeds.shape == (len(adus), self.ADU_EMBEDDING_DIM)
                # assert all([e.shape == (self.ADU_EMBEDDING_DIM * 2 + 2,) for e in inner_pairs_embed])
                # assert all([e.shape == (self.ADU_EMBEDDING_DIM * 2 + 2,) for e in inter_pairs_embed])

                # 計算 ADU Attention 和 pair attention
                # 1. ADU Attention
                #    備註: ADU Attention 並不需要計算上一篇的 ADU，也不需要計算沒有和他人連結的 adu，用 local_adu_in_use_masks 篩選
                if len(adus) == 0:
                    attended_adu = torch.zeros(size=(self.ADU_EMBEDDING_DIM,)).to(device)
                else:
                    local_adu_in_use_masks_post = local_adu_in_use_masks[comment_idx]
                    adu_embed_concat = [torch.cat([current_comment_embed, embed]) for embed_idx, embed in enumerate(adu_embeds) \
                                                    if local_adu_in_use_masks_post[embed_idx]]
                    adu_embed_masked_concat = [embed for embed_idx, embed in enumerate(adu_embeds) \
                                               if local_adu_in_use_masks_post[embed_idx]]
                    num_local_adu_in_adu = len(adu_embed_concat)
                    
                    if num_local_adu_in_adu > 0:
                        adu_embed_concat = torch.stack(adu_embed_concat)
                        adu_embed_masked_concat = torch.stack(adu_embed_masked_concat)
                        # assert adu_embed_concat.shape == (len(adus), self.SPAN_EMBEDDING_DIM + self.ADU_EMBEDDING_DIM)
                        adu_attention_scores = self.adu_attention(adu_embed_concat).view(1, num_local_adu_in_adu)
                        attended_adu = torch.matmul(adu_attention_scores, adu_embed_masked_concat).view(self.ADU_EMBEDDING_DIM)
                    else:
                        attended_adu = torch.zeros(size=(self.ADU_EMBEDDING_DIM,)).to(device)

                # 2. Inner Pair Attention
                if len(inner_pairs_comment) == 0:
                    attended_inner = torch.zeros(size=(self.ADU_EMBEDDING_DIM * 2 + 2,)).to(device)
                else:
                    inner_concat = torch.stack([torch.cat([current_comment_embed, embed]) for embed in inner_pairs_embed])
                    # assert inner_concat.shape == (len(inner_pairs_comment), self.SPAN_EMBEDDING_DIM + (self.ADU_EMBEDDING_DIM * 2 + 2))
                    inner_attention_scores = self.inner_adu_relation_attention(inner_concat).view(1, len(inner_pairs_comment))
                    attended_inner = torch.matmul(inner_attention_scores, torch.stack(inner_pairs_embed)).view(self.ADU_EMBEDDING_DIM * 2 + 2)

                # Inter Pair Attention
                if len(inter_pairs_comment) == 0:
                    attended_inter = torch.zeros(size=(self.ADU_EMBEDDING_DIM * 2 + 2,)).to(device)
                else:
                    inter_concat = torch.stack([torch.cat([current_comment_embed, embed]) for embed in inter_pairs_embed])
                    # assert inter_concat.shape == (len(inter_pairs_comment), self.SPAN_EMBEDDING_DIM + (self.ADU_EMBEDDING_DIM * 2 + 2))
                    inter_attention_scores = self.inter_adu_relation_attention(inter_concat).view(1, len(inter_pairs_comment))
                    attended_inter = torch.matmul(inter_attention_scores, torch.stack(inter_pairs_embed)).view(self.ADU_EMBEDDING_DIM * 2 + 2)
                
            # info scores tensor
            info_scores_tensor = info_scores[comment_idx]
            info_scores_tensor = torch.FloatTensor(info_scores_tensor).to(device)

            # 計算 integrated comment
            ic_components = [
                attended_adu,
                attended_inner,
                attended_inter,
                info_scores_tensor,
                current_comment_embed
            ]
            comment_withou_context = torch.cat(ic_components).view(self.INTEGRATED_COMMENT_EMBEDDING_DIM_WITHOUT_CONTEXT)
            comments_without_context.append(comment_withou_context)

        # 所有 comments 的處理已完成。透過 comment_compressor 計算 context
        comment_contexts = torch.stack(comments_without_context)
        comment_contexts, _ = self.comment_compressor(comment_contexts)
        
        if not return_all:
            last_comment_context = comment_contexts[num_comments - 1, :]
            return torch.cat([last_comment_context, comments_without_context[-1]])
        else:
            assert len(comments_without_context) == len(comment_contexts)
            return [torch.cat([comment_contexts[i], comments_without_context[i]]) \
                    for i in range(num_comments)]
        
        # if not return_all:
        #     return comments_without_context[-1]
        # else:
        #     return comments_without_context

class DebateModelBaseline(nn.Module):
    def __init__(self, longformer_model_name, lstm_hidden_dim):
        super(DebateModelBaseline, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lf = LongformerModel.from_pretrained(longformer_model_name)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=lstm_hidden_dim, bidirectional=False)
        self._weight_init()

    def _weight_init(self, lower=-0.5, upper=0.5):
        for name, parameter in self.lf.named_parameters():
            parameter.requires_grad = ('pooler' in name)

        # LSTM
        for _, parameter in self.lstm.named_parameters():
            parameter.data.uniform_(lower, upper)

    def forward(self, comment_ids, comment_mask):
        lf_output = self.lf(input_ids=comment_ids, attention_mask=comment_mask)
        comment_embed = lf_output.pooler_output
        lstm_output, _ = self.lstm(comment_embed)
        lstm_output = lstm_output[-1].view(self.lstm_hidden_dim)

        return lstm_output

       
