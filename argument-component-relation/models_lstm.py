import torch
import torch.nn as nn
from transformers import BertModel, LongformerModel

EMBEDDING_DIM = 768

class Model_Adu_Relation(nn.Module):

    def __init__(self, 
                 model_class=BertModel, 
                 model_name='bert-base-uncased', 
                 bert_dropout=0.5, 
                 lstm_dropout=0, 
                 lstm_hidden=256, 
                 linear_hidden=32, 
                 task='link',
                 do_inners_embed=False, 
                 do_adu_order=False):
        super(Model_Adu_Relation, self).__init__()
        self.bert = model_class.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=bert_dropout)
        self.bilstm = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=lstm_hidden, bidirectional=True, batch_first=True, dropout=lstm_dropout)
        linear_dim = lstm_hidden * 8 + (1 if do_inners_embed else 0) + (1 if do_adu_order else 0)
        linear_output = 1 if task == 'link' else 2
        self.link_predictor_1 = nn.Linear(linear_dim, linear_hidden)
        self.link_predictor_2 = nn.Linear(linear_hidden, linear_output)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.is_longformer = (model_class == LongformerModel)
        
        self.do_inners_embed = do_inners_embed
        self.do_adu_order = do_adu_order
        self.lstm_hidden = lstm_hidden
        self.task = task

    def _span_representation(self, sequence, index):
        # ADUs are in the range (i, j) inclusive.
        # both representations have size [batch, 2 * hidden_size]
        batch_size = index.shape[0]
        forward_sequence = sequence[:,:,:self.lstm_hidden]
        backward_sequence = sequence[:,:,self.lstm_hidden:]

        forward_j =  torch.cat([forward_sequence[b,index[b,1],:].view(1, -1) for b in range(batch_size)], dim=0)
        forward_1i = torch.cat([forward_sequence[b,index[b,0]-1,:].view(1, -1) for b in range(batch_size)], dim=0)
        forward_diff = forward_j - forward_1i

        backward_i =  torch.cat([backward_sequence[b,index[b,0],:].view(1, -1) for b in range(batch_size)], dim=0)
        backward_j1 = torch.cat([backward_sequence[b,index[b,1]+1,:].view(1, -1) for b in range(batch_size)], dim=0)
        backward_diff = backward_i - backward_j1

        output = torch.cat([forward_diff, backward_diff, forward_1i, backward_j1], dim=1)
        return output

        # 原本是使用這個。
        # i_representation = [sequence[b,index[b,0],:].view(1, -1) for b in range(index.shape[0])]
        # j_representation = [sequence[b,index[b,1],:].view(1, -1) for b in range(index.shape[0])]
        # return torch.cat(j_representation, dim=0) - torch.cat(i_representation, dim=0)

    def forward(self, post_1, post_2, post_1_mask, post_2_mask, adu_1_index, adu_2_index, adu_orders, is_inners_embed):
        # post: [batch_size, sequence_length] where sequence_length <= 512
        self.bert.eval()
        with torch.no_grad():
            post_1_embed = self.bert(input_ids=post_1, token_type_ids=None, attention_mask=post_1_mask).last_hidden_state
            post_2_embed = self.bert(input_ids=post_2, token_type_ids=None, attention_mask=post_2_mask).last_hidden_state if post_2 is not None else None
            post_1_embed = self.dropout(post_1_embed)
            post_2_embed = self.dropout(post_2_embed) if post_2_embed is not None else None

        # post_embed: [batch_size, sequence_length, embedding_dim] where embedding_dim = 768
        bilstm_1_output, _ = self.bilstm(post_1_embed)
        bilstm_2_output, _ = self.bilstm(post_2_embed) if post_2_embed is not None else (bilstm_1_output, None)

        # bilstm_output: [batch_size, sequence_length, 2 * hidden_size] where hidden_size = lstm_hidden
        # adu_1_index & adu_2_index: [batch_size, 2]
        # adu_1_embed & adu_2_embed: [batch_size, 2 * hidden_size]
        adu_1_embed = self._span_representation(bilstm_1_output, adu_1_index)
        adu_2_embed = self._span_representation(bilstm_2_output, adu_2_index)
        if self.do_inners_embed and self.do_adu_order:
            adus_embed = torch.cat((adu_1_embed, adu_2_embed, is_inners_embed, adu_orders), dim=1)
        elif self.do_inners_embed:
            adus_embed = torch.cat((adu_1_embed, adu_2_embed, is_inners_embed), dim=1)
        elif self.do_adu_order:
            adus_embed = torch.cat((adu_1_embed, adu_2_embed, adu_orders), dim=1)
        else:
            adus_embed = torch.cat((adu_1_embed, adu_2_embed), dim=1)

        # adus_embed: [batch_size, 4 * hidden_size]
        linear_hidden = self.relu(self.link_predictor_1(adus_embed))
        prediction = self.link_predictor_2(linear_hidden)
        
        if self.task == 'link':
            return self.sigmoid(prediction.view(-1))
        else:
            return prediction


