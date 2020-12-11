import torch.nn as nn

from config import *
from bert.layers.Gelu import GELU
from bert.layers.Transformer import Transformer
from bert.layers.BertEmbeddings import TokenEmbedding, PositionEmbedding, BertEmbeddings, TypeEmbedding


class BertClassify(nn.Module):
    def __init__(self,
                 oce_kinds_num,
                 ocn_kinds_num,
                 tnews_kinds_num,
                 vocab_size=VocabSize,
                 hidden=HiddenSize,
                 max_len=SentenceLength,
                 num_hidden_layers=HiddenLayerNum,
                 attention_heads=AttentionHeadNum,
                 dropout_prob=DropOut,
                 intermediate_size=IntermediateSize
                 ):
        super(BertClassify, self).__init__()
        self.oce_kinds_num = oce_kinds_num
        self.ocn_kinds_num = ocn_kinds_num
        self.tnews_kinds_num = tnews_kinds_num
        self.vocab_size = vocab_size
        self.hidden_size = hidden
        self.max_len = max_len
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_num = attention_heads
        self.dropout_prob = dropout_prob
        self.attention_head_size = hidden // attention_heads
        self.intermediate_size = intermediate_size

        # 申明网络
        self.type_emb = TypeEmbedding()
        self.token_emb = TokenEmbedding()
        self.position_emb = PositionEmbedding()
        self.transformer_blocks = nn.ModuleList(
            Transformer(
                hidden_size=self.hidden_size,
                attention_head_num=self.attention_head_num,
                attention_head_size=self.attention_head_size,
                intermediate_size=self.intermediate_size).to(device)
            for _ in range(self.num_hidden_layers)
        )
        self.transformer_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.transformer_act = GELU()
        self.dropout0 = nn.Dropout(0.1)
        self.layer_norm0 = nn.LayerNorm(self.hidden_size)

        self.oce_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.ocn_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.tnews_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)

        self.oce_layer2 = nn.Linear(self.hidden_size, 7)
        self.ocn_layer2 = nn.Linear(self.hidden_size, 3)
        self.tnews_layer2 = nn.Linear(self.hidden_size, 15)

    @staticmethod
    def gen_attention_masks(segment_ids):
        def gen_attention_mask(segment_id):
            dim = segment_id.size()[-1]
            attention_mask = torch.zeros([dim, dim], dtype=torch.int64)
            end_point = 0
            for i, segment in enumerate(segment_id.tolist()):
                if segment:
                    end_point = i
                else:
                    break
            for i in range(end_point + 1):
                for j in range(end_point + 1):
                    attention_mask[i][j] = 1
            return attention_mask
        attention_masks = []
        segment_ids = segment_ids.tolist()
        for segment_id in segment_ids:
            attention_mask = gen_attention_mask(torch.tensor(segment_id))
            attention_masks.append(attention_mask.tolist())
        return torch.tensor(attention_masks)

    def load_pretrain(self, path=LocalPretrainPath):
        pretrain_model_dict = torch.load(path)
        self.load_state_dict(pretrain_model_dict.state_dict())

    def forward(self, type_id, input_token, segment_ids, oce_end_id, ocn_end_id, tnews_end_id):
        # embedding
        embedding_x = self.type_emb(type_id) + self.token_emb(input_token) + self.position_emb()
        if AttentionMask:
            attention_mask = self.gen_attention_masks(segment_ids).to(device)
        else:
            attention_mask = None
        feedforward_x = None

        # transformer
        for i in range(self.num_hidden_layers):
            if i == 0:
                feedforward_x = self.transformer_blocks[i](embedding_x, attention_mask)
            else:
                feedforward_x = self.transformer_blocks[i](feedforward_x, attention_mask)

        # classify
        # 统一优化
        # transformer_output = self.transformer_layer(feedforward_x)
        # transformer_output = self.transformer_act(transformer_output)
        # transformer_output = self.dropout0(transformer_output)
        # transformer_output = self.layer_norm0(transformer_output)

        if oce_end_id > 0:
            transformer_oce = feedforward_x[:oce_end_id, 0, :]
            transformer_oce = self.oce_layer1(transformer_oce)
            transformer_oce = self.dropout1(transformer_oce)
            transformer_oce = self.layer_norm1(transformer_oce)
            oce_output = self.oce_layer2(transformer_oce)
        else:
            oce_output = None
        if ocn_end_id > 0:
            transformer_ocn = feedforward_x[oce_end_id:oce_end_id+ocn_end_id, 0, :]
            transformer_ocn = self.ocn_layer1(transformer_ocn)
            transformer_ocn = self.dropout1(transformer_ocn)
            transformer_ocn = self.layer_norm1(transformer_ocn)
            ocn_output = self.ocn_layer2(transformer_ocn)
        else:
            ocn_output = None
        if tnews_end_id > 0:
            transformer_tnews = feedforward_x[oce_end_id+ocn_end_id:oce_end_id+ocn_end_id+tnews_end_id, 0, :]
            transformer_tnews = self.tnews_layer1(transformer_tnews)
            transformer_tnews = self.dropout1(transformer_tnews)
            transformer_tnews = self.layer_norm1(transformer_tnews)
            tnews_output = self.tnews_layer2(transformer_tnews)
        else:
            tnews_output = None

        return oce_output, ocn_output, tnews_output
