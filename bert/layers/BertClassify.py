import torch.nn as nn

from config import *
from bert.layers.Gelu import GELU
from bert.layers.Transformer import Transformer
from bert.layers.BertEmbeddings import TokenEmbedding, PositionEmbedding, TypeEmbedding


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
        self.batch_size = OceBatchSize + OcnBatchSize + TnewsBatchSize

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
        self.oce_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.ocn_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.tnews_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.softmax_d1 = nn.Softmax(dim=-1)
        self.gelu = GELU()

        self.oce_layer2 = nn.Linear(self.hidden_size, self.batch_size)
        self.ocn_layer2 = nn.Linear(self.hidden_size, self.batch_size)
        self.tnews_layer2 = nn.Linear(self.hidden_size, self.batch_size)
        self.dropout2 = nn.Dropout(0.1)
        self.layer_norm2 = nn.LayerNorm(self.batch_size)

        self.oce_layer3 = nn.Linear(self.hidden_size, (self.batch_size) * 7)
        self.ocn_layer3 = nn.Linear(self.hidden_size, (self.batch_size) * 3)
        self.tnews_layer3 = nn.Linear(self.hidden_size, (self.batch_size) * 15)

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

    def load_finetune(self, path=FinetunePath):
        pretrain_model_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(pretrain_model_dict.state_dict())

    def load_pretrain(self, path):
        pretrain_model_dict = torch.load(path)
        finetune_model_dict = self.state_dict()
        new_parameter_dict = {}
        # 加载transformerblock层参数
        for i in range(self.num_hidden_layers):
            for key in local2target_transformer:
                local = key % i
                target = local2target_transformer[key] % i
                new_parameter_dict[local] = pretrain_model_dict[target]
        finetune_model_dict.update(new_parameter_dict)
        self.load_state_dict(finetune_model_dict)

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
        if oce_end_id > 0:
            transformer_oce = feedforward_x[:oce_end_id, 0, :]
            # transformer_oce = self.oce_layer1(transformer_oce)
            # transformer_oce = self.gelu(transformer_oce)
            # transformer_oce = self.dropout1(transformer_oce)
            # transformer_oce = self.layer_norm1(transformer_oce)
            oce_attention = self.oce_layer2(transformer_oce)
            oce_attention = self.dropout2(self.softmax_d1(oce_attention).unsqueeze(1))
            oce_attention = self.layer_norm2(oce_attention)
            oce_value = self.oce_layer3(transformer_oce).contiguous().view(-1, self.batch_size, 7)
            oce_output = torch.matmul(oce_attention, oce_value).squeeze(1)
        else:
            oce_output = None
        if ocn_end_id > 0:
            transformer_ocn = feedforward_x[oce_end_id:oce_end_id+ocn_end_id, 0, :]
            transformer_ocn = self.gelu(self.ocn_layer1(transformer_ocn))
            transformer_ocn = self.dropout1(transformer_ocn)
            transformer_ocn = self.layer_norm1(transformer_ocn)
            ocn_attention = self.dropout2(self.softmax_d1(self.ocn_layer2(transformer_ocn)).unsqueeze(1))
            ocn_attention = self.layer_norm2(ocn_attention)
            ocn_value = self.ocn_layer3(transformer_ocn).contiguous().view(-1, self.batch_size, 3)
            ocn_output = torch.matmul(ocn_attention, ocn_value).squeeze(1)
        else:
            ocn_output = None
        if tnews_end_id > 0:
            transformer_tnews = feedforward_x[oce_end_id+ocn_end_id:oce_end_id+ocn_end_id+tnews_end_id, 0, :]
            transformer_tnews = self.gelu(self.tnews_layer1(transformer_tnews))
            transformer_tnews = self.dropout1(transformer_tnews)
            transformer_tnews = self.layer_norm1(transformer_tnews)
            tnews_attention = self.dropout2(self.softmax_d1(self.tnews_layer2(transformer_tnews)).unsqueeze(1))
            tnews_attention = self.layer_norm2(tnews_attention)
            tnews_value = self.tnews_layer3(transformer_tnews).contiguous().view(-1, self.batch_size, 15)
            tnews_output = torch.matmul(tnews_attention, tnews_value).squeeze(1)
        else:
            tnews_output = None

        return oce_output, ocn_output, tnews_output
