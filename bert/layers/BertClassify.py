import torch.nn as nn

from config import *
from bert.layers.Transformer import Transformer
from bert.layers.BertEmbeddings import BertEmbeddings


class BertClassify(nn.Module):
    def __init__(self,
                 oce_kinds_num,
                 ocn_kinds_num,
                 tnews_kinds_num,
                 vocab_size=VocabSize,
                 hidden=HiddenSize,
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
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_num = attention_heads
        self.dropout_prob = dropout_prob
        self.attention_head_size = hidden // attention_heads
        self.intermediate_size = intermediate_size
        self.batch_size = OceBatchSize + OcnBatchSize + TnewsBatchSize

        # 申明网络
        self.bert_emb = BertEmbeddings()
        self.transformer_blocks = nn.ModuleList(
            Transformer(
                hidden_size=self.hidden_size,
                attention_head_num=self.attention_head_num,
                attention_head_size=self.attention_head_size,
                intermediate_size=self.intermediate_size).to(device)
            for _ in range(self.num_hidden_layers)
        )
        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.softmax_d1 = nn.Softmax(dim=-1)

        self.oce_layer2 = nn.Linear(self.hidden_size, self.batch_size)
        self.ocn_layer2 = nn.Linear(self.hidden_size, self.batch_size)
        self.tnews_layer2 = nn.Linear(self.hidden_size, self.batch_size)
        # self.attention_layer = nn.Linear(self.hidden_size, self.batch_size)
        self.dropout2 = nn.Dropout(0.1)

        self.oce_layer3 = nn.Linear(self.hidden_size, (self.batch_size) * 7)
        self.ocn_layer3 = nn.Linear(self.hidden_size, (self.batch_size) * 3)
        self.tnews_layer3 = nn.Linear(self.hidden_size, (self.batch_size) * 15)

    @staticmethod
    def gen_attention_masks(segment_ids):
        return segment_ids[:, None, None, :]

    def load_finetune(self, path=FinetunePath):
        pretrain_model_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(pretrain_model_dict.state_dict())

    def load_pretrain(self, path):
        pretrain_model_dict = torch.load(path)
        finetune_model_dict = self.state_dict()
        new_parameter_dict = {}
        # 加载embedding层参数
        for key in local2target_emb:
            local = key
            target = local2target_emb[key]
            new_parameter_dict[local] = pretrain_model_dict[target]
        # 加载transformerblock层参数
        for i in range(self.num_hidden_layers):
            for key in local2target_transformer:
                local = key % i
                target = local2target_transformer[key] % i
                new_parameter_dict[local] = pretrain_model_dict[target]
        for key, value in new_parameter_dict.items():
            if key in finetune_model_dict:
                finetune_model_dict[key] = new_parameter_dict[key]
        self.load_state_dict(finetune_model_dict)

    def forward(self, input_token, position_ids, segment_ids, oce_end_id, ocn_end_id, tnews_end_id):
        # embedding
        embedding_x = self.bert_emb(input_token, position_ids)
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
        feedforward_x = self.tanh(self.pooler(feedforward_x))

        if oce_end_id > 0:
            transformer_oce = feedforward_x[:oce_end_id, 0, :]
            oce_attention = self.oce_layer2(transformer_oce)
            oce_attention = self.dropout2(self.softmax_d1(oce_attention).unsqueeze(1))
            oce_value = self.oce_layer3(transformer_oce).contiguous().view(-1, self.batch_size, 7)
            oce_output = torch.matmul(oce_attention, oce_value).squeeze(1)
        else:
            oce_output = None
        if ocn_end_id > 0:
            transformer_ocn = feedforward_x[oce_end_id:oce_end_id+ocn_end_id, 0, :]
            ocn_attention = self.ocn_layer2(transformer_ocn)
            ocn_attention = self.dropout2(self.softmax_d1(ocn_attention).unsqueeze(1))
            ocn_value = self.ocn_layer3(transformer_ocn).contiguous().view(-1, self.batch_size, 3)
            ocn_output = torch.matmul(ocn_attention, ocn_value).squeeze(1)
        else:
            ocn_output = None
        if tnews_end_id > 0:
            transformer_tnews = feedforward_x[oce_end_id+ocn_end_id:oce_end_id+ocn_end_id+tnews_end_id, 0, :]
            tnews_attention = self.tnews_layer2(transformer_tnews)
            tnews_attention = self.dropout2(self.softmax_d1(tnews_attention).unsqueeze(1))
            tnews_value = self.tnews_layer3(transformer_tnews).contiguous().view(-1, self.batch_size, 15)
            tnews_output = torch.matmul(tnews_attention, tnews_value).squeeze(1)
        else:
            tnews_output = None

        return oce_output, ocn_output, tnews_output
