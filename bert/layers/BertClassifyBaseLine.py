import torch.nn as nn

from config import *
from transformers import BertModel, BertTokenizer


class BertClassifyBaseLine(nn.Module):
    def __init__(self,
                 oce_kinds_num,
                 ocn_kinds_num,
                 tnews_kinds_num,
                 pretrain_model_path
                 ):
        super(BertClassifyBaseLine, self).__init__()
        self.oce_kinds_num = oce_kinds_num
        self.ocn_kinds_num = ocn_kinds_num
        self.tnews_kinds_num = tnews_kinds_num
        self.batch_size = OceBatchSize + OcnBatchSize + TnewsBatchSize

        # 申明网络
        self.bert = BertModel.from_pretrained(pretrain_model_path)

        self.attention_layer = nn.Linear(self.hidden_size, self.batch_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout2 = nn.Dropout(0.1)
        self.layer_norm2 = nn.LayerNorm(self.batch_size)

        self.oce_layer3 = nn.Linear(self.hidden_size, (self.batch_size) * 7)
        self.ocn_layer3 = nn.Linear(self.hidden_size, (self.batch_size) * 3)
        self.tnews_layer3 = nn.Linear(self.hidden_size, (self.batch_size) * 15)

    def forward(self, type_id, input_token, segment_ids, oce_end_id, ocn_end_id, tnews_end_id):
        bert_output = self.bert(
            input_ids=input_token,
            token_type_ids=type_id,
            attention_mask=segment_ids)[0][:, 0, :].squeeze(1)

        # classify
        if oce_end_id > 0:
            transformer_oce = bert_output[:oce_end_id, 0, :]
            oce_attention = self.attention_layer(transformer_oce)
            oce_attention = self.dropout2(self.softmax(oce_attention).unsqueeze(1))
            oce_value = self.oce_layer3(transformer_oce).contiguous().view(-1, self.batch_size, 7)
            oce_output = torch.matmul(oce_attention, oce_value).squeeze(1)
        else:
            oce_output = None
        if ocn_end_id > 0:
            transformer_ocn = bert_output[oce_end_id:oce_end_id+ocn_end_id, 0, :]
            ocn_attention = self.attention_layer(transformer_ocn)
            ocn_attention = self.dropout2(self.softmax(ocn_attention).unsqueeze(1))
            ocn_value = self.ocn_layer3(transformer_ocn).contiguous().view(-1, self.batch_size, 3)
            ocn_output = torch.matmul(ocn_attention, ocn_value).squeeze(1)
        else:
            ocn_output = None
        if tnews_end_id > 0:
            transformer_tnews = bert_output[oce_end_id+ocn_end_id:oce_end_id+ocn_end_id+tnews_end_id, 0, :]
            tnews_attention = self.attention_layer(transformer_tnews)
            tnews_attention = self.dropout2(self.softmax(tnews_attention).unsqueeze(1))
            tnews_value = self.tnews_layer3(transformer_tnews).contiguous().view(-1, self.batch_size, 15)
            tnews_output = torch.matmul(tnews_attention, tnews_value).squeeze(1)
        else:
            tnews_output = None

        return oce_output, ocn_output, tnews_output
