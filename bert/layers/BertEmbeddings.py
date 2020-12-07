import torch
import torch.nn as nn

from config import device, HiddenSize, SentenceLength, VocabSize


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, dropout_prob=0.1):
        super(BertEmbeddings, self).__init__()
        self.max_len = max_len
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(self.max_len, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_token):
        token_embeddings = self.token_embeddings(input_token)
        # 生成固定位置信息
        position_ids = []
        input_count = list(input_token.size())[0]
        for i in range(input_count):
            tmp = [x for x in range(self.max_len)]
            position_ids.append(tmp)
        position_ids = torch.tensor(position_ids).to(device)
        postion_embeddings = self.position_embeddings(position_ids)
        embedding_x = token_embeddings + postion_embeddings
        embedding_x = self.emb_normalization(embedding_x)
        embedding_x = self.emb_dropout(embedding_x)
        return embedding_x


class PositionEmbedding(nn.Module):
    def __init__(self):
        super(PositionEmbedding, self).__init__()
        self.demb = HiddenSize
        self.inv_freq = 1 / (10000 ** (torch.arange(0.0, self.demb, 2.0) / self.demb)).to(device)

    def forward(self):
        pos_seq = torch.arange(1.0, SentenceLength + 1).to(device)
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq).to(device)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1).to(device)
        return pos_emb


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size=VocabSize, hidden_size=HiddenSize, dropout_prob=0.1):
        super(TokenEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_token):
        token_embeddings = self.token_embeddings(input_token)
        embedding_x = self.emb_normalization(token_embeddings)
        embedding_x = self.emb_dropout(embedding_x)
        return embedding_x
