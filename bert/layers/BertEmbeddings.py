import torch
import torch.nn as nn

from config import device, HiddenSize, SentenceLength, VocabSize


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=VocabSize, max_len=SentenceLength, hidden_size=HiddenSize, dropout_prob=0.1):
        super(BertEmbeddings, self).__init__()
        self.type_embeddings = nn.Embedding(3, hidden_size)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.part_embeddings = nn.Embedding(3, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, type_ids, input_token, position_ids, part_ids):
        type_embeddings = self.type_embeddings(type_ids)
        token_embeddings = self.token_embeddings(input_token)
        position_embeddings = self.position_embeddings(position_ids)
        part_embeddings = self.part_embeddings(part_ids)
        embedding_x = type_embeddings + token_embeddings + position_embeddings + part_embeddings
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


class TypeEmbedding(nn.Module):
    def __init__(self, type_size=3, hidden_size=HiddenSize, dropout_prob=0.1):
        super(TypeEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(type_size, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, type_token):
        token_embeddings = self.token_embeddings(type_token)
        embedding_x = self.emb_normalization(token_embeddings)
        embedding_x = self.emb_dropout(embedding_x)
        return embedding_x
