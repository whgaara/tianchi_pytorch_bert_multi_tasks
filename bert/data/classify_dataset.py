import math
import pickle
import string
import random
import pkuseg
import numpy as np

from tqdm import tqdm
from pretrain_config import *
from torch.utils.data import Dataset
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def tfidf(file_path='../../data/train_data/train_demo.csv'):
    descriptions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line:
                line = line.strip()
                line = line.split(',')
                if line[0] and line[1]:
                    descriptions.append(line[1])
    tfidfdict = {}
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(descriptions))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    for i in range(len(weight)):
        for j in range(len(word)):
            getword = word[j]
            getvalue = weight[i][j]
            if getvalue != 0:
                if getword in tfidfdict:
                    tfidfdict[getword] += int(getvalue)
                else:
                    tfidfdict.update({getword: getvalue})
    with open('../../data/train_data/tfidfdict.pickle', 'wb') as f:
        pickle.dump(tfidfdict, f)


class BertDataSetHead512(Dataset):
    def __init__(self, corpus_path):
        self.labels = []
        self.corpus_path = corpus_path
        self.descriptions = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split(',')
                    if line[0] and line[1]:
                        self.labels.append(int(line[0]))
                        self.descriptions.append(line[1].split(' ')[:SentenceLength-1])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        label_text = self.labels[item]
        token_text = self.descriptions[item]
        tokens_id = [7549] + [int(x) for x in token_text]
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(7550)
        segment_ids = [1 if x!=7550 else 0 for x in tokens_id]
        output['input_token_ids'] = tokens_id
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


class BertTestSetHead512(Dataset):
    def __init__(self, test_path):
        self.corpus_path = test_path
        self.labels = []
        self.descriptions = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split(',')
                    if line[0] and line[1]:
                        self.labels.append(int(line[0]))
                        self.descriptions.append(line[1].split(' ')[:SentenceLength-1])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        label_text = self.labels[item]
        token_text = self.descriptions[item]
        tokens_id = [7549] + [int(x) for x in token_text]
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(7550)
        segment_ids = [1 if x!=7550 else 0 for x in tokens_id]
        output['input_token_ids'] = tokens_id
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


class BertDataSetHeadTfidf(Dataset):
    def __init__(self, corpus_path, tfdict_path):
        self.labels = []
        self.corpus_path = corpus_path
        self.descriptions = []
        with open(tfdict_path, 'rb') as f:
            self.tfidfdict = pickle.load(f)
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split(',')
                    if line[0] and line[1]:
                        self.labels.append(int(line[0]))
                        self.descriptions.append(line[1])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        label_text = self.labels[item]
        token_text = self.descriptions[item].split(' ')
        tokens_id = [7549]
        for token in token_text:
            if int(token) not in self.tfidfdict:
                continue
            if self.tfidfdict[int(token)] > 0.5:
                tokens_id.append(int(token))
            if len(tokens_id) == 512:
                break
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(7550)
        segment_ids = [1 if x!=7550 else 0 for x in tokens_id]
        output['input_token_ids'] = tokens_id
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


class BertTestSetHeadTfidf(Dataset):
    def __init__(self, corpus_path, tfdict_path):
        self.labels = []
        self.corpus_path = corpus_path
        self.descriptions = []
        with open(tfdict_path, 'rb') as f:
            self.tfidfdict = pickle.load(f)
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split(',')
                    if line[0] and line[1]:
                        self.labels.append(int(line[0]))
                        self.descriptions.append(line[1])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        label_text = self.labels[item]
        token_text = self.descriptions[item].split(' ')
        tokens_id = [7549]
        for token in token_text:
            if int(token) not in self.tfidfdict:
                continue
            if self.tfidfdict[int(token)] > 0.5:
                tokens_id.append(int(token))
            if len(tokens_id) == 512:
                break
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(7550)
        segment_ids = [1 if x!=7550 else 0 for x in tokens_id]
        output['input_token_ids'] = tokens_id
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


if __name__ == '__main__':
    # tf = BertDataSetHeadTfidf('../../data/train_data/train_demo.csv')
    tfidf()
