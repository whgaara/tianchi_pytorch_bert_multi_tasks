import jieba
import pickle

from config import *
from torch.utils.data import Dataset


jieba.load_userdict('../../data/key.txt')


class BertDataSetByWords(Dataset):
    def __init__(self, corpus_path, c2n_pickle_path, w2n_pickle_path):
        self.labels = []
        self.corpus_path = corpus_path
        self.descriptions = []
        with open(c2n_pickle_path, 'rb') as f:
            self.classes2num = pickle.load(f)
        with open(w2n_pickle_path, 'rb') as f:
            self.words2num = pickle.load(f)
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split('\t')
                    if line[0] and line[1]:
                        self.labels.append(self.classes2num[line[0]])
                        self.descriptions.append(line[1])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        label_text = self.labels[item]
        token_text = self.descriptions[item]

        current_words = []
        for word in jieba.lcut(token_text):
            if word.isdigit() or word.replace('.', '').isdigit():
                current_words.append('isdigit')
            else:
                current_words.append(word)
        current_words = ['[cls]'] + current_words

        tokens_id = []
        for word in current_words:
            tokens_id.append(self.words2num.get(word, self.words2num['[unk]']))
        for i in range(SentenceLength - len(tokens_id)):
            tokens_id.append(self.words2num['[pad]'])

        segment_ids = [1 if x else 0 for x in tokens_id]
        output['input_token_ids'] = tokens_id
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


class BertEvalSetByWords(Dataset):
    def __init__(self, eval_path, c2n_pickle_path, w2n_pickle_path):
        self.labels = []
        self.corpus_path = eval_path
        self.descriptions = []
        with open(c2n_pickle_path, 'rb') as f:
            self.classes2num = pickle.load(f)
        with open(w2n_pickle_path, 'rb') as f:
            self.words2num = pickle.load(f)
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split('\t')
                    if line[0] and line[1]:
                        self.labels.append(self.classes2num[line[0]])
                        self.descriptions.append(line[1])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        label_text = self.labels[item]
        token_text = self.descriptions[item]

        current_words = []
        for word in jieba.lcut(token_text):
            if word.isdigit() or word.replace('.', '').isdigit():
                current_words.append('isdigit')
            else:
                current_words.append(word)
        current_words = ['[cls]'] + current_words

        tokens_id = []
        for word in current_words:
            tokens_id.append(self.words2num.get(word, self.words2num['[unk]']))
        for i in range(SentenceLength - len(tokens_id)):
            tokens_id.append(self.words2num['[pad]'])

        segment_ids = [1 if x else 0 for x in tokens_id]
        output['input_token_ids'] = tokens_id
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


if __name__ == '__main__':
    data = BertDataSetByWords('../../data/train_data/oce_train.txt',
                              '../../data/classes2num.pickle',
                              '../../data/words2num.pickle')
    for x in data:
        print(x)
