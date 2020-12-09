import jieba

from config import *
from torch.utils.data import Dataset
from bert.common.tokenizers import Tokenizer


jieba.load_userdict(UserDict)
with open(StopDict, 'r', encoding='utf-8') as f:
    stop_list = f.readlines()
    stop_list = [x.strip() for x in stop_list]


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
        current_words = jieba.lcut(token_text)
        new_words = []
        for word in current_words:
            if word not in stop_list:
                new_words.append(word)
        current_words = ['[cls]'] + new_words

        tokens_id = []
        for word in current_words:
            tokens_id.append(self.words2num.get(word, self.words2num['[unk]']))
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(self.words2num['[pad]'])
        else:
            tokens_id = tokens_id[:SentenceLength]

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
        current_words = jieba.lcut(token_text)
        new_words = []
        for word in current_words:
            if word not in stop_list:
                new_words.append(word)
        current_words = ['[cls]'] + new_words

        tokens_id = []
        for word in current_words:
            tokens_id.append(self.words2num.get(word, self.words2num['[unk]']))
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(self.words2num['[pad]'])
        else:
            tokens_id = tokens_id[:SentenceLength]

        segment_ids = [1 if x else 0 for x in tokens_id]
        output['sentence'] = token_text
        output['cut_words'] = current_words
        output['input_token_ids'] = torch.tensor(tokens_id, dtype=torch.long)
        output['segment_ids'] = torch.tensor(segment_ids, dtype=torch.long)
        output['token_ids_labels'] = torch.tensor(label_text, dtype=torch.long)
        return output


class BertDataSetByChars(Dataset):
    def __init__(self, corpus_path, vocab_path, c2n_pickle_path):
        self.labels = []
        self.corpus_path = corpus_path
        self.descriptions = []
        self.tokenizer = Tokenizer(vocab_path)
        with open(c2n_pickle_path, 'rb') as f:
            self.classes2num = pickle.load(f)
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

        current_words = jieba.lcut(token_text)
        new_text = ''.join(current_words)
        tokens_id = self.tokenizer.tokens_to_ids(list(new_text))
        tokens_id = [101] + tokens_id

        for i in range(SentenceLength - len(tokens_id)):
            tokens_id.append(0)

        segment_ids = [1 if x else 0 for x in tokens_id]
        output['input_token_ids'] = tokens_id
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


class BertEvalSetByChars(Dataset):
    def __init__(self, eval_path, vocab_path, c2n_pickle_path):
        self.labels = []
        self.corpus_path = eval_path
        self.descriptions = []
        self.tokenizer = Tokenizer(vocab_path)
        with open(c2n_pickle_path, 'rb') as f:
            self.classes2num = pickle.load(f)
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

        current_words = jieba.lcut(token_text)
        new_text = ''.join(current_words)
        tokens_id = self.tokenizer.tokens_to_ids(list(new_text))
        tokens_id = [101] + tokens_id

        for i in range(SentenceLength - len(tokens_id)):
            tokens_id.append(0)

        segment_ids = [1 if x else 0 for x in tokens_id]
        output['sentence'] = new_text
        output['input_token_ids'] = torch.tensor(tokens_id, dtype=torch.long)
        output['segment_ids'] = torch.tensor(segment_ids, dtype=torch.long)
        output['token_ids_labels'] = torch.tensor(label_text, dtype=torch.long)
        return output


if __name__ == '__main__':
    # data = BertDataSetByWords('../../data/train_data/oce_train.txt',
    #                           '../../data/classes2num.pickle',
    #                           '../../data/words2num.pickle')
    # for x in data:
    #     y = 1

    # import pkuseg
    # pk = pkuseg.pkuseg(user_dict='../../data/key.txt')
    # xxx = pk.cut('(╯□╰)happinessislikeapebble1droppedintoapooltosetinmotionanever-wideningcircleofripples2.asstevensonhassaid,"beinghappyisaduty."快乐好似掷入池塘里的一枚鹅卵石,会激起不断扩散的一圈圈涟漪。斯蒂文生曾说过:“快乐是一种责任。”[求关注]')
    # print(xxx)

    x = 1
    import jieba
    jieba.load_userdict('../../data/key.txt')
    xxx = jieba.lcut('expression1orz(╯□╰)happinessislikeapebble1droppedintoapooltosetinmotionanever-wideningcircleofripples2.asstevensonhassaid,"beinghappyisaduty."快乐好似掷入池塘里的一枚鹅卵石,会激起不断扩散的一圈圈涟漪。斯蒂文生曾说过:“快乐是一种责任。”[求关注]')
    print(xxx)
