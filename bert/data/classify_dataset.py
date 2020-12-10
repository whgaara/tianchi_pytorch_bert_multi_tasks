from config import *
from torch.utils.data import Dataset
from bert.common.tokenizers import Tokenizer


class BertDataSetByWords(Dataset):
    def __init__(self, corpus_path, c2n_pickle_path, w2n_pickle_path):
        self.labels = []
        self.corpus_path = corpus_path
        self.descriptions = []
        if self.corpus_path == OceTrainPath:
            self.type_id = 0
        if self.corpus_path == OcnTrainPath:
            self.type_id = 1
        if self.corpus_path == TnewsTrainPath:
            self.type_id = 2
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
        current_words = token_text.split(' ')
        current_words = ['[cls]'] + current_words

        tokens_id = []
        for word in current_words:
            tokens_id.append(self.words2num.get(word, self.words2num['[unk]']))
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(self.words2num['[pad]'])
        else:
            tokens_id = tokens_id[:SentenceLength]

        segment_ids = [1 if x else 0 for x in tokens_id]
        output['type_id'] = [self.type_id]
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
        if self.corpus_path == OceEvalPath:
            self.type_id = 0
        if self.corpus_path == OcnEvalPath:
            self.type_id = 1
        if self.corpus_path == TnewsEvalPath:
            self.type_id = 2
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
        current_words = token_text.split(' ')
        current_words = ['[cls]'] + current_words

        tokens_id = []
        for word in current_words:
            tokens_id.append(self.words2num.get(word, self.words2num['[unk]']))
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(self.words2num['[pad]'])
        else:
            tokens_id = tokens_id[:SentenceLength]

        segment_ids = [1 if x else 0 for x in tokens_id]
        output['type_id'] = torch.tensor([self.type_id], dtype=torch.long)
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
        new_text = token_text.replace(' ', '')
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
        new_text = token_text.replace(' ', '')
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
    x = 1
    import jieba
    jieba.load_userdict('../../data/key.txt')
    xxx = jieba.lcut('expression1orz(╯□╰)happinessislikeapebble1droppedintoapooltosetinmotionanever-wideningcircleofripples2.asstevensonhassaid,"beinghappyisaduty."快乐好似掷入池塘里的一枚鹅卵石,会激起不断扩散的一圈圈涟漪。斯蒂文生曾说过:“快乐是一种责任。”[求关注]')
    print(xxx)
