import re
import json
import jieba
import random
import pickle

from config import TrainRate, C2NPicklePath, BalanceNum, UserDict, CharsVocabPath, EmojiPicklePath
from config import StopDict
from string import punctuation as str_punctuation
from zhon.hanzi import punctuation as zhon_punctuation


jieba.load_userdict(UserDict)
with open(StopDict, 'r', encoding='utf-8') as f:
    stop_list = f.readlines()
    stop_list = [x.strip() for x in stop_list]


class DataAnalysis(object):
    def __init__(self):
        self.label_group = {
            'oce': ["sadness", "happiness", "like", "anger", "fear", "surprise", "disgust"],
            'ocn': ["0", "1", "2"],
            'tnews': ["100", "101", "102", "103", "104", "106", "107", "108", "109", "110",
                      "112", "113", "114", "115", "116"]
        }
        self.fr_oce = open('data/OCEMOTION/source_data/oce_source.csv', 'r', encoding='utf-8')
        self.fr_ocn = open('data/OCNLI/source_data/ocn_source.csv', 'r', encoding='utf-8')
        self.fr_tnews = open('data/TNEWS/source_data/tnews_source.csv', 'r', encoding='utf-8')

        self.tr_oce = open('data/OCEMOTION/source_data/oce_test.csv', 'r', encoding='utf-8')
        self.tr_ocn = open('data/OCNLI/source_data/ocn_test.csv', 'r', encoding='utf-8')
        self.tr_tnews = open('data/TNEWS/source_data/tnews_test.csv', 'r', encoding='utf-8')

        self.fw_oce = open('data/OCEMOTION/test_data/oce_test.txt', 'w', encoding='utf-8')
        self.fw_ocn = open('data/OCNLI/test_data/ocn_test.txt', 'w', encoding='utf-8')
        self.fw_tnews = open('data/TNEWS/test_data/tnews_test.txt', 'w', encoding='utf-8')

        self.g = open('data/assistant.txt', 'w', encoding='utf-8')
        self.f_vocab = open('data/vocab.txt', 'r', encoding='utf-8')
        self.vocabs = [x.strip() for x in self.f_vocab.readlines()]
        self.vocabs_extend = []

        self.oce_src_lines = []
        self.ocn_src_lines = []
        self.tnews_src_lines = []

        self.oce_test_lines = []
        self.ocn_test_lines = []
        self.tnews_test_lines = []

        self.oce_sentences = []
        self.ocn_sentences = []
        self.tnews_sentences = []

        self.sens_len_by_char = []

        self.classes = []
        self.classes2num = {}
        self.classes2count = {}
        self.classes2sentences = {}

        self.__traverse()
        self.__count_oce()
        self.__count_ocn()
        self.__count_tnews()

        self.classes = sorted(list(set(self.classes)))
        self.sens_len_by_char = sorted(list(set(self.sens_len_by_char)))

        # self.__balance()

    def __traverse(self):
        for line in self.fr_oce:
            if line:
                self.oce_src_lines.append(line.strip())
        for line in self.fr_ocn:
            if line:
                self.ocn_src_lines.append(line.strip())
        for line in self.fr_tnews:
            if line:
                self.tnews_src_lines.append(line.strip())

        for line in self.tr_oce:
            if line:
                self.oce_test_lines.append(line.strip())
        for line in self.tr_ocn:
            if line:
                self.ocn_test_lines.append(line.strip())
        for line in self.tr_tnews:
            if line:
                self.tnews_test_lines.append(line.strip())

    def __count_oce(self):
        # 对训练数据进行统计
        for line in self.oce_src_lines:
            _, sentence, label = tuple(line.lower().split('\t'))
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 记录label及其编号
            if label not in self.classes2num:
                self.classes2num[label] = len(self.classes2num)
            # 记录更新后的训练数据集
            if label in self.classes2sentences:
                self.classes2sentences[label].append(sentence)
            else:
                self.classes2sentences[label] = [sentence]
            self.oce_sentences.append(sentence)
            self.sens_len_by_char.append(len(sentence))
            self.vocabs_extend += list(sentence)
            self.classes.append(label)
            if label in self.classes2count:
                self.classes2count[label] += 1
            else:
                self.classes2count[label] = 1

        # 对测试数据进行统计
        for line in self.oce_test_lines:
            num, sentence = tuple(line.lower().split('\t'))
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            self.fw_oce.write(num + '\t' + sentence + '\n')
            self.oce_sentences.append(sentence)
            self.vocabs_extend += list(sentence)
            self.sens_len_by_char.append(len(sentence))

    def __count_ocn(self):
        # 对训练数据进行统计
        for line in self.ocn_src_lines:
            _, sentence1, sentence2, label = tuple(line.lower().split('\t'))
            sentence = sentence1 + '\t' + sentence2
            # 记录label及其编号
            if label not in self.classes2num:
                self.classes2num[label] = len(self.classes2num)
            # 记录更新后的训练数据集
            if label in self.classes2sentences:
                self.classes2sentences[label].append(sentence)
            else:
                self.classes2sentences[label] = [sentence]
            self.ocn_sentences.append(sentence)
            self.sens_len_by_char.append(len(sentence))
            self.vocabs_extend += list(sentence)
            self.classes.append(label)
            if label in self.classes2count:
                self.classes2count[label] += 1
            else:
                self.classes2count[label] = 1

        # 对测试数据进行统计
        for line in self.ocn_test_lines:
            num, sentence1, sentence2 = tuple(line.lower().split('\t'))
            sentence = sentence1 + '\t' + sentence2
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            self.fw_ocn.write(num + '\t' + sentence + '\n')
            self.ocn_sentences.append(sentence)
            self.vocabs_extend += list(sentence)
            self.sens_len_by_char.append(len(sentence))

    def __count_tnews(self):
        # 对训练数据进行统计
        for line in self.tnews_src_lines:
            _, sentence, label = tuple(line.lower().split('\t'))
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 记录label及其编号
            if label not in self.classes2num:
                self.classes2num[label] = len(self.classes2num)
            # 记录更新后的训练数据集
            if label in self.classes2sentences:
                self.classes2sentences[label].append(sentence)
            else:
                self.classes2sentences[label] = [sentence]
            self.tnews_sentences.append(sentence)
            self.sens_len_by_char.append(len(sentence))
            self.vocabs_extend += list(sentence)
            self.classes.append(label)
            if label in self.classes2count:
                self.classes2count[label] += 1
            else:
                self.classes2count[label] = 1

        # 对测试数据进行统计
        for line in self.tnews_test_lines:
            num, sentence = tuple(line.lower().split('\t'))
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            self.fw_tnews.write(num + '\t' + sentence + '\n')
            self.tnews_sentences.append(sentence)
            self.sens_len_by_char.append(len(sentence))
            self.vocabs_extend += list(sentence)

    def __balance(self):
        for label in self.classes2sentences:
            tmp_list = self.classes2sentences[label]
            if self.classes2count[label] > BalanceNum:
                continue
            else:
                while True:
                    if len(tmp_list) >= BalanceNum:
                        break
                    random.shuffle(self.classes2sentences[label])
                    tmp_list = tmp_list + self.classes2sentences[label][:100]
            self.classes2sentences[label] = tmp_list

    def gen_new_vocab(self):
        self.vocabs_extend = list(set(self.vocabs_extend))
        for ch in self.vocabs_extend:
            if ch not in self.vocabs and ch:
                self.vocabs.append(ch)
        with open(CharsVocabPath, 'w', encoding='utf-8') as f:
            for char in self.vocabs:
                f.write(char + '\n')

    def print_info(self):
        self.g.write('OCE共有数据总数：%s\n' % len(self.oce_sentences))
        self.g.write('OCN共有数据总数：%s\n' % len(self.ocn_sentences))
        self.g.write('Tnews共有数据总数：%s\n\n' % len(self.tnews_sentences))
        self.g.write('最短文本长度：%s\n' % self.sens_len_by_char[0])
        self.g.write('最长文本长度：%s\n' % self.sens_len_by_char[-1])
        self.g.write('共有类别总数：%s\n' % len(self.classes))
        self.g.write('类别数据分部：%s\n' % json.dumps(self.classes2count))
        with open(C2NPicklePath, 'wb') as f:
            pickle.dump(self.classes2num, f)

    def gen_train_eval(self):
        f_oce_train = open('data/OCEMOTION/train_data/oce_train.txt', 'w', encoding='utf-8')
        f_oce_eval = open('data/OCEMOTION/eval_data/oce_eval.txt', 'w', encoding='utf-8')

        f_ocn_train = open('data/OCNLI/train_data/ocn_train.txt', 'w', encoding='utf-8')
        f_ocn_eval = open('data/OCNLI/eval_data/ocn_eval.txt', 'w', encoding='utf-8')

        f_tnews_train = open('data/TNEWS/train_data/tnews_train.txt', 'w', encoding='utf-8')
        f_tnews_eval = open('data/TNEWS/eval_data/tnews_eval.txt', 'w', encoding='utf-8')

        for label in self.classes2sentences:
            if label in self.label_group['oce']:
                random.shuffle(self.classes2sentences[label])
                train_len = int(len(self.classes2sentences[label]) * TrainRate)
                tmp_train = self.classes2sentences[label][:train_len]
                tmp_eval = self.classes2sentences[label][train_len:]
                for sentence in tmp_train:
                    f_oce_train.write(label + '\t' + sentence + '\n')
                for sentence in tmp_eval:
                    f_oce_eval.write(label + '\t' + sentence + '\n')

            if label in self.label_group['ocn']:
                random.shuffle(self.classes2sentences[label])
                train_len = int(len(self.classes2sentences[label]) * TrainRate)
                tmp_train = self.classes2sentences[label][:train_len]
                tmp_eval = self.classes2sentences[label][train_len:]
                for sentence in tmp_train:
                    f_ocn_train.write(label + '\t' + sentence + '\n')
                for sentence in tmp_eval:
                    f_ocn_eval.write(label + '\t' + sentence + '\n')

            if label in self.label_group['tnews']:
                random.shuffle(self.classes2sentences[label])
                train_len = int(len(self.classes2sentences[label]) * TrainRate)
                tmp_train = self.classes2sentences[label][:train_len]
                tmp_eval = self.classes2sentences[label][train_len:]
                for sentence in tmp_train:
                    f_tnews_train.write(label + '\t' + sentence + '\n')
                for sentence in tmp_eval:
                    f_tnews_eval.write(label + '\t' + sentence + '\n')


if __name__ == '__main__':
    da = DataAnalysis()
    da.print_info()
    da.gen_new_vocab()
    da.gen_train_eval()
