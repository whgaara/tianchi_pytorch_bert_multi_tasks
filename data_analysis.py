import re
import json
import jieba
import random

from config import TrainRate
from string import punctuation as str_punctuation
from zhon.hanzi import punctuation as zhon_punctuation


class DataAnalysis(object):
    def __init__(self):
        self.f = open('data/source_data/oce_source.csv', 'r', encoding='utf-8')
        self.t = open('data/source_data/oce_test.csv', 'r', encoding='utf-8')
        self.v = open('data/vocab.txt', 'r', encoding='utf-8')
        self.g = open('data/assistant.txt', 'w', encoding='utf-8')
        self.f_train = open('data/train_data/oce_train.txt', 'w', encoding='utf-8')
        self.f_eval = open('data/eval_data/oce_eval.txt', 'w', encoding='utf-8')
        self.f_test = open('data/test_data/oce_test.txt', 'w', encoding='utf-8')

        self.words = []
        self.vocabs = []
        self.classes = []
        self.sentences = []
        self.src_lines = []
        self.test_lines = []
        self.missed_chars = []
        self.sens_len_by_char = []
        self.sens_len_by_words = []

        self.classes2count = {}
        self.classes2sentences = {}

        self.__traverse()
        self.__count()

    def __traverse(self):
        for char in self.v:
            if char:
                char = char.strip()
                self.vocabs.append(char)
        for line in self.f:
            if line:
                self.src_lines.append(line.strip())
        for line in self.t:
            if line:
                self.test_lines.append(line.strip())

    def __count(self):
        # 对训练数据进行统计
        for line in self.src_lines:
            _, sentence, label = tuple(line.split('\t'))

            # ##文本必做预处理操作## #
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation, r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 使用数字符号代替纯数字
            current_words = []
            for words in jieba.lcut(sentence):
                if words.isdigit():
                    current_words.append('isdigit')
                else:
                    current_words.append(words)
            self.words.extend(current_words)
            #######################

            # 记录更新后的训练数据集
            if label in self.classes2sentences:
                self.classes2sentences[label].append(sentence)
            else:
                self.classes2sentences[label] = [sentence]

            self.sentences.append(sentence)
            self.classes.append(label)
            if label in self.classes2count:
                self.classes2count[label] += 1
            else:
                self.classes2count[label] = 1
            self.sens_len_by_char.append(len(sentence))
            self.sens_len_by_words.append(len(current_words))

        # 对测试数据进行统计
        for line in self.test_lines:
            _, sentence = tuple(line.split('\t'))

            # ##文本必做预处理操作## #
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation, r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 使用数字符号代替纯数字
            current_words = []
            for words in jieba.lcut(sentence):
                if words.isdigit():
                    current_words.append('isdigit')
                else:
                    current_words.append(words)
            self.words.extend(current_words)
            self.f_test.write(sentence + '\n')
            #######################

            self.sentences.append(sentence)
            self.words.extend(jieba.lcut(sentence))
            self.sens_len_by_char.append(len(sentence))
            self.sens_len_by_words.append(len(current_words))

        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        self.sens_len_by_char = sorted(list(set(self.sens_len_by_char)))
        self.sens_len_by_words = sorted(list(set(self.sens_len_by_words)))

    def __gen_new_vocab(self):
        with open('data/new_vocab.txt', 'w', encoding='utf-8') as f:
            for s_char in self.vocabs + self.missed_chars:
                if s_char:
                    f.write(s_char + '\n')

    def check_char(self):
        for sentence in self.sentences:
            for s_char in sentence:
                if s_char in self.vocabs:
                    continue
                else:
                    if s_char not in self.missed_chars:
                        self.missed_chars.append(s_char)
        self.__gen_new_vocab()

    def check_words(self):
        with open('data/new_words.txt', 'w', encoding='utf-8') as f:
            for word in self.words:
                if word:
                    f.write(word + '\n')

    def print_info(self):
        self.g.write('共有数据总数：%s\n' % len(self.sentences))
        self.g.write('最短单字文本长度：%s\n' % self.sens_len_by_char[0])
        self.g.write('最长单字文本长度：%s\n' % self.sens_len_by_char[-1])
        self.g.write('最短单词文本长度：%s\n' % self.sens_len_by_words[0])
        self.g.write('最长单词文本长度：%s\n' % self.sens_len_by_words[-1])
        self.g.write('共有类别总数：%s\n' % len(self.classes))
        self.g.write('类别数据分部：%s\n' % json.dumps(self.classes2count))

    def gen_train_eval(self):
        for label in self.classes2sentences:
            random.shuffle(self.classes2sentences[label])
            train_len = int(len(self.classes2sentences[label]) * TrainRate)
            tmp_train = self.classes2sentences[label][:train_len]
            tmp_eval = self.classes2sentences[label][train_len:]
            for sentence in tmp_train:
                self.f_train.write(label + '\t' + sentence + '\n')
            for sentence in tmp_eval:
                self.f_eval.write(label + '\t' + sentence + '\n')


if __name__ == '__main__':
    da = DataAnalysis()
    da.print_info()
    # da.check_char()
    # da.check_words()
    da.gen_train_eval()
