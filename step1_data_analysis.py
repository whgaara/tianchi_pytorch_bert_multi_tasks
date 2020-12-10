import re
import json
import jieba
import random
import pickle

from config import TrainRate, C2NPicklePath, W2NPicklePath, BalanceNum, UserDict, WordsVocabPath, EmojiPicklePath
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

        # 替换符号表情
        emojis = set()
        with open('data/expression_chars.txt', 'r', encoding='utf-8') as f:
            for word in f:
                if word:
                    word = word.strip()
                    emojis.add(word)
        emojis = list(emojis)
        self.emojis_dict = {}
        for i, emoji in enumerate(emojis):
            self.emojis_dict[emoji] = 'charexpress' + str(i)
        print('!!!如果表情符号新增，注意新增key.txt中的表情符号编码!!!\n')

        with open(EmojiPicklePath, 'wb') as f:
            pickle.dump(self.emojis_dict, f)

        self.fr_oce = open('data/OCEMOTION/source_data/oce_source.csv', 'r', encoding='utf-8')
        self.fr_ocn = open('data/OCNLI/source_data/ocn_source.csv', 'r', encoding='utf-8')
        self.fr_tnews = open('data/TNEWS/source_data/tnews_source.csv', 'r', encoding='utf-8')

        self.tr_oce = open('data/OCEMOTION/source_data/oce_test.csv', 'r', encoding='utf-8')
        self.tr_ocn = open('data/OCNLI/source_data/ocn_test.csv', 'r', encoding='utf-8')
        self.tr_tnews = open('data/TNEWS/source_data/tnews_test.csv', 'r', encoding='utf-8')

        self.fw_oce = open('data/OCEMOTION/test_data/oce_test.txt', 'w', encoding='utf-8')
        self.fw_ocn = open('data/OCNLI/test_data/ocn_test.txt', 'w', encoding='utf-8')
        self.fw_tnews = open('data/TNEWS/test_data/tnews_test.txt', 'w', encoding='utf-8')

        self.v = open('data/vocab.txt', 'r', encoding='utf-8')
        self.g = open('data/assistant.txt', 'w', encoding='utf-8')

        self.oce_classes = []
        self.ocn_classes = []
        self.tnews_classes = []

        self.oce_sentences = []
        self.ocn_sentences = []
        self.tnews_sentences = []

        self.oce_src_lines = []
        self.ocn_src_lines = []
        self.tnews_src_lines = []

        self.oce_test_lines = []
        self.ocn_test_lines = []
        self.tnews_test_lines = []

        self.words = []
        self.vocabs = []
        self.classes = []
        self.missed_chars = []
        self.sens_len_by_char = []
        self.sens_len_by_words = []

        self.words2num = {'[pad]': 0, '[cls]': 1, '[unk]': 2, '[sep]': 3}
        self.classes2num = {}
        self.classes2count = {}
        self.classes2sentences = {}

        self.__traverse()
        self.__count_oce()
        self.__count_ocn()
        self.__count_tnews()

        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        self.sens_len_by_char = sorted(list(set(self.sens_len_by_char)))
        self.sens_len_by_words = sorted(list(set(self.sens_len_by_words)))

        self.__balance()

    def __traverse(self):
        for char in self.v:
            if char:
                char = char.strip()
                self.vocabs.append(char)

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
            try:
                _, sentence, label = tuple(line.lower().split('\t'))
            except:
                x = 1

            # ########################文本必做预处理操作######################## #
            # 使用专用符号代替符号表情
            for emoji in self.emojis_dict:
                if emoji in sentence:
                    sentence = sentence.replace(emoji, self.emojis_dict[emoji])
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 使用数字符号代替纯数字
            current_words = []
            for word in jieba.lcut(sentence):
                if word in stop_list:
                    continue
                if word.isdigit() or word.replace('.', '').isdigit():
                    current_words.append('*')
                else:
                    current_words.append(word)

            new_current_words = []
            for word in current_words:
                if 'charexpress' in word:
                    new_current_words.append(word)
                else:
                    for ch in word:
                        new_current_words.append(ch)

            sentence = ' '.join(new_current_words)
            self.words.extend(new_current_words)
            ###################################################################

            # 记录label及其编号
            if label not in self.classes2num:
                self.classes2num[label] = len(self.classes2num)

            # 记录更新后的训练数据集
            if label in self.classes2sentences:
                self.classes2sentences[label].append(sentence)
            else:
                self.classes2sentences[label] = [sentence]

            self.oce_sentences.append(sentence)
            self.classes.append(label)
            if label in self.classes2count:
                self.classes2count[label] += 1
            else:
                self.classes2count[label] = 1
            self.sens_len_by_char.append(len(sentence))
            self.sens_len_by_words.append(len(new_current_words))

        # 对测试数据进行统计
        for line in self.oce_test_lines:
            num, sentence = tuple(line.lower().split('\t'))

            # ########################文本必做预处理操作######################## #
            # 使用专用符号代替符号表情
            for emoji in self.emojis_dict:
                if emoji in sentence:
                    sentence = sentence.replace(emoji, self.emojis_dict[emoji])
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 使用数字符号代替纯数字
            current_words = []
            for word in jieba.lcut(sentence):
                if word in stop_list:
                    continue
                if word.isdigit() or word.replace('.', '').isdigit():
                    current_words.append('*')
                else:
                    current_words.append(word)

            new_current_words = []
            for word in current_words:
                if 'charexpress' in word:
                    new_current_words.append(word)
                else:
                    for ch in word:
                        new_current_words.append(ch)

            sentence = ' '.join(new_current_words)
            self.words.extend(new_current_words)
            self.fw_oce.write(num + '\t' + sentence + '\n')
            ###################################################################

            self.oce_sentences.append(sentence)
            self.sens_len_by_char.append(len(sentence))
            self.sens_len_by_words.append(len(new_current_words))

    def __count_ocn(self):
        # 对训练数据进行统计
        for line in self.ocn_src_lines:
            _, sentence1, sentence2, label = tuple(line.lower().split('\t'))
            sentence = sentence1 + '|' + sentence2

            # ########################文本必做预处理操作######################## #
            # 使用专用符号代替符号表情
            for emoji in self.emojis_dict:
                if emoji in sentence:
                    sentence = sentence.replace(emoji, self.emojis_dict[emoji])
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 使用数字符号代替纯数字
            current_words = []
            for word in jieba.lcut(sentence):
                if word in stop_list:
                    continue
                if word.isdigit() or word.replace('.', '').isdigit():
                    current_words.append('*')
                else:
                    current_words.append(word)

            new_current_words = []
            for word in current_words:
                if 'charexpress' in word:
                    new_current_words.append(word)
                else:
                    for ch in word:
                        new_current_words.append(ch)

            sentence = ' '.join(new_current_words)
            self.words.extend(new_current_words)
            ###################################################################

            # 记录label及其编号
            if label not in self.classes2num:
                self.classes2num[label] = len(self.classes2num)

            # 记录更新后的训练数据集
            if label in self.classes2sentences:
                self.classes2sentences[label].append(sentence)
            else:
                self.classes2sentences[label] = [sentence]

            self.ocn_sentences.append(sentence)
            self.classes.append(label)
            if label in self.classes2count:
                self.classes2count[label] += 1
            else:
                self.classes2count[label] = 1
            self.sens_len_by_char.append(len(sentence))
            self.sens_len_by_words.append(len(new_current_words))

        # 对测试数据进行统计
        for line in self.ocn_test_lines:
            num, sentence1, sentence2 = tuple(line.lower().split('\t'))
            sentence = sentence1 + '|' + sentence2

            # ########################文本必做预处理操作######################## #
            # 使用专用符号代替符号表情
            for emoji in self.emojis_dict:
                if emoji in sentence:
                    sentence = sentence.replace(emoji, self.emojis_dict[emoji])
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 使用数字符号代替纯数字
            current_words = []
            for word in jieba.lcut(sentence):
                if word in stop_list:
                    continue
                if word.isdigit() or word.replace('.', '').isdigit():
                    current_words.append('*')
                else:
                    current_words.append(word)

            new_current_words = []
            for word in current_words:
                if 'charexpress' in word:
                    new_current_words.append(word)
                else:
                    for ch in word:
                        new_current_words.append(ch)

            sentence = ' '.join(new_current_words)
            self.words.extend(new_current_words)
            self.fw_ocn.write(num + '\t' + sentence + '\n')
            ###################################################################

            self.ocn_sentences.append(sentence)
            self.sens_len_by_char.append(len(sentence))
            self.sens_len_by_words.append(len(new_current_words))

    def __count_tnews(self):
        # 对训练数据进行统计
        for line in self.tnews_src_lines:
            _, sentence, label = tuple(line.lower().split('\t'))

            # ########################文本必做预处理操作######################## #
            # 使用专用符号代替符号表情
            for emoji in self.emojis_dict:
                if emoji in sentence:
                    sentence = sentence.replace(emoji, self.emojis_dict[emoji])
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 使用数字符号代替纯数字
            current_words = []
            for word in jieba.lcut(sentence):
                if word in stop_list:
                    continue
                if word.isdigit() or word.replace('.', '').isdigit():
                    current_words.append('*')
                else:
                    current_words.append(word)

            new_current_words = []
            for word in current_words:
                if 'charexpress' in word:
                    new_current_words.append(word)
                else:
                    for ch in word:
                        new_current_words.append(ch)

            sentence = ' '.join(new_current_words)
            self.words.extend(new_current_words)
            ###################################################################

            # 记录label及其编号
            if label not in self.classes2num:
                self.classes2num[label] = len(self.classes2num)

            # 记录更新后的训练数据集
            if label in self.classes2sentences:
                self.classes2sentences[label].append(sentence)
            else:
                self.classes2sentences[label] = [sentence]

            self.tnews_sentences.append(sentence)
            self.classes.append(label)
            if label in self.classes2count:
                self.classes2count[label] += 1
            else:
                self.classes2count[label] = 1
            self.sens_len_by_char.append(len(sentence))
            self.sens_len_by_words.append(len(new_current_words))

        # 对测试数据进行统计
        for line in self.tnews_test_lines:
            num, sentence = tuple(line.lower().split('\t'))

            # ########################文本必做预处理操作######################## #
            # 使用专用符号代替符号表情
            for emoji in self.emojis_dict:
                if emoji in sentence:
                    sentence = sentence.replace(emoji, self.emojis_dict[emoji])
            # 去除常见的重复标点符号
            sentence = re.sub(r"([%s])+" % str_punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', ''), r"\1", sentence)
            sentence = re.sub(r"([%s])+" % zhon_punctuation, r"\1", sentence)
            # 使用数字符号代替纯数字
            current_words = []
            for word in jieba.lcut(sentence):
                if word in stop_list:
                    continue
                if word.isdigit() or word.replace('.', '').isdigit():
                    current_words.append('*')
                else:
                    current_words.append(word)

            new_current_words = []
            for word in current_words:
                if 'charexpress' in word:
                    new_current_words.append(word)
                else:
                    for ch in word:
                        new_current_words.append(ch)

            sentence = ' '.join(new_current_words)
            self.words.extend(new_current_words)
            self.fw_tnews.write(num + '\t' + sentence + '\n')
            ###################################################################

            self.tnews_sentences.append(sentence)
            self.sens_len_by_char.append(len(sentence))
            self.sens_len_by_words.append(len(new_current_words))

    def __gen_new_vocab(self):
        with open('data/new_vocab.txt', 'w', encoding='utf-8') as f:
            for s_char in self.vocabs + self.missed_chars:
                if s_char:
                    f.write(s_char + '\n')

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

    def check_char(self):
        for sentence in self.oce_sentences + self.ocn_sentences + self.tnews_sentences:
            for s_char in sentence:
                if s_char in self.vocabs:
                    continue
                else:
                    if s_char not in self.missed_chars:
                        self.missed_chars.append(s_char)
        self.__gen_new_vocab()

    def check_words(self):
        with open(WordsVocabPath, 'w', encoding='utf-8') as f:
            for word in self.words:
                if word:
                    self.words2num[word] = len(self.words2num)
                    f.write(word + '\n')
        with open(W2NPicklePath, 'wb') as f:
            pickle.dump(self.words2num, f)

    def print_info(self):
        self.g.write('OCE共有数据总数：%s\n' % len(self.oce_sentences))
        self.g.write('OCN共有数据总数：%s\n' % len(self.ocn_sentences))
        self.g.write('Tnews共有数据总数：%s\n\n' % len(self.tnews_sentences))
        self.g.write('最短单字文本长度：%s\n' % self.sens_len_by_char[0])
        self.g.write('最长单字文本长度：%s\n' % self.sens_len_by_char[-1])
        self.g.write('最短单词文本长度：%s\n' % self.sens_len_by_words[0])
        self.g.write('最长单词文本长度：%s\n' % self.sens_len_by_words[-1])
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
    # da.check_char()
    da.check_words()
    da.gen_train_eval()
