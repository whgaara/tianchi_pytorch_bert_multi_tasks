import time
import torch
import pickle

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

# ## 模型文件路径 ## #
CorpusPath = 'data/train_data/oce_train.txt'
# TDemoPath = 'data/train_data/train_demo.csv'
EvalPath = 'data/eval_data/oce_eval.txt'
# EDemoPath = 'data/test_data/eval_demo.csv'
TestPath = 'data/test_data/oce_test.csv'
C2NPicklePath = 'data/classes2num.pickle'
W2NPicklePath = 'data/words2num.pickle'
WordsVocabPath = 'data/new_words.txt'

# 保存最大句长，字符数，类别数
Assistant = 'data/train_data/assistant.txt'

# ## 训练调试参数开始 ## #
Epochs = 32
BatchSize = 32
TrainRate = 0.95
LearningRate = 1e-4
AttentionMask = True
HiddenLayerNum = 2
SentenceLength = 128
BalanceNum = 5000
PretrainPath = 'checkpoint/finetune/bert_classify_%s.model' % SentenceLength
# ## 训练调试参数结束 ## #

# ## 通用参数 ## #
DropOut = 0.1
try:
    with open(W2NPicklePath, 'rb') as f:
        words2num = pickle.load(f)
        VocabSize = len(words2num)
except:
    VocabSize = 0
HiddenSize = 768
IntermediateSize = 3072
AttentionHeadNum = 12


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
