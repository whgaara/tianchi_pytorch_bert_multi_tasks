import time
import torch
import pickle

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

# ## 模型文件路径 ## #
UserDict = 'data/key.txt'
StopDict = 'data/stop.txt'
CorpusPath = 'data/train_data/oce_train.txt'
EvalPath = 'data/eval_data/oce_eval.txt'
TestPath = 'data/eval_data/oce_test.csv'
C2NPicklePath = 'data/classes2num.pickle'
W2NPicklePath = 'data/words2num.pickle'
EmojiPicklePath = 'data/emoji.pickle'
WordsVocabPath = 'data/new_words.txt'
CharsVocabPath = 'data/new_vocab.txt'

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
FinetunePath = 'checkpoint/finetune/bert_classify_%s.model' % SentenceLength
# ## 训练调试参数结束 ## #

# ## 通用参数 ## #
DropOut = 0.1
try:
    # 使用分词训练
    with open(W2NPicklePath, 'rb') as f:
        words2num = pickle.load(f)
        VocabSize = len(words2num)

    # 使用分字训练
    # VocabSize = len(open(CharsVocabPath, 'r', encoding='utf-8').readlines())
except:
    VocabSize = 0
HiddenSize = 768
IntermediateSize = 3072
AttentionHeadNum = 12


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
