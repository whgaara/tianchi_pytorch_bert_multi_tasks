import time
import torch
import pickle

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

# ## 模型文件路径 ## #
UserDict = 'data/key.txt'
StopDict = 'data/stop.txt'
WordsVocabPath = 'data/new_words.txt'
CharsVocabPath = 'data/new_vocab.txt'
C2NPicklePath = 'data/classes2num.pickle'
W2NPicklePath = 'data/words2num.pickle'
EmojiPicklePath = 'data/emoji.pickle'

OceTrainPath = 'data/OCEMOTION/train_data/oce_train.txt'
OceEvalPath = 'data/OCEMOTION/eval_data/oce_eval.txt'
OceTestPath = 'data/OCEMOTION/test_data/oce_test.txt'
OcnTrainPath = 'data/OCNLI/train_data/ocn_train.txt'
OcnEvalPath = 'data/OCNLI/eval_data/ocn_eval.txt'
OcnTestPath = 'data/OCNLI/test_data/ocn_test.txt'
TnewsTrainPath = 'data/TNEWS/train_data/tnews_train.txt'
TnewsEvalPath = 'data/TNEWS/eval_data/tnews_eval.txt'
TnewsTestPath = 'data/TNEWS/test_data/tnews_test.txt'

# 保存最大句长，字符数，类别数
Assistant = 'data/train_data/assistant.txt'

# ## 训练调试参数开始 ## #
Epochs = 32
TrainRate = 0.95
LearningRate = 5e-5
AttentionMask = True
HiddenLayerNum = 6
SentenceLength = 256
BalanceNum = 5000

# 计算BatchSize
NormalSteps = 2000
OceTrainCount = len(open(OceTrainPath, 'r', encoding='utf-8').readlines())
OcnTrainCount = len(open(OcnTrainPath, 'r', encoding='utf-8').readlines())
TnewsTrainCount = len(open(TnewsTrainPath, 'r', encoding='utf-8').readlines())
OceBatchSize = OceTrainCount // 2000
OcnBatchSize = OcnTrainCount // 2000
TnewsBatchSize = TnewsTrainCount // 2000
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
