import time
import torch

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

# ## 模型文件路径 ## #
UserDict = 'data/key.txt'
StopDict = 'data/stop.txt'
CharsVocabPath = 'data/new_vocab.txt'
C2NPicklePath = 'data/classes2num.pickle'
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
Epochs = 16
TrainRate = 0.95
BalanceNum = 5000
LearningRate = 1e-5
AttentionMask = True
HiddenLayerNum = 12
SentenceLength = 512

# 计算BatchSize
NormalSteps = 10000
OceTrainCount = len(open(OceTrainPath, 'r', encoding='utf-8').readlines())
OcnTrainCount = len(open(OcnTrainPath, 'r', encoding='utf-8').readlines())
TnewsTrainCount = len(open(TnewsTrainPath, 'r', encoding='utf-8').readlines())
# OceBatchSize = OceTrainCount // NormalSteps
# OcnBatchSize = OcnTrainCount // NormalSteps
# TnewsBatchSize = TnewsTrainCount // NormalSteps
OceBatchSize = 1
OcnBatchSize = 1
TnewsBatchSize = 1
FinetunePath = 'checkpoint/finetune/bert_classify_%s_%s.model' % (SentenceLength, HiddenLayerNum)
PretrainPath = 'checkpoint/pretrain/pytorch_model.bin'
# ## 训练调试参数结束 ## #

# ## 通用参数 ## #
DropOut = 0.1
try:
    VocabSize = len(open(CharsVocabPath, 'r', encoding='utf-8').readlines())
except:
    VocabSize = 0
HiddenSize = 768
IntermediateSize = 3072
AttentionHeadNum = 12

local2target_emb = {
    'bert_emb.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
    'bert_emb.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
    'bert_emb.emb_normalization.weight': 'bert.embeddings.LayerNorm.gamma',
    'bert_emb.emb_normalization.bias': 'bert.embeddings.LayerNorm.beta'
}

local2target_transformer = {
    'transformer_blocks.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
    'transformer_blocks.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
    'transformer_blocks.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
    'transformer_blocks.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
    'transformer_blocks.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
    'transformer_blocks.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
    'transformer_blocks.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
    'transformer_blocks.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
    'transformer_blocks.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.gamma',
    'transformer_blocks.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.beta',
    'transformer_blocks.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
    'transformer_blocks.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
    'transformer_blocks.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
    'transformer_blocks.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
    'transformer_blocks.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.gamma',
    'transformer_blocks.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.beta',
}


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
