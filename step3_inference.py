# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

from tqdm import tqdm
from pretrain_config import PretrainPath, device, SentenceLength, VocabPath
from bert.common.tokenizers import Tokenizer


class Inference(object):
    def __init__(self):
        self.sen_count = 0
        self.sen_acc = 0
        self.model = torch.load(PretrainPath).to(device).eval()
        print('加载模型完成！')

    def inference_single(self, text):
        text2id = [7549] + [int(x) for x in text][:511]
        segments = []
        with torch.no_grad():
            output_tensor = self.model(text2id, segments)
            output_tensor = torch.nn.Softmax(dim=-1)(output_tensor)
            output_topk_prob = torch.topk(output_tensor, 1).values.squeeze(0).tolist()
            output_topk_indice = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()
        return output_topk_indice[0], output_topk_prob[0]

    def inference_batch(self, file_path):
        f = open(file_path, 'r', encoding='utf-8')
        for i, line in enumerate(tqdm(f)):
            if i == 0:
                continue
            if line:
                line = line.strip()
                self.sen_count += 1
                line = line.split(' ')
                output, prob = self.inference_single(line)


if __name__ == '__main__':
    bert_infer = Inference()
    bert_infer.inference_batch('../../data/test_data/test.txt')
