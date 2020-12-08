import os
import math
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from tqdm import tqdm
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from bert.data.classify_dataset import *


class FGM(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, emb_g, epsilon=0.25, emb_name='embed.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(emb_g)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embed.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class textCNN(nn.Module):
    def __init__(self, class_num):
        super(textCNN, self).__init__()
        ci = 1
        kernel_num = 100
        kernel_size = [2, 3, 4, 5]
        vocab_size = VocabSize
        embed_dim = 256
        dropout = 0.1
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.conv14 = nn.Conv2d(ci, kernel_num, (kernel_size[3], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x4 = self.conv_and_pool(x, self.conv14)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3, x4), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    with open(C2NPicklePath, 'rb') as f:
        classes2num = pickle.load(f)
    num2classes = {}
    for x, y in classes2num.items():
        num2classes[y] = x

    with open(C2NPicklePath, 'rb') as f:
        class2num = pickle.load(f)
        labelcount = len(class2num)

    model = textCNN(labelcount).to(device)
    fgm = FGM(model)

    testset = BertEvalSetByWords(EvalPath, WordsVocabPath, C2NPicklePath, W2NPicklePath)
    dataset = BertDataSetByWords(CorpusPath, WordsVocabPath, C2NPicklePath, W2NPicklePath)
    dataloader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True, drop_last=False)

    optim = Adam(model.parameters(), lr=LearningRate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(Epochs):
        # train
        model.train()
        data_iter = tqdm(enumerate(dataloader),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataloader),
                         bar_format='{l_bar}{r_bar}')
        print_loss = 0.0
        for i, data in data_iter:
            data = {k: v.to(device) for k, v in data.items()}
            input_token = data['input_token_ids']
            for i in range(len(input_token)):
                input_token[i] = torch.cat((input_token[i][1:], torch.tensor([0]).to(device)))
            label = data['token_ids_labels']

            # output = model(input_token)
            # mask_loss = criterion(output, label)
            # print_loss = mask_loss.item()
            # optim.zero_grad()
            # mask_loss.backward()
            # optim.step()

            # 正常训练
            output = model(input_token)
            emb_g = None

            def extract(g):
                global emb_g
                emb_g = g

            mask_loss = criterion(output, label)
            model.embed.weight.register_hook(extract)
            mask_loss.backward()

            # 对抗训练，反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.attack(emb_g)  # 在embedding上添加对抗扰动
            fgm_output = fgm.model(input_token,)
            mask_loss = criterion(fgm_output, label)
            print_loss = mask_loss.item()
            fgm.restore()  # 恢复embedding参数
            optim.zero_grad()
            mask_loss.backward()
            optim.step()

        print('EP_%d mask loss:%s' % (epoch, print_loss))

        # save
        output_path = FinetunePath + '.ep%d' % epoch
        torch.save(model.cpu(), output_path)
        model.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # test
        with torch.no_grad():
            model.eval()
            test_count = 0
            accuracy_count = 0
            for test_data in testset:
                test_count += 1
                token_text = test_data['sentence']
                current_words = test_data['cut_words']
                input_token = test_data['input_token_ids'].unsqueeze(0).to(device)
                for i in range(len(input_token)):
                    input_token[i] = torch.cat((input_token[i][1:], torch.tensor([0]).to(device)))
                label = test_data['token_ids_labels'].tolist()
                output = model(input_token)
                output_tensor = torch.nn.Softmax(dim=-1)(output)
                output_topk = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()

                # 累计数值
                if label == output_topk[0]:
                    accuracy_count += 1
                else:
                    if epoch > 30:
                        print(token_text)
                        print(current_words)
                        print(num2classes[label])
                        print(num2classes[output_topk[0]])

            if test_count:
                acc_rate = float(accuracy_count) / float(test_count)
                acc_rate = round(acc_rate, 2)
                print('情感判断正确率：%s' % acc_rate)
