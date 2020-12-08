import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from config import *
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from bert.data.classify_dataset import *


with open(W2NPicklePath, 'rb') as f:
        words2num = pickle.load(f)
        LstmVocabSize = len(words2num)


class LstmAttention(nn.Module):
    def __init__(self, num_embeddings=LstmVocabSize, embedding_dim=768, hidden_size=16, num_layers=1):
        super(LstmAttention, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.w = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.dense = nn.Linear(hidden_size, 7)

        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.u, -0.1, 0.1)

    def forward(self, inputs):
        """
        :param inputs: [batch_size,seq_len]
        :return:
        """
        embedding = self.embedding(inputs)  # [batch_size,seq_len,embedding_dim]
        output, _ = self.lstm(embedding)  # [batch_size,seq_len,hidden_size*2]

        # 添加 attention
        u = torch.tanh(torch.matmul(output, self.w))  # [batch_size,seq_len,hidden_size*2]
        attention = torch.matmul(u, self.u)  # [batch_size,seq_len,1]
        score = F.softmax(attention, dim=-1)  # [batch_size,seq_len,1]
        output = output * score  # [batch_size,seq_len,hidden_size*2]
        # attention 结束

        output = torch.sum(output, dim=1)  # [batch_size,hidden_size*2]
        output = self.dense(output)  # [batch_size,1]

        return output


if __name__ == '__main__':
    with open(C2NPicklePath, 'rb') as f:
        classes2num = pickle.load(f)
    num2classes = {}
    for x, y in classes2num.items():
        num2classes[y] = x

    with open(C2NPicklePath, 'rb') as f:
        class2num = pickle.load(f)
        labelcount = len(class2num)

    model = LstmAttention().to(device)
    testset = BertEvalSetByWords(EvalPath, C2NPicklePath, W2NPicklePath)
    dataset = BertDataSetByWords(CorpusPath, C2NPicklePath, W2NPicklePath)
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

            output = model(input_token)
            mask_loss = criterion(output, label)
            print_loss = mask_loss.item()
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
