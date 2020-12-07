import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from bert.data.classify_dataset import *
from bert.layers.BertClassify import BertClassify


if __name__ == '__main__':
    with open(C2NPicklePath, 'rb') as f:
        class2num = pickle.load(f)
        labelcount = len(class2num)

    bert = BertClassify(kinds_num=labelcount).to(device)

    testset = BertEvalSetByWords(EvalPath, WordsVocabPath, C2NPicklePath, W2NPicklePath)
    dataset = BertDataSetByWords(CorpusPath, WordsVocabPath, C2NPicklePath, W2NPicklePath)
    dataloader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True, drop_last=False)

    optim = Adam(bert.parameters(), lr=LearningRate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(Epochs):
        # train
        bert.train()
        data_iter = tqdm(enumerate(dataloader),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataloader),
                         bar_format='{l_bar}{r_bar}')
        print_loss = 0.0
        for i, data in data_iter:
            data = {k: v.to(device) for k, v in data.items()}
            input_token = data['input_token_ids']
            segment_ids = data['segment_ids']
            label = data['token_ids_labels']
            output = bert(input_token, segment_ids)
            mask_loss = criterion(output, label)
            print_loss = mask_loss.item()
            optim.zero_grad()
            mask_loss.backward()
            optim.step()
        print('EP_%d mask loss:%s' % (epoch, print_loss))

        # save
        output_path = PretrainPath + '.ep%d' % epoch
        torch.save(bert.cpu(), output_path)
        bert.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # test
        with torch.no_grad():
            bert.eval()
            test_count = 0
            accuracy_count = 0
            for test_data in testset:
                test_count += 1
                input_token = test_data['input_token_ids'].unsqueeze(0).to(device)
                segment_ids = test_data['segment_ids'].unsqueeze(0).to(device)
                label = test_data['token_ids_labels'].tolist()
                output = bert(input_token, segment_ids)
                output_tensor = torch.nn.Softmax(dim=-1)(output)
                output_topk = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()

                # 累计数值
                if label == output_topk[0]:
                    accuracy_count += 1

            if test_count:
                acc_rate = float(accuracy_count) / float(test_count)
                acc_rate = round(acc_rate, 2)
                print('情感判断正确率：%s' % acc_rate)
