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
        classes2num = pickle.load(f)
        labelcount = len(classes2num)
        num2classes = {}
        for x, y in classes2num.items():
            num2classes[y] = x

    bert = BertClassify(oce_kinds_num=7, ocn_kinds_num=3, tnews_kinds_num=15).to(device)

    # 使用分词训练
    oce_eval_set = BertEvalSetByWords(OceEvalPath, C2NPicklePath, W2NPicklePath)
    oce_train_set = BertDataSetByWords(OceTrainPath, C2NPicklePath, W2NPicklePath)
    ocn_eval_set = BertEvalSetByWords(OcnEvalPath, C2NPicklePath, W2NPicklePath)
    ocn_train_set = BertDataSetByWords(OcnTrainPath, C2NPicklePath, W2NPicklePath)
    tnews_eval_set = BertEvalSetByWords(TnewsEvalPath, C2NPicklePath, W2NPicklePath)
    tnews_train_set = BertDataSetByWords(TnewsTrainPath, C2NPicklePath, W2NPicklePath)
    # 使用分字训练
    # evalset = BertEvalSetByChars(EvalPath, CharsVocabPath, C2NPicklePath)
    # dataset = BertDataSetByChars(CorpusPath, CharsVocabPath, C2NPicklePath)

    oce_dataloader = DataLoader(dataset=oce_train_set, batch_size=OceBatchSize, shuffle=True, drop_last=True)
    ocn_dataloader = DataLoader(dataset=ocn_train_set, batch_size=OcnBatchSize, shuffle=True, drop_last=True)
    tnews_dataloader = DataLoader(dataset=tnews_train_set, batch_size=TnewsBatchSize, shuffle=True, drop_last=True)

    optim = Adam(bert.parameters(), lr=LearningRate)
    oce_criterion = nn.CrossEntropyLoss().to(device)
    ocn_criterion = nn.CrossEntropyLoss().to(device)
    tnews_criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(Epochs):
        # train
        bert.train()
        oce_data_iter = tqdm(enumerate(oce_dataloader), desc='EP_%s:%d' % ('train', epoch),
                             total=len(oce_dataloader), bar_format='{l_bar}{r_bar}')
        ocn_data_iter = tqdm(enumerate(ocn_dataloader), desc='EP_%s:%d' % ('train', epoch),
                             total=len(ocn_dataloader), bar_format='{l_bar}{r_bar}')
        tnews_data_iter = tqdm(enumerate(tnews_dataloader), desc='EP_%s:%d' % ('train', epoch),
                               total=len(tnews_dataloader), bar_format='{l_bar}{r_bar}')
        print_loss = 0.0

        for (i, oce_data), (j, ocn_data), (k, tnews_data) in zip(oce_data_iter, ocn_data_iter, tnews_data_iter):
            oce_data = {k: v.to(device) for k, v in oce_data.items()}
            ocn_data = {k: v.to(device) for k, v in ocn_data.items()}
            tnews_data = {k: v.to(device) for k, v in tnews_data.items()}

            oce_type_id = oce_data['type_id']
            oce_input_token = oce_data['input_token_ids']
            oce_segment_ids = oce_data['segment_ids']
            oce_label = oce_data['token_ids_labels']

            ocn_type_id = ocn_data['type_id']
            ocn_input_token = ocn_data['input_token_ids']
            ocn_segment_ids = ocn_data['segment_ids']
            ocn_label = ocn_data['token_ids_labels']

            tnews_type_id = tnews_data['type_id']
            tnews_input_token = tnews_data['input_token_ids']
            tnews_segment_ids = tnews_data['segment_ids']
            tnews_label = tnews_data['token_ids_labels']

            # 对数据进行叠加和拼接
            oce_end_id = len(oce_input_token)
            ocn_end_id = len(ocn_input_token)
            tnews_end_id = len(tnews_input_token)

            type_id = torch.cat([oce_type_id, ocn_type_id, tnews_type_id])
            input_token = torch.cat([oce_input_token, ocn_input_token, tnews_input_token])
            segment_ids = torch.cat([oce_segment_ids, ocn_segment_ids, tnews_segment_ids])

            output = bert(
                type_id,
                input_token,
                segment_ids,
                oce_end_id,
                ocn_end_id,
                tnews_end_id
            )

            # 多任务损失
            mask_loss = oce_criterion(output, label)
            print_loss = mask_loss.item()
            optim.zero_grad()
            mask_loss.backward()
            optim.step()

        print('EP_%d mask loss:%s' % (epoch, print_loss))

        # save
        output_path = FinetunePath + '.ep%d' % epoch
        torch.save(bert.cpu(), output_path)
        bert.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # test
        with torch.no_grad():
            bert.eval()
            test_count = 0
            accuracy_count = 0
            for eval_data in evalset:
                test_count += 1
                sentence = eval_data['sentence']
                input_token = eval_data['input_token_ids'].unsqueeze(0).to(device)
                segment_ids = eval_data['segment_ids'].unsqueeze(0).to(device)
                label = eval_data['token_ids_labels'].tolist()
                output = bert(input_token, segment_ids)
                output_tensor = torch.nn.Softmax(dim=-1)(output)
                output_topk = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()

                # 累计数值
                if label == output_topk[0]:
                    accuracy_count += 1
                else:
                    if epoch > 30:
                        print(sentence)
                        print('gt：', num2classes[label])
                        print('inf：', num2classes[output_topk[0]])

            if test_count:
                acc_rate = float(accuracy_count) / float(test_count)
                acc_rate = round(acc_rate, 2)
                print('情感判断正确率：%s' % acc_rate)
