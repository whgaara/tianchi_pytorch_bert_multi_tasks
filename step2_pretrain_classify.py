import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import pickle
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from bert.data.classify_dataset import *
from bert.layers.BertClassify import BertClassify
from sklearn.metrics import f1_score


def get_f1(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    return marco_f1_score


if __name__ == '__main__':
    with open(C2NPicklePath, 'rb') as f:
        classes2num = pickle.load(f)
        num2classes = {}
        for x, y in classes2num.items():
            num2classes[y] = x

    bert = BertClassify(oce_kinds_num=7, ocn_kinds_num=3, tnews_kinds_num=15).to(device)

    if os.path.exists(FinetunePath):
        print('开始加载本地预训练模型！')
        bert.load_finetune(FinetunePath)
        bert = bert.to(device)
        print('完成加载本地预训练模型！\n')
    else:
        print('开始加载外部预训练模型！')
        bert.load_pretrain(PretrainPath)
        print('完成加载外部预训练模型！')

    # 使用分词训练
    oce_eval_set = BertEvalSetByWords(OceEvalPath, C2NPicklePath)
    oce_train_set = BertDataSetByWords(OceTrainPath, C2NPicklePath)
    ocn_eval_set = BertEvalSetByWords(OcnEvalPath, C2NPicklePath)
    ocn_train_set = BertDataSetByWords(OcnTrainPath, C2NPicklePath)
    tnews_eval_set = BertEvalSetByWords(TnewsEvalPath, C2NPicklePath)
    tnews_train_set = BertDataSetByWords(TnewsTrainPath, C2NPicklePath)

    oce_dataloader = DataLoader(dataset=oce_train_set, batch_size=OceBatchSize, shuffle=True, drop_last=False)
    ocn_dataloader = DataLoader(dataset=ocn_train_set, batch_size=OcnBatchSize, shuffle=True, drop_last=False)
    tnews_dataloader = DataLoader(dataset=tnews_train_set, batch_size=TnewsBatchSize, shuffle=True, drop_last=False)

    optim = Adam(bert.parameters(), lr=LearningRate)
    oce_criterion = nn.CrossEntropyLoss().to(device)
    ocn_criterion = nn.CrossEntropyLoss().to(device)
    tnews_criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(Epochs):
        oce_pred_list = []
        oce_label_list = []
        ocn_pred_list = []
        ocn_label_list = []
        tnews_pred_list = []
        tnews_label_list = []

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
            if i == min(len(oce_data_iter), len(ocn_data_iter), len(tnews_data_iter)):
                break

            oce_data = {k: v.to(device) for k, v in oce_data.items()}
            ocn_data = {k: v.to(device) for k, v in ocn_data.items()}
            tnews_data = {k: v.to(device) for k, v in tnews_data.items()}

            oce_type_id = oce_data['type_id']
            oce_input_token = oce_data['input_token_ids']
            oce_segment_ids = oce_data['segment_ids']
            oce_label = oce_data['token_ids_labels']

            ocn_type_id = ocn_data['type_id']
            # ocn_separator = ocn_data['separator']
            ocn_input_token = ocn_data['input_token_ids']
            ocn_segment_ids = ocn_data['segment_ids']
            # -7为后续求损失做准备
            ocn_label = ocn_data['token_ids_labels'] - 7

            tnews_type_id = tnews_data['type_id']
            tnews_input_token = tnews_data['input_token_ids']
            tnews_segment_ids = tnews_data['segment_ids']
            # -10为后续求损失做准备
            tnews_label = tnews_data['token_ids_labels'] - 10

            # 对数据进行叠加和拼接
            oce_end_id = len(oce_input_token)
            ocn_end_id = len(ocn_input_token)
            tnews_end_id = len(tnews_input_token)

            type_id = torch.cat([oce_type_id, ocn_type_id, tnews_type_id])
            input_token = torch.cat([oce_input_token, ocn_input_token, tnews_input_token])
            segment_ids = torch.cat([oce_segment_ids, ocn_segment_ids, tnews_segment_ids])

            oce_output, ocn_output, tnews_output = bert(
                type_id,
                input_token,
                segment_ids,
                oce_end_id,
                ocn_end_id,
                tnews_end_id
            )

            # 多任务损失
            oce_loss = oce_criterion(oce_output, oce_label)
            ocn_loss = ocn_criterion(ocn_output, ocn_label)
            tnews_loss = tnews_criterion(tnews_output, tnews_label)
            oce_loss_weight = torch.div(oce_loss, oce_loss + ocn_loss + tnews_loss)
            ocn_loss_weight = torch.div(ocn_loss, oce_loss + ocn_loss + tnews_loss)
            tnews_loss_weight = torch.div(tnews_loss, oce_loss + ocn_loss + tnews_loss)
            total_loss = oce_loss * oce_loss_weight \
                         + ocn_loss * ocn_loss_weight \
                         + tnews_loss * tnews_loss_weight
            total_loss.backward()
            print_loss = total_loss.item()

            optim.step()
            optim.zero_grad()

            # 统计训练结果
            oce_label = oce_label.tolist()
            ocn_label = ocn_label.tolist()
            tnews_label = tnews_label.tolist()
            oce_output = torch.nn.Softmax(dim=-1)(oce_output)
            ocn_output = torch.nn.Softmax(dim=-1)(ocn_output)
            tnews_output = torch.nn.Softmax(dim=-1)(tnews_output)
            oce_topk = torch.topk(oce_output, 1).indices.squeeze(0).tolist()
            ocn_topk = torch.topk(ocn_output, 1).indices.squeeze(0).tolist()
            tnews_topk = torch.topk(tnews_output, 1).indices.squeeze(0).tolist()
            if isinstance(oce_topk[0], list):
                oce_topk = [x[0] for x in oce_topk]
            if isinstance(ocn_topk[0], list):
                ocn_topk = [x[0] for x in ocn_topk]
            if isinstance(tnews_topk[0], list):
                tnews_topk = [x[0] for x in tnews_topk]

            oce_pred_list += oce_topk
            oce_label_list += oce_label
            ocn_pred_list += ocn_topk
            ocn_label_list += ocn_label
            tnews_pred_list += tnews_topk
            tnews_label_list += tnews_label

            # 调试验证部分
            # break

        print('EP_%d mask loss:%s' % (epoch, print_loss))

        oce_f1 = get_f1(oce_label_list, oce_pred_list)
        ocn_f1 = get_f1(ocn_label_list, ocn_pred_list)
        tnews_f1 = get_f1(tnews_label_list, tnews_pred_list)
        avg_f1 = (oce_f1 + ocn_f1 + tnews_f1) / 3
        print(epoch, 'oce f1 is:', oce_f1)
        print(epoch, 'ocn f1 is:', ocn_f1)
        print(epoch, 'tnews f1 is:', tnews_f1)
        print(epoch, 'average f1 is:', avg_f1)

        # save
        output_path = FinetunePath + '.ep%d' % epoch
        torch.save(bert.cpu(), output_path)
        bert.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # test
        bert.eval()
        with torch.no_grad():
            oce_correct = 0
            oce_total = 0
            ocn_correct = 0
            ocn_total = 0
            tnews_correct = 0
            tnews_total = 0
            oce_pred_list = []
            oce_label_list = []
            ocn_pred_list = []
            ocn_label_list = []
            tnews_pred_list = []
            tnews_label_list = []

            # oce验证部分
            for eval_data in oce_eval_set:
                oce_total += 1
                type_id = eval_data['type_id'].to(device)
                sentence = eval_data['sentence']
                input_token = eval_data['input_token_ids'].unsqueeze(0).to(device)
                segment_ids = eval_data['segment_ids'].unsqueeze(0).to(device)
                label = eval_data['token_ids_labels'].tolist()
                oce_output, _, _ = bert(
                    type_id,
                    input_token,
                    segment_ids,
                    1,
                    0,
                    0
                )
                oce_output = torch.nn.Softmax(dim=-1)(oce_output)
                oce_topk = torch.topk(oce_output, 1).indices.squeeze(0).tolist()[0]
                oce_pred_list.append(oce_topk)
                oce_label_list.append(label)
                # 累计数值
                if label == oce_topk:
                    oce_correct += 1
                # break
            acc_rate = float(oce_correct) / float(oce_total)
            acc_rate = round(acc_rate, 2)
            print('oce验证集正确率：%s' % acc_rate)

            # ocn验证部分
            for eval_data in ocn_eval_set:
                ocn_total += 1
                type_id = eval_data['type_id'].to(device)
                sentence = eval_data['sentence']
                input_token = eval_data['input_token_ids'].unsqueeze(0).to(device)
                segment_ids = eval_data['segment_ids'].unsqueeze(0).to(device)
                label = eval_data['token_ids_labels'].tolist() - 7
                _, ocn_output, _ = bert(
                    type_id,
                    input_token,
                    segment_ids,
                    0,
                    1,
                    0
                )
                ocn_output = torch.nn.Softmax(dim=-1)(ocn_output)
                ocn_topk = torch.topk(ocn_output, 1).indices.squeeze(0).tolist()[0]
                ocn_pred_list.append(ocn_topk)
                ocn_label_list.append(label)
                # 累计数值
                if label == ocn_topk:
                    ocn_correct += 1
                # break
            acc_rate = float(ocn_correct) / float(ocn_total)
            acc_rate = round(acc_rate, 2)
            print('ocn验证集正确率：%s' % acc_rate)

            # tnews验证部分
            for eval_data in tnews_eval_set:
                tnews_total += 1
                type_id = eval_data['type_id'].to(device)
                sentence = eval_data['sentence']
                input_token = eval_data['input_token_ids'].unsqueeze(0).to(device)
                segment_ids = eval_data['segment_ids'].unsqueeze(0).to(device)
                label = eval_data['token_ids_labels'].tolist() - 10
                _, _, tnews_output = bert(
                    type_id,
                    input_token,
                    segment_ids,
                    0,
                    0,
                    1
                )
                tnews_output = torch.nn.Softmax(dim=-1)(tnews_output)
                tnews_topk = torch.topk(tnews_output, 1).indices.squeeze(0).tolist()[0]
                tnews_pred_list.append(tnews_topk)
                tnews_label_list.append(label)
                # 累计数值
                if label == tnews_topk:
                    tnews_correct += 1
                # break
            acc_rate = float(tnews_correct) / float(tnews_total)
            acc_rate = round(acc_rate, 2)
            print('tnews验证集正确率：%s' % acc_rate)

            oce_f1 = get_f1(oce_label_list, oce_pred_list)
            ocn_f1 = get_f1(ocn_label_list, ocn_pred_list)
            tnews_f1 = get_f1(tnews_label_list, tnews_pred_list)
            avg_f1 = (oce_f1 + ocn_f1 + tnews_f1) / 3
            print(epoch, 'oce f1 is:', oce_f1)
            print(epoch, 'ocn f1 is:', ocn_f1)
            print(epoch, 'tnews f1 is:', tnews_f1)
            print(epoch, 'average f1 is:', avg_f1)
            print('\n')
