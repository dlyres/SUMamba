import torch
import torch.nn as nn
import numpy as np
import sys
import time
from torch import Tensor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def train_one_epoch(model, optimizer, data_loader, device, epoch, in_c, dataset_name):
    model.train()
    model.to(device)
    loss_cross = nn.CrossEntropyLoss()
    total_train_loss = 0
    total_train_accuracy = 0
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        datas, targets = data

        sample_num += datas.shape[0]
        output = model(datas.to(device))
        accuracy = (output.argmax(1) == targets.to(device)).sum()
        total_train_accuracy = total_train_accuracy + accuracy
        train_loss = loss_cross(output, targets.to(device))
        train_loss.backward()
        total_train_loss += train_loss
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.4f}".format(epoch + 1,
                                                                               total_train_loss,
                                                                               total_train_accuracy.item() / sample_num)
        optimizer.step()
        optimizer.zero_grad()

    return total_train_loss


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, in_c, dataset_name, num_classes):
    model.eval()
    model.to(device)
    loss_cross = nn.CrossEntropyLoss()
    total_test_accuracy = 0
    total_test__loss = 0
    sample_num = 0
    total_acc = 0  # 总体预测正确率
    class_correct = list(0. for _ in range(num_classes))  # 单个分类正确的个数
    class_total = list(10E-8 for _ in range(num_classes))  # 单个分类预测的样本总数
    data_loader = tqdm(data_loader, file=sys.stdout)
    targets_roc = []
    predictions_roc = []

    itr_list = []

    for step, data in enumerate(data_loader):
        datas, targets = data

        sample_num += datas.shape[0]
        start_time = time.time()
        output = model(datas.to(device))
        end_time = time.time() - start_time
        itr_list.append(end_time)

        test_loss = loss_cross(output, targets.to(device))
        total_test_loss += test_loss
        accuracy = (output.argmax(1) == targets.to(device)).sum()
        total_test_accuracy = total_test_accuracy + accuracy
        total_acc = total_test_accuracy.item() / sample_num
        data_loader.desc = "[test  epoch {}] loss: {:.3f}, acc: {:.4f}".format(epoch + 1,
                                                                               total_test_loss,
                                                                               total_acc)

        for index, i in enumerate(output.argmax(1)):
            class_total[i] += 1
            if i == targets[index]:
                class_correct[i] += 1

        for target, prediction in zip(targets, output):
            targets_roc.append(Tensor.cpu(target))
            predictions_roc.append(Tensor.cpu(prediction))

    class_acc = np.array(class_correct) / np.array(class_total)  # 统计每一轮各个分类准确率

    # 计算auc
    if dataset_name == 'JFPM':
        targets_roc = label_binarize(targets_roc, classes=[i for i in range(12)])
    else:
        targets_roc = label_binarize(targets_roc, classes=[i for i in range(40)])
    micro_auc = round(roc_auc_score(targets_roc, predictions_roc, average='micro'), 4)

    return total_test_loss, class_acc, total_acc, micro_auc, itr_list

