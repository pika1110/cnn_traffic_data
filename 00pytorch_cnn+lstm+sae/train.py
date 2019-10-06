import os
import sys
import torch
import torch.autograd as autograd
from torch.optim import lr_scheduler
import torch.nn.functional as F
import logging
import numpy
import model
import time


def sub_class_acc(logit, target, size, n_class):
    classes = (0, 1, 2, 3, 4)
    c = []
    class_correct = list(0. for i in range(n_class))   #test 预测正确情况
    class_total = list(0. for i in range(n_class))   #test label情况
    class_precision_logit = list(0. for i in range(n_class))   #test 预测情况

    list_logit = torch.max(logit, 1)[1].view(target.size()).cpu().numpy().tolist()
    list_target = target.cpu().numpy().tolist()

    for i in range(size):
        if list_logit[i] == list_target[i]:
            c.append(1)
        else:
            c.append(0)
    for i in range(size):
        # recall相关参数
        target = list_target[i]
        class_correct[target] += c[i]
        class_total[target] += 1

        # precision 相关指标
        test_logit = list_logit[i]
        class_precision_logit[test_logit] += 1
    for i in range(n_class):
        recall = 0
        precision = 0
        f1_score = 0
        if class_total[i] != 0:
            if class_precision_logit[i] != 0:
                recall = float(100 * class_correct[i] / class_total[i])
                precision = float(100 * class_correct[i]/class_precision_logit[i])
                if recall + precision != 0:
                    f1_score = float(2 * recall * precision/(100*(recall + precision)))
                print('class%2s:  recall:%.4f%%     presision:%.4f%%     f1_score:%.4f ' %(classes[i], recall, precision, f1_score))
            else:
                print('class%2s:  recall:%.4f%%     presision:%.4f%%     f1_score:%.4f   ' % (classes[i], recall, precision, f1_score))
        else:
            print('class%5s 没有出现' % (classes[i]))
    print(class_correct)  # 测试集每个类别正确分类的数量
    print(class_total)   # 标签
    print(class_precision_logit)  # 测试集的各类别分类数量


def train(model, train_loader, batch_size, device, test_loader, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 需要调学习率
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 50 epochs 后lr=0.1*lr
    model.train()
    batch = 0
    lamda = 0.002
    #logging.info("----------------------------train  start--------------------------------")
    for epoch in range(100):
        running_loss = 0.0
        L1_regulation_loss = 0.0
        scheduler.step()
        # print(epoch+1, scheduler.get_lr()[0])  打印每一轮的学习率
        for i, data in enumerate(train_loader):
            inputs, target = data
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logit = model(inputs)
            #import pdb; pdb.set_trace()

            #for param in model.parameters():   # L1正则化项
               #L1_regulation_loss += torch.sum(torch.abs(param))
            loss = F.cross_entropy(logit, target) + lamda * L1_regulation_loss   #loss = F.cross_entropy(logit, target)+0.01*L1_regulation_loss  # 需要调损失函数
            L1_regulation_loss = 0
            loss.backward()
            optimizer.step()
            running_loss = running_loss+loss.item()

            if i % 2000 == 1999:
                batch = batch+1
                out_running_loss = running_loss / 2000   # 每2000次迭代，输出loss的平均值
                running_loss = 0
                corrects = (torch.max(logit, 1)[1].view(target.size()) == target).float().sum()  # torch.max(a,1)[1] 只返回最大值索引
                accuracy = 100 * corrects/batch_size
                sys.stdout.write(
                    '\repoch[{}] - Batch[{}]  train loss: {:.6f}  train acc: {:.4f}%({}/{})   \n'.format(
                        epoch+1,
                        2000 * batch,
                        out_running_loss,
                        accuracy,
                        int(corrects),
                        batch_size))

        test_corrects = 0
        test_size = 0
        i = 0

        logging.info("----------------------------test  start--------------------------------")
        with torch.no_grad():
            for test_data in test_loader:
                i += 1
                test_inputs, test_targets = test_data
                test_inputs = test_inputs.to(device)
                test_targets = test_targets.to(device)
                test_outputs = model(test_inputs)
                test_corrects += (torch.max(test_outputs, 1)[1].view(test_targets.size()) == test_targets).float().sum()

                if i == 1:
                    test_outputs_result = test_outputs
                    test_targets_result = test_targets
                else:
                    test_outputs_result = torch.cat((test_outputs_result, test_outputs), 0)
                    test_targets_result = torch.cat((test_targets_result, test_targets), 0)

                test_size += len(test_targets)
                test_acc = 100 * test_corrects / test_size

            #import pdb; pdb.set_trace()
            sys.stdout.write(
                '\repoch[{}]  test acc: {:.4f}%({}/{})   \n'.format(
                    epoch+1,
                    test_acc,
                    int(test_corrects),
                    test_size))
            sub_class_acc(test_outputs_result, test_targets_result, test_size, 5)  #输出每个类别的指标

    print('Finished Training')


