from sklearn.metrics import f1_score
from tqdm import tqdm
from time import sleep
from visdom import Visdom
import torch
import os
import pdb
import smtplib
from email.mime.text import MIMEText
import shutil
import time
import torch
import pickle
import json
import os
import shutil
import collections
import pdb
import time




def train(model, trainloader, validloader, testloader, lr, epoch, describe=None, only_save_trainable=True):
    valid_f1_score = -1
    
    if not os.path.exists("log"):
        os.mkdir("log")
        
    stamp = int(round(time.time() * 1000))
    log_path = './log/{}_{}_{}_{}'.format(stamp, describe, lr, epoch)
    log = open(log_path, "w")
    if describe == None:
        model_path = 'unnamed_model'
    else:
        model_path = "./model/{}_{}_{}_{}.pth".format(stamp, describe, lr, epoch)
        print('# ========================================')
        print('model = {} lr = {} epoch = {}'.format(describe, lr, epoch))
        print('# ========================================')
        log.write('model = {} lr = {} epoch = {}\n'.format(describe, lr, epoch))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    num_epochs = epoch
    train_losses = []
    interval = len(trainloader) if len(trainloader) < 6 else (len(trainloader)//6)
    valid_losses = []
    valid_f1 = []

    for epoch in range(num_epochs):
        labels = []
        predicts = []
        training_loss = 0.0
        training_acc = 0.0
        n_items = 0

        model.train()
        for i, (data, target) in enumerate(trainloader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(output, axis=1)
            training_loss += loss.item() * target.size(0)
            n_items += target.size(0)

            labels.extend(target.tolist())
            predicts.extend(pred.tolist())

        train_f1 = f1_score(labels, predicts)
        train_losses.append(training_loss/n_items)
        print('epoch [{}/{}] loss: {:.6f} train_f1_score: {:.6f}'.format(epoch + 1, num_epochs, 
                                                                training_loss / n_items, train_f1), end=" ")
        log.write('epoch [{}/{}] loss: {:.6f} train_f1_score: {:.6f} '.format(epoch + 1, num_epochs, 
                                                                training_loss / n_items, train_f1))
        labels = []
        predicts = []

        # model.load_state_dict(torch.load(model_path), strict=False)

        testing_loss = 0.0
        testing_acc = 0.0
        n_items = 0

        model.eval()
        with torch.no_grad():

            for i, (data, target) in enumerate(validloader):
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)

                pred = torch.argmax(output, axis=1)

                testing_loss += loss.item() * target.size(0)
                testing_acc += torch.sum(pred == target).item()
                n_items += target.size(0)

                labels.extend(target.tolist())
                predicts.extend(pred.tolist())


        valid_losses.append(testing_loss / n_items)

        f1 = f1_score(labels, predicts)
        print('valid loss: {:.6f}, valid_f1_score: {:.6f}'.format(testing_loss / n_items, f1), end=" ")
        log.write('valid loss: {:.6f}, valid_f1_score: {:.6f} '.format(testing_loss / n_items, f1))

        if valid_f1_score < f1:
            print('f1: {:.6f} -> {:.6f}'.format(valid_f1_score, f1))
            log.write('f1: {:.6f} -> {:.6f}\n'.format(valid_f1_score, f1))
            valid_f1_score = f1
            if only_save_trainable==True:
                state_dict = model.state_dict()
                trainable_layers = set([name.rsplit('.', 1)[0] for name, layer in model.named_parameters() if layer.requires_grad])
                state_dict = collections.OrderedDict([(name, parameter) for name, parameter in state_dict.items() if name.rsplit('.', 1)[0] in trainable_layers])

                torch.save(state_dict, model_path)
            else:
                torch.save(model.state_dict(), model_path)
        else:
            print("not saved")
            log.write("not saved\n")
            
        valid_f1.append(valid_f1_score)

    labels = []
    predicts = []

    model.load_state_dict(torch.load(model_path), strict=False)

    print('\ntesting...')
    log.write('\ntesting...\n')

    testing_loss = 0.0
    testing_acc = 0.0
    n_items = 0

    model.to(device)
    model.eval()
    with torch.no_grad():

        for i, (data, target) in enumerate(testloader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            pred = torch.argmax(output, axis=1)


            testing_loss += loss.item() * target.size(0)
            testing_acc += torch.sum(pred == target).item()
            n_items += target.size(0)

            labels.extend(target.tolist())
            predicts.extend(pred.tolist())

    f1 = f1_score(labels, predicts)
    print('f1_score on testset', f1)
    log.write('f1_score on testset: {:.6f}\n'.format(f1))
    log.close()
    os.rename(log_path, "{}_{:.6f}".format(log_path, f1))
    return f1


def validate(model, validloader):

    # 当GPU可用时，使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # 测试
    valid_loss = 0.0
    n_items = 0

    # 真实值和预测值
    y_true = []
    y_pred = []

    # 进度条
    pbar = tqdm(validloader)
    pbar.set_description('validation ')

    model.eval()
    with torch.no_grad():
        for data, target in pbar:
            data = data.to(device)
            target = target.to(device)

            # 计算输出和损失
            output = model(data)
            loss = criterion(output, target)

            # 记录真实值和预测值
            y_true.extend(target.tolist())
            y_pred.extend(torch.argmax(output, axis=1).tolist())

            # 计算损失和准确率
            valid_loss += loss.item() * target.size(0)
            n_items += target.size(0)

    # loss
    valid_loss /= n_items

    # f1-score
    f1 = f1_score(y_true, y_pred)

    return valid_loss, f1


def train_msml(model, trainloader, validloader, testloader, lr, epoch, describe=None, only_save_trainable=True):
    valid_f1_score = -1
    
    if not os.path.exists("log"):
        os.mkdir("log")
        
    stamp = int(round(time.time() * 1000))
    log_path = './log/{}_{}_{}_{}'.format(stamp, describe, lr, epoch)
    log = open(log_path, "w")
    if describe == None:
        model_path = 'unnamed_model'
    else:
        model_path = "./model/{}_{}_{}_{}.pth".format(stamp, describe, lr, epoch)
        print('# ========================================')
        print('model = {} lr = {} epoch = {}'.format(describe, lr, epoch))
        print('# ========================================')
        log.write('model = {} lr = {} epoch = {}\n'.format(describe, lr, epoch))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    num_epochs = epoch
    train_losses = []
    interval = len(trainloader) if len(trainloader) < 6 else (len(trainloader)//6)
    valid_losses = []
    valid_f1 = []

    for epoch in range(num_epochs):
        labels = []
        predicts = []
        training_loss = 0.0
        training_acc = 0.0
        n_items = 0

        model.train()
        for i, (data, target) in enumerate(trainloader):
            data = [data[i].to(device) for i in range(len(data))]
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(output, axis=1)
            training_loss += loss.item() * target.size(0)
            n_items += target.size(0)

            labels.extend(target.tolist())
            predicts.extend(pred.tolist())

        train_f1 = f1_score(labels, predicts)
        train_losses.append(training_loss/n_items)
        print('epoch [{}/{}] loss: {:.6f} train_f1_score: {:.6f}'.format(epoch + 1, num_epochs, 
                                                                training_loss / n_items, train_f1), end=" ")
        log.write('epoch [{}/{}] loss: {:.6f} train_f1_score: {:.6f} '.format(epoch + 1, num_epochs, 
                                                                training_loss / n_items, train_f1))
        labels = []
        predicts = []

        # model.load_state_dict(torch.load(model_path), strict=False)

        testing_loss = 0.0
        testing_acc = 0.0
        n_items = 0

        model.eval()
        with torch.no_grad():

            for i, (data, target) in enumerate(validloader):
                data = [data[i].to(device) for i in range(len(data))]
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)

                pred = torch.argmax(output, axis=1)

                testing_loss += loss.item() * target.size(0)
                testing_acc += torch.sum(pred == target).item()
                n_items += target.size(0)

                labels.extend(target.tolist())
                predicts.extend(pred.tolist())


        valid_losses.append(testing_loss / n_items)

        f1 = f1_score(labels, predicts)
        print('valid loss: {:.6f}, valid_f1_score: {:.6f}'.format(testing_loss / n_items, f1), end=" ")
        log.write('valid loss: {:.6f}, valid_f1_score: {:.6f} '.format(testing_loss / n_items, f1))

        if valid_f1_score < f1:
            print('f1: {:.6f} -> {:.6f}'.format(valid_f1_score, f1))
            log.write('f1: {:.6f} -> {:.6f}\n'.format(valid_f1_score, f1))
            valid_f1_score = f1
            if only_save_trainable==True:
                state_dict = model.state_dict()
                trainable_layers = set([name.rsplit('.', 1)[0] for name, layer in model.named_parameters() if layer.requires_grad])
                state_dict = collections.OrderedDict([(name, parameter) for name, parameter in state_dict.items() if name.rsplit('.', 1)[0] in trainable_layers])

                torch.save(state_dict, model_path)
            else:
                torch.save(model.state_dict(), model_path)
        else:
            print("not saved")
            log.write("not saved\n")
            
        valid_f1.append(valid_f1_score)

    labels = []
    predicts = []

    model.load_state_dict(torch.load(model_path), strict=False)

    print('\ntesting...')
    log.write('\ntesting...\n')

    testing_loss = 0.0
    testing_acc = 0.0
    n_items = 0

    model.to(device)
    model.eval()
    with torch.no_grad():

        for i, (data, target) in enumerate(testloader):
            data = [data[i].to(device) for i in range(len(data))]
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            pred = torch.argmax(output, axis=1)


            testing_loss += loss.item() * target.size(0)
            testing_acc += torch.sum(pred == target).item()
            n_items += target.size(0)

            labels.extend(target.tolist())
            predicts.extend(pred.tolist())

    f1 = f1_score(labels, predicts)
    print('f1_score on testset', f1)
    log.write('f1_score on testset: {:.6f}\n'.format(f1))
    log.close()
    os.rename(log_path, "{}_{:.6f}".format(log_path, f1))
    return f1




def validate_msml(model, validloader):

    # 当GPU可用时，使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # 测试
    valid_loss = 0.0
    n_items = 0

    # 真实值和预测值
    y_true = []
    y_pred = []

    # 进度条
    pbar = tqdm(validloader)
    model.eval()
    with torch.no_grad():
        for data, target in pbar:
            data = [data[i].to(device) for i in range(len(data))]
            target = target.to(device)

            # 计算输出和损失
            output = model(data)
            loss = criterion(output, target)

            # 记录真实值和预测值
            y_true.extend(target.tolist())
            y_pred.extend(torch.argmax(output, axis=1).tolist())

            # 计算损失和准确率
            valid_loss += loss.item() * target.size(0)
            n_items += target.size(0)

    # loss
    valid_loss /= n_items

    # f1-score
    f1 = f1_score(y_true, y_pred)

    return valid_loss, f1

