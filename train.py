from dataloader import *
from sklearn.metrics import *
import time
import torch
import pickle
import json
import os
import shutil
import collections
import pdb


def train(model, trainloader, validloader, testloader, describe=None, only_save_trainable=True):


    valid_f1_score = -1


    if describe == None:
        dst_path = 'unnamed_model'
    else:
        dst_path = describe
        print('# ========================================')
        print('# ' + describe)
        print('# ========================================')
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
            #time.sleep(4)
        os.mkdir(dst_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    lr = 0.005
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    num_epochs = 20
    train_losses = []

    interval = len(trainloader) if len(trainloader) < 20 else (len(trainloader)//10)

    print('training...')
    for epoch in range(num_epochs):
        labels = []
        predicts = []
        training_loss = 0.0
        training_acc = 0.0
        n_items = 0

        model.train()
        print('\n')
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
            training_acc += torch.sum(pred == target).item()
            n_items += target.size(0)

            labels.extend(target.tolist())
            predicts.extend(pred.tolist())

            if interval==len(trainloader) or i%interval==0:
                print('epoch [{}/{}] {}/{} loss: {} acc: {}'.format(epoch + 1, num_epochs, i + 1, len(trainloader),
                                                                training_loss / n_items, training_acc / n_items))

        print('f1_score on trainset', f1_score(labels, predicts))
        print('\n')

        train_losses.append(training_loss/n_items)

        labels = []
        predicts = []

        # model.load_state_dict(torch.load(model_path), strict=False)

        print('validating...')

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

                if interval==len(validloader) or i%interval==0:
                    print('epoch [{}/{}] {}/{} loss: {} acc: {}'.format(epoch + 1, num_epochs, i + 1, len(validloader),
                                                                testing_loss / n_items, testing_acc / n_items))

        f1 = f1_score(labels, predicts)
        print('f1_score on validset', f1)

        model_path = '{}/{}.pth'.format(dst_path, dst_path)

        if valid_f1_score < f1:
            print('f1 score increased from {} to {}, save model...'.format(valid_f1_score, f1))
            valid_f1_score = f1
            if only_save_trainable==True:
                state_dict = model.state_dict()
                trainable_layers = set([name.rsplit('.', 1)[0] for name, layer in model.named_parameters() if layer.requires_grad])
                state_dict = collections.OrderedDict([(name, parameter) for name, parameter in state_dict.items() if name.rsplit('.', 1)[0] in trainable_layers])

                torch.save(state_dict, model_path)
            else:
                torch.save(model.state_dict(), model_path)
        else:
            print('f1 score decreased from {} to {}, without saving model...'.format(valid_f1_score, f1))


    labels = []
    predicts = []

    model.load_state_dict(torch.load(model_path), strict=False)

    print('\ntesting...')

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


            if interval==len(testloader) or i%interval==0:
                print('epoch [{}/{}] {}/{} loss: {} acc: {}'.format(epoch + 1, num_epochs, i + 1, len(testloader),
                                                            testing_loss / n_items, testing_acc / n_items))


    f1 = f1_score(labels, predicts)
    print('f1_score on testset', f1)

    return f1


def extract_feature(model, dataloader, dst='dataset'):

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    index = 0
    print('extracting features...')
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(dataloader):
            data = data.to(device)
            features = model(data).cpu().numpy()
            # features = [f.unsqueeze(1) for f in features]
            # features = torch.cat(features, dim=1)  # N*d

            for i in range(len(target)):
                with open('{}/{}'.format(dst, target[i]), 'wb') as f:
                    pickle.dump(features[i], f)
                index = index + 1
                print(index, target[i], features[i].shape)

    print('extracting features finished')


def test(model, weight_path, testloader):


    valid_f1_score = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.CrossEntropyLoss().to(device)
    model = model.to(device)

    interval = len(testloader) if len(testloader) < 20 else (len(testloader)//10)


    labels = []
    predicts = []

    model.load_state_dict(torch.load(weight_path), strict=False)

    print('\ntesting...')

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


            if interval==len(testloader) or i%interval==0:
                print('{}/{} loss: {} acc: {}'.format(i + 1, len(testloader),
                                                            testing_loss / n_items, testing_acc / n_items))


    f1 = f1_score(labels, predicts)
    print('f1_score on testset', f1)
    return f1

