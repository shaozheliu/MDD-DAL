from models.MyTransformer import ST_Transformer,Embedding_ST_Trans
# from EEGnet import EEGNET as EEGNet
from models.EEGModels import EEGNet, EEGNet_2b, EEGNet_500interval, DeepConvNet, DeepConvNet_2b, Shallow_Net, Shallow_Net_2b
from models.MMCNN import MMCNN_model
import sys, time, copy
import torch
import torch.nn as nn
import tqdm
from models.model import Mymodel
import os
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, roc_auc_score

head = 4
d_model = 750
d_ff = 512
ff_hide = 1024
mode1 = "T"
mode2 = "C"
n_layer = 10


def prepare_training(d_model, d_ff, head, n_layer, dropout=0.3, lr = 0.001, model_name='EEGNET', type = 'all', emb_feature=30):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model_name == "ST_Transformer":
        model = Embedding_ST_Trans(d_model, d_ff, head, n_layer, dropout).to(device)
        # model = ST_Transformer(d_model, d_ff, head, n_layer, 0.1).to(device)
    elif model_name == "EEGNET":
        model = EEGNet().to(device)
    elif model_name == "EEGNet_500interval":
        model = EEGNet_500interval().to(device)
    elif model_name == "Mymodel":
        model = Mymodel(d_model, d_ff, head, n_layer, dropout, type).to(device)
    elif model_name == "Mymodel_ablation":
        model = Mymodel_ablation(d_model, d_ff, head, n_layer, dropout, type).to(device)
    elif model_name == 'Mymodel_concat':
        model = Mymodel_concat(d_model, d_ff, head, n_layer, dropout, emb_feature, type).to(device)
    elif model_name == "DeepConvNet":
        model = DeepConvNet(22,dropout,4).to(device)
    elif model_name == 'Shallow_Net':
        model = Shallow_Net(22,dropout,4).to(device)
    elif model_name == 'MMCNN':
        model = MMCNN_model(22,1000,dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    print(device)
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain



def prepare_training(d_model, d_ff, head, n_layer, dropout=0.3, lr = 0.001, model_name='EEGNET', type = 'all', emb_feature=30, domain_num=2):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model_name == "ST_Transformer":
        model = Embedding_ST_Trans(d_model, d_ff, head, n_layer, dropout).to(device)
        # model = ST_Transformer(d_model, d_ff, head, n_layer, 0.1).to(device)
    elif model_name == "EEGNET":
        model = EEGNet_2b().to(device)
    elif model_name == "EEGNet_500interval":
        model = EEGNet_500interval().to(device)
    elif model_name == 'Mymodel':
        model = Mymodel(d_model, d_ff, head, n_layer, dropout, emb_feature, type, clf_class=2, domain_class=domain_num).to(device)
    elif model_name == "DeepConvNet":
        model = DeepConvNet_2b(3,dropout,2).to(device)
    elif model_name == 'Shallow_Net':
        model = Shallow_Net_2b(3,dropout,2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # criterion = nn.BCEWithLogitsLoss()  # 自带sigmoid
    criterion = nn.CrossEntropyLoss()   # 自带softmax
    criterion_domain = nn.CrossEntropyLoss()
    print(device)
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain



def train_model(model, criterion, optimizer, scheduler, device, dataloaders, n_epoch,
                args={'dataset_sizes': {'train': 1800, 'val': 200, 'test':304}}):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(n_epoch):
        sys.stdout.flush()
        print('Epoch {}/{}'.format(epoch+1, 300))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # for phase in ['train', 'val', 'test']:
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            test = dataloaders['train']
            print(1)
            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.type(torch.cuda.FloatTensor).to(device)
                labels = labels.type(torch.cuda.LongTensor).to(device).squeeze(1)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # _, label_idx = torch.max(labels,1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == label_idx)   # 这里源码有问题
                running_corrects += torch.sum(preds.data == labels.data)  # 这里源码有问题

            # if phase == 'train':
            #     scheduler.step(epoch=epoch)

            epoch_loss = running_loss / args['dataset_sizes'][phase]
            epoch_acc = running_corrects / args['dataset_sizes'][phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def train_model_with_domain(model, criterion, criterion_domain, optimizer, scheduler, device, dataloaders, n_epoch,
                            args={'dataset_sizes': {'train': 1800, 'val': 200, 'test':304}},
                            ):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in tqdm.tqdm(range(n_epoch)):
        sys.stdout.flush()
        print('Epoch {}/{}'.format(epoch+1, n_epoch))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            domain_running_loss = 0.0
            domain_running_corrects = 0
            print(phase)
            # Iterate over data.
            # for i, (inputs, labels, domain_labels) in enumerate(dataloaders[phase]):
            for i, (inputs, labels, domain_labels) in enumerate(dataloaders[phase]):
                p = float(i + epoch * len(dataloaders[phase])) / n_epoch / len(dataloaders[phase])
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                inputs = inputs.type(torch.cuda.FloatTensor).to(device)
                labels = labels.type(torch.cuda.LongTensor).to(device)
                domain_labels = domain_labels.type(torch.cuda.LongTensor).to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, domain_output = model(inputs,alpha)
                    _, preds = torch.max(torch.softmax(outputs,dim=1), 1)
                    _, domain_preds = torch.max(torch.softmax(domain_output,dim=1), 1)
                    # label predictor loss
                    loss = criterion(outputs, labels)
                    # domain classifier loss
                    loss_domain = criterion_domain(domain_output, domain_labels)
                    # compute total loss
                    total_loss = loss + loss_domain
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()
                        # scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == label_idx)   # 这里源码有问题
                running_corrects += torch.sum(preds.data == labels.data)  # 这里源码有问题

                # domain loss & corrects
                domain_running_loss += loss_domain.item() * inputs.size(0)
                domain_running_corrects += torch.sum(domain_preds.data == domain_labels.data)

            # if phase == 'train':
            #     scheduler.step(epoch=epoch)

            epoch_loss = running_loss / args['dataset_sizes'][phase]
            epoch_acc = running_corrects / args['dataset_sizes'][phase]
            domain_epoch_loss = domain_running_loss / args['dataset_sizes'][phase]
            domain_epoch_acc = domain_running_corrects / args['dataset_sizes'][phase]

            print('Predictor {} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Domain classifier {} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, domain_epoch_loss, domain_epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model_baseline(model, criterion, criterion_domain, optimizer, scheduler, device, dataloaders, n_epoch,
                         args={'dataset_sizes': {'train': 1800, 'val': 200, 'test':304}}):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in tqdm.tqdm(range(n_epoch)):
        sys.stdout.flush()
        print('Epoch {}/{}'.format(epoch+1, 300))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print(phase)
            # Iterate over data.
            # for i, (inputs, labels, domain_labels) in enumerate(dataloaders[phase]):
            for i, (inputs, labels, domain_labels) in enumerate(dataloaders[phase]):
                inputs = inputs.type(torch.cuda.FloatTensor).to(device)
                labels = labels.type(torch.cuda.LongTensor).to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(torch.softmax(outputs,dim=1), 1)
                    # label predictor loss
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == label_idx)   # 这里源码有问题
                running_corrects += torch.sum(preds.data == labels.data)  # 这里源码有问题

            # if phase == 'train':
            #     scheduler.step(epoch=epoch)

            epoch_loss = running_loss / args['dataset_sizes'][phase]
            epoch_acc = running_corrects / args['dataset_sizes'][phase]

            print('Predictor {} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model_baseline_2b(model, criterion, criterion_domain, optimizer, scheduler, device, dataloaders, n_epoch,
                         args={'dataset_sizes': {'train': 1800, 'val': 200, 'test':304}}):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in tqdm.tqdm(range(n_epoch)):
        sys.stdout.flush()
        print('Epoch {}/{}'.format(epoch+1, 300))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print(phase)
            # Iterate over data.
            # for i, (inputs, labels, domain_labels) in enumerate(dataloaders[phase]):
            for i, (inputs, labels, domain_labels) in enumerate(dataloaders[phase]):
                inputs = inputs.type(torch.cuda.FloatTensor).to(device)
                labels = labels.type(torch.cuda.LongTensor).to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.squeeze(dim=-1)
                    # _, preds = torch.max(torch.sigmoid(outputs,dim=1), 1)
                    preds = torch.sigmoid(outputs)
                    zeros = torch.zeros_like(preds,dtype=int)
                    ones = torch.ones_like(preds,dtype=int)
                    preds = torch.where(preds>0.5, ones, zeros)
                    # label predictor loss
                    loss = criterion(outputs, labels.float())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == label_idx)   # 这里源码有问题
                running_corrects += torch.sum(preds.data == labels.data)  # 这里源码有问题

            # if phase == 'train':
            #     scheduler.step(epoch=epoch)

            epoch_loss = running_loss / args['dataset_sizes'][phase]
            epoch_acc = running_corrects / args['dataset_sizes'][phase]

            print('Predictor {} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# 保存模型权重的函数
def save_best_model(model, filename='best_model.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Best model weights saved to {filename}")


def test_evaluate(model, device, X, y, model_name):
    inputs = torch.from_numpy(X).type(torch.cuda.FloatTensor).to(device)
    labels = torch.from_numpy(y).type(torch.cuda.LongTensor).to(device)
    if model_name == "Mymodel" or model_name == 'Mymodel_concat':
        outputs, domain_outputs = model(inputs,0.1)
    else:
        outputs = model(inputs)
    te = torch.softmax(outputs, dim=1)
    _, preds = torch.max(torch.softmax(outputs, dim=0), 1)
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    te = te.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    ka = cohen_kappa_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    roc_auc = roc_auc_score(labels, te, multi_class='ovr')
    return acc, ka, prec,recall, roc_auc

def test_evaluate_2b(model, device, X, y, model_name):
    inputs = torch.from_numpy(X).type(torch.cuda.FloatTensor).to(device)
    labels = torch.from_numpy(y).type(torch.cuda.FloatTensor).to(device)
    if model_name == "Mymodel" or model_name == 'Mymodel_concat':
        outputs, domain_outputs = model(inputs,0.1)
    else:
        outputs = model(inputs)
    te = torch.softmax(outputs, dim=1)
    _, preds = torch.max(torch.softmax(outputs, dim=0), 1)
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    te = te.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    ka = cohen_kappa_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    roc_auc = roc_auc_score(labels, preds)
    return acc, ka, prec,recall, roc_auc