'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
from models import *
import torch.optim as optim
import numpy as np
import copy
import time, sys, glob
import torch.backends.cudnn as cudnn

from torchvision.datasets import MNIST, CIFAR10

import random
import pickle

import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from utils import progress_bar
from model_gradient import concat_param_grad, aver_grad_1D, aver_grad_net
from main import classes, device


def save_objects_of_class(data_loader, label, train):
    print(data_loader)
    total = 0
    total_objects_cls = 0
    _X = None
    _Y = None
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        index = (targets == label).nonzero()[:, 0]
        inputs_cls = inputs[index, :, :]
        targets_cls = targets[index]
        total_objects_cls += targets_cls.size(0)
        total += 1

        if _X is None:
            _X = inputs_cls
            _Y = targets_cls
        else:
            _X = torch.cat((_X, inputs_cls), dim=0)
            _Y = torch.cat((_Y, targets_cls), dim=0)

    print('total' + classes[label] + ' is ', total_objects_cls)
    print('_X.shape=', _X.shape)
    print('_Y.shape=', _Y.shape)

    if train:
        torch.save(_X, '../class_interference/data/cifar-10/class_'+str(label)+'_X.pt')
        torch.save(_Y, '../class_interference/data/cifar-10/class_' + str(label) + '_Y.pt')
    else:
        torch.save(_X, '../class_interference/data/cifar-10/class_'+str(label)+'_X_test.pt')
        torch.save(_Y, '../class_interference/data/cifar-10/class_' + str(label) + '_Y_test.pt')

def compute_sample_output(net, criterion, class_loader, total_samples):
    '''class_loader is the training data for a class without transform
    Compute the softmax for each sample for the class objects
    todo: this is actually not the softmax; it is the output of the linear layer
    '''
    net.eval()
    train_loss = 0
    total = 0
    correct = 0
    softmax_outputs = np.zeros((total_samples, len(classes)))
    total_till_last_batch = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(class_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(class_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            softmax_outputs[total_till_last_batch:total, :] = outputs.cpu().numpy()
            total_till_last_batch = total

    return softmax_outputs

def save_objects_all_classes(data_loader, train_or_not=True):
    for cl in range(len(classes)):
        save_objects_of_class(data_loader, label=cl, train=train_or_not)

def dataset_XY(label, folder='data/', train=True):
    '''this is the samples from training set because save_objects_all_classes is used with trainloader'''
    # print('folder=', folder)

    if train:
        print('loading Training data for Class {}.'.format(classes[label].upper()))
        _X = torch.load(folder + 'cifar-10/class_'+str(label)+'_X.pt')
        _Y = torch.load(folder + 'cifar-10/class_'+str(label)+'_Y.pt')
    else:
        print('loading Testing data for Class {}.'.format(classes[label].upper()))
        _X = torch.load(folder + 'cifar-10/class_'+str(label)+'_X_test.pt')
        _Y = torch.load(folder + 'cifar-10/class_'+str(label)+'_Y_test.pt')
    print('_X.shape=', _X.shape)
    print('_Y.shape=', _Y.shape)
    return _X, _Y

def dataset_class(label, folder='data/', train=True):
    _X, _Y = dataset_XY(label, folder=folder, train=train)
    dataset = TensorDataset(_X, _Y)
    return dataset

def class_load(label, folder='data/', batch_size=1000, train=True):
    dataset = dataset_class(label, folder=folder, train=train)
    #don't shuffle
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2), len(dataset)


def find_model_file(path, model, lr, lr_mode):
    '''path needs / in the end'''
    '''using the first model starting with this header pattern'''
    pattern = path+'model_' + model + '_alpha_' + str(lr) + '_lrmode_' + lr_mode + '_momentum_decayed_testacc_??.??.pyc'
    print('pattern=', pattern)
    for filename in glob.glob(pattern):
        return filename
    return None

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(1)

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--model', default='VGG19', type=str, help='model name')
    parser.add_argument('--lr_mode', default='constant', type=str, help='lr mode')
    parser.add_argument('--output', default='gradient', type=str, help='compute all the class gradients')
    args = parser.parse_args()

    args.model = args.model.lower()
    args.lr_mode = args.lr_mode.lower()

    print('@@model=', args.model)
    print('@@lr=', args.lr)
    print('@@lr_mode=', args.lr_mode)
    if args.output == 'gradient':
        print('@@Computing the gradient of all the class gradients')
    elif args.output == 'softmax':
        print('@@Computing the softmax output for each sample')
    else:
        print('unknown mode for args.output=', args.output)
        sys.exit(1)

    if args.model == 'vgg19':
        net = VGG('VGG19')
    elif args.model == 'resnet18':
        net = ResNet18()
    else:
        print('not run yet')
        sys.exit(1)

    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True
        # Data
        print('==> Preparing data..')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        # trainset = torchvision.datasets.CIFAR10(
        #     root='./data', train=True, download=True, transform=transform_train)
        #It appears data transform is applied in dataloader, not in the CIFAR: transform has no effect here yet
        # trainset = CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        #However, on MNIST, the data transform is effective in the dataset already.
        # trainset = MNIST('./data', transform=transforms.ToTensor(), download=True)

        #so here I used a solution that first retrieve from the dataloader, save, and then load; this guarantees using the same transformed data as trainloader

        #this is just one time running.
        # trainset = CIFAR10(root='./data', train=True, transform=transform_test, download=True)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=False, num_workers=2)
        # save_objects_all_classes(trainloader)
        #testing
        testset = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4096, shuffle=False, num_workers=2)
        save_objects_all_classes(testloader, train_or_not=False)
        sys.exit(1)

        # train_cls = dataset_class(label=3)
        # train_cls_loader = torch.utils.data.DataLoader(train_cls, batch_size=1000, shuffle=False, num_workers=2)
        # for batch_idx, (inputs, targets) in enumerate(train_cls_loader):
        #     print('targets=', targets)
        #     print('inputs.shape=', inputs.shape)

        net = torch.nn.DataParallel(net)
        model_path = find_model_file('results/', args.model, args.lr, args.lr_mode)

        print('loading model at path:', model_path)
        net.load_state_dict(torch.load(model_path))
        # sys.exit(1)
        # print(net)

        criterion = nn.CrossEntropyLoss(reduction='sum')  # by default. it's mean.

        # todo check the momentum and weight does not matter because we only use the forward prediction and gradient zero; no step()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        #This computes the gradient for each class and saves it
        # grad_cls_norm2 = np.zeros(len(classes))

        for cl in range(len(classes)):
            print('class: ', classes[cl])
            trainloader_cls, num_samples_cls = train_loader_class(label=cl)
            print('total class samples:', num_samples_cls)

            # saving gradient as a nd array
            # grad = aver_grad_1D(trainloader_cls, net, optimizer, criterion)
            # print('grads.shape=', grad.shape)
            # np.save(model_path+'_grad_'+classes[cl]+'.npy', grad.cpu().numpy())
            # grad_cls_norm2[cl] = torch.norm(grad)

            if args.output == 'gradient':
                #saving gradient as a dictionary
                grad_net = aver_grad_net(trainloader_cls, net, optimizer, criterion)
                f = open(model_path+'_grad_'+classes[cl]+'.pkl', "wb")
                pickle.dump(grad_net, f)
                f.close()
            elif args.output == 'softmax':
                softmax_cls = compute_sample_output(net, trainloader_cls, num_samples_cls)
                f = open(model_path + '_softmax_' + classes[cl] + '.pkl', "wb")
                pickle.dump(softmax_cls, f)
                f.close()
            else:
                print('unknown mode for args.output=', args.output)
                sys.exit(1)

        # np.save(model_path+'_grad_norm2_classes.npy', grad_cls_norm2)
