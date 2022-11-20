'''
Train CIFAR10 with PyTorch.

This supports different optimizers and a number of models.

'''
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

inv_classes = dict((cl, i) for i, cl in enumerate(classes))

def train(epoch, net, criterion, optimizer, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def train_and_save_epoch_model(epoch, net, criterion, optimizer, trainloader, save_at_epochs):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if epoch in save_at_epochs:
        acc = 100.*correct/total
        model_file_name = 'results/model_' + args.model + '_alpha_' + str(args.lr) \
                          + '_lrmode_' + args.lr_mode + 'batchsize_' + str(args.batchsize) + \
                          '_momentum_decayed_model_at_epoch_' + str(epoch) + "_acc_{:.2f}".format(acc, 2) + '.pyc'
        print('saving model at epoch ' + str(epoch) +' to file:', model_file_name)
        torch.save(net.state_dict(), model_file_name)

def train_and_save_every_epoch(epoch, net, criterion, optimizer, trainloader, saved_dir, optimizer_name):
    print('\nEpoch: %d' % epoch)

    #save initial model
    if epoch == 0:
        if optimizer_name == 'sgd':
            model_file_name = saved_dir + '/model_' + args.model + '_alpha_' + str(args.lr) \
                              + '_lrmode_' + args.lr_mode + 'batchsize_' + str(args.batchsize) + \
                              '_momentum_decayed_model_at_epoch_' + str(epoch) + "_acc_0.0.pyc" #just give a void 0 training err
        else:
            model_file_name = saved_dir + '/model_' + args.model + '_' + optimizer_name + '_batchsize_' + str(args.batchsize) + \
                              '_at_epoch_' + str(epoch) + "_acc_0.0.pyc" #just give a void 0 training err
        torch.save(net.state_dict(), model_file_name)

    net.train()
    train_loss = 0
    correct, total = 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100. * correct / total
    if optimizer == 'sgd':
        model_file_name = saved_dir + '/model_' + args.model + '_alpha_' + str(args.lr) \
                          + '_lrmode_' + args.lr_mode + 'batchsize_' + str(args.batchsize) + \
                          '_momentum_decayed_model_at_epoch_' + str(epoch) + "_acc_{:.2f}".format(acc, 2) + '.pyc'
    else:
        model_file_name = saved_dir + '/model_' + args.model + '_' + optimizer_name + '_batchsize_' + str(args.batchsize) + \
                          '_at_epoch_' + str(epoch) + "_acc_{:.2f}".format(acc, 2) + '.pyc'
    print('saving model at epoch ' + str(epoch) +' to file:', model_file_name)
    torch.save(net.state_dict(), model_file_name)


def test(net, testloader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total


def evaluate_f():
    '''evaluate the loss on the whole training dataset: no training. '''
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_big):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()
    return train_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--lr_mode', default='schedule', type=str, help='lr mode')

    parser.add_argument('--model', default='vgg19', type=str, help='model name')

    parser.add_argument('--optimizer', default='Rmsprop', type=str, help='optimizer such as sgd or adam')
    parser.add_argument('--saved_dir', default='results/run1_save_model_every_epoch_vgg19_Rmsprop', type=str, help='directory to save the data: make sure it is created before running')

    args = parser.parse_args()

    print('model=', args.model)
    print('@@lr=', args.lr)
    print('@@batchsize=', args.batchsize)
    args.lr_mode=args.lr_mode.lower()
    print('lr mode=', args.lr_mode)
    args.optimizer = args.optimizer.lower()
    print('optimizer=', args.optimizer)

    isExist = os.path.exists(args.saved_dir)
    if not isExist: # Create a new directory because it does not exist
        os.makedirs(args.saved_dir)
        print("The new directory {} is created!".format(args.saved_dir))

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epochf
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    trainloader_big = torch.utils.data.DataLoader(
        trainset, batch_size=1024, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    # Models
    print('==> Building model..')

    args.model = args.model.lower()
    print('running model:', args.model)

    if args.model=='vgg19':
        net = VGG('VGG19')
    elif args.model == 'vgg11':
        net = VGG('VGG11')
    elif args.model == 'vgg16':
        net = VGG('VGG16')
    elif args.model=='resnet18':
        net = ResNet18()
    elif args.model=='resnet50':
        net = ResNet50()
    elif args.model == 'googlenet':
        net = GoogLeNet()#so slow to train
    elif args.model == 'dla':
        net = SimpleDLA()
    else:
        print('not run yet')
        sys.exit(1)
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters())#default parameters
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(net.parameters())#default parameters
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters())
    elif args.optimizer == 'asgd':
        optimizer = optim.ASGD(net.parameters())
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters())
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(net.parameters())
    elif args.optimizer == 'rprop':
        optimizer = optim.Rprop(net.parameters())
    elif args.optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(net.parameters())
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(net.parameters())
    elif args.optimizer == 'lbfgs':
        #todo: cannot get working
        loc_param = pyro.param("auto_loc").unconstrained()
        optimizer = optim.LBFGS(net.parameters())
        print('LBFGS optimizer object generated. ')
    elif args.optimizer == 'nadam':
        optimizer = optim.NAdam(net.parameters())
    elif args.optimizer == 'radam':
        optimizer = optim.RAdam(net.parameters())#default parameters
    else:
        print('this optimizer name is not supported yet')
        sys.exit(1)

    if args.lr_mode == 'constant' or args.lr_mode == 'fixed':
        scheduler = None
    elif args.lr_mode=='schedule' or args.lr_mode=='anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        print('lr mode not supported this yet. ')

    for epoch in range(200):
        # train(epoch, net, criterion, optimizer, trainloader)

        train_and_save_every_epoch(epoch, net, criterion, optimizer, trainloader, saved_dir=args.saved_dir, optimizer_name=args.optimizer)

        # train(epoch, net, criterion, optimizer, trainloader)

        acc_test = test(net, testloader, criterion)

        if (args.lr_mode=='schedule' or args.lr_mode=='anneal') and args.optimizer == "sgd":
            scheduler.step()

    print('final test acc:', acc_test)

    if args.optimizer == 'sgd':
        model_file_name = args.saved_dir + '/model_' + args.model+ '_alpha_'+str(args.lr) \
                      + '_lrmode_'+ args.lr_mode +'batchsize_'+ str(args.batchsize) + \
                      '_momentum_decayed_testacc_' + "{:.2f}".format(acc_test, 2)  +'.pyc'
    else:
        model_file_name = args.saved_dir + '/model_' + args.model + args.optimizer + 'batchsize_' + str(args.batchsize) + \
                          '_momentum_decayed_testacc_' + "{:.2f}".format(acc_test, 2) + '.pyc'

    print(model_file_name)
    torch.save(net.state_dict(), model_file_name)
    # np.save(model_file_name + '_epoch_saved.npy', np.array(epoch_saved))

