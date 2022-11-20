import argparse, sys
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from models import *
from utils import progress_bar
from main import device, classes, inv_classes
from cifar10_with_id import CIFAR10WithID

'''
study the final model's low confidence samples. 

Given a final model, study the low confidence samples. See if they are the same 

(1) across different runs. 

(2) change model; do they change? 

(3) change optimizer do they change?

'''

def test(net, testloader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, ids) in enumerate(testloader):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='vgg19', type=str, help='model name')
    parser.add_argument('--saved_dir', default='results/run2_save_model_every_epoch_vgg19/', type=str, help='saved directory of the final model')

    args = parser.parse_args()

    print('model=', args.model)
    print('saved_dir=', args.saved_dir)

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

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    #just use the training set to study samples; not for training perpose
    trainset = CIFAR10WithID(root='~/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    test(net, trainloader, criterion)



