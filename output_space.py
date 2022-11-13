import torch
import torch.nn as nn
from models import *
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import copy
import argparse
import time
import sys, glob
import pickle
import torch.backends.cudnn as cudnn
from main import classes,inv_classes, device, test
from utils import progress_bar

from extract_test_objects_classwise import class_load, compute_sample_output
from generate_cctm import compute_class_accuracies

'''
This script generates the embedding features for all the training samples. 

'''

def compute_sample_outputs(net, criterion, class_loader, total_samples):
    '''class_loader is the training data for a class without transform
    Compute the softmax for each sample for the class objects
    todo: this is actually not the softmax; it is the output of the linear layer
    '''
    net.eval()
    train_loss = 0
    total = 0
    correct = 0
    lin_outputs = np.zeros((total_samples, len(classes)))
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

            lin_outputs[total_till_last_batch:total, :] = outputs.cpu().numpy()
            total_till_last_batch = total

    return lin_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    #todo: out of mem
    # parser.add_argument('--model', default='resnet50', type=str, help='model name')
    # parser.add_argument('--saved_dir', default='results/run1_save_model_every_epoch_resnet50', type=str,
    #                     help='directory to load the data')

    parser.add_argument('--model', default='vgg19', type=str, help='model name')
    parser.add_argument('--saved_dir', default='results/run1_save_model_every_epoch_vgg19_Rmsprop', type=str,
                        help='directory to load the data')

    parser.add_argument('--train_or_test', default='train', type=str, help='training data set or testing data set?')
    parser.add_argument('--all_models', default='no', type=str, help='all the models or only the final model')


    args = parser.parse_args()
    args.model = args.model.lower()
    args.train_or_test = args.train_or_test.lower()

    print('@@model=', args.model)
    print('@@train_or_test', args.train_or_test)

    if args.model == 'vgg19':
        net = VGG('VGG19')
    elif args.model == 'vgg11':
        net = VGG('VGG11')
    elif args.model == 'vgg16':
        net = VGG('VGG16')
    elif args.model == 'resnet18':
        net = ResNet18()
    elif args.model == 'resnet50':
        net = ResNet50()
    elif args.model == 'dla':
        net = SimpleDLA()
    else:
        print('not run yet')
        sys.exit(1)

    # get the features
    features = {}
    def get_features_linear(name):
        def hook(net, input, output):
            features[name] = output.detach()
        return hook

    def get_features_avg_pool(name):
        def hook(net, input, output):
            out = output.detach()
            print('name=', name)
            print('hook out.shape=', out.shape)
            out = F.avg_pool2d(out, 4)
            features[name] = out.view(out.size(0), -1) #from ResNet code
            print('hook featuers.shape=', features[name].shape)
        return hook

    if args.model == 'vgg19' or args.model == 'vgg11' or args.model == 'vgg16':
        net.features.register_forward_hook(get_features_linear('features'))
    elif args.model == 'resnet18' or args.model == 'resnet50':
        net.layer4.register_forward_hook(get_features_avg_pool('features'))
    elif args.model == 'dla':
        net.layer6.register_forward_hook(get_features_avg_pool('features'))

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    #model_path = find_model_file('../class_interference/results/', args.model, args.lr, args.lr_mode)

    if args.all_models == 'yes':
        model_list = glob.glob(args.saved_dir+'/*.pyc')
    else:
        model_list = glob.glob(args.saved_dir+'/*testacc*.pyc')

    print(model_list)

    print('how many models:', len(model_list))

    for model_path in model_list:
        print(model_path)

        print('loading model at path:', model_path)
        net.load_state_dict(torch.load(model_path))

        #saving the last layer's parameters
        if args.model == 'vgg19' or args.model == 'vgg11' or args.model == 'vgg16':
            W_f = net.module.classifier.weight.detach().cpu().numpy()
            bias_f = net.module.classifier.bias.detach().cpu().numpy()
            print(net.module.classifier.weight.shape)
        elif args.model == 'resnet18' or args.model == 'resnet50':#just different naming of the last layer from vgg19
            W_f = net.module.linear.weight.detach().cpu().numpy()
            bias_f = net.module.linear.bias.detach().cpu().numpy()
            print(net.module.linear.weight.shape)
        elif args.model == 'dla':
            W_f = net.module.linear.weight.detach().cpu().numpy()
            bias_f = net.module.linear.bias.detach().cpu().numpy()
            print(net.module.linear.weight.shape)

        np.save(model_path + '_W.npy', np.array(W_f))
        np.save(model_path + '_b.npy', np.array(bias_f))

        # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        criterion = nn.CrossEntropyLoss()  # by default. it's mean.

        feature_size = 512 #VGG19
        if args.train_or_test == 'train':
            num_samples_per_class = 5000
        else:
            num_samples_per_class = 1000

        sample_features = -np.ones((len(classes), num_samples_per_class, feature_size))
        sample_lin = -np.ones((len(classes), num_samples_per_class, len(classes)))

        #test overall train acc.
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
        print('training acc:')
        test(net, trainloader, criterion)

        print('test acc:')
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)#verified: batch size doesn't influence acc.
        test(net, testloader, criterion)

        class_test_acc = []
        cctm_list = []
        with torch.no_grad():
            for c in range(len(classes)):
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@  {}  @@@@@@@@@@@@@@@@@@@@@@@@@@@@'.format(classes[c].upper()))
                dataloader_c, num_samples_c = class_load(folder='/home/hsh/conditioning/class_interference/data/', label=c, batch_size=num_samples_per_class,
                                                         train=args.train_or_test=='train')
                # get the features for all samples, organized by classes
                compute_sample_outputs(net, criterion, dataloader_c, num_samples_c)
                print('features.size()=', features['features'].size())#features is a placeholder
                if args.model == 'vgg19' or args.model == 'vgg11' or args.model == 'vgg16':
                    sample_features[c] = features['features'][:, :, 0, 0].cpu().numpy() #destroy the last flat dimensions
                else:#resnet has no flat dimensions
                    sample_features[c] = features['features'].cpu().numpy()
                print('sample features of first sample:', sample_features[c, 0, :10])
                #compute the linear output
                sample_lin[c] = compute_sample_output(net, criterion, dataloader_c, num_samples_c)

                # also compute the test acc per class (class recall rate), and the cctm matrix per class
                if args.train_or_test == 'test':
                    test_acc_c = test(net, dataloader_c, criterion)
                    class_test_acc.append(test_acc_c)

                    cctm = compute_class_accuracies(net, dataloader=dataloader_c)
                    cctm_list.append(cctm)

        if args.train_or_test == 'train':
            np.save(model_path + '_features.npy', np.array(sample_features))
            np.save(model_path + '_outputs.npy', np.array(sample_lin))
        else:
            np.save(model_path + '_features_test.npy', np.array(sample_features))
            np.save(model_path + '_outputs_test.npy', np.array(sample_lin))
            np.save(model_path + '_class_recall_acc.npy', np.array(class_test_acc))
            np.save(model_path + '_class_cctm.npy', np.array(cctm_list))


#https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
#todo: check the feature extraction by hook is correct. Just run softmax over the product with the W and b and see if the predictions are correct. DONE. YES.











