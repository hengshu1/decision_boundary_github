import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

'''This can be applied to any trained model and do not need training. It just needs a training dataset to compute the measure'''

from models import *

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_test_loader(batch_size):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader


def compute_class_accuracies(net, dataloader=get_test_loader(batch_size=1000)):
    '''compute the accu for each class
    compute the Class Matrix Prediction (CMP): (i,j): the times class i is predicted as class j
    todo this can be speed up by first grouping the samples according to classes
    '''
    net.eval()
    print('compute_class_accuracies: len(dataloader) =', len(dataloader))
    # criterion = nn.CrossEntropyLoss()#loss used: no change.
    cmp = np.zeros((len(classes), len(classes)))
    total = np.zeros(len(classes))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            for i in range(len(classes)):
                cls_ = (targets == i).nonzero()[:, 0]
                total[i] += len(cls_)
                # loss = criterion(outputs[cls_], targets[cls_])

                # test_loss += loss.item()
                _, predicted = outputs[cls_].max(1)
                for j in range(len(classes)):
                    cmp[i, j] += predicted.eq(j).sum()
        return cmp

def load_model(net, net_name_string):
    path = './original_models/' + net_name_string.lower() + '.pth'
    net.load_state_dict(torch.load(path))
    return net

def measure_net(net_name_string):
    cmp = compute_class_accuracies(net)
    print(net_name_string+'-class accuracies:', cmp)
    file_acc = './original_models/'+net_name_string+'_class_cmp.npy'
    np.save(file_acc, cmp)

if __name__ == '__main__':
    net = VGG('VGG19').to(device)
    net = load_model(net, 'vgg19')
    net.eval()
    measure_net('vgg19')
