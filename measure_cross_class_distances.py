import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

'''This can be applied to any trained model and do not need training. It just needs a training dataset to compute the measure'''

# from models import *

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_same_index(target, label):
    label_indices = []
    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)
    return label_indices


def get_train_cats(trainset, batch_size, label):
    index = get_same_index(trainset.targets, label=label)
    sampler = torch.utils.data.sampler.SubsetRandomSampler(index)
    trainloader = torch.utils.data.DataLoader(
        trainset, sampler=sampler, batch_size=batch_size, num_workers=2)
    return trainloader

def get_train_loader(batch_size):
    transform_train = transforms.Compose([
        # this has randomness even if shuffle is disabled.
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader


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


def compute_class_means_for_layer(layer, channels, dataloader, is_last_layer=True):
    '''
    This is for any layer as long as there is a shortcut definition like this:
    this layer should take the network input and outputs the this layer's output.

    todo:
    Is it easy to put this operation at the layer such as the forward method?
    It seems not. There will be different number of sample for each class in a mini-batch. So it's hard to vectorize.
    Anyhow, let's first do this for any layer, say outside of the layer just like this
    Note the layer can also be a relu layer, which is not pramatric.
    I need the channel size (64), call the layer to query the output.
    '''

    class_means = torch.zeros((len(classes), channels)).to(device)
    # zeros. some class may not be in the batch. the good thing is class_means is also zero.
    total = torch.ones(len(classes)).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = layer(inputs)
            # if is_last_layer:
            #     outputs = F.softmax(outputs, dim=1)
            for i in range(len(classes)):
                cls_ = (targets == i).nonzero()[:, 0]
                total[i] += len(cls_)
                if len(outputs.shape) == 4:
                    # use the mean of the channel as the feature for an input
                    x = outputs[cls_].mean(dim=(2, 3)).sum(dim=0)
                else:
                    x = outputs[cls_].sum(dim=0)
                class_means[i] += x

        # for i in range(len(classes)):#todo: check if the same
        #     class_means[i] /= total[i]
        class_means /= total[:, None]

        return class_means


def compute_class_means_for_layer_softmax(layer, channels, dataloader):
    '''
    with softmax
    '''

    class_means = torch.zeros((len(classes), channels)).to(device)
    # zeros. some class may not be in the batch. the good thing is class_means is also zero.
    total = torch.ones(len(classes)).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = layer(inputs)
            outputs = F.softmax(outputs, dim=1)
            for i in range(len(classes)):
                cls_ = (targets == i).nonzero()[:, 0]
                total[i] += len(cls_)
                if len(outputs.shape) == 4:
                    # use the mean of the channel as the feature for an input
                    x = outputs[cls_].mean(dim=(2, 3)).sum(dim=0)
                else:
                    x = outputs[cls_].sum(dim=0)
                class_means[i] += x

        # for i in range(len(classes)):#todo: check if the same
        #     class_means[i] /= total[i]
        class_means /= total[:, None]

        return class_means


def compute_class_means_for_layer_on_batch(layer, channels, inputs, targets, is_last_layer=True):
    '''
    This computes the means on a minibatch
    todo: can we parameterize this (remove the no_grad())?
    '''

    class_means = torch.zeros((len(classes), channels)).to(device)
    # zeros. some class may not be in the batch. the good thing is class_means is also zero.
    total = torch.ones(len(classes)).to(device)
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = layer(inputs)
        # if is_last_layer:
        #     outputs = F.softmax(outputs, dim=1)
        for i in range(len(classes)):
            cls_ = (targets == i).nonzero()[:, 0]
            total[i] += len(cls_)
            if len(outputs.shape) == 4:
                # use the mean of the channel as the feature for an input
                x = outputs[cls_].mean(dim=(2, 3)).sum(dim=0)
            else:
                x = outputs[cls_].sum(dim=0)
            class_means[i] += x

        # for i in range(len(classes)):#todo: check if the same
        #     class_means[i] /= total[i]
        class_means /= total[:, None]

        return class_means


def compute_class_means_for_layer_on_batch_cat_dog(layer, channels, inputs, targets, is_last_layer=True):
    '''
    This computes the means on a minibatch
    todo: can we parameterize this (remove the no_grad())?
    '''

    class_means = torch.zeros((2, channels)).to(device)
    # zeros. some class may not be in the batch. the good thing is class_means is also zero.
    total = torch.ones(2).to(device)

    inputs, targets = inputs.to(device), targets.to(device)
    outputs = layer(inputs)
    # if is_last_layer:
    #     outputs = F.softmax(outputs, dim=1)
    for i, cl in enumerate([3, 5]):
        cls_ = (targets == cl).nonzero()[:, 0]
        total[i] += len(cls_)
        if len(outputs.shape) == 4:
            # use the mean of the channel as the feature for an input
            x = outputs[cls_].mean(dim=(2, 3)).sum(dim=0)
        else:
            x = outputs[cls_].sum(dim=0)
        class_means[i] += x

    # for i in range(len(classes)):#todo: check if the same
    #     class_means[i] /= total[i]
    class_means /= total[:, None]

    return class_means


def wcd_for_layer(layer, cls_centers, dataloader, is_last_layer=True):
    '''
    within class distance: this is a centerness measure
    layer is a 4D (conv) or 2D (linear) shape
    todo: this can be speed up by pre-grouping samples acc. to classes
    '''
    with torch.no_grad():
        within_class_distances = torch.zeros(len(classes)).to(device)
        total = torch.zeros(len(classes)).to(device)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = layer(inputs)
            # if is_last_layer:
            #     outputs = F.softmax(outputs, dim=1)
            for i in range(len(classes)):
                cls_ = (targets == i).nonzero()[:, 0]
                total[i] += len(cls_)
                if len(outputs.shape) == 4:
                    x = outputs[cls_].mean(dim=(2, 3))
                else:
                    assert(len(outputs.shape) == 2)
                    x = outputs[cls_]
                # see k-means definition
                sum_ = ((x - cls_centers[i])**2).sum(dim=1).sum(dim=0)
                within_class_distances[i] += sum_
        # print('total=', total)
        return (within_class_distances/total).cpu().numpy()


def ccd_for_layer(layer, cls_centers, dataloader, is_last_layer=True):
    '''
    cross class distances: compute cross centerness to all the classes
    layer is a 4D (conv) or 2D (linear) shape

    The diagonal part of ccd is just the wcd
    '''
    with torch.no_grad():
        ccd = torch.zeros(len(classes), len(classes)).to(device)
        total = torch.zeros(len(classes)).to(device)
        # print('len(dataloader)=', len(dataloader))
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = layer(inputs)
            # if is_last_layer:
            #     outputs = F.softmax(outputs, dim=1)

            for i in range(len(classes)):
                cls_ = (targets == i).nonzero()[:, 0]
                total[i] += len(cls_)
                #x: samples for class i
                if len(outputs.shape) == 4:
                    x = outputs[cls_].mean(dim=(2, 3))
                else:
                    assert(len(outputs.shape) == 2)
                    x = outputs[cls_]
                for j in range(len(classes)):
                    sum_ = ((x - cls_centers[j])**2).sum(dim=1).sum(dim=0)
                    ccd[i, j] += sum_
                # print('x.shape=', x.shape)
                # print('cls_centers.shape=', cls_centers.shape)
                # sum_ = ((x - cls_centers) ** 2).sum(dim=1).sum(dim=0)
                # print('sum_.shape=', sum_.shape)
                # sys.exit(1)
        # print('total=', total)
        return (ccd/total).cpu().numpy()


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


def measure_net(net_name_string, layer_1st, layer_last, channels_1st, channels_last, mode):
    if mode == 'train':
        dataloader = get_train_loader(batch_size=1000)
        print('Training Mode')
    else:
        dataloader = get_train_loader(batch_size=1000)
        print('Testing Mode')

    # class_means_1st = compute_class_means_for_layer(layer_1st, channels=channels_1st)
    # wcd_1st = wcd_for_layer(layer_1st, class_means_1st)
    # print(net_name_string+'-wcd 1st:', wcd_1st)
    # file_wcd_1st = './original_models/' + net_name_string + '_wcd_1st.npy'
    # np.save(file_wcd_1st, wcd_1st)

    class_means_last = compute_class_means_for_layer(
        layer=layer_last, channels=channels_last, dataloader=dataloader)
    print('class_means_last[:3]', class_means_last[:3])
    print('class_means_last.shape=', class_means_last.shape)
    file_center_last = './original_models/' + \
        net_name_string + '_centers_last_' + mode + '.npy'
    np.save(file_center_last, class_means_last.cpu().numpy())

    wcd_last = wcd_for_layer(
        layer_last, cls_centers=class_means_last, dataloader=dataloader)
    print(net_name_string+'-wcd last', wcd_last)
    file_wcd_last = './original_models/' + \
        net_name_string + '_wcd_last_' + mode + '.npy'
    np.save(file_wcd_last, wcd_last)

    ccd_last = ccd_for_layer(
        layer_last, cls_centers=class_means_last, dataloader=dataloader)
    print(net_name_string + '-ccd last', ccd_last)
    print('diag(ccd_last) should match wcd ', np.diag(ccd_last))  # YES!
    file_ccd_last = './original_models/' + \
        net_name_string + '_ccd_last_' + mode + '.npy'
    np.save(file_ccd_last, ccd_last)

    if mode == 'test':
        cmp = compute_class_accuracies(net)
        print(net_name_string+'-class accuracies:', cmp)
        file_acc = './original_models/'+net_name_string+'_class_cmp.npy'
        np.save(file_acc, cmp)


if __name__ == '__main__':
    net = VGG('VGG11').to(device)
    net = load_model(net, 'vgg11')
    net.eval()
    measure_net('vgg11', net.features[0], net, 64, 10, mode='train')
    measure_net('vgg11', net.features[0], net, 64, 10, mode='test')
