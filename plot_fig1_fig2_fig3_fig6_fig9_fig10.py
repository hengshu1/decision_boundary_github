import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from main import classes
import sys, time
from utils import progress_bar
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

header = 'results/'

def check_features_are_correct(output_file, feature_file, Wfile, bfile):
    outs = np.load(header + output_file)
    print(outs.shape)

    features = np.load(header + feature_file)
    W = np.load(header + Wfile)
    b = np.load(header + bfile)
    outs2 = np.dot(features, W.T) + b
    print('norm of out - out2=', np.linalg.norm(outs -outs2))

    f_softmax = torch.nn.Softmax(dim=2)
    softmax1 = f_softmax(torch.as_tensor(outs)).cpu().numpy()
    softmax2 = f_softmax(torch.as_tensor(outs2)).cpu().numpy()
    print('softmax1.shape=', softmax1.shape)
    print('softmax2.shape=', softmax2.shape)

    plt.plot(np.reshape(softmax2, (softmax2.shape[0]*softmax2.shape[1], softmax2.shape[2])), '--ko')
    plt.plot(np.reshape(softmax1, (softmax1.shape[0]*softmax1.shape[1], softmax1.shape[2])), '-b+')

    plt.figure()
    for i in range(len(classes)):
        plt.plot(softmax1[i, :, i], label=classes[i].upper())#should be all close to 1: YES.
    plt.legend()

    plt.figure()
    for i in range(len(classes)):
        plt.plot(softmax2[i, :, i], label=classes[i].upper())#should be all close to 1: YES.
    plt.legend()

    plt.show()

    print(features.shape)
    print(W.shape)
    print(b.shape)

#seems exactly matching except small numerical errors
# check_features_are_correct('model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_outputs.npy',
#                            'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_features.npy',
#                            'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_W.npy',
#                            'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_b.npy',
#                            )
# check_features_are_correct('model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.18.pyc_outputs.npy',
#                            'model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.18.pyc_features.npy',
#                            'model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.18.pyc_W.npy',
#                            'model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.18.pyc_b.npy',
#                            )
# sys.exit(1)

def plot_prob_output(train_accuracies, head_path, final_file_header, c1=3, c2=5):
    '''train_accuracy is string of float with .2 accuracy'''
    f_softmax = torch.nn.Softmax(dim=2)
    # fig, ax = plt.subplots()
    for i, acc in enumerate(train_accuracies):
        if acc == 'final':
            all_outs = np.load(head_path + final_file_header + 'outputs.npy')
            # all_outs = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.80.pyc_outputs.npy')
            textstr = acc + ' (100%) model'
        else:
            all_outs = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile'+acc+'.pyc_outputs.npy')
            textstr = acc + '% model'

        print(all_outs.shape)

        softmax = f_softmax(torch.as_tensor(all_outs)).cpu().numpy()
        ax = plt.subplot(2, 3, i+1)
        plt.scatter(softmax[c1, :, c1], softmax[c1, :, c2], color='b', marker ='o', alpha=0.6, label='{}s'.format(classes[c1]))
        plt.scatter(softmax[c2, :, c1], softmax[c2, :, c2], color = 'r', marker = '+', alpha=0.6, label='{}s'.format(classes[c2]))
        plt.xlabel('Prob({})'.format(classes[c1].upper()))
        plt.ylabel('Prob({})'.format(classes[c2].upper()))
        plt.legend()

        # place a text box in upper left in axes coords

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.45, 0.70, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    # plt.savefig('decision_boundary_vanishing_scatter.png')


#figure 9
header = 'results/run2/'
train_accuracies = reversed(['85.16', '90.62', '99.48', '99.87', '99.97', 'final'])
final_file_header = 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.89.pyc_'
plot_prob_output(train_accuracies, head_path=header, final_file_header=final_file_header, c1=3, c2=5)#cats and dogs
# plot_prob_output(accuracies, c1=1, c2=9)#cars and trucks
# plot_prob_output# plot_percentile_model(accuracies, c1=3, c2=6)#cats and frogs
plt.show()
sys.exit(1)

#below study features instead of the decision space above

def get_grids_in_pca_space(features_PCA, pca, f1, f2, grids=200):
    print('f1=', f2, 'f2=', f2)
    data_selected_1, data_selected_2 = features_PCA[:, f1][:, None], features_PCA[:, f2][:, None]
    data_selected = np.concatenate((data_selected_1, data_selected_2), axis=1)
    print('data_selected.shape=', data_selected.shape)

    # lows = np.min(data_selected, axis=0) - 1.
    # highs = np.max(data_selected, axis=0) + 1.
    # print('lows=', lows)
    # print('highs=', highs)

    # use same bounds across plots
    lows = [-7, -4]
    highs = [7., 6.]

    x_1 = np.linspace(lows[0], highs[0], grids)
    x_2 = np.linspace(lows[1], highs[1], grids)

    # generate a synthetic feature sample matrix
    Phi = np.zeros((len(x_1) * len(x_2), 2))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            Phi[n, 0], Phi[n, 1] = x_1[i], x_2[j]
            n += 1
    print(Phi.shape)

    #Phi_back = pca.inverse_transform(Phi)
    comp_selected = select_pca_components(pca, f1, f2)  # 2, 512
    Phi_back = Phi.dot(comp_selected) + pca.mean_
    print('Phi_back.shape=', Phi_back.shape)
    return Phi, Phi_back, x_1, x_2

def get_softmax_output(Phi_back, W, b, c1, c2):
    '''
    Given a 512-d feature vector, get the softmax output according to the networks
    '''
    outputs = np.dot(Phi_back, W.T) + b
    print('outputs.shape=', outputs.shape)

    #use binary classification: because the other classes are not involved.
    outputs_c1 = outputs[:, c1][:, None]
    outputs_c2 = outputs[:, c2][:, None]
    outputs = np.concatenate((outputs_c1, outputs_c2), axis=1)
    f_softmax = torch.nn.Softmax(dim=1)
    softmax = f_softmax(torch.as_tensor(outputs)).cpu().numpy()
    print('softmax.shape=', softmax.shape)
    return softmax

def get_prob_c1_matrix(softmax, x_1, x_2):
    '''
    softmax: num_samples x 2
    x_1, x_2: grids in PCA(2)
    return matrix elements in the same order as for x1 for x2
    '''
    prob_c1, prob_c2 = softmax[:, 0], softmax[:, 1]
    is_c1 = prob_c1 >= prob_c2
    print('is_c1.shape=', is_c1.shape)

    #method 1:
    # is_c1_matrix = np.reshape(is_c1, (len(x_1), len(x_2)))

    #method 2: guaranteed the same order
    is_c1_matrix = np.zeros((len(x_1), len(x_2)))
    prob_c1_matrix = np.zeros((len(x_1), len(x_2)))
    prob_c2_matrix = np.zeros((len(x_1), len(x_2)))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            is_c1_matrix[i, j] = is_c1[n]
            prob_c1_matrix[i, j] = prob_c1[n]
            prob_c2_matrix[i, j] = prob_c2[n]
            n += 1

    return is_c1_matrix, prob_c1_matrix, prob_c2_matrix


def pca_features_to_network_output(pca, features_PCA, f1, f2, W, b, c1, c2):
    '''
    Given a pca feature vector (low-d),
    recover the 512 features by low-rank approximation,
    then feed it into the last layer of the networks to get the output, to get a prediction whether it is a CAT or DOG

    W, b: the parameters of the last (linear) layer before softmax

    This shows perturbation in the first two PCA space is not a big deal: even a large perturbation.

    How about the third, even the last components?
    '''

    Phi, Phi_back, x_1, x_2 = get_grids_in_pca_space(features_PCA, pca, f1, f2)

    outputs = np.dot(Phi_back, W.T) + b
    print('outputs.shape=', outputs.shape)

    #use binary classification: because the other classes are not involved.
    softmax = get_softmax_output(Phi_back, W, b, c1, c2)

    is_c1_matrix, prob_c1_matrix, prob_c2_matrix = get_prob_c1_matrix(softmax, x_1, x_2)

    return is_c1_matrix, x_1, x_2, prob_c1_matrix, prob_c2_matrix


def pca_features_to_db_predictor(pca_net, pca_features_2d, c1, c2):
    '''
    Given a pca feature vector (low-d),
    recover the 512 features by low-rank approximation,
    then feed it into the last layer of the networks to get the output, to get a prediction whether it is a CAT or DOG

    W, b: the parameters of the last (linear) layer before softmax
    '''

    # lows = np.min(pca_features_2d, axis=0) - 1.
    # highs = np.max(pca_features_2d, axis=0) + 1.
    # print('lows=', lows)
    # print('highs=', highs)

    #use same bounds across plots
    lows = [-7, -4]
    highs = [7., 6.]

    grids = 100
    x_1 = np.linspace(lows[0], highs[0], grids)
    x_2 = np.linspace(lows[1], highs[1], grids)
    print('x_1=', x_1.shape)

    #generate a synthetic feature sample matrix
    Phi = np.zeros((len(x_1)*len(x_2), 2))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            Phi[n, 0], Phi[n, 1] = x_1[i], x_2[j]
            n += 1
    print(Phi.shape)
    print(pca_features_2d.shape)

    pca_net.eval()
    f_softmax = torch.nn.Softmax(dim=1)

    outputs_samples = pca_net(torch.as_tensor(pca_features_2d).float())
    print('outputs_samples.shape=', outputs_samples.shape)
    softmax_samples = f_softmax(torch.as_tensor(outputs_samples)).detach().cpu().numpy()
    print('softmax_samples.shape=', softmax_samples.shape)
    plt.subplot(1, 2, 2)
    # labels_pred = np.argmax(softmax_samples, axis=1)
    # print('labels_pred=', labels_pred.shape)
    # plt.plot(labels_pred, 'b+')#correct
    plt.plot(softmax_samples[:5000, c1], 'b+', label='cats, prob(CAT)')
    plt.plot(softmax_samples[:5000, c2], '-co', label='cats, prob(DOG)')

    plt.plot(softmax_samples[5000:10000, c2], 'rx',label='dogs, prob(DOG)')#why in some run smaller than prob(CAT)
    plt.plot(softmax_samples[5000:10000, c1], '-ms' ,label='dogs, prob(CAT)')


    plt.legend()
    # sys.exit(1)

    outputs = pca_net(torch.as_tensor(Phi).float())
    print('outputs.shape from Phi:', outputs.shape)

    #add:do softmax only for c1 and c2
    outputs_c1 = outputs[:, c1][:, None]
    outputs_c2 = outputs[:, c2][:, None]
    outputs = torch.cat((outputs_c1, outputs_c2), dim=1)

    softmax = f_softmax(torch.as_tensor(outputs)).detach().cpu().numpy()
    print('softmax.shape=', softmax.shape)

    # prob_c1, prob_c2 = softmax[:, c1], softmax[:, c2]
    prob_c1, prob_c2 = softmax[:, 0], softmax[:, 1]
    is_c1 = prob_c1 >= prob_c2
    print('is_c1.shape=', is_c1.shape)

    #method 2: guaranteed the same order
    is_c1_matrix = np.zeros((len(x_1), len(x_2)))
    prob_c1_matrix = np.zeros((len(x_1), len(x_2)))
    prob_c2_matrix = np.zeros((len(x_1), len(x_2)))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            is_c1_matrix[i, j] = is_c1[n]
            prob_c1_matrix[i, j] = prob_c1[n]
            prob_c2_matrix[i, j] = prob_c2[n]
            n += 1

    return is_c1_matrix, x_1, x_2, prob_c1_matrix, prob_c2_matrix

def pca_features_least_squares(pca_features_2d, features_10d, c1, c2):
    '''
    Given a pca feature vector (low-d),
    recover the 512 features by low-rank approximation,
    then feed it into the last layer of the networks to get the output, to get a prediction whether it is a CAT or DOG

    pca_features_2d: 100 x 2
    features_10d: 100 x 10
    we want to approximate: pca_features_2d * W = features_10d
    This is solved by a least-squares procedure: W = inv(pca_features_2d.T * pca_features_2d) * (pca_features_2d.T * features_10d): 2 x 10
    Given a feature matrix X: n x 2,  the prediction is then given by X*W: n x 10

    oh. The other eight dimensions influence W. So that may be the reason that pca_low_rank can explain Y with only two features.
    '''

    print('features_10d.shape=', features_10d.shape)
    A = np.dot(pca_features_2d.T, pca_features_2d)
    b = pca_features_2d.T.dot(features_10d)
    print('A.shape=', A.shape)
    print('b.shape=', b.shape)
    print('cond(A)=', np.linalg.cond(A))
    W = np.linalg.solve(A, b)
    print('W.shape=', W.shape)

    #let's check the approximation error:
    features_10d_appr = pca_features_2d.dot(W)
    print('approximation error: ', np.linalg.norm(features_10d_appr - features_10d))#YES. It is very high. About 800!

    # lows = np.min(pca_features_2d, axis=0) - 1.
    # highs = np.max(pca_features_2d, axis=0) + 1.
    # print('lows=', lows)
    # print('highs=', highs)

    #use same bounds across plots
    lows = [-7, -4]
    highs = [7., 6.]

    grids = 10
    x_1 = np.linspace(lows[0], highs[0], grids)
    x_2 = np.linspace(lows[1], highs[1], grids)
    print('x_1=', x_1)

    #generate a synthetic feature sample matrix
    Phi = np.zeros((len(x_1)*len(x_2), 2))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            Phi[n, 0], Phi[n, 1] = x_1[i], x_2[j]
            n += 1
    print(Phi.shape)

    Yhat = Phi.dot(W)
    print('Yhat.shape=', Yhat.shape)

    f_softmax = torch.nn.Softmax(dim=1)
    softmax = f_softmax(torch.as_tensor(Yhat)).cpu().numpy()
    print('softmax.shape=', softmax.shape)

    prob_c1, prob_c2 = softmax[:, 0], softmax[:, 1]#binary; c1 is 0 and c2 is 1
    is_c1 = prob_c1 >= prob_c2
    print('is_c1.shape=', is_c1.shape)

    #method 1:
    # is_c1_matrix = np.reshape(is_c1, (len(x_1), len(x_2)))

    #method 2: guaranteed the same order
    is_c1_matrix = np.zeros((len(x_1), len(x_2)))
    prob_c1_matrix = np.zeros((len(x_1), len(x_2)))
    prob_c2_matrix = np.zeros((len(x_1), len(x_2)))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            is_c1_matrix[i, j] = is_c1[n]
            prob_c1_matrix[i, j] = prob_c1[n]
            prob_c2_matrix[i, j] = prob_c2[n]
            n += 1

    return is_c1_matrix, x_1, x_2, prob_c1_matrix, prob_c2_matrix

class EmbedNet(torch.nn.Module):
    '''
    from embedding features build a good approximation to Y (classification features)

    '''
    def __init__(self, input_size=512, output_size=10):
        super(EmbedNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size//2)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(input_size//2, input_size//4)
        self.relu2 = torch.nn.ReLU()
        self.fc_neck = torch.nn.Linear(input_size // 4, 32)
        # self.relu2 = torch.nn.Sigmoid()
        # self.fc3 = torch.nn.Linear(input_size//4, input_size//8)
        self.fc3 = torch.nn.Linear(32, input_size//8)#bottle neck
        self.relu3 = torch.nn.ReLU()
        # self.fc_neck = torch.nn.Linear(input_size // 8, 32)#still linear db
        # self.fc_neck = torch.nn.Linear(input_size//8, 8)#two PCA features out of 8 is already accurate in the scatter plot of cats and dogs for prob(CAT), it's all linear boundary for all predecessor models.
        # self.fc_neck = torch.nn.Linear(input_size//8, 2)
        self.fc4 = torch.nn.Linear(32, output_size)
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output_neck = self.fc_neck(output)
        output = self.fc3(output_neck)
        output = self.relu3(output)
        # output_neck = self.fc_neck(output)

        output = self.fc4(output_neck)
        return output, output_neck

    def forward_neck(self, output_neck):
        return self.fc4(output_neck)


class PCANet(torch.nn.Module):
    '''
    from PCA features build a good approximation to Y (classification features)
    input_size is the number of components of PCA, or the size of PCA features
    '''

    def __init__(self, n_components=64, output_size=10):
        super(PCANet, self).__init__()
        print('!!!!!PCANet: n_components=', n_components)
        self.fc1 = torch.nn.Linear(n_components, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.relu3 = torch.nn.ReLU()
        # self.relu3 = torch.nn.Sigmoid()
        self.fc4 = torch.nn.Linear(64, 16)
        self.bn4 = torch.nn.BatchNorm1d(16)
        self.relu4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(16, output_size)


    def forward(self, x):
        '''I suspect that batch size when big generalizing poor is because some mythical effect of BN'''
        output = self.fc1(x)
        # output = self.relu1(self.bn1(output))
        output = self.relu1(output)
        output2 = self.fc2(output)
        # output = self.relu2(self.bn2(output2))
        output = self.relu2(output2)
        output = self.fc3(output)
        # output = self.relu3(self.bn3(output))
        output = self.relu3(output)
        output = self.fc4(output)
        # output = self.relu4(self.bn4(output))
        output = self.relu4(output)
        output = self.fc5(output)
        return output


def pca_features_to_network_output_with_net(pca, pca_features_2d, low_rank_net, c1, c2):
    '''
    Given a pca feature vector (low-d),
    recover the 512 features by low-rank approximation,
    then feed it into the last layer of the networks to get the output, to get a prediction whether it is a CAT or DOG

    feed into the net directly
    '''

    # lows = np.min(pca_features_2d, axis=0) - 1.
    # highs = np.max(pca_features_2d, axis=0) + 1.
    # print('lows=', lows)
    # print('highs=', highs)

    #use same bounds across plots
    lows = [-7, -3]
    highs = [8., 4.]

    grids = 10
    x_1 = np.linspace(lows[0], highs[0], grids)
    x_2 = np.linspace(lows[1], highs[1], grids)
    print('x_1=', x_1)

    #generate a synthetic feature sample matrix
    Phi = np.zeros((len(x_1)*len(x_2), 2))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            Phi[n, 0], Phi[n, 1] = x_1[i], x_2[j]
            n += 1
    print(Phi.shape)

    Phi_back = torch.as_tensor(pca.inverse_transform(Phi)).float()#numpy used float64 by default; need to down to 32 for pytorch
    print('Phi_back.shape=', Phi_back.shape)
    outputs = low_rank_net.forward_neck(Phi_back)
    print('outputs.shape=', outputs.shape)

    f_softmax = torch.nn.Softmax(dim=1)
    softmax = f_softmax(outputs).detach().cpu().numpy()
    print('softmax.shape=', softmax.shape)

    prob_c1, prob_c2 = softmax[:, c1], softmax[:, c2]
    is_c1 = prob_c1 >= prob_c2
    print('is_c1.shape=', is_c1.shape)

    #method 1:
    # is_c1_matrix = np.reshape(is_c1, (len(x_1), len(x_2)))

    #method 2: guaranteed the same order
    is_c1_matrix = np.zeros((len(x_1), len(x_2)))
    prob_c1_matrix = np.zeros((len(x_1), len(x_2)))
    prob_c2_matrix = np.zeros((len(x_1), len(x_2)))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            is_c1_matrix[i, j] = is_c1[n]
            prob_c1_matrix[i, j] = prob_c1[n]
            prob_c2_matrix[i, j] = prob_c2[n]
            n += 1

    return is_c1_matrix, x_1, x_2, prob_c1_matrix, prob_c2_matrix

def embed_low_rank(features, Y):
    '''
    This is the low-rank approximation done in the network way.
    features: 1-dimensional feature vectors (input of the networks)
    Y the target signal vector
    '''

    def train(epoch, net, criterion, optimizer, trainloader, device='cpu'):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    low_ranker = EmbedNet()
    print(low_ranker)
    loss = torch.nn.CrossEntropyLoss()#torch.nn.MSELoss()
    # optimizer = optim.SGD(low_ranker.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-3)
    optimizer = optim.SGD(low_ranker.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-2)

    #generate data loader
    print(torch.as_tensor(features).size(0))
    print(torch.as_tensor(Y).size(0))
    dataset = TensorDataset(torch.as_tensor(features).float(), torch.as_tensor(Y).long())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for epoch in range(100):
        train(epoch, low_ranker, loss, optimizer, train_loader)

    return low_ranker


def train_pca_low_rank(low_ranker, ncomponents, pca_features, Y):
    '''
    This is the low-rank approximation done in the network way.
    pca_features: 1-dimensional PCA feature vectors (input of the networks)
    Y the target signal vector
    '''

    def train(low_ranker, epoch, criterion, optimizer, trainloader, device='cpu'):
        low_ranker.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # print('batch_idx=', batch_idx)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = low_ranker(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('\nEpoch: %d' % epoch)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        return total-correct

    print('net w in pca_low_rank')
    print(low_ranker.fc1.weight[0, :5])
    loss = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(low_ranker.parameters(), lr=1e-2, momentum=0.99, weight_decay=5e-3)#need small lr for fine details; that is, overfit it
    optimizer = optim.Adam(low_ranker.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    #generate data loader

    #data normalization: also do this when testing.
    scaler = preprocessing.StandardScaler().fit(pca_features)
    print('scaler._mean=', scaler.mean_)
    pca_features_scaled = scaler.transform(pca_features)
    print('pca_features_scaled.mean(axis=0)=', pca_features_scaled.mean(axis=0))
    print('pca_features_scaled.std(axis=0)=', pca_features_scaled.std(axis=0))
    print(torch.as_tensor(pca_features).size(0))
    print(torch.as_tensor(Y).size(0))
    dataset = TensorDataset(torch.as_tensor(pca_features_scaled).float(), torch.as_tensor(Y).long())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    for epoch in range(200):
        num_wrongs = train(low_ranker, epoch, loss, optimizer, train_loader)
        # scheduler.step()
        print('num_wrongs=', num_wrongs)

    return num_wrongs

def get_outputs(features, low_ranker):
    features_tensor = torch.as_tensor(features).float()
    c1_outputs, c1_outputs_neck = low_ranker(features_tensor[:5000])
    print('c1_outputs.shape=', c1_outputs.shape)
    print('c1_outputs_neck.shape=', c1_outputs_neck.shape)
    c2_outputs, c2_outputs_neck = low_ranker(features_tensor[5000:10000])
    print('c2_outputs.shape=', c2_outputs.shape)
    print('c2_outputs_neck.shape=', c2_outputs_neck.shape)
    return c1_outputs_neck.detach().cpu().numpy(), c2_outputs_neck.detach().cpu().numpy()

def get_outputs_pca(features, low_ranker):
    features_tensor = torch.as_tensor(features).float()
    c1_outputs = low_ranker(features_tensor[:5000])
    print('c1_outputs.shape=', c1_outputs.shape)
    c2_outputs = low_ranker(features_tensor[5000:10000])
    print('c2_outputs.shape=', c2_outputs.shape)
    return c1_outputs.detach().cpu().numpy(), c2_outputs.detach().cpu().numpy()

def standardization(outputs, c1, c2):
    '''
    softmax_outputs: 10 x 5000 x 10
    not working for now
    '''

    f_softmax = torch.nn.Softmax(dim=2)
    softmax_outputs = f_softmax(torch.as_tensor(outputs))
    print('softmax_outputs.shape=', softmax_outputs.shape)

    prob_cats_is_CAT = softmax_outputs[c1, :, c1][:, None]
    prob_cats_is_DOG = softmax_outputs[c1, :, c2][:, None]
    prob_cats = np.concatenate((prob_cats_is_CAT, prob_cats_is_DOG), axis=1)
    # print('prob_cats.shape=, should be 5000 x 2', prob_cats.shape)#YES

    prob_dogs_is_DOG = softmax_outputs[c2, :, c2][:, None]
    prob_dogs_is_CAT = softmax_outputs[c2, :, c1][:, None]
    prob_dogs = np.concatenate((prob_dogs_is_CAT, prob_dogs_is_DOG), axis=1)#features should be in the same order as cats
    print('prob_dogs=', prob_dogs)

    data = np.concatenate((prob_cats, prob_dogs), axis=0)
    # print('data.shape=; should be 10000 x 2', data.shape)

    # scaler = preprocessing.StandardScaler().fit(data)
    # print('scaler.mean_=', scaler.mean_)
    # print('scaler.scale_=', scaler.scale_)
    # data_scaled = scaler.transform(data)
    # print('data_scaled.mean(axis=0)=', data_scaled.mean(axis=0))
    # print('data_scaled.std(axis=0)=', data_scaled.std(axis=0))

    #method 2: svd: not working well
    # u, sigma, vh = np.linalg.svd(data, full_matrices=False)
    # print('u=', u.shape)
    # print('vh=', vh.shape)
    # print('sigma=', sigma)
    # return u

    #method 3: manual scaling
    # f1_cats = (prob_cats_is_CAT - prob_cats_is_CAT.mean(axis=0)) / prob_cats_is_CAT.std(axis=0)
    # f2_cats = (prob_cats_is_DOG - prob_cats_is_DOG.mean(axis=0)) / prob_cats_is_DOG.std(axis=0)
    # print('prob_cats_is_CAT.mean(axis=0) close to 1.0?:', prob_cats_is_CAT.mean(axis=0))
    # print('prob_cats_is_DOG.std(axis=0) small?:', prob_cats_is_DOG.std(axis=0))
    # f_cats = np.concatenate((f1_cats, f2_cats), axis=1)
    # print('f_cats.shape=', f_cats.shape)
    #
    # f1_dogs = (prob_dogs_is_CAT - prob_dogs_is_CAT.mean(axis=0)) / prob_dogs_is_CAT.std(axis=0)
    # f2_dogs = (prob_dogs_is_DOG - prob_dogs_is_DOG.mean(axis=0)) / prob_dogs_is_DOG.std(axis=0)
    # f_dogs = np.concatenate((f1_dogs, f2_dogs), axis=1)
    # print('f_dogs.shape=', f_dogs.shape)

    # return np.concatenate((f_cats, f_dogs), axis=0)#pattern is too simple though cats and dogs getting closer, not a typical clustering look.
    # return np.concatenate((np.cos(f_cats), np.cos(f_dogs)), axis=0)#add some nonlinearity: some random looks: not good.

    # m3 = np.concatenate((f_cats, f_dogs), axis=0)
    # print('m3.shape=', m3.shape)
    # return np.concatenate((u[:, 0][:, None], m3[:, 1][:, None]), axis=1)# a mixture of svd result and m3 scaling result: still too simple pattern

    return np.concatenate(((data[:, 0][:, None]-.5)**2, 0.01*((data[:, 1]-data[:, 0]**2)**2)[:, None]), axis=1)

def select_pca_components(pca, f1, f2):
    print('f1=', f1, ', f2=', f2)
    VT = pca.components_
    comp_selected_f1, comp_selected_f2 = VT[f1, :][None, :], VT[f2, :][None, :]
    comp_selected = np.concatenate((comp_selected_f1, comp_selected_f2), axis=0)
    print('comp_selected.shape=', comp_selected.shape)
    print('pca.mean=', pca.mean_.shape)
    # x_back = x_test.dot(comp_selected) + pca.mean_
    # print('x_back.shape=', x_back.shape)

    return comp_selected

def select_2features_from_least_squares(features, outputs, c1, c2):
    '''
    featuers: num_samples, num_features
    Interesting, this shows the best feature pair is actually (0,1): first two components
    -just very few predecessor models have (0, 2)
    '''
    num_features = features.shape[1]
    outputs_1_1, outputs_1_2 = outputs[c1, :, c1][:, None], outputs[c1, :, c2][:, None]
    outputs_1 = np.concatenate((outputs_1_1, outputs_1_2), axis=1)
    print(outputs_1.shape)

    outputs_2_1, outputs_2_2 = outputs[c2, :, c1][:, None], outputs[c2, :, c2][:, None]
    outputs_2 = np.concatenate((outputs_2_1, outputs_2_2), axis=1)
    print('outputs2.shape=', outputs_2.shape)

    Y = np.concatenate((outputs_1, outputs_2), axis=0)
    print('Y.shape=', Y.shape)
    min_err = np.infty
    selected = None
    for i in range(num_features):
        for j in range(i+1, num_features):
            Phi_1, Phi_2 = features[:, i][:, None], features[:, j][:, None]
            Phi = np.concatenate((Phi_1, Phi_2), axis=1)
            print('Phi.shape=', Phi.shape)
            w = np.linalg.solve(Phi.T.dot(Phi), Phi.T.dot(Y))
            outputs_appr = Phi.dot(w)
            err = np.linalg.norm(outputs_appr - Y)
            print('approximation error by LS:', err)
            if err<min_err:
                min_err = err
                selected = (i, j)

    print('########################best feature pair is ', selected, ', err:', min_err)
    return selected

def find_neighbors(phi, sample_features, n_neighbors):
    '''
    find a few nearest neighbors, then get their feature vectors in sample_features;
    todo: phi can be Phi: the batched sample features
    '''
    distances_by_row = np.linalg.norm(phi - sample_features, axis=1)
    # print('distances_by_row.shape should be number_samples', distances_by_row.shape)

    index = np.argsort(distances_by_row)#ascending order
    # print('increading order?', distances_by_row[index[:n_neighbors+5]])

    return index[:n_neighbors]


def nearest_neighbors(sample_features, sample_features_hd, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=5):
    '''
    add noise for a point in the PCA(2) space (first two components), by the nearest neighbors in the sample set
    sample_features: PCA features of samples (2D)
    sample_features_hd: original high-dimensional feature vectors

    This has almost same signature with pca_features_to_network_output
    '''

    Phi, Phi_back, x_1, x_2 = get_grids_in_pca_space(sample_features, pca, f1, f2)

    print('Phi.shape=', Phi.shape)
    print('sample_features_hd.shape=', sample_features_hd.shape)

    #for each row (feature vector) in Phi, find a few nearest neighbors, then get their feature vectors in sample_features;
    #todo: maybe worthwhile to optimize this search process like sorting?
    t0 = time.time()
    Phi_hd = np.zeros((Phi.shape[0], sample_features_hd.shape[1]))#10000 x 512
    for i in range(Phi.shape[0]):
        phi = Phi[i, :]
        closest_neighors = find_neighbors(phi, sample_features, n_neighbors)
        # print('closest neighbors are:', closest_neighors)
        phi_neighbors = sample_features[closest_neighors, :]
        # print('phi_neighbors.shape=', phi_neighbors.shape)
        Phi_hd[i, :] = sample_features_hd[closest_neighors, :].mean(axis=0)
        # print('phi_hd.shape should be 512?', phi_hd.shape)#YES!

    print('time:', time.time()-t0)

    softmax = get_softmax_output(Phi_hd, W, b, c1, c2)
    is_c1_matrix, prob_c1_matrix, prob_c2_matrix = get_prob_c1_matrix(softmax, x_1, x_2)

    return is_c1_matrix, x_1, x_2, prob_c1_matrix, prob_c2_matrix



def plot_PCA_pairwise(train_accuracies, n_components, c1, c2, head_path=header, f1=0, f2=1):

    plt.figure()

    for i, acc in enumerate(train_accuracies):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@features for ', acc)

        if acc == 'final':
            features = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_features.npy')
            outputs = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_outputs.npy')
            W = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_W.npy')
            b = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_b.npy')
            # features = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.80.pyc_features.npy')
        else:
            features = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')
            outputs = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_outputs.npy')
            W = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_W.npy')
            b = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_b.npy')

        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
        print('c1 and c2 combined featurte.shape=', c1_c2_features.shape)

        # PCA on the embedding features directly
        pca = PCA(n_components=n_components)
        pca.fit(c1_c2_features)
        print('pca.components_=', pca.components_)
        print('pca.explained_variance_=', pca.explained_variance_)
        features_PCA = pca.transform(c1_c2_features)
        print('pca features:', features_PCA.shape)
        features_back = pca.inverse_transform(features_PCA)
        print('pca transform_back features:', features_back.shape)
        print('@@@@@@low-rank appr err:', np.linalg.norm(features_back-c1_c2_features))


        plt.subplot(2, 3, i+1)
        plt.title(str(acc))

        #plots c1 and c2 objects in the PCA space
        plt.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        plt.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], color='r', marker='+', label=classes[c2]+'s', alpha=0.6)

        #method 5: use nearest sample neighbors for any point in the PCA(2) space
        is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=10)
        # is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=5)
        # is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=2)

        #select features by least-squares
        # select_2features_from_least_squares(features_PCA, outputs, c1, c2)

        #test the pca and svd
        # U, S, VT = np.linalg.svd(c1_c2_features - c1_c2_features.mean(0), full_matrices=False)
        # print('VT.shape=', VT.shape)
        # print('pca.components_.shape=', pca.components_.shape)
        # flip eigenvectors' sign to enforce deterministic output
        # U, VT = svd_flip(U, VT)#todo: need to import this
        # print('difference:', np.linalg.norm(VT[:n_components] - pca.components_))
        # np.testing.assert_array_almost_equal(VT[:n_components], pca.components_)#not hold probably because didn't do svd_flip

        #this shows how to use 1st and 3rd components to transform a 2-dimensional component feature vector back to 512 features to be fed into networks
        # comp_selected = select_pca_components(pca, f1=f1, f2=f2) #2, 512
        # x_test = np.random.randn(4, 2)
        # x_back = x_test.dot(comp_selected) + pca.mean_
        # print('x_back.shape=', x_back.shape)
        # sys.exit(1)

        #important: how how much approx error for the Y (outputs)?
        # outs_appr = np.dot(features_back, W.T) + b
        # print('outs_appr.shape=', outs_appr.shape)
        # #get only cat and dog prob
        # outs_appr_c1, outs_appr_c2 = outs_appr[:, c1][:, None], outs_appr[:, c2][:, None]
        # outs_appr = np.concatenate((outs_appr_c1, outs_appr_c2), axis=1)
        # print('outs_appr.shape=', outs_appr.shape)
        # softmax = torch.nn.Softmax(dim=1)
        # outs_appr = softmax(torch.as_tensor(outs_appr))
        #
        # outputs_c1_c1 = outputs[c1, :, c1][:, None]
        # outputs_c1_c2 = outputs[c1, :, c2][:, None]
        # outputs_c1 = np.concatenate((outputs_c1_c1, outputs_c1_c2), axis=1)
        #
        # outputs_c2_c1 = outputs[c2, :, c1][:, None]
        # outputs_c2_c2 = outputs[c2, :, c2][:, None]
        # outputs_c2 = np.concatenate((outputs_c2_c1, outputs_c2_c2), axis=1)
        #
        # outputs_true = np.concatenate((outputs_c1, outputs_c2), axis=0)
        # outputs_true = softmax(torch.as_tensor(outputs_true))
        # print('outputs_true.shape=', outputs_true.shape)
        # print('@@@@outputs approximation error:', np.linalg.norm(outs_appr - outputs_true))

        #now get the most two noisy dimensions: to study the sensitivity of the db


        # plt.plot(outputs_true[:5000, 0], 'b+', label='true cats prob(CAT)')#this shows the first two principle components are not accurate enough for predecessor models, but for the final model, it is! in the end, the db in the two pca space is characteristic enough.
        # plt.plot(outs_appr[:5000, 0], 'go', label='approx cats prob(CAT)')
        # plt.legend()


        #plot the most noisy dimensions: this shows the last two components out of three components are already not separable: this shows the first two components are very strong for separation of two classes
        # which means the two components are telling the major difference, and all the remaining components are perhaps telling small differences
        # plt.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        # plt.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], color='r', marker='+', label=classes[c2]+'s', alpha=0.6)



        #method 1:
        #below plots the prob(CAT) in the PCA space: this shows PCA space is not accurate enough with two features.
        # is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = pca_features_to_network_output(pca, features_PCA, f1= f1, f2=f2, W=W, b=b, c1=c1, c2=c2)
        # dx = (real_x[1] - real_x[0]) / 2.
        # dy = (real_y[1] - real_y[0]) / 2.
        extent = [real_x[0], real_x[-1], real_y[0], real_y[-1]]
        # plt.imshow(is_c1_matrix.T, extent=extent)
        plt.imshow(prob_c1.T, extent=extent)#why transpose?
        # plt.imshow(prob_c2.T, extent=extent)
        # plt.colorbar()#why some prob_cat is so low even for regions near cats: because PCA is only two features. PCA does not consider approximating Y.
        # plt.clim(0., 1.0)

        #method 3: apply least-squares fit with the two features: saved in db_evolution_method3.png. This shows the approximation of the target softmax features is really high.
        #confirms that using network embedding space have a similar results: just two components are not explainable enough. -- just found because all the 10 dims are used. used binary, it can explain.
        #What should we do? Bottleneck features? This shows actually the PCA components from the embedding space is explainable enough. Just the weights of the last layer needs to be adjusted. This also shows at the last stage, training is mostly for the last layer.
        # c1_c2_outputs = np.concatenate((outputs[c1, :, :], outputs[c2, :, :]), axis=0)#all 10 classes are fitted.
        # c1_outputs = np.concatenate((outputs[c1, :, c1][:, None], outputs[c1, :, c2][:, None]), axis=1)
        # c2_outputs = np.concatenate((outputs[c2, :, c1][:, None], outputs[c2, :, c2][:, None]), axis=1)
        # c1_c2_outputs = np.concatenate((c1_outputs, c2_outputs), axis=0)#use binary classification
        # print('c1_c2_outputs.shape=', c1_c2_outputs.shape)
        # is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = pca_features_least_squares(features_PCA, c1_c2_outputs, c1, c2)
        # print('real_x=', real_x)
        # print('real_y=', real_y)
        # # dx = (real_x[1] - real_x[0]) / 2.
        # # dy = (real_y[1] - real_y[0]) / 2.
        # extent = [real_x[0], real_x[-1], real_y[0], real_y[-1]]
        # # plt.imshow(is_c1_matrix.T, extent=extent)
        # plt.imshow(prob_c1.T, extent=extent)#why transpose?
        # # plt.imshow(prob_c1.T-prob_c2.T, extent=extent)#why transpose?
        # # plt.imshow(prob_c2.T, extent=extent)
        # plt.colorbar()

        #method 4: use standardization: not working
        # features_prob_scaled = standardization(outputs, c1, c2)
        # plt.scatter(features_prob_scaled[:5000, f1], features_prob_scaled[:5000, f2], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        # plt.scatter(features_prob_scaled[5000:10000, f1], features_prob_scaled[5000:10000, f2], color='r', marker='+', label=classes[c2]+'s', alpha=0.6)



        #
        # #method 2: use networks: could use         c1_c2_outputs = np.concatenate((outputs[c1, :, :], outputs[c2, :, :]), axis=0), but use labels are fine
        # c1_c2_outputs = np.zeros(10000)
        # c1_c2_outputs[:5000] = c1
        # c1_c2_outputs[5000:10000] = c2

        # print(c1_c2_outputs[:5, :])#should be close to one hot of c1: Yes
        # print(c1_c2_outputs[5000:5005, :])#should be close to one hot of c2

        #run PCA on the "neck" features output by the low-ranker
        # low_rank_net = embed_low_rank(c1_c2_features, c1_c2_outputs)
        # c1_outputs_neck, c2_outputs_neck = get_outputs(c1_c2_features, low_rank_net)
        # c1_c2_neck = np.concatenate((c1_outputs_neck, c2_outputs_neck), axis=0)
        # pca = PCA(n_components=n_components)
        # pca.fit(c1_c2_neck)
        # print(pca.components_)
        # print(pca.explained_variance_)
        # features_PCA_neck = pca.transform(c1_c2_neck)
        # print('features_PCA_neck:', features_PCA_neck.shape)
        #
        # # this transforms the pca features (low-d) back to the original feature size (high-d)
        # features_back = pca.inverse_transform(features_PCA_neck)
        # print('pca features_back_neck features:', features_back.shape)
        # print('low-rank appr err:', np.linalg.norm(features_back - c1_c2_neck))
        #
        # #
        # # #note we aim to plot the samples and boundary now in the neck space instead of the PCA space now.
        # #oh, this looks just extract the prob(CAT) and prob(DOG). The features lie in a straight line. Anyway, this could be a cool method for cardinality minimization.
        # plt.scatter(c1_outputs_neck[:, 0], c1_outputs_neck[:, 1], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        # plt.scatter(c2_outputs_neck[:, 0], c2_outputs_neck[:, 1], color='r', marker='+', label=classes[c2]+'s', alpha=0.6)
        #
        # #plot c1 and c2 objects in the PCA on distilled low rank space
        # plt.scatter(features_PCA_neck[:5000, 0], features_PCA_neck[:5000, 1], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        # plt.scatter(features_PCA_neck[5000:10000, 0], features_PCA_neck[5000:10000, 1], color='r', marker='+', label=classes[c2]+'s', alpha=0.6)
        #
        #
        # is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = pca_features_to_network_output_with_net(pca,
        #                                                                                 pca_features_2d=features_PCA_neck,
        #                                                                                 low_rank_net=low_rank_net, c1=c1, c2=c2)
        # print('real_x=', real_x)
        # print('real_y=', real_y)
        # # dx = (real_x[1] - real_x[0]) / 2.
        # # dy = (real_y[1] - real_y[0]) / 2.
        # extent = [real_x[0], real_x[-1], real_y[0], real_y[-1]]
        # # plt.imshow(is_c1_matrix.T, extent=extent)
        # plt.imshow(prob_c1.T, extent=extent)  # why transpose?
        # plt.imshow(prob_c2.T, extent=extent)
        # plt.colorbar()  # why some prob_cat is so low even for regions near cats: because PCA is only two features. PCA does not consider approximating Y.
        # plt.show()
        # sys.exit(1)
        #generate points in the low-rank space


        #method 2.1: similar to method 2 but use the pca components as network input instead of all the 512 features: this contracts with method 3!
        # I found this method cannot explain the model
        # c1_c2_outputs = np.zeros(10000)
        # c1_c2_outputs[:5000] = c1
        # c1_c2_outputs[5000:10000] = c2
        # low_ranker = PCANet(n_components=n_components)
        # print('before learning: w is')
        # print(low_ranker.fc1.weight[0, :5])
        # num_wrongs = train_pca_low_rank(low_ranker, n_components, features_PCA, c1_c2_outputs)
        # print('after learning: w is')
        # print(low_ranker.fc1.weight[0, :5])
        # is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = pca_features_to_db_predictor(low_ranker, features_PCA, c1, c2)
        # dx = (real_x[1] - real_x[0]) / 2.
        # dy = (real_y[1] - real_y[0]) / 2.
        # extent = [real_x[0], real_x[-1], real_y[0], real_y[-1]]
        # plt.subplot(1, 2, 1)
        # plt.imshow(prob_c1.T, extent=extent)#why transpose?
        # plt.colorbar()
        # # plt.clim(0.4, 0.6)
        # plt.title('num_wrongs={}'.format(num_wrongs))

        # for i in range(len(real_x)):
        #     for j in range(len(real_y)):
        #         if is_c1_matrix[i, j]:
        #             plt.plot(real_x[i], real_y[j], 'ys')
        #         else:
        #             plt.plot(real_x[i], real_y[j], 'mx')


        #plot using contour
        # F1, F2 = np.meshgrid(real_x, real_y)
        # # F1, F2 = np.meshgrid(real_y, real_x)
        # print('matrix.shape=', is_c1_matrix.shape)
        # print(is_c1_matrix)
        # # plt.contour(F1, F2, is_c1_matrix)
        # plt.contour(F1, F2, prob_c1)
        # print(F1)

        # if i == 0:
        #     plt.legend(loc='lower center', fontsize=14)

#Fig 1
#use a few percentile models
#run 1
train_accuracies = accuracies#['final', '99.98', '99.89', '99.63', '99.24', '95.16']
# plot_PCA_pairwise(train_accuracies, n_components=3, c1=3, c2=5)
# plot_PCA_pairwise(train_accuracies, n_components=2, c1=3, c2=5)#it seems two-components is already precise.
# plot_PCA_pairwise(train_accuracies, n_components=3, c1=1, c2=9)
# plot_PCA_pairwise(train_accuracies, n_components=3, c1=7, c2=8)

#for PCANet method
plot_PCA_pairwise(train_accuracies, n_components=2, c1=3, c2=5)#this shows two components can still explaine prob(CAT) and prob(DOG): it's just the weights of components were not good before
# plot_PCA_pairwise(train_accuracies, n_components=3, c1=3, c2=5, f1=0, f2=2)#one major one noisy
# plot_PCA_pairwise(train_accuracies, n_components=10, c1=3, c2=5, f1=0, f2=9)#two noisy; however, what about approximation of Y? will the first two components accurate enough to capture Y (though approximation the feature matrix is poor)
plt.show()

#run 2
# accuracies = reversed(['95.06', '99.48', '99.87', '99.92', 'final'])
# train_accuracies = accuracies#['final', '99.98', '99.89', '99.63', '99.24', '95.16']
# plot_PCA_pairwise(train_accuracies, n_components=3, head_path='results/second_run/', c1=3, c2=5)
# plot_PCA_pairwise(train_accuracies, n_components=3, head_path='results/second_run/', c1=1, c2=9)
# plot_PCA_pairwise(train_accuracies, n_components=3, head_path='results/second_run/',  c1=7, c2=8)


def plot_PCA_one_vs_allothers(acc, n_components, c1, f1, f2, colors, markers, path_head=header):
    '''
    use one model specified by acc to plot a class c1's boundary against all others
    Note: very important. PCA are applied pairwise, instead of fixed.
    '''
    # features = np.load(path_head + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')
    features = np.load(path_head + 'model_dla_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')
    plt.figure()
    pca = PCA(n_components=n_components)
    counter = 0
    for c2 in range(len(classes)):
        if c2 == c1:
            continue
        else:
            counter += 1
        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
        print('c1 and c2 combined featurte.shape=', c1_c2_features.shape)
        pca.fit(c1_c2_features)
        print(pca.components_)
        print(pca.explained_variance_)
        features_PCA = pca.transform(c1_c2_features)
        print('pca features:', features_PCA.shape)

        plt.subplot(3, 3, counter)
        plt.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        plt.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], color=colors[counter-1], marker=markers[counter-1], label=classes[c2]+'s', alpha=0.6)

        # plt.legend(loc='best', fontsize=14)

#cats vs. all other classes
colors   =['g', 'k', 'y', 'm', 'r', 'c', 'tab:brown', 'tab:olive', 'tab:purple']
markers = ['s', 'p', '*', '<', '+', 'v', 'x', 'h', '1']
# plot_PCA_one_vs_allothers('99.63', n_components=2, c1=3, f1=0, f2=1, colors=colors, markers=markers)
# plot_PCA_one_vs_allothers('99.45', n_components=2, c1=3, f1=0, f2=1, colors=colors, markers=markers, path_head='results/')#first run
# plot_PCA_one_vs_allothers('99.48', n_components=2, c1=3, f1=0, f2=1, colors=colors, markers=markers, path_head='results/second_run/')#second run
#compare with DLA at 99.5
# plot_PCA_one_vs_allothers('99.53', n_components=2, c1=3, f1=0, f2=1, colors=colors, markers=markers, path_head='results/')
#this shows mostly boundaries are similar. except cat vs. frog
# plt.show()

def plot_PCA_tripplewise(acc, tripples, n_components, colors, markers, f1=0, f2=1, f3=2):
    pca = PCA(n_components=n_components)
    if acc == 'final':
        features = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_features.npy')
    else:
        features = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')

    fig = plt.figure()
    for i, tr in enumerate(tripples):
        c1, c2, c3 = tr[0], tr[1], tr[2]
        c1_c2_c3_features = np.concatenate((features[c1, :, :], features[c2, :, :], features[c3, :, :]), axis=0)
        print('c1 and c2 and c3 combined feature.shape=', c1_c2_c3_features.shape)
        pca.fit(c1_c2_c3_features)
        print(pca.components_)
        print(pca.explained_variance_)
        features_PCA = pca.transform(c1_c2_c3_features)
        print('pca features:', features_PCA.shape)

        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2], features_PCA[:5000, f3], color=colors[c1], marker=markers[c1], label=classes[c1]+'s', alpha=0.6)
        ax.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], features_PCA[5000:10000, f3], color=colors[c2], marker=markers[c2], label=classes[c2]+'s', alpha=0.6)
        ax.scatter(features_PCA[10000:15000, f1], features_PCA[10000:15000, f2], features_PCA[10000:15000, f3], color=colors[c3], marker=markers[c3], label=classes[c3] + 's', alpha=0.6)
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        ax.set_zlabel('f3')
        # plt.legend(loc='best', fontsize=14)

#plot class triples
# triples = [[3, 5, 0], [3, 5, 6], [3, 5, 1],
#             [1, 9, 3], [1, 9, 0], [1, 9, 8]]
colors = ['g', 'k', 'y', 'b', 'm', 'r', 'c', 'tab:brown', 'tab:olive', 'tab:purple']
markers = ['s', 'p', '*', 'o', '<', '+', 'v', 'x', 'h', '1']
# plot_PCA_tripplewise('99.45', triples, n_components=3, colors=colors, markers=markers)

#A closer look by extracting the boundary samples.
#two extraction methods: first is by the prob(CAT) and prob(DOG); the other is by the class center and distance to the centers.

def prob_resistors(softmax, c1, c2, prob_threshold=0.5):
    '''
    training resistor extraction according to the softmax output
    '''
    c1_predict_as_c1 = softmax[c1, :, c1]
    c1_predict_as_c2 = softmax[c1, :, c2]
    print(c1_predict_as_c1.shape)
    print(c1_predict_as_c2.shape)#5000
    index_c1 = np.where(c1_predict_as_c1 <= c1_predict_as_c2 + prob_threshold)[0]
    return index_c1, c1_predict_as_c1, c1_predict_as_c2

def extract_by_prob_c1_c2(acc, colors, markers,c1=3, c2=5, closest_k=100):
    '''
    acc: which percentile model
    '''

    f_softmax = torch.nn.Softmax(dim=2)

    # if acc == 'final':
    #     all_outs = np.load(
    #         header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_outputs.npy')
    # else:
    all_outs = np.load(
        header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_outputs.npy')
    features = np.load(
        header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')

    print(all_outs.shape)

    softmax = f_softmax(torch.as_tensor(all_outs)).cpu().numpy()

    #look for ambiguity in the predictions: method 1
    index_c1, c1_predict_as_c1, c1_predict_as_c2 = prob_resistors(softmax, c1, c2, prob_threshold=0.5)
    index_c2, c2_predict_as_c2, c2_predict_as_c1= prob_resistors(softmax, c2, c1, prob_threshold=0.5)
    print('how many boundary samples for class {}:{}'.format(classes[c1], len(index_c1)))
    print('how many boundary samples for class {}:{}'.format(classes[c2], len(index_c2)))

    plt.figure()
    plt.plot(c1_predict_as_c1[index_c1], 'b+', label='prob({}) of {} resistors'.format(classes[c1].upper(), classes[c1].upper()))
    plt.plot(c1_predict_as_c2[index_c1], 'bo', label='prob({}) of {} resistors'.format(classes[c2].upper(), classes[c1].upper()))
    plt.legend()
    plt.figure()
    plt.plot(c1_predict_as_c1[index_c1] + c1_predict_as_c2[index_c1], 'rx', label='sum of two probs for {} resistors'.format(classes[c1].upper()))
    plt.legend()

    #now plot these boundary c1 and c2 objects
    #first do PCA
    c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
    print('c1 and c2 combined feature.shape=', c1_c2_features.shape)
    pca = PCA(n_components=2)
    pca.fit(c1_c2_features)
    print(pca.components_)
    print(pca.explained_variance_)
    features_PCA = pca.transform(c1_c2_features)
    print('pca features:', features_PCA.shape)

    # #method 2: by to-center distance
    # c1_features = features_PCA[:5000, :]
    # c1_center = np.mean(c1_features, axis=0)
    # print('c1_center.shape=', c1_center.shape)
    # c2_features = features_PCA[5000:10000, :]
    # c2_center = np.mean(c2_features, axis=0)
    # print('c2_center.shape=', c2_center.shape)
    #
    # c1_to_c2_center = np.linalg.norm(c1_features - c2_center, axis=1)
    # index_sorted_c1 = np.argsort(c1_to_c2_center)#increasing order
    # index_c1 = index_sorted_c1[:closest_k] #smallest distance to c2
    # print('c1_to_c2_center.shape=', c1_to_c2_center.shape)
    # print('index_sorted_c1.shape=', index_sorted_c1)
    # print(c1_to_c2_center[index_sorted_c1[:10]])#indeed increasing
    #
    # c2_to_c1_center = np.linalg.norm(c2_features - c1_center, axis=1)
    # index_sorted_c2 = np.argsort(c2_to_c1_center)
    # index_c2 = index_sorted_c2[:closest_k]

    # plt.scatter(features_PCA[index_c1, 0], features_PCA[index_c1, 1], c=colors[c1], marker = markers[c1], label='boundary {}s'.format(classes[c1]))
    # plt.scatter(features_PCA[5000+index_c2, 0], features_PCA[5000+index_c2, 1], c=colors[c2], marker = markers[c2], label='boundary {}s'.format(classes[c2]))
    # plt.legend()

    return

# extract_by_prob_c1_c2(acc='99.98')
# extract_by_prob_c1_c2(acc='99.45', closest_k=5000) # just to confirm is the same as the pca feature space for the whole samples: Yes!
# plt.figure()
# plt.subplot(1, 3, 1)
# extract_by_prob_c1_c2(acc='99.45', closest_k=250, colors=colors, markers=markers)
# plt.subplot(1, 3, 2)
# extract_by_prob_c1_c2(acc='99.45', closest_k=250, c1=3, c2=6, colors=colors, markers=markers)
# plt.subplot(1, 3, 3)
# extract_by_prob_c1_c2(acc='99.45', closest_k=250, c1=1, c2=9, colors=colors, markers=markers)

#todo: see how training resistors look like

#compare with small-lr and big-lr
def plot_prob_output_file(output_file, c1=3, c2=5):
    '''train_accuracy is string of float with .2 accuracy'''
    f_softmax = torch.nn.Softmax(dim=2)
    all_outs = np.load(header + output_file)
    print(all_outs.shape)

    softmax = f_softmax(torch.as_tensor(all_outs)).cpu().numpy()
    plt.scatter(softmax[c1, :, c1], softmax[c1, :, c2], color='b', marker ='o', alpha=0.6, label='{}s'.format(classes[c1]))
    plt.scatter(softmax[c2, :, c1], softmax[c2, :, c2], color = 'r', marker = '+', alpha=0.6, label='{}s'.format(classes[c2]))
    plt.xlabel('Prob({})'.format(classes[c1].upper()))
    plt.ylabel('Prob({})'.format(classes[c2].upper()))
    plt.legend()

def plot_PCA_pairwise_file(feature_file, colors, markers, n_components=2, c1=3, c2=5, f1=0, f2=1):
    pca = PCA(n_components=n_components)
    features = np.load(header + feature_file)

    c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
    print('c1 and c2 combined featurte.shape=', c1_c2_features.shape)
    pca.fit(c1_c2_features)
    print(pca.components_)
    print(pca.explained_variance_)
    features_PCA = pca.transform(c1_c2_features)
    print('pca features:', features_PCA.shape)

    plt.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2], color=colors[c1], marker=markers[c1], label=classes[c1]+'s', alpha=0.6)
    plt.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], color=colors[c2], marker=markers[c2], label=classes[c2]+'s', alpha=0.6)
    # plt.legend()
    plt.xlim([-6, 7.])
    plt.ylim([-3, 4.5])

# plt.figure()

#small lr
# plt.subplot(2, 2, 1)
# plt.title('small-lr')
# plot_prob_output_file(output_file='model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_outputs.npy', c1=3, c2=5)
# plt.subplot(2, 2, 2)
# plt.title('small-lr')
# plot_PCA_pairwise_file(feature_file='model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_features.npy', n_components=2, c1=3, c2=5, f1=0, f2=1)

#big lr
# plt.subplot(2, 2, 3)
# plt.title('big-lr')
# plot_prob_output_file(output_file='model_vgg19_alpha_0.01_lrmode_constantbatchsize_128_momentum_decayed_testacc_90.12.pyc_outputs.npy', c1=3, c2=5)
# plt.subplot(2, 2, 4)
# plt.title('big-lr')
# plot_PCA_pairwise_file(feature_file='model_vgg19_alpha_0.01_lrmode_constantbatchsize_128_momentum_decayed_testacc_90.12.pyc_features.npy', n_components=2, c1=3, c2=5, f1=0, f2=1)

#compare different classes: small-lr
# small_lr_outfile = 'model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_outputs.npy'
# plt.subplot(2, 2, 1)
# plt.title('small-lr')
# plot_prob_output_file(output_file=small_lr_outfile, c1=3, c2=5)
# plt.subplot(2, 2, 2)
# plt.title('small-lr')
# plot_prob_output_file(output_file=small_lr_outfile, c1=3, c2=6)
# plt.subplot(2, 2, 3)
# plt.title('small-lr')
# plot_prob_output_file(output_file=small_lr_outfile, c1=3, c2=0)
# plt.subplot(2, 2, 4)
# plt.title('small-lr')
# plot_prob_output_file(output_file=small_lr_outfile, c1=3, c2=1)


def plot_PCA_one_vs_allothers_file(feature_file, n_components, c1, f1, f2, colors, markers):
    '''
    similar to plot_PCA_one_vs_allothers
    '''
    features = np.load(header + feature_file)
    print('features.shape=', features.shape)
    pca = PCA(n_components=n_components)
    counter = 0
    for c2 in range(len(classes)):
        if c2 == c1:
            continue
        else:
            counter += 1
        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
        print('c1 and c2 combined featurte.shape=', c1_c2_features.shape)
        pca.fit(c1_c2_features)
        print(pca.components_)
        print(pca.explained_variance_)
        features_PCA = pca.transform(c1_c2_features)
        print('pca features:', features_PCA.shape)

        plt.subplot(3, 3, counter)
        # plt.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        plt.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2], color='b', marker='o', alpha=0.6)
        plt.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], color=colors[counter-1], marker=markers[counter-1], label=classes[c2]+'s', alpha=0.6)
        # plt.legend(loc='best', fontsize=14)
        plt.xlim([-6, 6])

# colors = ['g', 'k', 'y', 'm', 'r', 'c', 'tab:brown', 'tab:olive', 'tab:purple']
# markers = ['s', 'p', '*', '<', '+', 'v', 'x', 'h', '1']

# fig=plt.figure()
# # fig.suptitle('cats(blue circles) vs. all other classes: small lr 0.0001')
# # small_lr_feature_file = 'model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_features.npy'
# # plot_PCA_one_vs_allothers_file(feature_file=small_lr_feature_file, n_components=2, c1=3, f1=0, f2=1, colors=colors, markers=markers)
#
# small_lr_feature_same_train_acc_file = 'model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_model_at_epoch_120_acc_98.29.pyc_features.npy'
# plot_PCA_one_vs_allothers_file(feature_file=small_lr_feature_same_train_acc_file, n_components=2, c1=3, f1=0, f2=1, colors=colors, markers=markers)

# fig=plt.figure()
# # fig.suptitle('cats(blue circles) vs. all other classes: big lr 0.01')
# big_lr_feature_file = 'model_vgg19_alpha_0.01_lrmode_constantbatchsize_128_momentum_decayed_testacc_90.12.pyc_features.npy'#about 98.25 train accuracy
# plot_PCA_one_vs_allothers_file(feature_file=big_lr_feature_file, n_components=2, c1=3, f1=0, f2=1, colors=colors, markers=markers)

#compare vgg, resnet18 and dla
# fig = plt.figure()
# vgg_feature_file = 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.45.pyc_features.npy'
# resnet_feature_file = 'model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.48.pyc_features.npy'
# dla_feature_file = 'model_dla_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.53.pyc_features.npy'
# vgg_feature_file = 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.89.pyc_features.npy'
# resnet_feature_file = 'model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.89.pyc_features.npy'
# dla_feature_file = 'model_dla_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.86.pyc_features.npy'
# plt.subplot(3, 3, 1)
# plot_PCA_pairwise_file(vgg_feature_file, colors, markers, c1=3, c2=5)
# plt.subplot(3, 3, 2)
# plot_PCA_pairwise_file(vgg_feature_file, colors, markers, c1=3, c2=6)
# plt.subplot(3, 3, 3)
# plot_PCA_pairwise_file(vgg_feature_file, colors, markers, c1=3, c2=4)
#
# plt.subplot(3, 3, 4)
# plot_PCA_pairwise_file(resnet_feature_file, colors, markers, c1=3, c2=5)
# plt.subplot(3, 3, 5)
# plot_PCA_pairwise_file(resnet_feature_file, colors, markers, c1=3, c2=6)
# plt.subplot(3, 3, 6)
# plot_PCA_pairwise_file(resnet_feature_file, colors, markers, c1=3, c2=4)
#
# plt.subplot(3, 3, 7)
# plot_PCA_pairwise_file(dla_feature_file, colors, markers, c1=3, c2=5)
# plt.subplot(3, 3, 8)
# plot_PCA_pairwise_file(dla_feature_file, colors, markers, c1=3, c2=6)
# plt.subplot(3, 3, 9)
# plot_PCA_pairwise_file(dla_feature_file, colors, markers, c1=3, c2=4)
# plt.show()
