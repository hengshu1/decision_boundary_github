import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from main import classes
import sys, time

header = 'results/'

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


#Fig 9: decision boundary in the decision space
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

        # Use nearest sample neighbors for any point in the PCA(2) space
        is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=10)
        # is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=5)
        # is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=2)

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

#Fig 1
train_accuracies = accuracies#['final', '99.98', '99.89', '99.63', '99.24', '95.16']
# plot_PCA_pairwise(train_accuracies, n_components=3, c1=3, c2=5)
# plot_PCA_pairwise(train_accuracies, n_components=2, c1=3, c2=5)#it seems two-components is already precise.
# plot_PCA_pairwise(train_accuracies, n_components=3, c1=1, c2=9)
# plot_PCA_pairwise(train_accuracies, n_components=3, c1=7, c2=8)

plot_PCA_pairwise(train_accuracies, n_components=2, c1=3, c2=5)#this shows two components can still explaine prob(CAT) and prob(DOG): it's just the weights of components were not good before
# plot_PCA_pairwise(train_accuracies, n_components=3, c1=3, c2=5, f1=0, f2=2)#one major one noisy
# plot_PCA_pairwise(train_accuracies, n_components=10, c1=3, c2=5, f1=0, f2=9)#two noisy; however, what about approximation of Y? will the first two components accurate enough to capture Y (though approximation the feature matrix is poor)
plt.show()


def plot_PCA_one_vs_allothers(acc, n_components, c1, f1, f2, colors, markers, path_head=header):
    '''
    use one model specified by acc to plot a class c1's boundary against all others
    Note: PCA is applied pairwise classes
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

#Fig 2
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

#Fig 10: 3d decision boundary
#plot class triples
# triples = [[3, 5, 0], [3, 5, 6], [3, 5, 1],
#             [1, 9, 3], [1, 9, 0], [1, 9, 8]]
colors = ['g', 'k', 'y', 'b', 'm', 'r', 'c', 'tab:brown', 'tab:olive', 'tab:purple']
markers = ['s', 'p', '*', 'o', '<', '+', 'v', 'x', 'h', '1']
# plot_PCA_tripplewise('99.45', triples, n_components=3, colors=colors, markers=markers)

#Fig 3: learning rate effect for cluster size
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

#Fig 6
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
