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
from scipy.stats import entropy

#todo: try to train the model and save the 90% training accuracy's model, and then plot the decision boundary

#todo: google drive seems need Google API
# header = '~/Google\ Drive/My\ Drive/decision_boundary_results/' #not working
# header = 'results/run1/'
# train_accuracies = reversed(['90.62', '99.50', '99.80', '99.87', '99.97', 'final'])

header = 'results/run1/'
train_accuracies = reversed(['90.62', '99.50', '99.80', '99.87', '99.97', 'final'])
final_file_header = 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.31.pyc_'

# header = 'results/run2/'
# train_accuracies = reversed(['90.62', '99.48', '99.84', '99.87', '99.97', 'final'])
# train_accuracies = reversed(['71.09', '74.54', '81.25', '85.16'])
# train_accuracies = reversed(['85.16', '90.62', '99.48', '99.87', '99.97', 'final'])
# final_file_header = 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.89.pyc_'

# header = 'results/run3/'
# train_accuracies = reversed(['90.23', '99.48', '99.84', '99.87', '99.97', 'final'])
# final_file_header = 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.99.pyc_'

# header = 'results/run4/'
# train_accuracies = reversed(['91.67', '99.48', '99.80', '99.86', '99.97', 'final'])
# final_file_header = 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.97.pyc_'

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

def get_grids_in_pca_space(features_PCA, pca, f1, f2, lows, highs, grids):
    print('f1=', f1, 'f2=', f2)
    data_selected_1, data_selected_2 = features_PCA[:, f1][:, None], features_PCA[:, f2][:, None]
    data_selected = np.concatenate((data_selected_1, data_selected_2), axis=1)
    print('data_selected.shape=', data_selected.shape)

    # lows = np.min(data_selected, axis=0) - 1.
    # highs = np.max(data_selected, axis=0) + 1.
    # print('lows=', lows)
    # print('highs=', highs)

    # use same bounds across plots
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


def get_grids_in_svd_space(features_svd, VT, lows, highs, grids):
    print('get_grids_in_svd_space')
    data_selected_1, data_selected_2 = features_svd[:, 0][:, None], features_svd[:, 1][:, None]
    data_selected = np.concatenate((data_selected_1, data_selected_2), axis=1)
    print('data_selected.shape=', data_selected.shape)

    # lows = np.min(data_selected, axis=0) - 1.
    # highs = np.max(data_selected, axis=0) + 1.
    # print('lows=', lows)
    # print('highs=', highs)

    # use same bounds across plots
    x_1 = np.linspace(lows[0], highs[0], grids)
    x_2 = np.linspace(lows[1], highs[1], grids)

    # generate a synthetic feature sample matrix
    Phi = np.zeros((len(x_1) * len(x_2), 2))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            Phi[n, 0], Phi[n, 1] = x_1[i], x_2[j]
            n += 1

    print('Phi.shape=', Phi.shape)
    print('VT.shape=', VT.shape)
    Phi_back = Phi.dot(VT[:2, :])
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


def pca_features_to_network_output(pca, features_PCA, f1, f2, W, b, c1, c2, lows, highs):
    '''
    Given a pca feature vector (low-d),
    recover the 512 features by low-rank approximation,
    then feed it into the last layer of the networks to get the output, to get a prediction whether it is a CAT or DOG

    W, b: the parameters of the last (linear) layer before softmax

    This shows perturbation in the first two PCA space is not a big deal: even a large perturbation.

    How about the third, even the last components?
    '''

    Phi, Phi_back, x_1, x_2 = get_grids_in_pca_space(features_PCA, pca, f1, f2, lows=lows, highs=highs)

    outputs = np.dot(Phi_back, W.T) + b
    print('outputs.shape=', outputs.shape)

    #use binary classification: because the other classes are not involved.
    softmax = get_softmax_output(Phi_back, W, b, c1, c2)

    is_c1_matrix, prob_c1_matrix, prob_c2_matrix = get_prob_c1_matrix(softmax, x_1, x_2)

    return is_c1_matrix, x_1, x_2, prob_c1_matrix, prob_c2_matrix


def find_neighbors_sort(phi, sample_features, n_neighbors):
    '''
    find a few nearest neighbors, then get their feature vectors in sample_features;
    todo: phi can be Phi: the batched sample features
    '''
    distances_by_row = np.linalg.norm(phi - sample_features, axis=1)
    # print('distances_by_row.shape should be number_samples', distances_by_row.shape)

    index = np.argsort(distances_by_row)#ascending order
    # print('increading order?', distances_by_row[index[:n_neighbors+5]])
    # print('index=', index[:n_neighbors])

    return index[:n_neighbors]

def find_neighbors_by_topk(phi, sample_features, n_neighbors):
    '''
    note the results are not in order
    a bit faster but not much than find_neighbors_sort
    '''
    print('phi.shape=', phi.shape)
    print('(phi - sample_features).shape=', (phi - sample_features).shape)#num_samples, 2
    sys.exit(1)
    distances_by_row = np.linalg.norm(phi - sample_features, axis=1)
    ind = np.argpartition(-distances_by_row, -n_neighbors)[-n_neighbors:]#get the topk smallest
    # print('ind=', ind)
    return ind


def nearest_neighbors(sample_features, sample_features_hd, pca, W, b, c1, c2, lows, highs, f1=0, f2=1, n_neighbors=5, decay=0.5, grids=200, mode='svd'):
    '''
    add noise for a point in the PCA(2) space (first two components), by the nearest neighbors in the sample set
    sample_features: PCA features of samples (2D)
    sample_features_hd: original high-dimensional feature vectors

    This has almost same signature with pca_features_to_network_output
    '''
    if mode == 'svd':
        Phi, Phi_back, x_1, x_2 = get_grids_in_svd_space(features_svd=sample_features, VT=pca, lows=lows, highs=highs, grids=grids)
    else:
        Phi, Phi_back, x_1, x_2 = get_grids_in_pca_space(sample_features, pca, f1, f2, lows, highs, grids=grids)

    print('Phi.shape=', Phi.shape)
    print('sample_features_hd.shape=', sample_features_hd.shape)
    print('n_neighbors=', n_neighbors)

    #for each row (feature vector) in Phi, find a few nearest neighbors, then get their feature vectors in sample_features;
    #todo: maybe worthwhile to optimize this search process like sorting?
    t0 = time.time()
    Phi_hd = np.zeros((Phi.shape[0], sample_features_hd.shape[1]))#10000 x 512
    for i in range(Phi.shape[0]):
        phi = Phi[i, :]

        #double checked: they returned the same results
        closest_neighors = find_neighbors_sort(phi, sample_features, n_neighbors)
        # closest_neighors = find_neighbors_by_topk(phi, sample_features, n_neighbors)

        # print('closest neighbors are:', closest_neighors)
        # phi_neighbors = sample_features[closest_neighors, :]
        # print('phi_neighbors.shape=', phi_neighbors.shape)
        # Phi_hd[i, :] = sample_features_hd[closest_neighors, :].mean(axis=0)
        # print('sample_features_hd[closest_neighors, :].shape=', sample_features_hd[closest_neighors, :].shape)

        #method 2:
        exponents = np.array(range(n_neighbors))
        weights = decay ** exponents
        # assert n_neighbors == closest_neighors.shape[0]#passed
        products = sample_features_hd[closest_neighors, :].T * weights #512, 5
        # print('products.shape=', products.shape)
        sum = products.sum(axis=1)
        # scale = (1-decay**n_neighbors) / (1-decay)
        #print('sums.shape=', sum.shape) #512
        Phi_hd[i, :] = sum

        #method 1: simply mean
        #Phi_hd[i, :] = sample_features_hd[closest_neighors, :].mean(axis=0)

        # print('Phi_hd.shape should be 512?', Phi_hd.shape)#YES!

    # print('time:', time.time()-t0)
    # sys.exit(1)

    softmax = get_softmax_output(Phi_hd, W, b, c1, c2)
    is_c1_matrix, prob_c1_matrix, prob_c2_matrix = get_prob_c1_matrix(softmax, x_1, x_2)

    return is_c1_matrix, x_1, x_2, prob_c1_matrix, prob_c2_matrix


def nearest_neighbor_entropy(sample_features, sample_features_hd, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=5, decay=0.5):
    '''
    use the shannon entropy to measure the distribution of labels in the neighbourhood of a feature vector in PCA(2) space.
    '''

    Phi, Phi_back, x_1, x_2 = get_grids_in_pca_space(sample_features, pca, f1, f2)

    # print('Phi.shape=', Phi.shape)
    print('sample_features.shape=', sample_features.shape)
    print('sample_features_hd.shape=', sample_features_hd.shape)

    #for each row (feature vector) in Phi, find a few nearest neighbors, then get their feature vectors in sample_features;
    #todo: maybe worthwhile to optimize this search process like sorting?
    t0 = time.time()
    labels_prob = -np.ones(Phi.shape[0])
    entropies = -np.ones(Phi.shape[0])
    for i in range(Phi.shape[0]):
        phi = Phi[i, :]

        #double checked: they returned the same results
        closest_neighors = find_neighbors_sort(phi, sample_features, n_neighbors)
        # closest_neighors = find_neighbors_by_topk(phi, sample_features, n_neighbors)

        num_c1 = len(np.where(closest_neighors < sample_features.shape[0]//2)[0])

        # if num_c1 == 9:
        #     print('num_c1=', num_c1)
        #     print(phi)
        #     print('closest neighbors are:', closest_neighors)
        #     print('num_c1=', num_c1)
        #     print('sample_features.shape[0]//2=', sample_features.shape[0]//2)

        labels_prob[i] = num_c1 / len(closest_neighors)
        p = [labels_prob[i], 1. - labels_prob[i]]
        entropies[i] = entropy(p, base=2)

    print('time:', time.time()-t0)

    # method 2: guaranteed the same order
    entropy_matrix = np.zeros((len(x_1), len(x_2)))
    n = 0
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            entropy_matrix[i, j] = entropies[n]
            n += 1

    return entropy_matrix, x_1, x_2 #, labels_prob



#
# def nearest_neighbors_efficient(sample_features, sample_features_hd, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=5, decay=0.5):
#     '''
#     making nearest_neighbors more efficient
#     todo: not done yet. time is not so important for now.
#     '''
#
#     Phi, _, x_1, x_2 = get_grids_in_pca_space(sample_features, pca, f1, f2)
#
#     print('Phi.shape=', Phi.shape)
#     print('sample_features_hd.shape=', sample_features_hd.shape)
#
#     t0 = time.time()
#     # Phi_hd = np.zeros((Phi.shape[0], sample_features_hd.shape[1]))#10000 x 512
#
#     distances_by_row = np.linalg.norm(Phi - sample_features, axis=1)
#     print('distances_by_row.shape should be number_samples:', distances_by_row.shape)
#     sys.exit(1)
#
#     index = np.argsort(distances_by_row)  # ascending order
#     # print('increading order?', distances_by_row[index[:n_neighbors+5]])
#     # print('index=', index[:n_neighbors])
#
#     closest_neighors = find_neighbors_sort(phi, sample_features, n_neighbors)
#     # closest_neighors = find_neighbors_by_topk(phi, sample_features, n_neighbors)
#
#     # print('closest neighbors are:', closest_neighors)
#     # phi_neighbors = sample_features[closest_neighors, :]
#     # print('phi_neighbors.shape=', phi_neighbors.shape)
#     # Phi_hd[i, :] = sample_features_hd[closest_neighors, :].mean(axis=0)
#     # print('sample_features_hd[closest_neighors, :].shape=', sample_features_hd[closest_neighors, :].shape)
#     exponents = np.array(range(n_neighbors))
#     weights = decay ** exponents
#     # assert n_neighbors == closest_neighors.shape[0]#passed
#     products = sample_features_hd[closest_neighors, :].T * weights #512, 5
#     # print('products.shape=', products.shape)
#     sum = products.sum(axis=1)
#     # scale = (1-decay**n_neighbors) / (1-decay)
#     #print('sums.shape=', sum.shape) #512
#     Phi_hd[i, :] = sum
#
#
#     softmax = get_softmax_output(Phi_hd, W, b, c1, c2)
#     is_c1_matrix, prob_c1_matrix, prob_c2_matrix = get_prob_c1_matrix(softmax, x_1, x_2)
#
#     return is_c1_matrix, x_1, x_2, prob_c1_matrix, prob_c2_matrix


def plot_PCA_pairwise(train_accuracies, n_components, c1, c2, head_path=header, f1=0, f2=1, n_neighbors=5, grids=200):
    print('in unflipping function')
    print(train_accuracies)

    for i, acc in enumerate(train_accuracies):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@features for ', acc)

        if acc == 'final':
            features = np.load(header + final_file_header + 'features.npy')
            outputs = np.load(header + final_file_header + 'outputs.npy')
            W = np.load(header + final_file_header + 'W.npy')
            b = np.load(header + final_file_header + 'b.npy')
        else:
            features = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')
            outputs = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_outputs.npy')
            W = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_W.npy')
            b = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_b.npy')

        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
        print('c1 and c2 combined featurte.shape=', c1_c2_features.shape)

        # PCA on the embedding features
        pca = PCA(n_components=n_components)
        pca.fit(c1_c2_features)
        print('pca.components_=', pca.components_)
        print('pca.explained_variance_=', pca.explained_variance_)
        features_PCA = pca.transform(c1_c2_features)
        print('features_PCA:', features_PCA.shape)
        # features_back = pca.inverse_transform(features_PCA)
        # print('pca transform_back features:', features_back.shape)
        # print('@@@@@@low-rank appr err:', np.linalg.norm(features_back-c1_c2_features))

        plt.subplot(2, 3, i+1)
        plt.title(str(acc))

        #plots c1 and c2 objects in the PCA space
        plt.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        plt.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], color='r', marker='+', label=classes[c2]+'s', alpha=0.6)

        #use nearest sample neighbors for any point in the PCA(2) space
        lows = [-8, -5.5]
        highs = [10.0, 7.]
        plt.xlim([lows[0], highs[0]])
        plt.ylim([lows[1], highs[1]])
        is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2,
                                                                           lows=lows, highs=highs, f1=0, f2=1, n_neighbors=n_neighbors, grids=grids)
        extent = [real_x[0], real_x[-1], real_y[0], real_y[-1]]
        plt.imshow(prob_c1.T, extent=extent)#why transpose?

        # plt.imshow(prob_c2.T, extent=extent)
        # plt.colorbar()#why some prob_cat is so low even for regions near cats: because PCA is only two features. PCA does not consider approximating Y.
        # plt.clim(0., 1.0)


# Figure 1 in the paper
# plot_PCA_pairwise(train_accuracies, n_components=2, c1=3, c2=5, n_neighbors=5)#this shows two components can still explaine prob(CAT) and prob(DOG): it's just the weights of components were not good before
# plt.show()
# sys.exit(1)


def plot_PCA_pairwise_flip(train_accuracies, n_components, c1, c2, head_path=header, f1=0, f2=1, n_neighbors=5, grids=200):
    '''
    flip plot_PCA_pairwise in such a way that cats are always on the left side of the panel
    '''
    print('In flipping function')
    print(train_accuracies)

    for i, acc in enumerate(train_accuracies):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@features for ', acc)

        if acc == 'final':
            features = np.load(header + final_file_header + 'features.npy')
            outputs = np.load(header + final_file_header + 'outputs.npy')
            W = np.load(header + final_file_header + 'W.npy')
            b = np.load(header + final_file_header + 'b.npy')
        else:
            features = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')
            outputs = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_outputs.npy')
            W = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_W.npy')
            b = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_b.npy')

        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
        print('c1 and c2 combined feature.shape=', c1_c2_features.shape)

        # PCA on the embedding features
        pca = PCA(n_components=n_components)
        pca.fit(c1_c2_features)
        print('pca.components_=', pca.components_)
        print('pca.explained_variance_=', pca.explained_variance_)
        features_PCA = pca.transform(c1_c2_features)
        print('features_PCA:', features_PCA.shape)
        # features_back = pca.inverse_transform(features_PCA)
        # print('pca transform_back features:', features_back.shape)
        # print('@@@@@@low-rank appr err:', np.linalg.norm(features_back-c1_c2_features))

        c1_features = features_PCA[:5000, :]
        c2_features = features_PCA[5000:10000, :]
        c1_center = c1_features.mean(axis=0)
        c2_center = c2_features.mean(axis=0)
        print('c1_center.shape=', c1_center.shape)
        print('c2_center.shape=', c2_center.shape)

        if c1_center[0] >= c2_center[0]:
            # c1_center *= -1
            # c2_center *= -1
            # c1_features *= -1
            # c2_features *= -1
            c1_features[0] *= -1
            c2_features[0] *= -1
            features_PCA = np.concatenate((c1_features, c2_features), axis=0)
            print('the flipped features_PCA.shape=', features_PCA.shape)

        plt.subplot(2, 3, i+1)
        plt.title(str(acc))

        #plots c1 and c2 objects in the PCA space
        plt.scatter(c1_features[:, f1], c1_features[:, f2], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        plt.scatter(c2_features[:, f1], c2_features[:, f2], color='r', marker='+', label=classes[c2]+'s', alpha=0.6)

        #use nearest sample neighbors for any point in the PCA(2) space
        lows = [-8, -5.5]
        highs = [10.0, 7.]
        plt.xlim([lows[0], highs[0]])
        plt.ylim([lows[1], highs[1]])
        is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2, lows=lows, highs=highs, f1=0, f2=1,
                                                                           n_neighbors=n_neighbors, grids=grids)
        extent = [real_x[0], real_x[-1], real_y[0], real_y[-1]]
        plt.imshow(prob_c1.T, extent=extent)#why transpose?

        # plt.imshow(prob_c2.T, extent=extent)
        # plt.colorbar()#why some prob_cat is so low even for regions near cats: because PCA is only two features. PCA does not consider approximating Y.
        # plt.clim(0., 1.0)

#unflipped
# plt.figure()
# plot_PCA_pairwise(train_accuracies, n_components=2, c1=3, c2=5, n_neighbors=5, grids=50)
# #flipped
# plt.figure()
# plot_PCA_pairwise_flip(train_accuracies, n_components=2, c1=3, c2=5, n_neighbors=5, grids=50)
# plt.show()
# sys.exit(1)

def plot_PCA_one_vs_allothers(acc, n_components, c1, f1, f2, colors, markers, path_head=header, n_neighbors=5, decay=0.5):
    '''
    use one model specified by acc to plot a class c1's boundary against all others
    Note: very important. PCA are applied pairwise, instead of fixed.
    '''
    print('@@@@@n_neighbors=', n_neighbors)
    features = np.load(path_head + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')
    # features = np.load(path_head + 'model_dla_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')
    W = np.load(path_head + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_W.npy')
    b = np.load(path_head + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_b.npy')

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
        plt.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], color=colors[counter - 1], marker=markers[counter - 1], label=classes[c2] + 's', alpha=0.6)
        plt.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2],           color='b', marker='o', label=classes[c1]+'s', alpha=0.6)

        # plt.legend(loc='best', fontsize=14)

        lows = [-8, -5.]
        highs = [8., 6.5]
        plt.xlim([lows[0], highs[0]])
        plt.ylim([lows[1], highs[1]])

        is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2, lows=lows, highs=highs, f1=f1, f2=f2, n_neighbors=n_neighbors, decay=decay)
        # entropy_matrix, real_x, real_y = nearest_neighbor_entropy(features_PCA, c1_c2_features, pca, W, b, c1, c2, f1=0, f2=1, n_neighbors=10)

        extent = [real_x[0], real_x[-1], real_y[0], real_y[-1]]
        # plt.imshow(is_c1_matrix.T, extent=extent)
        plt.imshow(prob_c1.T, extent=extent)  # why transpose?
        # plt.imshow(entropy_matrix.T, extent=extent)  # why transpose?



#cats vs. all other classes
colors   =['g', 'k', 'tab:pink', 'm', 'r', 'c', 'tab:brown', 'tab:orange', 'tab:purple']#9 colors
markers = ['s', 'p', '*', '<', '+', 'v', 'x', 'h', '1']
# Figure 3
# plot_PCA_one_vs_allothers('99.87', n_components=2, c1=3, f1=0, f2=1, colors=colors, markers=markers, n_neighbors=5, decay=0.3)

# plt.show()
# sys.exit(1)


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

#figure 4: 3d
triples = [[3, 5, 0], [3, 5, 6], [3, 5, 1],
            [1, 9, 3], [1, 9, 0], [1, 9, 8]]
colors  =['g', 'k', 'tab:pink', 'b', 'm', 'r', 'c', 'tab:brown', 'tab:orange', 'tab:purple']#only adds the cat for the above
markers = ['s', 'p', '*', 'o', '<', '+', 'v', 'x', 'h', '1']
# plot_PCA_tripplewise('99.45', triples, n_components=3, colors=colors, markers=markers)
# plt.show()

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

def plot_prob_output_file(output_file, c1=3, c2=5, header=header):
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
    plt.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], color=colors[c2], marker=markers[c2], label=classes[c2]+'s', alpha=0.2)
    # plt.legend()

    #for 99.9 model
    # plt.xlim([-6, 7.])
    # plt.ylim([-3, 4.5])

    #99.5% model
    plt.xlim([-7, 7.])
    plt.ylim([-5., 6.5])



# small lr
# plt.figure()
# plt.subplot(2, 2, 1)
# plt.title('small-lr')
# plot_prob_output_file(output_file='model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_outputs.npy', c1=3, c2=5)
# plt.subplot(2, 2, 2)
# plt.title('small-lr')
# plot_PCA_pairwise_file(feature_file='model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_features.npy', n_components=2, c1=3, c2=5, f1=0, f2=1)
#
# #big lr
# plt.subplot(2, 2, 3)
# plt.title('big-lr')
# plot_prob_output_file(output_file='model_vgg19_alpha_0.01_lrmode_constantbatchsize_128_momentum_decayed_testacc_90.12.pyc_outputs.npy', c1=3, c2=5)
# plt.subplot(2, 2, 4)
# plt.title('big-lr')
# plot_PCA_pairwise_file(feature_file='model_vgg19_alpha_0.01_lrmode_constantbatchsize_128_momentum_decayed_testacc_90.12.pyc_features.npy', n_components=2, c1=3, c2=5, f1=0, f2=1)


# compare prob out in the decision space different classes: small-lr
# small_lr_outfile = 'model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_outputs.npy'
# plt.subplot(2, 2, 1)
# plt.title('small-lr')
# plot_prob_output_file(output_file=small_lr_outfile, c1=3, c2=5, header='results/')
# plt.subplot(2, 2, 2)
# plt.title('small-lr')
# plot_prob_output_file(output_file=small_lr_outfile, c1=3, c2=6, header='results/')
# plt.subplot(2, 2, 3)
# plt.title('small-lr')
# plot_prob_output_file(output_file=small_lr_outfile, c1=3, c2=0, header='results/')
# plt.subplot(2, 2, 4)
# plt.title('small-lr')
# plot_prob_output_file(output_file=small_lr_outfile, c1=3, c2=1, header='results/')
#
# plt.show()
# sys.exit(1)

def plot_PCA_one_vs_allothers_file(feature_file, n_components, c1, f1, f2, colors, markers, one_c2=-1, header=header):
    '''
    similar to plot_PCA_one_vs_allothers
    '''
    features = np.load(header + feature_file)
    print('features.shape=', features.shape)
    pca = PCA(n_components=n_components)
    counter = 0

    if one_c2 != -1:
        #only one c2
        c2_list = [one_c2]
    else:
        c2_list = range(len(classes))

    for c2 in c2_list:
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

        if one_c2 == -1:
            plt.subplot(3, 3, counter)

        # plt.scatter(features_PCA[:5000, f1], features_PCA[:5000, f2], color='b', marker='o', alpha=0.6)
        # plt.scatter(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], color=colors[c2], marker=markers[c2], label=classes[c2]+'s', alpha=0.6)
        # plt.legend(loc='best', fontsize=14)

        # plt.xlim([-25, 25])
        # plt.ylim([-20, 20])

        # plt.xlim([-5, 8])
        # plt.ylim([-8, 9])

        # if counter == 9:
        #     # plt.xticks([-5, 0, 5, 8])
        #     plt.xticks([-20, -10, 10, 20])
        # else:
        #     plt.xticks([])
        #
        # # plt.yticks([-8, 0, 9])
        # plt.yticks([-30, 0, 30])

        if one_c2 !=-1:
            return features_PCA


# Figure 5 Fig 5: compare optimizers
# fig=plt.figure()
# fig.suptitle('cats(blue circles) vs. all other classes: small lr 0.0001')

def plot_train_and_test(train_file, test_file, big_range=True, header=header):
    f_small_lr_final = plot_PCA_one_vs_allothers_file(feature_file=train_file, n_components=2, c1=3, f1=0,
                                                      f2=1, colors=colors, markers=markers, one_c2=5, header=header)
    plt.subplot(1, 2, 1)
    plt.title('training')
    plt.scatter(f_small_lr_final[:5000, 0], f_small_lr_final[:5000, 1], color='b', marker='o', alpha=0.6)
    plt.scatter(f_small_lr_final[5000:10000, 0], f_small_lr_final[5000:10000, 1], color='r', marker='*', label=classes[5] + 's', alpha=0.2)
    if big_range:
        plt.xlim([-24, 28])
        plt.ylim([-22, 25])
    # else:
    #     plt.xlim([-7, 8])
    #     plt.ylim([-4, 7])


    plt.subplot(1, 2, 2)
    plt.title('testing')
    f_small_lr_final_test = plot_PCA_one_vs_allothers_file(feature_file=test_file, n_components=2, c1=3, f1=0,
                                                           f2=1, colors=colors, markers=markers, one_c2=5, header=header)
    print('f_small_lr_final.shape=', f_small_lr_final.shape)
    print('f_small_lr_final_test.shape=', f_small_lr_final_test.shape)
    plt.scatter(f_small_lr_final_test[:1000, 0], f_small_lr_final_test[:1000, 1], color='b', marker='s', alpha=0.6)
    plt.scatter(f_small_lr_final_test[1000:2000, 0], f_small_lr_final_test[1000:2000, 1], color='r', marker='*', label=classes[5] + 's', alpha=0.2)
    if big_range:
        plt.xlim([-24, 28])
        plt.ylim([-22, 25])
    # else:
    #     plt.xlim([-7, 8])
    #     plt.ylim([-4, 7])

plt.figure(1)
plot_train_and_test(train_file='model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_features.npy',
                    test_file= 'model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_features_test.npy', header='results/Fig5/')


plt.figure(2)
plot_train_and_test(train_file='model_vgg19_alpha_0.01_lrmode_constantbatchsize_128_momentum_decayed_testacc_90.12.pyc_features.npy',
                    test_file='model_vgg19_alpha_0.01_lrmode_constantbatchsize_128_momentum_decayed_testacc_90.12.pyc_features_test.npy', big_range=False, header='results/Fig5/' )

plt.figure(3)
plot_train_and_test(train_file='model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.93.pyc_features.npy',
                    test_file='model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.93.pyc_features_test.npy', big_range=False, header='results/Fig5/')

plt.figure(4)
plot_train_and_test(train_file='model_vgg19adambatchsize_128_momentum_decayed_testacc_92.64.pyc_features.npy',
                    test_file='model_vgg19adambatchsize_128_momentum_decayed_testacc_92.64.pyc_features_test.npy', big_range=False, header='results/Fig5/')


def plot_train_and_predecessor(train_file, predecessor_file, big_range=True):
    f_small_lr_final = plot_PCA_one_vs_allothers_file(feature_file=train_file, n_components=2, c1=3, f1=0,
                                                      f2=1, colors=colors, markers=markers, one_c2=5)
    plt.subplot(1, 2, 1)
    plt.title('training, 99.5%-model', fontsize=24)
    plt.plot(f_small_lr_final[:5000, 0], f_small_lr_final[:5000, 1], 'bo', alpha=0.6, markersize=12)
    plt.plot(f_small_lr_final[5000:10000, 0], f_small_lr_final[5000:10000, 1], 'r*', label=classes[5] + 's', alpha=0.2, markersize=12)
    if big_range:
        plt.xlim([-24, 28])
        plt.ylim([-22, 25])
    # else:
    #     plt.xlim([-7, 8])
    #     plt.ylim([-4, 7])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.subplot(1, 2, 2)
    plt.title('training; 99.95%-model', fontsize=24)
    f_small_lr_final_test = plot_PCA_one_vs_allothers_file(feature_file=predecessor_file, n_components=2, c1=3, f1=0, f2=1, colors=colors, markers=markers, one_c2=5)
    print('f_small_lr_final.shape=', f_small_lr_final.shape)
    print('f_small_lr_final_test.shape=', f_small_lr_final_test.shape)
    plt.plot(f_small_lr_final_test[:5000, 0], f_small_lr_final_test[:5000, 1], 'bs', alpha=0.6, markersize=12)
    plt.plot(f_small_lr_final_test[5000:10000, 0], f_small_lr_final_test[5000:10000, 1], 'r*', label=classes[5] + 's', alpha=0.2, markersize=12)
    if big_range:
        plt.xlim([-24, 28])
        plt.ylim([-22, 25])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # else:
    #     plt.xlim([-7, 8])
    #     plt.ylim([-4, 7])

#for adam, add two predecessor models.
# plt.figure(5)
# plot_train_and_predecessor(train_file='model_vgg19adambatchsize_128_momentum_decayed_train_percentile99.53.pyc_features.npy',
#                     # predecessor_file='model_vgg19adambatchsize_128_momentum_decayed_train_percentile99.80.pyc_features.npy',
#                     predecessor_file='model_vgg19adambatchsize_128_momentum_decayed_train_percentile99.95.pyc_features.npy',
#                            big_range=False)

plt.show()
sys.exit(1)

#Figure 6 compares vgg, resnet18 and dla
# fig = plt.figure()
# vgg_feature_file = 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.45.pyc_features.npy'
# resnet_feature_file = 'model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.48.pyc_features.npy'
# dla_feature_file = 'model_dla_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.53.pyc_features.npy'
# # vgg_feature_file = 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.89.pyc_features.npy'
# # resnet_feature_file = 'model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.89.pyc_features.npy'
# # dla_feature_file = 'model_dla_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile99.86.pyc_features.npy'
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
