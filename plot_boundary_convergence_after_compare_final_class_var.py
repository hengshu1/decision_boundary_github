import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse
from main import classes
from generate_mean_and_variance import PCA_on_feature_matrix, SVD_on_feature_matrix
from scipy.stats import spearmanr, pearsonr

'''
compare VGG19 and Resnet18 in the top singular values, only compare the final models

Now compare class-wise variance
'''

def get_var_for_c1_vs_others(c1, features, mode):
    '''
    no interesting pattern observed.
    '''
    c1_vars, c2_vars = [], []
    # print('features.shape=', features.shape)
    for c2 in range(len(classes)):
        if c2 == c1:
            continue
        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)#10000, 512
        if mode == 'pca':
            features_PCA, singulars = PCA_on_feature_matrix(c1_c2_features, n_components=2)
        else:
            features_PCA, singulars, _ = SVD_on_feature_matrix(c1_c2_features, k=2)
        # print('features_PCA.shape=', features_PCA.shape)
        c1_center = features_PCA[:5000,:].mean(axis=0)
        print('c1_center=', c1_center)
        c2_center = features_PCA[5000:, :].mean(axis=0)
        print('c2_center=', c2_center)
        c1_var = features_PCA[:5000, :].std(axis=0)
        c2_var = features_PCA[5000:, :].std(axis=0)
        print('c1_var.shape=', c1_var.shape)
        c1_vars.append(c1_var)
        c2_vars.append(c2_var)
    return np.array(c1_vars), np.array(c2_vars)

def get_correl_for_c1_vs_others(c1, features, mode, corel='spear'):
    '''
    no interesting pattern observed.
    '''
    print('corel=', corel)
    c1_correlations, c2_correlations = [], []
    # print('features.shape=', features.shape)
    for c2 in range(len(classes)):
        if c2 == c1:
            continue
        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)#10000, 512
        if mode=='pca':
            features_PCA, singulars = PCA_on_feature_matrix(c1_c2_features, n_components=2)
        else:
            features_PCA, singulars, _ = SVD_on_feature_matrix(c1_c2_features, k=2)
        # print('features_PCA.shape=', features_PCA.shape)
        c1_center = features_PCA[:5000,:].mean(axis=0)
        print('c1_center=', c1_center)
        c2_center = features_PCA[5000:, :].mean(axis=0)
        print('c2_center=', c2_center)
        if corel == 'spear':
            c1_corel, _ = spearmanr(features_PCA[:5000, 0], features_PCA[:5000, 1])
            c2_corel, _ = spearmanr(features_PCA[5000:, 0], features_PCA[5000:, 1])
        elif corel == 'pearson':
            c1_corel, _ = pearsonr(features_PCA[:5000, 0], features_PCA[:5000, 1])
            c2_corel, _ = pearsonr(features_PCA[5000:, 0], features_PCA[5000:, 1])
        else:
            print('features_PCA.shape=', features_PCA.shape)
            c1_corel = np.cov(features_PCA[:5000, :].T)
            c2_corel = np.cov(features_PCA[5000:, :].T)
        print('c1_corel=', c1_corel.shape)
        print('c1_corel=', c2_corel.shape)
        c1_correlations.append(c1_corel)
        c2_correlations.append(c2_corel)
    return np.array(c1_correlations), np.array(c2_correlations)


def get_centerness_for_c1_vs_others(c1, features, mode):
    c1_vars, c2_vars = [], []
    # print('features.shape=', features.shape)
    for c2 in range(len(classes)):
        if c2 == c1:
            continue
        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)  # 10000, 512
        if mode == 'pca':
            features_PCA, singulars = PCA_on_feature_matrix(c1_c2_features, n_components=2)
        else:
            features_PCA, singulars, _ = SVD_on_feature_matrix(c1_c2_features, k=2)
        # print('features_PCA.shape=', features_PCA.shape)
        c1_center = features_PCA[:5000, :].mean(axis=0)
        print('c1_center=', c1_center)
        c2_center = features_PCA[5000:, :].mean(axis=0)
        print('c2_center=', c2_center)
        c1_var = ((features_PCA[:5000, :] - c1_center) ** 2).sum(axis=1).mean(axis=0) #mean squared distance, scalar
        c2_var = ((features_PCA[5000:, :] - c2_center) ** 2).sum(axis=1).mean(axis=0)
        c1_vars.append(c1_var)
        c2_vars.append(c2_var)
    return np.array(c1_vars), np.array(c2_vars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mode', default='pca', type=str, help='svd or pca')

    args = parser.parse_args()

    print('@@mode=', args.mode)

    vgg_file = 'results/run1_save_model_every_epoch_vgg19/model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.57.pyc_features.npy'
    vgg_features = np.load(vgg_file)
    print('@@vgg_features=', vgg_features.shape)
    # c1_vars_vgg, c2_vars_vgg = get_var_for_c1_vs_others(c1=3, features=vgg_features, mode=args.mode)
    # c1_vars_vgg, c2_vars_vgg = get_correl_for_c1_vs_others(c1=3, features=vgg_features, mode=args.mode)
    # c1_vars_vgg, c2_vars_vgg = get_correl_for_c1_vs_others(c1=3, features=vgg_features, mode=args.mode, corel='cov')
    c1_vars_vgg, c2_vars_vgg = get_centerness_for_c1_vs_others(c1=3, features=vgg_features, mode=args.mode)

    res_file = 'results/run1_save_model_every_epoch_resnet18/model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.75.pyc_features.npy'
    res_features = np.load(res_file)
    print('@@res_features=', res_features.shape)
    # c1_vars_res, c2_vars_res = get_correl_for_c1_vs_others(c1=3, features=res_features, mode=args.mode)
    # c1_vars_res, c2_vars_res = get_correl_for_c1_vs_others(c1=3, features=res_features, mode=args.mode, corel = 'cov')
    c1_vars_res, c2_vars_res = get_centerness_for_c1_vs_others(c1=3, features=res_features, mode=args.mode)

    dla_file = 'results/run1_save_model_every_epoch_dla/model_dla_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.05.pyc_features.npy'
    dla_features = np.load(dla_file)
    print('@@dla_features=', dla_features.shape)
    # c1_vars_dla, c2_vars_dla = get_correl_for_c1_vs_others(c1=3, features=dla_features, mode=args.mode)
    # c1_vars_dla, c2_vars_dla = get_correl_for_c1_vs_others(c1=3, features=dla_features, mode=args.mode, corel = 'cov')
    c1_vars_dla, c2_vars_dla = get_centerness_for_c1_vs_others(c1=3, features=dla_features, mode=args.mode)
    print('c1_vars_dla.shape=', c1_vars_dla.shape)
    print('c2_vars_dla.shape=', c2_vars_dla.shape)

    # plt.figure()
    # plt.plot(c1_vars_vgg[:, 0], '-b+', label='VGG19-c1')
    # plt.plot(c1_vars_res[:, 0], '-ro', label='ResNet18-c1')
    # plt.plot(c1_vars_dla[:, 0], '--k*', label='DLA-c1')
    # plt.figure()
    # plt.plot(c2_vars_vgg[:, 0], '-b+', label='VGG19-c2')
    # plt.plot(c2_vars_res[:, 0], '-ro', label='ResNet18-c2')
    # plt.plot(c2_vars_dla[:, 0], '--k*', label='DLA-c2')

    # plt.figure()
    # plt.plot(c1_vars_vgg, '-b+', label='VGG19')
    # plt.plot(c1_vars_res, '-ro', label='ResNet18')
    # plt.plot(c1_vars_dla, '--k*', label='DLA')

    #plot covariance between f1 and f2
    plt.figure()
    # plt.plot(c1_vars_vgg[:, 0, 1], '-b+', label='VGG19')
    # plt.plot(c1_vars_res[:, 0, 1], '-ro', label='ResNet18')
    # plt.plot(c1_vars_dla[:, 0, 1], '--k*', label='DLA')
    # plt.plot(c1_vars_vgg[:, 0, 0], '-b+', label='VGG19, var-f1')
    # plt.plot(c1_vars_res[:, 0, 0], '-ro', label='ResNet18, var-f1')
    # plt.plot(c1_vars_dla[:, 0, 0], '--k*', label='DLA, var-f1')
    # plt.plot(c1_vars_vgg[:, 1, 1], '-b+', label='VGG19, var-f2')
    # plt.plot(c1_vars_res[:, 1, 1], '-ro', label='ResNet18, var-f2')
    # plt.plot(c1_vars_dla[:, 1, 1], '--k*', label='DLA, var-f2')

    #centerness plot
    print('c1_vars_vgg.shape=', c1_vars_vgg.shape)
    # plt.plot(c1_vars_vgg, '-b+', label='VGG19 cat centerness')
    # plt.plot(c1_vars_res, '-ro', label='ResNet18, cat centerness')
    # plt.plot(c1_vars_dla, '--k*', label='DLA, cat centerness')

    print('c2_vars_vgg.shape=', c2_vars_vgg.shape)
    plt.plot(c2_vars_vgg, '-b+', label='VGG19 other centerness')
    plt.plot(c2_vars_res, '-ro', label='ResNet18, other centerness')
    plt.plot(c2_vars_dla, '--k*', label='DLA, other centerness')

    plt.legend(fontsize=14)
    plt.xlabel('class index', fontsize=14)
    # if args.mode == 'SVD':
    #     plt.ylabel('singular values', fontsize=14)
    # else:
    #     plt.ylabel(r'$\sigma_1^2$', fontsize=14)
    plt.ylabel('correlation')
    classes_no_cat = list(classes)
    print('classes_no_cat=', classes_no_cat)
    del classes_no_cat[3]
    print('classes_no_cat=', classes_no_cat)
    print(len(classes))
    print(len(classes_no_cat))
    plt.xticks(range(9), classes_no_cat)
    plt.show()