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
import glob, argparse


def PCA_on_feature_matrix(c1_c2_features, n_components):
    # PCA on the embedding features
    pca = PCA(n_components=n_components)
    pca.fit(c1_c2_features)
    # print('pca.components_=', pca.components_)
    # print('pca.explained_variance_=', pca.explained_variance_)
    features_PCA = pca.transform(c1_c2_features)
    # print('features_PCA:', features_PCA.shape)
    return features_PCA, pca.explained_variance_

def SVD_on_feature_matrix(c1_c2_features, k=2):
    '''
    low rank approximation with SVD
    https://scikit-learn.org/stable/modules/decomposition.html#:~:text=TruncatedSVD%20is%20very%20similar%20to,matrix%20is%20equivalent%20to%20PCA.
    Note, if we substract the mean, then it will change the auto-correlation matrix to the covariance matrix:
    https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#covariance-matrix
    '''

    #todo: original code has minus mean; check if results the same when doing so
    # the link above says truncated SVD with mean substracted is the same as PCA: we need to double check this.
    # U, S, VT = np.linalg.svd(c1_c2_features - c1_c2_features.mean(0), full_matrices=False)
    U, S, VT = np.linalg.svd(c1_c2_features, full_matrices=False)

    # ATA = c1_c2_features.T.dot(c1_c2_features)
    # print('ATA.shape=', ATA.shape)
    # S2, eigv = np.linalg.eig(ATA)
    #the two should be the same: YES. This means the auto-correlation matrix are very important.
    # print(S[:3]**2)
    # print(S2[:3])

    # print('VT.shape=', VT.shape)
    # print('VT[:k, :].shape=', VT[:k, :].shape)
    #U, VT = svd_flip(U, VT)#todo: need to import this
    #print('difference:', np.linalg.norm(VT[:n_components] - pca.components_))
    #np.testing.assert_array_almost_equal(VT[:n_components], pca.components_)#not hold probably because didn't do svd_flip

    features_svd = np.dot(c1_c2_features, VT[:k, :].T)
    # print('features_svd.shape=', features_svd.shape)
    # print('S.shape=', S.shape)
    return features_svd, S, VT #note the full eigvalues are returned

def plot_PCA_pairwise_mean(train_accuracies, n_components, c1, c2, header, f1=0, f2=1):

    centers_c1, centers_c2 = [], []
    singular_values = []
    for i, acc in enumerate(train_accuracies):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@features for ', acc)

        if acc == 'final':
            # features = np.load(header + 'model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.39.pyc_features.npy')#run1_resnet18
            features = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.99.pyc_features.npy')#run 3
            # features = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.97.pyc_features.npy')#run4
            # features = np.load(header + 'model_vgg19adambatchsize_128_momentum_decayed_testacc_92.64.pyc_features.npy')#run1_vgg_adam
            # outputs = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.31.pyc_outputs.npy')
            # W = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.31.pyc_W.npy')
            # b = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.31.pyc_b.npy')
            # features = np.load(head_path + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.80.pyc_features.npy')
        else:
            features = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')#vgg19 sgd
            # features = np.load(header +'model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')
            # features = np.load(header + 'model_vgg19adambatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_features.npy')#run1_vgg_adam
            # outputs = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_outputs.npy')
            W = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_W.npy')
            # b = np.load(header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_train_percentile' + acc + '.pyc_b.npy')

        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
        print('c1 and c2 combined featurte.shape=', c1_c2_features.shape)

        # features = PCA_on_feature_matrix(c1_c2_features, n_components)
        features, singulars = SVD_on_feature_matrix(c1_c2_features, k=2)
        singular_values.append(singulars)

        # features_back = pca.inverse_transform(features_PCA)
        # print('pca transform_back features:', features_back.shape)
        # print('@@@@@@low-rank appr err:', np.linalg.norm(features_back-c1_c2_features))

        #plots c1 and c2 objects in the PCA space
        c1_features = features[:5000, :]
        c2_features = features[5000:10000, :]
        print('c1_features.shape=', c1_features.shape)
        print('c2_features.shape=', c2_features.shape)

        c1_center = c1_features.mean(axis=0)
        c2_center = c2_features.mean(axis=0)
        print('c1_center.shape=', c1_center.shape)
        print('c2_center.shape=', c2_center.shape)

        # switch sign for the principle components:
        if c1_center[0] >= c2_center[0]:
            #todo: which is the right?
            c1_center *= -1
            c2_center *= -1
            # c1_center[0] *= -1
            # c2_center[0] *= -1

        centers_c1.append(c1_center)
        centers_c2.append(c2_center)

    return np.array(centers_c1), np.array(centers_c2), np.array(singular_values)

#double check these files were generated by the same process
# train_accuracies = ['71.09', '74.54', '81.25', '85.16', '90.62', '96.09', '99.48', '99.84', '99.87', '99.97', 'final']#todo: I suspect these models are not generated in the same run
# train_accuracies = [ '99.87', '99.97', 'final']
# train_accuracies = ['71.09', '74.54', '81.25', '85.16', '90.62', '96.09', '99.48', '99.84', 'final']

#run 3
# train_accuracies = ['8.59',  '19.02',  '30.47',
#                     '37.21', '39.84', '45.70', '49.50', '55.08',
#                     '60.94', '64.58', '69.53', '75.00', '80.86',
#                     '85.94', '90.23', '94.53', '99.48', '99.84',
#                     '99.87', '99.97', 'final']
#run 4
# train_accuracies = ['10.94',  '14.04',  '19.53',
#                     '24.22', '31.25', '37.11', '39.59', '46.09',
#                     '49.57', '54.52', '60.55', '64.53', '71.09',
#                     '75.47', '80.47', '85.16', '91.67', '92.58',
#                     '94.53', '96.09', '96.61', '98.44', '99.22',
#                     '99.48', '99.80', '99.86', '99.95', '99.97',
#                     'final']

#run1_vgg_adam
# train_accuracies = ['9.38',  '14.22',  '19.05',
#                     '24.03', '41.02', '44.55', '49.52', '59.77',
#                     '66.15', '69.54', '79.69', '84.90', '90.31',
#                     '92.97', '94.53', '96.09', '96.88', '97.66',
#                     '99.22', '99.48', '99.80', '99.87', '99.95',
#                     '99.97', 'final']



def compute_center_and_variances(map_epoch_to_file, n_components, c1, c2, f1=0, f2=1, mode='svd'):
    centers_c1, centers_c2 = [], []
    singular_values = []
    print('mode=', mode)
    for k in sorted(map_epoch_to_file.keys()):
        # print('k=', k, '-->', map_epoch_to_file[k])
        features = np.load(map_epoch_to_file[k])
        # print('features.shape=', features.shape)

        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
        # print('c1 and c2 combined featurte.shape=', c1_c2_features.shape)

        if mode == 'svd':
            features, singulars = SVD_on_feature_matrix(c1_c2_features, k=n_components)
        elif mode == 'pca':
            features, singulars = PCA_on_feature_matrix(c1_c2_features, n_components)
        else:
            print('not supported')
            sys.exit(1)

        print('singulars.shape=', singulars.shape)
        singular_values.append(singulars)

        # features_back = pca.inverse_transform(features_PCA)
        # print('pca transform_back features:', features_back.shape)
        # print('@@@@@@low-rank appr err:', np.linalg.norm(features_back-c1_c2_features))

        #plots c1 and c2 objects in the PCA space
        c1_features = features[:5000, :]
        c2_features = features[5000:10000, :]
        # print('c1_features.shape=', c1_features.shape)
        # print('c2_features.shape=', c2_features.shape)

        c1_center = c1_features.mean(axis=0)
        c2_center = c2_features.mean(axis=0)
        # print('c1_center.shape=', c1_center.shape)
        # print('c2_center.shape=', c2_center.shape)

        # switch sign for the principle components:
        if c1_center[0] >= c2_center[0]:
            #todo: which is the right?
            c1_center *= -1
            c2_center *= -1
            # c1_center[0] *= -1
            # c2_center[0] *= -1

        centers_c1.append(c1_center)
        centers_c2.append(c2_center)

    return np.array(centers_c1), np.array(centers_c2), np.array(singular_values)

def parse_files_and_sort(args, file_type='features'):

    if args.optimizer == 'sgd':
        final_model = glob.glob((args.saved_dir + 'model_{}_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_*.pyc*' + file_type + '.npy').format(args.model))
        len_header = len(header + '/model_{}_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_model_at_epoch_'.format(args.model))
    else:
        file_name_pattern = 'model_*batchsize_128_momentum_decayed_testacc_*' + file_type + '.npy'
        print('file_name_pattern=', file_name_pattern)
        print('final name full pattern:', args.saved_dir + file_name_pattern)
        final_model = glob.glob((args.saved_dir + file_name_pattern).format(args.model))

        #header pattern is, model_resnet18_adagrad_batchsize_128_at_epoch_60
        len_header = len(args.saved_dir + '/model_{}_{}_batchsize_128_at_epoch'.format(args.model, args.optimizer))


    print('final_model is ', final_model)
    assert len(final_model) == 1
    final_model = final_model[0]
    print('final_model is ', final_model)

    model_list = glob.glob(args.saved_dir + '/*.pyc_' + file_type + '.npy')
    print('len(print(model_list))=', len(model_list))
    files = model_list

    count1, count2, count3 = 0, 0, 0
    map = {}
    for f in files:
        if f == final_model:
            print('encounter final model; skip')
            continue

        print(f)
        # if 'epoch_0_acc' in f:
        #     print('initial model encountered; skip')
        #     continue

        epoch_str = f[len_header:len_header + 3]
        print('epoch_str=', epoch_str)
        if epoch_str.endswith('_'):  # two digit
            epoch_str = epoch_str[:-1]
            count2 += 1
        elif epoch_str.endswith('_a'):  # one digit
            epoch_str = epoch_str[:-2]
            count1 += 1
        else:
            count3 += 1
        print('epoch_str=', epoch_str)
        map[int(epoch_str)] = f

    assert len(map.keys()) == 200  # total epochs.
    assert count1 == 11  # one digit: 0 -9; but epoch 0: both before epoch (initial model) and after epoch 0 are saved
    assert count2 == 90  # two digits: 10-99
    assert count3 == 100  # three digits: 100--199
    print('done passing map for ' +file_type)

    return map


if __name__ == "__main__":
    '''
    1. Need to first run output_space.py to get the features 
    2. then run generate_mean_and_variance.py to generate the centers and singulvar values during training
    3. then plot using generate_fig7.py
    '''

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mode', default='pca', type=str, help='svd or pca')
    parser.add_argument('--n_components', default=10, type=int, help='number of components or low-rank number')

    parser.add_argument('--model', default='resnet18', type=str, help='model name')
    parser.add_argument('--optimizer', default='asgd', type=str, help='optimizer')
    parser.add_argument('--saved_dir', default='results/run2_save_model_every_epoch_resnet18_ASGD/', type=str, help='svd or pca')

    args = parser.parse_args()
    print('model=', args.model)
    print('mode=', args.mode)
    print('n_components=', args.n_components)
    print('saved_dir=', args.saved_dir)

    map = parse_files_and_sort(args)

    c1_centers, c2_centers, singulars = compute_center_and_variances(map, n_components=512, c1=3, c2=5, mode=args.mode)

    np.save(args.saved_dir +'c1_centers_{}_{}comp_{}_{}.npy'.format(args.mode, args.n_components, args.model, args.optimizer), c1_centers)
    np.save(args.saved_dir +'c2_centers_{}_{}comp_{}_{}.npy'.format(args.mode, args.n_components, args.model, args.optimizer), c2_centers)
    np.save(args.saved_dir +'singulars_{}_{}comp_{}_{}.npy'.format(args.mode, args.n_components, args.model, args.optimizer), singulars)

    # c1_centers, c2_centers, singulars = plot_PCA_pairwise_mean_from_map(map, n_components=10, c1=3, c2=5, mode='pca')
    # np.save(saved_dir +'c1_centers_pca_10comp.npy', c1_centers)
    # np.save(saved_dir +'c2_centers_pca_10comp.npy', c2_centers)
    # np.save(saved_dir +'singulars_pca_10comp.npy', singulars)

    sys.exit(1)
    #run1_resnet18: same observations
    # train_accuracies = ['8.98', '14.03', '19.00', '24.05', '29.51',
    #                     '34.77', '40.49', '44.54', '54.84', '61.46',
    #                     '64.54', '70.02', '75.78', '80.47', '85.16',
    #                     '90.62', '93.75', '94.53', '96.09', '96.88',
    #                     '97.66', '99.22', '99.48', '99.80', '99.87',
    #                     '99.97', 'final'
    #                     ]
    centers_c1, centers_c2, singular_values = plot_PCA_pairwise_mean(train_accuracies, n_components=2, c1=3, c2=5, header=args.saved_dir)
    print('centers_c1.shape=', centers_c1.shape)
    print('centers_c2.shape=', centers_c2.shape)
    print('singular_values=', singular_values.shape)
    fig = plt.figure()
    print('centers_c1=', centers_c1)
    print('centers_c2=', centers_c2)

    #no arrow plot
    ax = fig.add_subplot(111)

    #for plotting in the pca space
    #plot with arrow between successive points
    # for i in range(centers_c1.shape[0]-1):
    #     # plt.arrow(centers_c1[i, 0], centers_c1[i, 1],
    #     #           centers_c1[i+1, 0] - centers_c1[i+1, 0],
    #     #           centers_c1[i + 1, 1] - centers_c1[i + 1, 1],
    #     #           head_width=3, length_includes_head=True, color='b')
    #     #
    #     # plt.arrow(centers_c2[i, 0], centers_c2[i, 1],
    #     #           centers_c2[i+1, 0] - centers_c2[i+1, 0],
    #     #           centers_c2[i + 1, 1] - centers_c2[i + 1, 1],
    #     #           head_width=3, length_includes_head=True, color='r')
    #
    #     ax.annotate('', xy=(centers_c1[i+1, 0], centers_c1[i+1, 1]),
    #                 xycoords='data',
    #                 xytext=(centers_c1[i, 0], centers_c1[i, 1]),#source
    #                 textcoords='data',
    #                 arrowprops=dict(#arrowstyle='->',
    #                                 color='blue',
    #                                 width=0.5,
    #                                 headwidth = 7.,
    #                                 headlength = 7.,
    #                                 )
    #                 )
    #
    #
    #     ax.annotate('', xy=(centers_c2[i+1, 0], centers_c2[i+1, 1]),
    #                 xycoords='data',
    #                 xytext=(centers_c2[i, 0], centers_c2[i, 1]),#source
    #                 textcoords='data',
    #                 arrowprops=dict(#arrowstyle='->',
    #                                 color='red',
    #                                 width=0.5,
    #                                 headwidth = 7.,
    #                                 headlength = 7.,
    #                                 )
    #                 )

    #for plotting in the svd(2) space
    for i in range(centers_c1.shape[0]-1):

        if i<5:
            color_c1 = 'cyan'
        elif i<10:
            color_c1 = 'blue'
        elif i < 15:
            color_c1 = 'brown'
        elif i< 20:
            color_c1 = 'black'
        else:
            color_c1 = 'red'

        ax.annotate('', xy=(centers_c1[i+1, 0], centers_c1[i+1, 1]),
                    xycoords='data',
                    xytext=(centers_c1[i, 0], centers_c1[i, 1]),#source
                    textcoords='data',
                    arrowprops=dict(#arrowstyle='->',
                                    color=color_c1,
                                    width=0.5,
                                    headwidth = 7.,
                                    headlength = 7.,
                                    )
                    )


        ax.annotate('', xy=(centers_c2[i+1, 0], centers_c2[i+1, 1]),
                    xycoords='data',
                    xytext=(centers_c2[i, 0], centers_c2[i, 1]),#source
                    textcoords='data',
                    arrowprops=dict(#arrowstyle='simple',
                                    color=color_c1,
                                    #width=0.5,
                                    headwidth = 10.,
                                    headlength = 9.
                                    )
                    )

    plt.plot(centers_c1[:, 0], centers_c1[:, 1], 'bo', fillstyle='none', label='CAT center')
    plt.plot(centers_c2[:, 0], centers_c2[:, 1], 'r+', label='DOG center', markersize=12)
    plt.legend(fontsize=16, labelcolor='linecolor', loc='best')

    #run 3
    pos_x, pos_y = [], []
    points_x = [centers_c1[6, 0], centers_c2[6, 0]]
    points_y = [centers_c1[6, 1], centers_c2[6, 1]]
    pos_x.append(points_x)
    pos_y.append(points_y)
    points_x = [centers_c1[11, 0], centers_c2[11, 0]]
    points_y = [centers_c1[11, 1], centers_c2[11, 1]]
    pos_x.append(points_x)
    pos_y.append(points_y)
    points_x = [centers_c1[12, 0], centers_c2[12, 0]]
    points_y = [centers_c1[12, 1], centers_c2[12, 1]]
    pos_x.append(points_x)
    pos_y.append(points_y)
    points_x = [centers_c1[-6, 0], centers_c2[-6, 0]]
    points_y = [centers_c1[-6, 1], centers_c2[-6, 1]]
    pos_x.append(points_x)
    pos_y.append(points_y)
    points_x = [centers_c1[-3, 0], centers_c2[-3, 0]]
    points_y = [centers_c1[-3, 1], centers_c2[-3, 1]]
    pos_x.append(points_x)
    pos_y.append(points_y)

    # for points_x, points_y in zip(pos_x, pos_y):
    #     plt.plot(points_x, points_y, '--k')
    #     mean_x = np.array(points_x).mean()
    #     mean_y = np.array(points_y).mean()
    #     print('({}, {})'.format(mean_x, mean_y))

    #plot singular values
    plt.figure()
    plt.plot(singular_values[:, 0], '-k+', label='singular values correponds to f1')
    plt.plot(singular_values[:, 1], '--ro', label='singular values correponds to f2')
    plt.legend()

    plt.show()

