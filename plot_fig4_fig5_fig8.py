import numpy as np
import matplotlib.pyplot as plt
import argparse, sys
from main import classes
from generate_mean_and_variance import PCA_on_feature_matrix, SVD_on_feature_matrix


'''
compare VGG19 and Resnet18 in the top singular values, only compare the final models

PCA and SVD have similar results
'''

#just for the x-ticks
classes_no_cat = list(classes)
print('classes_no_cat=', classes_no_cat)
del classes_no_cat[3]
print('classes_no_cat=', classes_no_cat)

def get_sigmas_for_c1_vs_others(c1, features, mode, full_spectrum=True):
    print('mode=', mode)
    sigma1s = []
    # print('features.shape=', features.shape)
    for c2 in range(len(classes)):
        if c2 == c1:
            continue
        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
        # print('c1_c2_features.shape=', c1_c2_features.shape)
        if mode=='pca':
            features_PCA, singulars = PCA_on_feature_matrix(c1_c2_features, n_components=512)
        else:
            features_PCA, singulars, _ = SVD_on_feature_matrix(c1_c2_features)
        if full_spectrum:
            sigma1s.append(singulars)
        else:
            sigma1s.append(singulars[0]/singulars.sum()) #using explained ratio of f1
    sigma1s = np.array(sigma1s)
    print('sigma1s.shape=', sigma1s.shape)
    return sigma1s

def get_cond_for_c1_vs_others(c1, features, mode='pca', cond_or_rank='rank', matrix='partial'):
    '''
    if matrix is full, it computes the full (512 by 512) autocorrelation matrix; otherwise, it computes the 2 by 2 matrix in the PCA(2) space
    '''
    print('mode=', mode)
    print('cond_or_rank=', cond_or_rank)
    print('matrix=', matrix)
    sigma1s = []
    # print('features.shape=', features.shape)
    for c2 in range(len(classes)):
        if c2 == c1:
            continue
        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
        print('c1_c2_features.shape=', c1_c2_features.shape)
        if mode =='pca':
            if matrix == 'full':
                features_lowd, singulars = PCA_on_feature_matrix(c1_c2_features, n_components=512)
            else:
                features_lowd, singulars = PCA_on_feature_matrix(c1_c2_features, n_components=2)
        else:
            features_lowd, singulars, _ = SVD_on_feature_matrix(c1_c2_features, k=2)
        # features_PCA, singulars, _ = SVD_on_feature_matrix(c1_c2_features, k=100)
        # sigma1s.append(singulars[0])

        if cond_or_rank == 'cond':
            if matrix == 'full':
                print('singulars.shape=', singulars.shape)
                sigma1s.append(singulars[0]/singulars[-1]) #condition number of the full matrix
            else:
                BTB = features_lowd.T.dot(features_lowd)  # 2 by 2
                singular_B, _ = np.linalg.eig(BTB)
                print('cond(ATA)=', singular_B[0]/singular_B[-1])
                print('cond(ATA)=', singulars[0]/singulars[1])#second way. Yes. exactly the same. So cond in 2d space, doesn't make sense. it's just the ratio of lambda1/lambda2
                sigma1s.append(singular_B[0]/singular_B[-1])
        else:
            if matrix == 'full':
                sigma1s.append(np.linalg.matrix_rank(c1_c2_features))  # rank
                # sigma1s.append(np.linalg.matrix_rank(c1_c2_features.T.dot(c1_c2_features))) #is it the same? yes.
            else:
                sigma1s.append(np.linalg.matrix_rank(features_lowd)) #must be 2: only 2 by 2 matrix
        # sigma1s.append(np.linalg.cond(c1_c2_features))
        # print('cond1:', np.linalg.cond(c1_c2_features))
        # print('cond2:', singulars[0]/singulars[-1])
    return np.array(sigma1s)


def plot_vgg_res_dla(args):
    '''
        plot the comparison between vgg, resnet, and dla
    '''
    vgg_file = 'results/run1_save_model_every_epoch_vgg19/model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.57.pyc_features.npy'
    vgg_features = np.load(vgg_file)
    print('vgg_features=', vgg_features.shape)
    sigma1_vgg = get_sigmas_for_c1_vs_others(c1=3, features=vgg_features, mode=args.mode)
    sigma1_vgg_full = get_sigmas_for_c1_vs_others(c1=3, features=vgg_features, mode=args.mode, full_spectrum=True)
    sigma1_vgg_cond = get_cond_for_c1_vs_others(c1=3, features=vgg_features, cond_or_rank=args.cond_or_rank)

    res_file = 'results/run1_save_model_every_epoch_resnet18/model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.75.pyc_features.npy'
    res_features = np.load(res_file)
    print('res_features=', res_features.shape)
    sigma1_res = get_sigmas_for_c1_vs_others(c1=3, features=res_features, mode=args.mode)
    sigma1_res_full = get_sigmas_for_c1_vs_others(c1=3, features=res_features, mode=args.mode, full_spectrum=True)
    sigma1_res_cond = get_cond_for_c1_vs_others(c1=3, features=res_features, cond_or_rank=args.cond_or_rank)

    dla_file = 'results/run1_save_model_every_epoch_dla/model_dla_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.05.pyc_features.npy'
    dla_features = np.load(dla_file)
    print('dla_features=', dla_features.shape)
    sigma1_dla = get_sigmas_for_c1_vs_others(c1=3, features=dla_features, mode=args.mode)
    sigma1_dla_full = get_sigmas_for_c1_vs_others(c1=3, features=dla_features, mode=args.mode, full_spectrum=True)
    sigma1_dla_cond = get_cond_for_c1_vs_others(c1=3, features=dla_features, cond_or_rank=args.cond_or_rank)

    plt.figure()
    plt.bar(range(9), sigma1_vgg, color='k', width=0.5, label='VGG19')
    plt.bar(np.array(range(9)) + 0.2, sigma1_dla, color='b', width=0.5, label='DLA')
    plt.bar(np.array(range(9)) + 0.4, sigma1_res, color='r', width=0.5, label='ResNet18')

    plt.legend(fontsize=14)
    if args.mode == 'SVD':
        plt.ylabel('singular values', fontsize=14)
    else:
        plt.ylabel(r'$\sigma_1^2/\Sigma \sigma_i^2$', fontsize=14)

    plt.xticks(range(len(classes_no_cat)), classes_no_cat, fontsize=14)
    plt.yticks(fontsize=12)

    if args.mode == 'pca':
        plt.ylim([0.9, 1.])
    else:
        plt.ylim([0.2, 0.5])

    plt.legend(fontsize=14)

    # plotting full-spectrum
    plt.figure()
    plt.title('largest singular value')
    print('sigma1_dla_full.shape.shape=', sigma1_dla_full.shape)
    plt.plot(sigma1_vgg_full[:, 0], '-k+', label=r'vgg-$\sigma1$')
    plt.plot(sigma1_dla_full[:, 0], '-bo', label=r'dla-$\sigma1$')
    plt.plot(sigma1_res_full[:, 0], '-rx', label=r'res-$\sigma1$')

    # plt.plot(sigma1_vgg_full[:, 1], '--k+', label='vgg-sigma2')
    # plt.plot(sigma1_dla_full[:, 1], '--bo', label='dla-sigma2')
    # plt.plot(sigma1_res_full[:, 1], '--rx', label='res-sigma2')
    # plt.legend(ncol=2)
    plt.legend()
    plt.xticks(range(len(classes_no_cat)), classes_no_cat, fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylabel('top-2 singular values', fontsize=14)
    # plt.ylabel('Biggest singular values', fontsize=14)

    plt.figure()
    plt.title('smallest singular value')
    plt.plot(sigma1_vgg_full[:, -1], '-k+', label=r'vgg-$\sigma$-last')
    plt.plot(sigma1_dla_full[:, -1], '-bo', label=r'dla-$\sigma$-last')
    plt.plot(sigma1_res_full[:, -1], '-rx', label=r'res-$\sigma$-last')
    plt.legend()
    plt.yscale('log')
    plt.xticks(range(len(classes_no_cat)), classes_no_cat, fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylabel('smallest singular values', fontsize=14)

    plt.figure()
    plt.title('condition number or rank')
    plt.plot(sigma1_vgg_cond, '-k+', label='VGG')
    plt.plot(sigma1_dla_cond, '-bo', label='DLA')
    plt.plot(sigma1_res_cond, '-rx', label='ResNet18')
    plt.xticks(range(len(classes_no_cat)), classes_no_cat, fontsize=14)
    plt.legend()

    # for cond mode
    # plt.yscale('log')

    print('VGG')
    print(sigma1_vgg_cond)
    print('DLA')
    print(sigma1_dla_cond)
    print('Res')
    print(sigma1_res_cond)

    plt.show()

def plot_db_complexity_for_optimizer(args):
    '''
    Plot the auto-correlation matrix for Fig 5
    Quantifying the decision boundary complexity using condition number
    This compares the final models by SGD-big-lr, SGD-small-lr, SGD-anneal and Adam, trained in 200 epochs.
    All the models are VGG19.
    '''

    header = 'results/Fig5/'
    small_lr_file = header + 'model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_features.npy'
    big_lr_file = header + 'model_vgg19_alpha_0.01_lrmode_constantbatchsize_128_momentum_decayed_testacc_90.12.pyc_features.npy'
    anneal_lr_file = header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.93.pyc_features.npy'
    adam_file = header + 'model_vgg19adambatchsize_128_momentum_decayed_testacc_92.64.pyc_features.npy'

    small_lr_phi = np.load(small_lr_file)
    big_lr_phi = np.load(big_lr_file)
    anneal_lr_phi = np.load(anneal_lr_file)
    adam_phi = np.load(adam_file)

    plt.figure()

    print('args.sigma=', args.sigma)
    if args.sigma == 0 or args.sigma == 1 or args.sigma == 511:
        plt.title('sigma'+str(args.sigma))
        small_lr_data = get_sigmas_for_c1_vs_others(c1=3, features=small_lr_phi, mode=args.mode)[:, args.sigma]
        big_lr_data = get_sigmas_for_c1_vs_others(c1=3, features=big_lr_phi, mode=args.mode)[:, args.sigma]
        anneal_lr_data = get_sigmas_for_c1_vs_others(c1=3, features=anneal_lr_phi, mode=args.mode)[:, args.sigma]
        adam_data = get_sigmas_for_c1_vs_others(c1=3, features=adam_phi, mode=args.mode)[:, args.sigma]
    else:
        plt.title('condition number or rank')
        small_lr_data = get_cond_for_c1_vs_others(c1=3, features=small_lr_phi, mode=args.mode, cond_or_rank=args.cond_or_rank, matrix=args.matrix)
        big_lr_data = get_cond_for_c1_vs_others(c1=3, features=big_lr_phi, mode=args.mode, cond_or_rank=args.cond_or_rank, matrix=args.matrix)
        anneal_lr_data = get_cond_for_c1_vs_others(c1=3, features=anneal_lr_phi, mode=args.mode, cond_or_rank=args.cond_or_rank, matrix=args.matrix)
        adam_data = get_cond_for_c1_vs_others(c1=3, features=adam_phi, mode=args.mode, cond_or_rank=args.cond_or_rank, matrix=args.matrix)


    plt.plot(small_lr_data, '-ko', label='SGD-small-lr')
    plt.plot(big_lr_data, '-.b*', label='SGD-big-lr')
    plt.plot(anneal_lr_data, '--r+', label='SGD-anneal-lr')
    plt.plot(adam_data, linestyle='dotted', color='m', label='Adam')
    plt.ylabel(r'$\kappa(A^TA)$ in PCA(2)')
    plt.xticks(range(len(classes_no_cat)), classes_no_cat, fontsize=14)
    plt.legend()

    # if args.sigma == 0 or args.sigma == 511:
    #     plt.yscale('log')

    plt.show()

def plot_db_complexity_for_optimizer_all_sigmas(args, generate_data=False):
    '''
    plot spectral profiling of the optimizers during training
    CAT-DOG, sigma1
    '''

    if generate_data:
        header = 'results/Fig5/'

        #these two models are not well trained. so does not count. especially big lr has osciallation and the test number is not stable.
        # small_lr_file = header + 'model_vgg19_alpha_0.0001_lrmode_constantbatchsize_128_momentum_decayed_testacc_84.77.pyc_features.npy'
        # big_lr_file = header + 'model_vgg19_alpha_0.01_lrmode_constantbatchsize_128_momentum_decayed_testacc_90.12.pyc_features.npy'

        anneal_lr_file = header + 'model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.93.pyc_features.npy'
        adam_file = header + 'model_vgg19adambatchsize_128_momentum_decayed_testacc_92.64.pyc_features.npy'
        adamW_file = 'results/run1_save_model_every_epoch_vgg19_adamW/model_vgg19adamwbatchsize_128_momentum_decayed_testacc_92.97.pyc_features.npy'

        rmpsprop_file = 'results/run1_save_model_every_epoch_vgg19_rmsprop/model_vgg19rmspropbatchsize_128_momentum_decayed_testacc_91.22.pyc_features.npy'
        asgd_file = 'results/run1_save_model_every_epoch_vgg19_ASGD/model_vgg19asgdbatchsize_128_momentum_decayed_testacc_90.60.pyc_features.npy'
        adadelta_file = 'results/run1_save_model_every_epoch_vgg19_Adadelta/model_vgg19adadeltabatchsize_128_momentum_decayed_testacc_91.91.pyc_features.npy'
        adagrad_file = 'results/run1_save_model_every_epoch_vgg19_Adagrad/model_vgg19adagradbatchsize_128_momentum_decayed_testacc_90.61.pyc_features.npy'
        # rprop_file = 'results/run1_save_model_every_epoch_vgg19_Rprop/model_vgg19rpropbatchsize_128_momentum_decayed_testacc_24.68.pyc_features.npy'#this is slow acc.

        # small_lr_phi = np.load(small_lr_file)
        # big_lr_phi = np.load(big_lr_file)
        anneal_lr_phi = np.load(anneal_lr_file)
        adam_phi = np.load(adam_file)
        adamW_phi = np.load(adamW_file)
        rmpsprop_phi = np.load(rmpsprop_file)
        asgd_phi = np.load(asgd_file)
        adadelta_phi = np.load(adadelta_file)
        adagrad_phi = np.load(adagrad_file)
        # rprop_phi = np.load(rprop_file)

        anneal_lr_data = get_sigmas_for_c1_vs_others(c1=3, features=anneal_lr_phi, mode=args.mode)
        adam_data = get_sigmas_for_c1_vs_others(c1=3, features=adam_phi, mode=args.mode)
        adamW_data = get_sigmas_for_c1_vs_others(c1=3, features=adamW_phi, mode=args.mode)
        rmsprop_data = get_sigmas_for_c1_vs_others(c1=3, features=rmpsprop_phi, mode=args.mode)
        asgd_data = get_sigmas_for_c1_vs_others(c1=3, features=asgd_phi, mode=args.mode)
        adadelta_data = get_sigmas_for_c1_vs_others(c1=3, features=adadelta_phi, mode=args.mode)
        adagrad_data = get_sigmas_for_c1_vs_others(c1=3, features=adagrad_phi, mode=args.mode)

        # c2 = 4 #DOG; -1 because CAT to itself is not included
        # c2 = 1

        # small_lr_data = get_sigmas_for_c1_vs_others(c1=3, features=small_lr_phi, mode=args.mode)[c2, :]
        # big_lr_data = get_sigmas_for_c1_vs_others(c1=3, features=big_lr_phi, mode=args.mode)[c2, :]
        # anneal_lr_data = get_sigmas_for_c1_vs_others(c1=3, features=anneal_lr_phi, mode=args.mode)[c2, :]
        # adam_data = get_sigmas_for_c1_vs_others(c1=3, features=adam_phi, mode=args.mode)[c2, :]
        # adamW_data = get_sigmas_for_c1_vs_others(c1=3, features=adamW_phi, mode=args.mode)[c2, :]
        # rmsprop_data = get_sigmas_for_c1_vs_others(c1=3, features=rmpsprop_phi, mode=args.mode)[c2, :]
        # asgd_data = get_sigmas_for_c1_vs_others(c1=3, features=asgd_phi, mode=args.mode)[c2, :]
        # adadelta_data = get_sigmas_for_c1_vs_others(c1=3, features=adadelta_phi, mode=args.mode)[c2, :]
        # adagrad_data = get_sigmas_for_c1_vs_others(c1=3, features=adagrad_phi, mode=args.mode)[c2, :]
        # rprop_data = get_sigmas_for_c1_vs_others(c1=3, features=rprop_phi, mode=args.mode)[4, :]

        np.save('results/Fig6/CAT_sigmas_anneal_lr.npy', anneal_lr_data)
        np.save('results/Fig6/CAT_sigmas_adam.npy', adam_data)
        np.save('results/Fig6/CAT_sigmas_adamW.npy', adamW_data)
        np.save('results/Fig6/CAT_sigmas_rmsprop_data.npy', rmsprop_data)
        np.save('results/Fig6/CAT_sigmas_asgd.npy', asgd_data)
        np.save('results/Fig6/CAT_sigmas_adadelta.npy', adadelta_data)
        np.save('results/Fig6/CAT_sigmas_adagrad.npy', adagrad_data)

        print('Data generated successfully. ')
        sys.exit(1)
    else:
        anneal_lr_data = np.load('results/Fig6/CAT_sigmas_anneal_lr.npy')
        adam_data = np.load('results/Fig6/CAT_sigmas_adam.npy')
        adamW_data = np.load('results/Fig6/CAT_sigmas_adamW.npy')
        rmsprop_data = np.load('results/Fig6/CAT_sigmas_rmsprop_data.npy')
        asgd_data = np.load('results/Fig6/CAT_sigmas_asgd.npy')
        adadelta_data = np.load('results/Fig6/CAT_sigmas_adadelta.npy')
        adagrad_data = np.load('results/Fig6/CAT_sigmas_adagrad.npy')

        plt.figure()
        plt.title(r'$\sigma$s')

        # c2 = 4
        # c2 = 0
        # c2 = 1
        # c2 = 2
        c2 = 3

        # plt.plot(small_lr_data, '-k', label='SGD-small-lr')
        # plt.plot(big_lr_data, '-.b*', label='SGD-big-lr')
        plt.plot(asgd_data[c2, :], linestyle='--', color='tab:brown', label='ASGD')
        plt.plot(adagrad_data[c2, :], linestyle='-.', color='b', label='Adagrad')
        plt.plot(rmsprop_data[c2, :], linestyle='-.', color='c', label='RMSProp')
        plt.plot(adadelta_data[c2, :], linestyle=':', color='g', label='Adadelta')
        plt.plot(adam_data[c2, :], linestyle='-', color='k', label='Adam')
        plt.plot(adamW_data[c2, :], linestyle='dashed', color='m', label='AdamW')
        plt.plot(anneal_lr_data[c2, :], '--r+', label='SGD-anneal-lr')

        # plt.plot(rprop_data, linestyle='--', color='tab:orange', label='Rprop')

        #for the zoom shot: blue window in the center of the figure
        # plt.legend(fontsize=16)
        # plt.xticks([0,1], fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.xlim([-0.3, 1.3])

        plt.xlabel(r'singular value index $i$', fontsize=16)
        plt.ylabel(r'$\sigma_i$', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.yscale('log')

        plt.show()


def plot_db_complexity_for_models_all_sigmas(args):
    '''
    plot during training, the sigma 1 changes.
    CAT-DOG, sigma1
    all models are trained by SGD-anneal
    '''

    header = 'results/'
    vgg_phi = np.load(header+'run1_save_model_every_epoch_vgg19/model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.57.pyc_features.npy')
    res_phi = np.load(header+'run1_save_model_every_epoch_resnet18/model_resnet18_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.75.pyc_features.npy')
    dla_phi = np.load(header+'run1_save_model_every_epoch_dla/model_dla_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_95.05.pyc_features.npy')

    vgg11_phi = np.load(
        header + 'run1_save_model_every_epoch_vgg11/model_vgg11_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_92.42.pyc_features.npy')

    vgg16_phi = np.load(
        header + 'run1_save_model_every_epoch_vgg16/model_vgg16_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.91.pyc_features.npy')

    plt.figure()
    plt.title(r'$\sigma$s')

    vgg11_data = get_sigmas_for_c1_vs_others(c1=3, features=vgg11_phi, mode=args.mode)[4, :]
    vgg16_data = get_sigmas_for_c1_vs_others(c1=3, features=vgg16_phi, mode=args.mode)[4, :]
    vgg_data = get_sigmas_for_c1_vs_others(c1=3, features=vgg_phi, mode=args.mode)[4, :]
    res_data = get_sigmas_for_c1_vs_others(c1=3, features=res_phi, mode=args.mode)[4, :]
    dla_data = get_sigmas_for_c1_vs_others(c1=3, features=dla_phi, mode=args.mode)[4, :]

    # plt.plot(vgg11_data, '--g', label='VGG11')
    # plt.plot(vgg_data, '-k', label='VGG19')
    # plt.plot(vgg16_data, '--m<', label='VGG16')
    # plt.plot(dla_data, '-.b*', label='DLA')
    # plt.plot(res_data, '-r', label='ResNet18')

    #plot ratio normalization over the sum of all others: see if the data point or the last data point gives a good ordering that is the same as test performance
    plt.plot(vgg11_data/vgg11_data.sum(), '--g', label='VGG11')
    plt.plot(vgg16_data / vgg16_data.sum(), '--m<', label='VGG16')
    plt.plot(vgg_data/vgg_data.sum(), '-k', label='VGG19')
    plt.plot(dla_data/dla_data.sum(), '-.b*', label='DLA')
    plt.plot(res_data/res_data.sum(), '-r', label='ResNet18')

    plt.xlabel(r'singular value index $i$', fontsize=16)
    plt.ylabel(r'$\sigma_i$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.yscale('log')
    plt.legend(fontsize=16)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    #option 1: how come the anneal-lr has the biggest condition number in PCA mode? todo
    # parser.add_argument('--mode', default='pca', type=str, help='svd or pca')
    # parser.add_argument('--cond_or_rank', default='cond', type=str, help='condition number or rank')
    # parser.add_argument('--matrix', default='partial', type=str, help='full or partial correlation matrix')

    #option 2: This is SVD mode. It makes more sense to me. anneal-lr has the smallest condition number. So opition 1 or option 2?
    # parser.add_argument('--mode', default='svd', type=str, help='svd or pca')
    # parser.add_argument('--cond_or_rank', default='cond', type=str, help='condition number or rank')
    # parser.add_argument('--matrix', default='partial', type=str, help='full or partial correlation matrix')

    #option 3: adam is really high.
    # parser.add_argument('--mode', default='pca', type=str, help='svd or pca')
    # parser.add_argument('--cond_or_rank', default='cond', type=str, help='condition number or rank')
    # parser.add_argument('--matrix', default='full', type=str, help='full or partial correlation matrix')

    #option 4: full, and rank mode
    #interesting. this shows for optimizers, ranks in decreasing order: small/big lr, adam, anneal: The smaller rank, the better? This is the opposite from the model comparison.
    # parser.add_argument('--mode', default='pca', type=str, help='svd or pca')
    # parser.add_argument('--cond_or_rank', default='rank', type=str, help='condition number or rank')
    # parser.add_argument('--matrix', default='full', type=str, help='full or partial correlation matrix')

    #option 5: study sigma 1 for optimizers
    # parser.add_argument('--mode', default='pca', type=str, help='svd or pca')
    # parser.add_argument('--sigma1', default='true', type=str, help='study sigma1 or not')

    #option 6: similar to 5, but svd mode
    # parser.add_argument('--mode', default='svd', type=str, help='svd or pca')
    # # parser.add_argument('--sigma', default=0, type=int, help='index of sigma')
    # parser.add_argument('--sigma', default=1, type=int, help='index of sigma')

    #plot 512 singular values for different optimizers. VGG19 model
    parser.add_argument('--mode', default='svd', type=str, help='svd or pca')
    # parser.add_argument('--sigma', default=0, type=int, help='index of sigma')
    # parser.add_argument('--sigma', default=1, type=int, help='index of sigma')
    args = parser.parse_args()
    plot_db_complexity_for_optimizer_all_sigmas(args)

    #plot 512 singular values for different models. SGD-anneal optimizer
    # parser.add_argument('--mode', default='pca', type=str, help='svd or pca')
    # args = parser.parse_args()
    # print('@@mode=', args.mode)
    # plot_db_complexity_for_models_all_sigmas(args)


    # plot_vgg_res_dla(args)

    # plot_db_complexity_for_optimizer(args)

    # plot_db_complexity_for_optimizer_all_sigmas(args)
