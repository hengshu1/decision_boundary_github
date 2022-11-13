import numpy as np
import matplotlib.pyplot as plt
import argparse

'''
compare VGG19 and Resnet18 in the top singular values
PCA and SVD has same observations
'''

'''
1. Need to first run output_space.py to get the features 
2. then run generate_mean_and_variance.py to generate the singulvar values during training
3. then plot using generate_fig7.py
'''

def load_data(saved_dir, model, mode, optimizer='sgd'):
    if optimizer == 'sgd':
        c1_center = np.load(saved_dir + 'c1_centers_{}_10comp_{}.npy'.format(mode, model))
        c2_center = np.load(saved_dir + 'c2_centers_{}_10comp_{}.npy'.format(mode, model))
        singualrs = np.load(saved_dir + 'singulars_{}_10comp_{}.npy'.format(mode, model))
    else:
        c1_center = np.load(saved_dir + 'c1_centers_{}_10comp_{}_{}.npy'.format(mode, model, optimizer))
        c2_center = np.load(saved_dir + 'c2_centers_{}_10comp_{}_{}.npy'.format(mode, model, optimizer))
        singualrs = np.load(saved_dir + 'singulars_{}_10comp_{}_{}.npy'.format(mode, model, optimizer))
    return c1_center, c2_center, singualrs

def compare_models_for_SGD(args):
    '''
    For Fig 7
    '''
    c1_center_vgg, c2_center_vgg, singualrs_vgg = load_data(
        saved_dir='results/run1_save_model_every_epoch_vgg19/',
        model='vgg19',
        mode=args.mode)

    print('c1_center_vgg.shape=', c1_center_vgg.shape)
    print('c2_center_vgg.shape=', c2_center_vgg.shape)
    print('singualrs_vgg.shape=', singualrs_vgg.shape)

    c1_center_resnet, c2_center_resnet, singualrs_resnet = load_data(
        saved_dir='results/run1_save_model_every_epoch_resnet18/',
        model='resnet18', mode=args.mode)
    print('c1_center_resnet.shape=', c1_center_resnet.shape)
    print('c2_center_resnet.shape=', c2_center_resnet.shape)
    print('singualrs_resnet.shape=', singualrs_resnet.shape)

    c1_center_dla, c2_center_dla, singualrs_dla = load_data(
        saved_dir='results/run1_save_model_every_epoch_dla/',
        model='dla',
        mode=args.mode)
    print('c1_center_dla.shape=', c1_center_dla.shape)
    print('c2_center_dla.shape=', c2_center_dla.shape)
    print('singualrs_dla.shape=', singualrs_dla.shape)

    plt.figure()  # cat-dog singular value evolution; also plot for other class pairs
    plt.title('singulars by ' + args.mode)

    # in the end, this ratio is about two times: vgg/resnet18
    # plt.plot(singualrs_vgg[:, 0], '-k', label=r'$\sigma_{{{}}}^2$-vgg19'.format(1))
    # plt.plot(singualrs_resnet[:, 0], '-b+', label=r'$\sigma_{{{}}}^2$-resnet18'.format(1))
    # plt.plot(singualrs_dla[:, 0], '-ro', label=r'$\sigma_{{{}}}^2$-dla'.format(1))
    #
    # plt.plot(singualrs_vgg[:, 1], '--', color='tab:orange', marker='x', label=r'$\sigma_{{{}}}^2$-vgg19'.format(2))
    # plt.plot(singualrs_resnet[:, 1], '--mo', label=r'$\sigma_{{{}}}^2$-resnet18'.format(2))
    # plt.plot(singualrs_dla[:, 1], '--', color='tab:brown', marker='s', label=r'$\sigma_{{{}}}^2$-dla'.format(2))
    #
    # plt.plot(singualrs_vgg[:, 2], '--g', label=r'$\sigma_{{{}}}^2$-vgg19'.format(3))
    # plt.plot(singualrs_resnet[:, 2], '--c', label=r'$\sigma_{{{}}}^2$-resnet18'.format(3))
    # plt.plot(singualrs_dla[:, 2], '--y', label=r'$\sigma_{{{}}}^2$-dla'.format(3))

    # var_sum_vgg = singualrs_vgg.sum(axis=1) - singualrs_vgg[:, 0]
    # var_sum_vgg = singualrs_vgg[:, 1]
    # print('var_sum_vgg.shape=', var_sum_vgg.shape)
    # var_vgg = singualrs_vgg[:, 0]/var_sum_vgg
    var_vgg = singualrs_vgg[:, 0]
    plt.plot(var_vgg, '-k+', label=r'$\sigma_{{{}}}^2$-vgg19'.format(1))
    # plt.plot(singualrs_vgg[:, 0] / singualrs_vgg[:, 1], '-k+', label=r'$\sigma_{{{}}}^2$-vgg19'.format(1))

    # var_sum_res = singualrs_resnet.sum(axis=1) - singualrs_resnet[:, 0]
    # var_sum_res = singualrs_resnet[:, 1]
    var_res = singualrs_resnet[:, 0]  # /var_sum_res
    plt.plot(var_res, '-bo', label=r'$\sigma_{{{}}}^2$-resnet18'.format(1))
    # plt.plot(singualrs_resnet[:, 0] / singualrs_resnet[:, 1], '-bo', label=r'$\sigma_{{{}}}^2$-resnet18'.format(1))

    # var_sum_dla = singualrs_dla.sum(axis=1) - singualrs_dla[:, 0]
    # var_sum_dla = singualrs_dla[:, 1]
    var_dla = singualrs_dla[:, 0]  # /var_sum_dla
    plt.plot(var_dla, '-rx', label=r'$\sigma_{{{}}}^2$-dla'.format(1))

    print('vgg last var:', var_vgg[-1])
    print('res last var:', var_res[-1])
    print('dla last var:', var_dla[-1])

    plt.legend(fontsize=14)
    plt.xlabel('epochs', fontsize=14)
    if args.mode == 'SVD':
        plt.ylabel('singular values', fontsize=14)
    else:
        plt.ylabel('explained variances', fontsize=14)
    plt.show()

def compare_optimizers_for_Resnet18(args):
    '''
    The observations also holds similarly to VGG19
    '''

    c1_center_sgd, c2_center_sgd, singualrs_sgd = load_data(
        saved_dir='results/run1_save_model_every_epoch_resnet18/',
        model='resnet18',
        mode=args.mode,
        optimizer='sgd'
    )

    c1_center_adam, c2_center_adam, singualrs_adam = load_data(
        saved_dir='results/run2_save_model_every_epoch_resnet18_Adam/',
        model='resnet18',
        mode=args.mode,
        optimizer='adam'
    )

    c1_center_adagrad, c2_center_adagrad, singualrs_adagrad = load_data(
        saved_dir='results/run2_save_model_every_epoch_resnet18_Adagrad/',
        model='resnet18',
        mode=args.mode,
        optimizer='adagrad'
    )

    c1_center_adamax, c2_center_adamax, singualrs_adamax = load_data(
        saved_dir='results/run2_save_model_every_epoch_resnet18_Adamax/',
        model='resnet18',
        mode=args.mode,
        optimizer='adamax'
    )

    print('singualrs_sgd.shape=', singualrs_sgd.shape)
    print('singualrs_adam.shape=', singualrs_adam.shape)
    print('singualrs_adagrad.shape=', singualrs_adagrad.shape)
    print('singualrs_adamax.shape=', singualrs_adamax.shape)


    plt.figure()
    plt.title('ResNet18')

    #sigma0, both are increasing
    # plt.plot(singualrs_sgd[:, 0], '-ko', label=r'SGD, $\sigma0$')
    plt.plot(singualrs_adam[:, 0], '-r+', label=r'Adam, $\sigma0$')
    plt.plot(singualrs_adamax[:, 0], '-.bs', label=r'Adamax, $\sigma0$')
    plt.plot(singualrs_adagrad[:, 0], '--ko', label=r'Adagrad, $\sigma0$')

    #sigma1:
    # plt.plot(singualrs_sgd[:, 1], '--b', label=r'SGD, $\sigma1$')  #
    # plt.plot(singualrs_adam[:, 1], '-r+', label=r'Adam, $\sigma1$')
    # plt.plot(singualrs_adagrad[:, 1], '--ko', label=r'Adagrad, $\sigma1$')
    # plt.plot(singualrs_adamax[:, 1], '-.bs', label=r'Adamax, $\sigma1$')

    # plt.plot(singualrs_adam[:, 2], '-r+', label=r'Adam, $\sigma2$')
    # plt.plot(singualrs_adagrad[:, 2], '--ko', label=r'Adagrad, $\sigma2$')
    # plt.plot(singualrs_adamax[:, 2], '-.bs', label=r'Adamax, $\sigma2$')

    # plt.plot(singualrs_adam[:, 0]/singualrs_adam[-1, :].sum(), '-r+', label=r'Adam, $\sigma2$')
    # plt.plot(singualrs_adamax[:, 0]/singualrs_adamax[-1, :].sum(), '-.bs', label=r'Adamax, $\sigma2$')
    # plt.plot(singualrs_adagrad[:, 0] / singualrs_adagrad[-1, :].sum(), '--ko', label=r'Adagrad, $\sigma2$')

    #let's show the spectrum in the end: oh. No. SGD's data is for different class pairs not during training.

    # plt.plot(singualrs_sgd[-1, :], '--b', label=r'SGD, $\sigma$s')
    # plt.plot(singualrs_adam[-1, :], '-.m', label='Adam, $\sigma$s')
    # plt.plot(singualrs_adamax[-1, :], ':g', label='Adamax, $\sigma$s')
    # plt.plot(singualrs_adagrad[-1, :], '--k', label='Adagrad, $\sigma$s')

    # plt.yscale('log')

    plt.legend()
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mode', default='pca', type=str, help='svd or pca')
    # parser.add_argument('--optimizer', default='adam', type=str, help='optimizers')

    args = parser.parse_args()

    print('@@mode=', args.mode)

    compare_models_for_SGD(args)

    compare_optimizers_for_Resnet18(args)
