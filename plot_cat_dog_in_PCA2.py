import numpy as np
import matplotlib.pyplot as plt
import argparse

'''
This plot the cat and dog samples in the PCA(2) space
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='resnet18', type=str, help='model name')
    parser.add_argument('--saved_dir', default='results/run1_save_model_every_epoch_resnet18/', type=str, help='directory to load the data')
    parser.add_argument('--mode', default='pca', type=str, help='pca or svd')

    args = parser.parse_args()

    print('@@model=',args.model)
    print('@@saved_dir=', args.saved_dir)
    print('@@mode=', args.mode)

    c1_center = np.load(args.saved_dir + 'c1_centers_{}_10comp_{}.npy'.format(args.mode, args.model))
    c2_center = np.load(args.saved_dir + 'c2_centers_{}_10comp_{}.npy'.format(args.mode, args.model))
    singualrs = np.load(args.saved_dir + 'singulars_{}_10comp_{}.npy'.format(args.mode, args.model))
    print('c1_center.shape=', c1_center.shape)
    print('c2_center.shape=', c2_center.shape)
    print('singulars.shape=', singualrs)

    plt.figure()
    plt.title('center-{}-{}'.format(args.model, args.mode))
    # plt.plot(c1_center[:, 0], c1_center[:, 1], '-bo', label='CAT center')
    # plt.plot(c2_center[:, 0], c2_center[:, 1], '-r+', label='DOG center')
    plt.plot(c1_center[:, 0], '-bo', label='CAT center-x')
    plt.plot(c1_center[:, 1], '--bo', label='CAT center-y')
    plt.plot(c2_center[:, 0], '-r+', label='DOG center-x')
    plt.plot(c2_center[:, 1], '--r+', label='DOG center-y')
    plt.legend()

    plt.figure()#cat-dog singular value evolution; also plot for other class pairs
    # plt.title('singulars by svd')
    plt.title('singulars by pca')

    print('singular.shape=', singualrs.shape)
    # plt.plot(singualrs_svd[:, 0], '-b+', label=r'$\sigma_1^2$')#svd
    # plt.plot(singualrs_svd[:, 1], '-ro', label=r'$\sigma_2^2$')
    # plt.plot(singualrs_svd[:, 2], '-gx', label=r'$\sigma_3^2$')

    # plt.plot(singualrs_svd[:, 0], '-r', label=r'$\sigma_{{{}}}^2$'.format(1))
    # for i in range(1, singualrs_svd.shape[1]):
    #     plt.plot(singualrs_svd[:, i], label=r'$\sigma_{{{}}}^2$'.format(i+1))

    for i in range(singualrs.shape[1]):#similar to svd
        plt.plot(singualrs[:, i], label=r'$\sigma_{{{}}}^2$'.format(i + 1))

    # plt.plot(singualrs_pca[:, 0]*20, '--b+', label='sigma_1, pca')#similar to svd
    # plt.plot(singualrs_pca[:, 1]*20, '--ro', label='sigma_2, pca')
    plt.legend(fontsize=14)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('singular values', fontsize=14)
    plt.show()