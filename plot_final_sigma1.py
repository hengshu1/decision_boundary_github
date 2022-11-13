import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys, time
from generate_mean_and_variance import PCA_on_feature_matrix, SVD_on_feature_matrix
from main import classes
'''

compute and compare sigma1 of the final model -- the variance of the first principle component for both models and class pairs

https://en.wikipedia.org/wiki/Singular_value

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--feature_mat_dir',
                        default='results/run1_save_model_every_epoch_vgg19/model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.57.pyc_features.npy',
                        # default='results/run2_save_model_every_epoch_vgg19/model_vgg19_alpha_0.1_lrmode_schedulebatchsize_128_momentum_decayed_testacc_93.76.pyc_features.npy',
                        type=str, help='feature_mat_dir')

    args = parser.parse_args()
    print('@@feature_mat_dir=', args.feature_mat_dir)
    time.sleep(3)

    feature_mat_file = args.feature_mat_dir

    features_all_c = np.load(feature_mat_file)
    print('features_all_c.shape=', features_all_c.shape)

    #table 1: cat vs. all the other classes: strange. No interesting pattern observed. instead, why cat-dog is among the smallest variance (whether first or sum of variances)?
    #I think it makes sense to for cat to have a large 1st variance, right? because this is also the conditioning of the feature matrix. cat-dog feature matrix is supposed to be ill conditioned.
    #maybe the first principle does not measure the conditioning of DB?
    c1 = 3
    # c1 = 5
    sigma1sA, sigma1s = [], []
    for c2 in range(len(classes)):
        if c2 == c1:
            continue
        c1_c2_features = np.concatenate((features_all_c[c1, :, :], features_all_c[c2, :, :]), axis=0)
        # features, singulars = PCA_on_feature_matrix(c1_c2_features, n_components=2)
        features, singulars, _ = SVD_on_feature_matrix(c1_c2_features, k=100)
        # print('singulars.shape=', singulars.shape)
        # print('{}-{}:{} (first)'.format(classes[c1], classes[c2], singulars[0]))
        # print('{}-{}:{} (sum)'.format(classes[c1], classes[c2], singulars.sum()))

        UA, eigsA, vhA = np.linalg.svd(c1_c2_features)
        print(eigsA[:3]**2)
        # sigma1sA.append(eigsA[0])
        sigma1sA.append(singulars[0])

        # print('c1_c2_features.shape=', c1_c2_features.shape)
        ATA = c1_c2_features.T.dot(c1_c2_features)
        # print('ATA.shape=', ATA.shape)#
        u, eigs, vh = np.linalg.svd(ATA)
        print(eigs[:3])

        #this seems make sense: cat-dog is largest.
        print('{}-{}:{} (sigma0)'.format(classes[c1], classes[c2], eigs[0]))
        sigma1s.append(eigs[0])

        # sigma1s.append(singulars.sum())
        # sigma1s.append(singulars[-1])
    # print('sigma1sA=', sigma1sA)
    # print('sigma1s=', sigma1s)

    classes_no_cat = list(classes)
    print('classes_no_cat=', classes_no_cat)
    del classes_no_cat[3]
    print('classes_no_cat=', classes_no_cat)

    plt.figure()
    plt.bar(sigma1sA, color='b', label='A ')
    plt.xticks(range(len(classes_no_cat)), classes_no_cat)
    plt.legend()

    plt.figure()
    plt.bar(sigma1s, color='k', label='ATA ')
    plt.xticks(range(len(classes_no_cat)), classes_no_cat)
    plt.legend()

    plt.show()







