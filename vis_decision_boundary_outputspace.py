import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from main import classes

# model = 'lr_small'
# model = 'lr_big'#why lr_big is more separatble than lr_small?
model = 'lr_anneal'

header = 'results/'

if model=='lr_big':
    print('lr_big model')
    outputs_lin = np.load(
        header + 'model_vgg19_alpha_0.01_lrmode_constant_momentum_decayed_testacc_88.76.pyc_outputs.npy')  # lr-big
    features = np.load(
        header + 'model_vgg19_alpha_0.01_lrmode_constant_momentum_decayed_testacc_88.76.pyc_features.npy')
elif model=='lr_small':
    print('lr_small model')
    outputs_lin = np.load(
        header + 'model_vgg19_alpha_0.0001_lrmode_constant_momentum_decayed_testacc_84.99.pyc_outputs.npy')  # lr-small
    features = np.load(
        header + 'model_vgg19_alpha_0.0001_lrmode_constant_momentum_decayed_testacc_84.99.pyc_features.npy')
elif model=='lr_anneal':
    print('lr_anneal model')
    outputs_lin = np.load(
        header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_outputs.npy')  # linear outputs by the net
    features = np.load(
        header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_features.npy')
else:
    print('unsupported model')
    sys.exit(1)


print(outputs_lin.shape)#10, 5000, 10

#this test shows the class' prediction is highest -- using softmax, it will be close to prob 1. YES, see plot_fig1_fig2_fig3_fig6_fig9_fig10.py
plt.figure()
plt.plot(outputs_lin[3, :, 3], '-r')
plt.plot(outputs_lin[3, :, 5], '-g')
plt.show()
sys.exit(1)

#
#

print('features.shape=', features.shape)#10, 5000, 512

# plt.figure()
# plt.plot(outputs_lin[0, :, 0], '-r', label='output')
# #plot some sample features: the first 10 features: some features have very small abs values
# for i in range(10):
#     plt.plot(features[0, :, i], label='feature '+str(i))
# plt.legend(labelcolor='linecolor')
# plt.show()

W = np.load(header+'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_W.npy')
print('W.shape=', W.shape)#10, 512

b = np.load(header+'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_b.npy')
print('b.shape=', b.shape)#10,

output2 = np.dot(features, W.T) + b
print(output2.shape)#10, 5000, 10

#seems pretty close
print(output2[3, 1, :])
print(outputs_lin[3, 1, :])
print('2-norm {} should be close to 0. '.format(np.linalg.norm(output2 - outputs_lin)))#it's 0.19: todo double check if it's caused by numerical error
print('{} max abs should be very close to 0. '.format(np.max(np.abs(output2 - outputs_lin))))#right.

#interesting figure showing interference
# plt.figure()
# #plot cats in two outputs space: picked randomly
# # index1 = 0#prob(PLANE)
# # index2 = 1#prob(CAR)
# #this shows in the CAT-DOG prob space, cats and dogs are clearly separated; however, in the prob(CAR) and prob(DOG) space, it's not.
# # index1 = 3#prob(CAT)
# # index2 = 5#prob(DOG)
# # index1 = 3#prob(CAT)
# # index2 = 6#prob(FROG)
# # index1 = 4#prob(DEER)
# # index2 = 6#prob(FROG)
# index1 = 7#prob(HORSE)#in this space it shows that frogs are more like CAT than DOG; and dogs are more like HORSE
# index2 = 6#prob(FROG)
# plt.plot(outputs_lin[3, :, index1], outputs_lin[3, :, index2], 'b+', label='cats')
# plt.plot(outputs_lin[5, :, index1], outputs_lin[5, :, index2], 'ro', label='dogs')
# plt.xlabel('~prob({})'.format(classes[index1].upper()))
# plt.ylabel('~prob({})'.format(classes[index2].upper()))
# plt.legend(labelcolor='linecolor')
# plt.show()

#plot the cats and dogs in two random feature space: feature 0 1 are all 0; phi(2) are very dynamic
#this has a clear separation
# index1 = 3
# index2 = 4
# # this messes cats and dogs together: capture their similarity I guess
# # index1 = 3
# # index2 = 5
# plt.plot(features[3, :, index1], features[3, :, index2], 'b+', label='cats')
# #plot dogs in two feature space
# plt.plot(features[5, :, index1], features[5, :, index2], 'ro', label='dogs')
# plt.xlabel('feature '+str(index1))
# plt.ylabel('feature '+str(index2))
# plt.legend(labelcolor='linecolor')

#run PCA
from sklearn.decomposition import PCA

def plot_PCA(features, n_components, c1, c2, f1, f2):
    pca = PCA(n_components=n_components)
    c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)
    print('c1 and c2 combined featurte.shape=', c1_c2_features.shape)
    pca.fit(c1_c2_features)
    print(pca.components_)
    print(pca.explained_variance_)
    features_PCA = pca.transform(c1_c2_features)
    print('pca features:', features_PCA.shape)

    plt.plot(features_PCA[:5000, f1], features_PCA[:5000, f2], 'b+', label=classes[c1]+'s')
    plt.plot(features_PCA[5000:10000, f1], features_PCA[5000:10000, f2], 'ro', label=classes[c2]+'s')
    plt.legend(labelcolor='linecolor')

#cats and dogs
plt.figure()
plt.subplot(1, 3, 1)
plot_PCA(features, n_components=3, c1=3, c2=5, f1=0, f2=1)#separatebly
plt.subplot(1, 3, 2)
plot_PCA(features, n_components=3, c1=3, c2=5, f1=0, f2=2)#separatebly
plt.subplot(1, 3, 3)
plot_PCA(features, n_components=3, c1=3, c2=5, f1=1, f2=2)#messed up all together

#horse ship
plt.figure()
plt.subplot(1, 3, 1)
plot_PCA(features, n_components=3, c1=7, c2=8, f1=0, f2=1)#separatebly
plt.subplot(1, 3, 2)
plot_PCA(features, n_components=3, c1=7, c2=8, f1=0, f2=2)#separatebly
plt.subplot(1, 3, 3)
plot_PCA(features, n_components=3, c1=7, c2=8, f1=1, f2=2)#messed up all together

#the above shows only the first component can already separate two classes. This is consistent with my SVD on Y approximation.
#it also shows cats and dogs are farther than horse and ships
#this confirms with my finding that deep learning has no "boundary structure". Instead, they are separatebly.

plt.show()
sys.exit(1)

#the above plots shows these features are a god space for decision boundary exploration

markers=['k2', 'kx', 'ks',
         'ro', 'm<', 'b+',
         'y*', 'rp', 'c1',
         'gh'
         ]

#use SVD: looks like the features need to consider class-dependent components
#this doesn't look like that classes have a separation. So just use PCA on the embedding class is not good.
# plt.figure()
# plt.title('SVD on features of each class')
# for c in range(len(classes)):
#     if c==3 or c==5:
#         A = features[c]#5000, 512
#         u, s, vh = np.linalg.svd(A)
#         # print(s)#ordered from high to low singular values
#         u_top2 = u[:, :2]
#         s_top2 = np.diag(s[:2])
#         print('u_top2.shape=', u_top2.shape)
#         two_features = np.dot(u_top2, s_top2)
#         plt.plot(two_features[:, 0], two_features[:, 1], markers[c], label=classes[c])
#         # Ahat = np.dot(np.dot(u_top2, s_top2), vh[:2, :])
#         # print('Ahat.shape=', Ahat.shape)
#         # sys.exit(1)
# plt.legend(labelcolor='linecolor')

#now use SVD and recover \hat{Y}
# plt.figure()
# #this shows only rank-1, it is already gives predictions that are near 1 and 0 -- perfect classification. Why? this is surprising.
# softmax = torch.nn.Softmax(dim=1)
# plt.title('SVD on features of each class')
# for c in range(len(classes)):
#     if c==3 or c==5:
#         A = features[c]#5000, 512
#         print('A.shape=', A.shape)
#         u, s, vh = np.linalg.svd(A)
#         # print(s)#ordered from high to low singular values
#         topk=2#1
#         u_top = u[:, :topk]
#         s_top = np.diag(s[:topk])
#         print('u_top.shape=', u_top.shape)
#         lowd_features = np.dot(u_top, s_top)
#
#         plt.subplot(1, 2, 1)
#         plt.plot(lowd_features[:, 0], lowd_features[:, 1], markers[c], label=classes[c])#however, the cats and dogs are messed together in the two feature space
#         plt.xlabel('feature 1')
#         plt.ylabel('feature 2')
#
#         features_appr = np.dot(lowd_features, vh[:topk, :])
#         output_approx = np.dot(features_appr, W.T) + b
#         output_approx = softmax(torch.as_tensor(output_approx)).cpu().numpy()
#         print('output_approx.shape=', output_approx.shape)
#         print(output_approx[0, :])
#         plt.subplot(1, 2, 2)
#         plt.plot(output_approx[:, 3], output_approx[:, 5], markers[c], label=classes[c]+'s')
#         plt.xlabel('prob(CAT)')
#         plt.ylabel('prob(DOG)')

plt.legend(labelcolor='linecolor')

#add noises into the features
# noise = np.random.randn(*features.shape) * 1.5 #how much noise to add?
# output_noisy = np.dot(features + noise, W.T) + b

#use softmax: the figure is not very typical of boundary.
# output_noisy_tensor = torch.as_tensor(output_noisy)
# softmax = torch.nn.Softmax(dim=2)
# output_noisy = softmax(output_noisy_tensor).numpy()

# print('output_noisy.shape=', output_noisy.shape)#10, 5000, 10
# #todo: do this for softmax. which is more easily interprettable
# def plot_c1_c2(c1, c2, output_noisy):
#     plt.plot(output_noisy[c1, :, c1], output_noisy[c1, :, c2], markers[c1], label=classes[c1]+'s')
#     plt.plot(output_noisy[c2, :, c1], output_noisy[c2, :, c2], markers[c2], label=classes[c2]+'s')
#     plt.xlabel('~prob({})'.format(classes[c1].upper()))
#     plt.ylabel('~prob({})'.format(classes[c2].upper()))
#     plt.legend(labelcolor='linecolor')

# plt.figure()
# plot_c1_c2(c1=5, c2=3, output_noisy=output_noisy)
#
# plt.figure()
# plot_c1_c2(c1=1, c2=3, output_noisy=output_noisy)
#
# plt.figure()
# plot_c1_c2(c1=6, c2=3, output_noisy=output_noisy)
#
# plt.figure()
# plot_c1_c2(c1=2, c2=3, output_noisy=output_noisy)
#this shows some interesting messed boundary; but the cats and dogs boundary should be bigger.

#Investigate this approach in a more principled way. Gaussian noise may be not the best you can do.


#use projected y
#NOW do SVD on the Y signals directly. Note Y softmaxed is super close to the one-hot encoding because of overfitting. So this doesn't sound a correct approach.
#but why ships are separated from the other classes?
#todo: think more about this. I think Y needs to form diag(Y1, Y2, ..., Y10)
#however, I don't know what's the results in 2D look like. remember Y is not one-hot; it is linear output
# plt.figure()
# #focus on cats and dogs ONLY to figure out a good space to visualize the boundary
# # A = np.concatenate((outputs_lin[3], outputs_lin[5], outputs_lin[2]), axis = 0)
# A = np.concatenate((outputs_lin[3], outputs_lin[5], outputs_lin[1], outputs_lin[2], outputs_lin[0]), axis = 0)
# print('A.shape=', A.shape)
#
# topk = 4 # if topk is equal to A's number of outputs, then cats and dogs are well separated; otherwise (topk is smaller), they are messed together.
#
# u, s, vh = np.linalg.svd(A)
# print('vh.shape', vh.shape)
# print('singular values:', s)#ordered from high to low singular values
# u_top2 = u[:, :topk]
# s_top2 = np.diag(s[:topk])
# print('u_top2.shape=', u_top2.shape)
# # two_features = np.dot(u_top2, s_top2)#all classes are messed together
# Y_hat = np.dot(np.dot(u_top2, s_top2), vh[:topk, :]) #this is actually \hat{Y}
# print('Y_hat.shape=', Y_hat.shape)
# # plt.plot(two_features[:5000, 0], two_features[:5000, 1], 'bo')
# # plt.plot(two_features[5000:, 0], two_features[5000:, 1], 'r+')
# plt.plot(Y_hat[:5000, 3], Y_hat[:5000, 5], 'bo', label='cats')
# plt.plot(Y_hat[5000:10000, 3], Y_hat[5000:10000, 5], 'r+', label='dogs')
# plt.plot(Y_hat[10000:15000, 3], Y_hat[10000:15000, 5], 'gs', label='cars')
# plt.plot(Y_hat[15000:20000, 3], Y_hat[15000:20000, 5], 'mx', label='birds')
# plt.plot(Y_hat[20000:25000, 3], Y_hat[20000:25000, 5], 'kp', label='planes')
# plt.legend(labelcolor='linecolor')
# plt.xlabel('Prob(CAT)')
# plt.ylabel('Prob(DOG)')

#I think using \hat{Y} from low-rank is not good. Y is already a low dimensional signal. 10 classes as dimension. It doesn't make sense to further to reduce it.
#we need to reduce the 512 features. However, we should reduce the number of features for a good approximation of Y. Not looking into SVD for the features independently.
#hope this is clear
#\Phi w = Y ==> U_k*Sigma_k*V_k w \approx Y; So this is not a typical SVD problem.

#study the condition number of the feature matrix for each class: cats and dogs are the biggest.
# cond = []
# largest_eig = []
# rank  = []
# for c in range(len(classes)):
#     A = features[c] #5000, 512
#     ATA = np.dot(A.T, A)
#     cond.append(np.linalg.cond(ATA))#infty
#     eign, eigvec = np.linalg.eig(ATA)
#     print('eigenvalues=', eign)
#     largest_eig.append(eign[0])
#     rank.append(np.linalg.matrix_rank(ATA))
# print('cond=', cond)
# plt.subplot(1, 2, 1)
# plt.plot(largest_eig, '-ko', label='largest eigenvalues of features')
# plt.subplot(1, 2, 2)#rank doesn't seem to have an interesting interpretation.
# plt.plot(rank, '--b+', label='rank of features')
# plt.xlabel('classes')
# plt.xticks(range(len(classes)), classes)


plt.show()



#loss contour






