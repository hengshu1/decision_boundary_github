import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_per_sample_loss import argmax3d, argmin3d

f_softmax = torch.nn.Softmax(dim=2)

file_header = 'results/'
#note this code: it actually the model output; now we need to compute the softmax
# lin_output = np.load(file_header + 'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyccat_dog_egomodels_limit_theta0.001_softmax.npy')
lin_output = np.load(file_header +  'model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_cat_dog_egomodels_limit_theta1.0_softmax.npy')#more disturbance/perturbation

tensor =  torch.as_tensor(lin_output)
print('tensor.shape=', tensor.shape)
softmax = f_softmax(tensor)


def plot_one_class_objects(class_softmax, color, marker, label):
     #this is the optimal model
     probCAT_opt = class_softmax[:, 3, 4, 4]
     probDOG_opt = class_softmax[:, 5, 4, 4]

     # this looks right: almost all cats have near one probability
     # plt.figure()
     # plt.plot(probCAT_opt, '-b+', label='prob(CAT)')
     # plt.plot(probDOG_opt, '--ko', label='prob(DOG)')
     # plt.xlabel('cats')
     # plt.legend()

     #prob(CAT) and prob(CAT) for the same samples
     cat_probCAT = np.moveaxis(class_softmax[:, 3, :, :], 0, -1)
     cat_probDOG = np.moveaxis(class_softmax[:, 5, :, :], 0, -1)
     # print('cat_probCAT.shape=', cat_probCAT.shape)
     # print('cat_probDOG.shape=', cat_probDOG.shape)
     #
     max_dog, i_dog, j_dog = argmax3d(cat_probDOG)
     print(i_dog.shape)
     print(j_dog.shape)
     max_dog2 = np.zeros((len(max_dog)))
     max_cat = np.zeros((len(max_dog)))
     for i in range(len(max_dog2)):
          max_dog2[i] = cat_probDOG[int(i_dog[i]), int(j_dog[i]), i]
          max_cat[i] = cat_probCAT[int(i_dog[i]), int(j_dog[i]), i]
     print(max_dog[:20])
     print(max_dog2[:20])

     #independent worst ego models
     # max_cat, i_cat, j_cat = argmin3d(cat_probCAT)
     # max_dog, i_dog, j_dog = argmin3d(cat_probDOG)

     plt.plot(max_cat, max_dog, color+marker, label=label)
     plt.xlabel('prob(CAT)')
     plt.ylabel('prob(DOG)')
     plt.legend(labelcolor='linecolor')
     plt.title('maximal prob(DOG): dog is near 1')

print('softmax.shape=', softmax.shape)


plt.figure()
plot_one_class_objects(class_softmax=softmax[0].numpy(), color='b', marker='+', label='cats')#cats
plt.figure()
plot_one_class_objects(class_softmax=softmax[1].numpy(), color='r', marker='o', label='dogs')#dogs

plt.show()

'''
conclusion:
no interesting boundary structure in the Prob(CAT) and Prob(DOG) space. 
'''



