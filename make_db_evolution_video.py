import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
from generate_mean_and_variance import parse_files_and_sort
from main import classes
from plot_predecessor_boundaries_nearest import nearest_neighbors
import time, os
from generate_mean_and_variance import SVD_on_feature_matrix



'''
1. sudo apt install ffmpeg
2. /usr/bin/ffmpeg -framerate 25 -i %d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -s 1024x768 db_evolution_vgg.mp4
3. Slower Video Speed: VLC is a very powerful video converter and can trim and modify videos quite easily. 
You just need to go Media->Convert/Save… Select your file and click ‘Show more options’ in the Edit Options box just type :rate=0.5 for half speed 
or :rate=2.0 for double speed at the end of the line. You’ll have to click through some options though.
'''

#video
def plot_PCA_pairwise_video(map_features, map_W, map_b, video_dir, n_components=2, c1=3, c2=5, f1=0, f2=1, n_neighbors=5, optimizer='sgd'):

    for k in sorted(map_features.keys()):
        #first make videos for the last few epochs.
        # if k <= 190:
        #     continue

        print('k=', k, '-->', map_features[k])
        features = np.load(map_features[k])
        W = np.load(map_W[k])
        b = np.load(map_b[k])
        # print('features.shape=', features.shape)
        # print('W.shape=', W.shape)
        # print('b.shape=', b.shape)

        # print(map_features[k])
        # print(map_W[k])
        # print(map_b[k])

        plt.figure()

        #pca
        c1_c2_features = np.concatenate((features[c1, :, :], features[c2, :, :]), axis=0)

        #svd
        # features_PCA, singulars, VT = SVD_on_feature_matrix(c1_c2_features, k=n_components)
        # pca_or_VT = VT

        print('c1 and c2 combined featurte.shape=', c1_c2_features.shape)
        pca = PCA(n_components=n_components)
        pca.fit(c1_c2_features)
        print('pca.components_=', pca.components_)
        print('pca.explained_variance_=', pca.explained_variance_)
        features_PCA = pca.transform(c1_c2_features)
        print('features_PCA:', features_PCA.shape)
        c1_features = features_PCA[:5000, :]
        c2_features = features_PCA[5000:10000, :]
        c1_center = c1_features.mean(axis=0)
        c2_center = c2_features.mean(axis=0)
        print('c1_center.shape=', c1_center.shape)
        print('c2_center.shape=', c2_center.shape)

        if c1_center[0] >= c2_center[0]:
            c1_features *= -1
            c2_features *= -1

        # if c1_center[0] >= c2_center[0]:
        #     c1_features[0] *= -1
        #     c2_features[0] *= -1
        # if c1_center[1] >= c2_center[1]:
        #     c1_features[1] *= -1
        #     c2_features[1] *= -1

        features_PCA = np.concatenate((c1_features, c2_features), axis=0)
        print('the flipped features_PCA.shape=', features_PCA.shape)

        plt.title(str('epoch ' + str(k)))

        #plots c1 and c2 objects in the PCA space
        plt.scatter(c1_features[:, f1], c1_features[:, f2], color='b', marker='o', label=classes[c1]+'s', alpha=0.6)
        plt.scatter(c2_features[:, f1], c2_features[:, f2], color='r', marker='+', label=classes[c2]+'s', alpha=0.6)

        #use nearest sample neighbors for any point in the PCA(2) space

        if optimizer == 'sgd':
            lows = [-10.0, -10.0]
            highs = [10.0, 10.]
        else:#adam
            lows = [-30.0, -30.0]
            highs = [30.0, 30.]

        plt.xlim([lows[0], highs[0]])
        plt.ylim([lows[1], highs[1]])
        is_c1_matrix, real_x, real_y, prob_c1, prob_c2 = nearest_neighbors(features_PCA, c1_c2_features, pca, W, b, c1, c2,
                                                                           lows=lows, highs=highs, f1=0, f2=1,
                                                                           n_neighbors=n_neighbors,
                                                                           grids=100, mode='pca')
        extent = [real_x[0], real_x[-1], real_y[0], real_y[-1]]
        plt.imshow(prob_c1.T, extent=extent)#why transpose?

        plt.savefig(video_dir + 'video_pca/' + str(k+1) + '.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--optimizer', default='sgd', type=str, help='sgd, adam')

    # parser.add_argument('--model', default='vgg19', type=str, help='model name')
    # parser.add_argument('--saved_dir', default='results/run1_save_model_every_epoch_vgg19', type=str, help='saving directory for the video')

    parser.add_argument('--model', default='vgg19', type=str, help='model name')
    parser.add_argument('--saved_dir', default='results/run2_save_model_every_epoch_vgg19/', type=str, help='saving directory for the video')

    args = parser.parse_args()
    print('args.optimizer=', args.optimizer)

    args = parser.parse_args()
    print('@@model=', args.model)
    print('@@saved_dir=', args.saved_dir)
    print('@@c1=', args.c1)
    print('@@c2=', args.c2)
    time.sleep(3)

    map_features = parse_files_and_sort(args)
    map_W = parse_files_and_sort(args, file_type='W')
    map_b = parse_files_and_sort(args, file_type='b')

    t0 = time.time()

    video_dir = args.saved_dir + '_' + classes[args.c1] + '_' + classes[args.c2] + '/'
    isExist = os.path.exists(video_dir)
    if not isExist:
        os.makedirs(video_dir)
        print("{} is created!".format(video_dir))
    plot_PCA_pairwise_video(map_features, map_W, map_b, video_dir = video_dir, c1=3, c2=5)

    print('time=', time.time()-t0)
