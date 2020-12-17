import argparse
import numpy as np
from sklearn.cluster import DBSCAN
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import seaborn as sns
sns.set()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correct data labels.')
    parser.add_argument('precomputed_vals', type=str,
                        help='Path to precomputed uncertainty/loss values.',
                        metavar='path/to/precomputed.npz')
    parser.add_argument('--uncertainty_threshold', type=float,
                        help='Uncertainty threshold after clustering.',
                        metavar='<threshold>',
                        default=0.3)
    args = parser.parse_args()

    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck')

    # Load precomputed uncertainty and loss values
    npzfile = np.load(args.precomputed_vals, allow_pickle=True)
    predictive_uncertainty_samples_array = npzfile['arr_0']
    aleatoric_uncertainty_samples_array = npzfile['arr_1']
    epistemic_uncertainty_samples_array = npzfile['arr_2']
    loss_samples_array = npzfile['arr_3']
    pred_labels_array = npzfile['arr_4']
    noisy_labels = npzfile['arr_5']
    clean_labels = npzfile['arr_6']

    uncertainty_loss_samples = np.stack((aleatoric_uncertainty_samples_array,
                                         loss_samples_array), axis=-1)

    # Uncertainty-Loss plot
    incorrect_idx = noisy_labels != clean_labels
    colors = [plt.cm.tab10(3) if i in np.argwhere(incorrect_idx)
              else plt.cm.tab10(0) for i in range(len(loss_samples_array))]
    legend_elements = [lines.Line2D([0], [0], marker='o',
                                    markerfacecolor=plt.cm.tab10(0),
                                    markeredgecolor=plt.cm.tab10(0),
                                    markersize=5,
                                    lw=0, label='Clean'),
                       lines.Line2D([0], [0], marker='o',
                                    markerfacecolor=plt.cm.tab10(3),
                                    markeredgecolor=plt.cm.tab10(3),
                                    markersize=5,
                                    lw=0, label='Noisy')]

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(uncertainty_loss_samples[:, 0], 
                          uncertainty_loss_samples[:, 1], s=1, c=colors)
    plt.xlabel('Aleatoric Uncertainty')
    plt.ylabel('Loss')
    plt.legend(handles=legend_elements, loc='lower right')
    plt.title('Uncertainty - Loss plot of training samples')

    # Estimated uncertainty per noisy class violin plot
    plt.figure()
    plt.violinplot(
        dataset=[aleatoric_uncertainty_samples_array[noisy_labels == i]
        for i in range(len(classes))],
        showmeans=False, showextrema=False, showmedians=True, quantiles=None,
        points=1000)
    plt.xticks(list(range(1, len(classes) + 1)), classes)
    plt.title('Estimated uncertainty per class')

    # Estimated uncertainty for noisy samples
    plt.figure()
    plt.violinplot(
        dataset=[aleatoric_uncertainty_samples_array[np.logical_and(
        noisy_labels == i, clean_labels != i)]
        for i in range(len(classes))],
        showmeans=False, showextrema=False, showmedians=True, quantiles=None,
        points=1000)
    plt.xticks(list(range(1, len(classes) + 1)), classes)
    plt.title('Estimated uncertainty for noisy samples')

    # Density-based clustering
    clustering = DBSCAN(eps=3e-2, min_samples=50, metric='wminkowski',
                        metric_params={'p': 2, 'w': [0.1, 100]})
    clustering.fit(uncertainty_loss_samples)

    # Plot clustering result
    colors = [plt.cm.tab10(i + 1) for i in clustering.labels_]
    legend_elements = [lines.Line2D([0], [0], marker='o',
                                    markerfacecolor=plt.cm.tab10(i + 1),
                                    markeredgecolor=plt.cm.tab10(i + 1),
                                    markersize=5,
                                    lw=0, label=('Cluster ' + str(i)))
                       for i in np.unique(clustering.labels_)]

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(uncertainty_loss_samples[:, 0], 
                          uncertainty_loss_samples[:, 1], s=1, c=colors)
    plt.xlabel('Aleatoric Uncertainty')
    plt.ylabel('Loss')
    plt.legend(handles=legend_elements, loc='lower right')
    plt.title('DBSCAN Clustering')

    # Select cluster closest to (0, 0) as the noisy sample cluster
    cluster_labels = np.unique(clustering.labels_)
    cluster_centroids = [
        np.mean(uncertainty_loss_samples[clustering.labels_ == i], axis=0)
        for i in cluster_labels]
    dist_from_origin = dist.cdist(cluster_centroids, np.array([[0, 0]]))
    noise_cluster = cluster_labels[np.argmin(np.squeeze(dist_from_origin))]
    clusters = clustering.labels_.copy()
    # Set other clusters as background and threshold uncertainty
    clusters[clusters != noise_cluster] = -1
    clusters[clusters == noise_cluster] = 1
    clusters[uncertainty_loss_samples[:, 0] > args.uncertainty_threshold] = -1

    # Plot noisy sample selection
    colors = [plt.cm.tab10(3) if clusters[i] == 1
              else plt.cm.tab10(0) for i in range(len(clusters))]

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(uncertainty_loss_samples[:, 0], 
                          uncertainty_loss_samples[:, 1], s=1, c=colors)
    plt.xlabel('Aleatoric Uncertainty')
    plt.ylabel('Loss')
    legend_elements = [lines.Line2D([0], [0], marker='o',
                                    markerfacecolor=plt.cm.tab10(0),
                                    markeredgecolor=plt.cm.tab10(0),
                                    markersize=5,
                                    lw=0, label='Clean'),
                       lines.Line2D([0], [0], marker='o',
                                    markerfacecolor=plt.cm.tab10(3),
                                    markeredgecolor=plt.cm.tab10(3),
                                    markersize=5,
                                    lw=0, label='Noisy')]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.title('Noisy Samples Selected from clustering')

    # Use predicted labels as true and compute noisy/clean rate
    fixed_labels = noisy_labels.copy()
    print('Noise rate before fixing', np.mean(fixed_labels != clean_labels))
    fixed_labels[clusters == 1] = pred_labels_array[clusters == 1]
    print('Noise rate after fixing', np.mean(fixed_labels != clean_labels))


    plt.show()

