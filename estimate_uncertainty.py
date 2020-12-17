import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data import *
from networks import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estimate uncertainty for data samples.')
    parser.add_argument('model_path', type=str,
                        help='Path to uncertainty estimation model.',
                        metavar='path/to/model.pth')
    parser.add_argument('subset', type=str,
                        help='Subset for which to estimate uncertainty.',
                        choices=['all', 'val'],
                        metavar='<all/val>')
    parser.add_argument('save_target', type=str,
                        help='File to save estimations.',
                        metavar='path/to/estimations.npz')
    parser.add_argument('--mc_dropout_samples', type=int,
                        help='Number of MC samples for dropout.',
                        metavar='<mc_samples>',
                        default=50)
    parser.add_argument('--visualize_estimations', action='store_true',
                        help='Plot samples and their uncertainty estimations.',
                        default=False)
    args = parser.parse_args()

    # Load data
    if args.subset == 'val':
        (_, data_loader, _, data_clean_loader, _) = load_cifar10(
            batch_size=1, train_shuffle=False)
    if args.subset == 'all':
        (data_loader, _, data_clean_loader, _, _) = load_cifar10(
            batch_size=1, split_train=False, train_shuffle=False)

    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck')

    # Load model state
    model = AleatoricCNN(in_channels=3, n_classes=10)
    criterion = nn.NLLLoss()
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = model.to(device)
    model.eval()

    predictive_uncertainty_samples = []
    aleatoric_uncertainty_samples = []
    epistemic_uncertainty_samples = []
    loss_samples = []
    pred_labels = []
    clean_labels = []
    noisy_labels = []

    # MC dropout Sampling - enable dropout during eval 
    model.apply(lambda m: m.train() if type(m) == nn.Dropout else None)
    images, _ = iter(data_loader).next()
    output_size = model(images.to(device)).size()

    print('[*] Predicting uncertainty for samples...')
    for i, (images, labels) in enumerate(data_loader):
        if (i + 1) % 100 == 0:
            print('[{}/{}]'.format(i + 1, len(data_loader)))
        images = images.to(device)
        labels = labels.to(device)

        mc_probs = torch.zeros(output_size, device=device)
        aleatoric = 0
        # MC samples of network outputs
        with torch.no_grad():
            for t in range(args.mc_dropout_samples):
                pred_log_probs = model(images)
                mc_probs = mc_probs + torch.exp(pred_log_probs)

                # Accumulate the sample entropies to estimate aleatoric uncertainty
                sample_entropy = torch.distributions.Categorical(
                    probs=torch.exp(pred_log_probs)).entropy().item()
                aleatoric = aleatoric + sample_entropy

            # Compute uncertainties and losses
            mc_probs = mc_probs / args.mc_dropout_samples
            aleatoric = aleatoric / args.mc_dropout_samples
            predictive = torch.distributions.Categorical(
                probs=mc_probs).entropy().item()
            epistemic = predictive - aleatoric
            model_loss = criterion(mc_probs, labels)

            predictive_uncertainty_samples.append(predictive)
            aleatoric_uncertainty_samples.append(aleatoric)
            epistemic_uncertainty_samples.append(epistemic)
            loss_samples.append(model_loss.item())
            pred_labels.append(np.argmax(mc_probs.cpu()))
            noisy_labels.append(labels.cpu().item())
            clean_labels.append(data_clean_loader.dataset[i][1])

            # Visualize input labels and predicted uncertainty
            if args.visualize_estimations:
                img_np = np.transpose(images.cpu().numpy()[0, :], (1, 2, 0))
                plt.imshow(img_np / 2 + 0.5)
                plt.title('Noisy {} Correct {}: Predicted: {}'.format(
                    classes[labels.cpu().item()],
                    classes[data_clean_loader.dataset[i][1]],
                    classes[np.argmax(mc_probs.cpu())]))
                print('Predictive: {:.4f}'.format(predictive), 
                      'Aleatoric: {:.4f}'.format(aleatoric),
                      'Epistemic: {:.4f}'.format(epistemic))
                plt.show()

# Save estimation results to target fle
predictive_uncertainty_samples_array = np.array(predictive_uncertainty_samples)
aleatoric_uncertainty_samples_array = np.array(aleatoric_uncertainty_samples)
epistemic_uncertainty_samples_array = np.array(epistemic_uncertainty_samples)
loss_samples_array = np.array(loss_samples)
pred_labels_array = np.array(pred_labels)
noisy_labels_array = np.array(noisy_labels)
clean_labels_array = np.array(clean_labels)

np.savez(args.save_target,
         predictive_uncertainty_samples_array,
         aleatoric_uncertainty_samples_array,
         epistemic_uncertainty_samples_array,
         loss_samples_array,
         pred_labels_array,
         noisy_labels_array,
         clean_labels_array)

