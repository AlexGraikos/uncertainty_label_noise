import argparse
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import seaborn as sns
from data import *
from networks import *
sns.set()

def aleatoric_train(aleatoric_net, aleatoric_optimizer, train_loader,
        val_loader, epochs, model_name, device, early_stopping_epochs=10,
        print_per=2000):
    best_val_loss = np.inf
    nll_criterion = nn.NLLLoss()
    aleatoric_net_train_loss = np.zeros(epochs)
    aleatoric_net_val_loss = np.zeros(epochs)
    train_timer = time.time()

    for e in range(epochs):
        print('[Epoch {}]'.format(e + 1))
        running_loss = 0.0
        train_loss = 0.0
        n_train_samples = 0
        epoch_timer = time.time()

        # Train over data
        for i, (images, labels) in enumerate(train_loader):
            aleatoric_net.train()
            images = images.to(device)
            labels = labels.to(device)

            # Perform training step
            aleatoric_net.zero_grad()
            pred_log_probs = aleatoric_net(images)
            aleatoric_net_loss = nll_criterion(pred_log_probs, labels)
            aleatoric_net_loss.backward()
            aleatoric_optimizer.step()

            # Log loss
            running_loss += aleatoric_net_loss.item()
            train_loss += aleatoric_net_loss.item()
            n_train_samples += pred_log_probs.shape[0]
            if (i + 1) % print_per == 0:
                print('[{}, {}]'.format(e + 1, i + 1),
                      'Train Loss: {:.3f}'.format(running_loss
                                                 / (print_per
                                                    * pred_log_probs.shape[0])))
                running_loss = 0.0

        aleatoric_net_train_loss[e] = train_loss / n_train_samples
        print('[Epoch {}]'.format(e + 1),
              'Train Loss: {:.3f}'.format(aleatoric_net_train_loss[e]),
              'Time: {:.3f}s'.format(time.time() - epoch_timer))

        # Compute validation loss
        val_loss = 0.0
        n_val_samples = 0
        val_timer = time.time()

        for i, (images, labels) in enumerate(val_loader):
            aleatoric_net.eval()
            images = images.to(device)
            labels = labels.to(device)

            # Compute loss
            pred_log_probs = aleatoric_net(images)
            aleatoric_net_loss = nll_criterion(pred_log_probs, labels)
            val_loss += aleatoric_net_loss.item()
            n_val_samples += pred_log_probs.shape[0]

        aleatoric_net_val_loss[e] = val_loss / n_val_samples
        print('[Epoch {}]'.format(e + 1),
              'Validation Loss: {:.3f}'.format(aleatoric_net_val_loss[e]),
              'Time: {:.3f}s'.format(time.time() - val_timer))

        # Save best validation loss models
        if aleatoric_net_val_loss[e] < best_val_loss:
            best_val_loss = aleatoric_net_val_loss[e]

            model_path = 'models/aleatoric_{}_{}.pth'.format(
                model_name, str(e + 1))
            print('[*] Saving model at', model_path)
            torch.save({'epoch': e,
                        'model_state_dict': aleatoric_net.state_dict(),
                        'optimizer_state_dict': aleatoric_optimizer.state_dict(),
                        'loss': aleatoric_net_train_loss[-1],},
                      model_path)
        # Stop early if no improvement over the last early_stopping_epochs
        elif (e + 1) >= early_stopping_epochs:
            start_idx = e - early_stopping_epochs + 1
            end_idx = e + 1
            if best_val_loss not in aleatoric_net_val_loss[start_idx:end_idx]:
                print('[*] Stopping early at epoch', e + 1)
                break

    print('[*] Total training time: {:.3f}'.format(time.time() - train_timer))
    return aleatoric_net_train_loss, aleatoric_net_val_loss
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Train the aleatoric uncertainty model.')
    parser.add_argument('model_name', type=str,
                        help='Model name to use when saving.',
                        metavar='<model_name>')
    parser.add_argument('--batch_size', type=int,
                        help='Train/validation set batch size.',
                        metavar='<batch_size>',
                        default=4)
    parser.add_argument('--logit_mc_samples', type=int,
                        help='Number of MC samples for logit expectation.',
                        metavar='<mc_samples>',
                        default=1000)
    parser.add_argument('--temperature', type=float,
                        help='Softmax temperature parameter.',
                        metavar='<temp>',
                        default=1.0)
    parser.add_argument('--skip_clean', action='store_true',
                        help='Skip training on clean dataset.',
                        default=False)
    parser.add_argument('--skip_noisy', action='store_true',
                        help='Skip training on noisy dataset.',
                        default=False)
    args = parser.parse_args()

    # Train aleatoric CNN on clean and noisy datasets
    print('[*] Loading data...')
    loader_tuple = load_cifar10(batch_size=args.batch_size)
    (train_loader, val_loader,
     train_clean_loader, val_clean_loader, _) = loader_tuple 

    if not args.skip_clean:
        print('[*] Training on clean dataset...')
        aleatoric_net = AleatoricCNN(in_channels=3, n_classes=10, 
                                     mc_samples=args.logit_mc_samples,
                                     temp=args.temperature)
        lr = 0.001
        aleatoric_optimizer = optim.Adam(aleatoric_net.parameters(), lr=lr,
                                         betas=(0.9, 0.999))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        aleatoric_net = aleatoric_net.to(device)
        aleatoric_clean_train_loss, aleatoric_clean_val_loss = aleatoric_train(
            aleatoric_net, aleatoric_optimizer, 
            train_clean_loader, val_clean_loader,
            epochs=1000, model_name=(args.model_name + '_clean'), 
            device=device)

        plt.figure()
        trained_epochs = len(
            aleatoric_clean_train_loss[aleatoric_clean_train_loss != 0])
        plt.plot(range(1, trained_epochs + 1),
                 aleatoric_clean_train_loss[:trained_epochs])
        plt.plot(range(1, trained_epochs + 1),
                 aleatoric_clean_val_loss[:trained_epochs])
        plt.title('Aleatoric Model Losses - Clean Dataset')
        plt.legend(['Train', 'Validation'])

    if not args.skip_noisy:
        print('[*] Training on noisy dataset...')
        aleatoric_net = AleatoricCNN(in_channels=3, n_classes=10, 
                                     mc_samples=args.logit_mc_samples,
                                     temp=args.temperature)
        lr = 0.001
        aleatoric_optimizer = optim.Adam(aleatoric_net.parameters(), lr=lr,
                                         betas=(0.9, 0.999))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        aleatoric_net = aleatoric_net.to(device)
        aleatoric_noisy_train_loss, aleatoric_noisy_val_loss = aleatoric_train(
            aleatoric_net, aleatoric_optimizer, 
            train_loader, val_loader,
            epochs=1000, model_name=(args.model_name + '_noisy'), 
            device=device)

        plt.figure()
        trained_epochs = len(
            aleatoric_noisy_train_loss[aleatoric_noisy_train_loss != 0])
        plt.plot(range(1, trained_epochs + 1),
                 aleatoric_noisy_train_loss[:trained_epochs])
        plt.plot(range(1, trained_epochs + 1),
                 aleatoric_noisy_val_loss[:trained_epochs])
        plt.title('Aleatoric Model Losses - Noisy Dataset')
        plt.legend(['Train', 'Validation'])

    plt.show()

