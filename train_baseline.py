import numpy as np
import torch
import torch.optim as optim
from data import *
from networks import *

def baseline_train(baseline_net, baseline_optimizer, train_loader,
        val_loader, epochs, model_name, device, early_stopping_epochs=10,
        print_per=2000):
    best_val_loss = np.inf
    ce_criterion = nn.CrossEntropyLoss()
    baseline_net_train_loss = np.zeros(epochs)
    baseline_net_val_loss = np.zeros(epochs)

    for e in range(epochs):
        print('[Epoch {}]'.format(e + 1))
        running_loss = 0.0
        train_loss = 0.0
        n_train_samples = 0
        # Train over data
        for i, (images, labels) in enumerate(train_loader):
            baseline_net.train()
            images = images.to(device)
            labels = labels.to(device)

            # Perform training step
            baseline_net.zero_grad()
            pred_labels = baseline_net(images)
            baseline_net_loss = ce_criterion(pred_labels, labels)
            baseline_net_loss.backward()
            baseline_optimizer.step()

            # Log loss
            running_loss += baseline_net_loss.item()
            train_loss += baseline_net_loss.item()
            n_train_samples += 1
            if (i + 1) % print_per == 0:
                print('[{}, {}]'.format(e + 1, i + 1),
                      'Train Loss: {:.3f}'.format(running_loss / print_per))
                running_loss = 0.0

        print('[Epoch {}]'.format(e + 1),
              'Train Loss: {:.3f}'.format(train_loss / n_train_samples))
        baseline_net_train_loss[e] = train_loss / n_train_samples

        # Compute validation loss
        val_loss = 0.0
        n_val_samples = 0
        for i, (images, labels) in enumerate(val_loader):
            baseline_net.eval()
            images = images.to(device)
            labels = labels.to(device)

            # Compute loss
            pred_labels = baseline_net(images)
            baseline_net_loss = ce_criterion(pred_labels, labels)
            val_loss += baseline_net_loss.item()
            n_val_samples += 1

        print('[Epoch {}]'.format(e + 1),
              'Validation Loss: {:.3f}'.format(val_loss / n_val_samples))
        baseline_net_val_loss[e] = val_loss / n_val_samples

        # Save best validation loss models
        if baseline_net_val_loss[e] < best_val_loss:
            best_val_loss = baseline_net_val_loss[e]

            model_path = 'models/baseline_{}_{}.pth'.format(
                model_name, str(e + 1))
            print('[*] Saving model at', model_path)
            torch.save({'epoch': e,
                        'model_state_dict': baseline_net.state_dict(),
                        'optimizer_state_dict': baseline_optimizer.state_dict(),
                        'loss': baseline_net_train_loss[-1],},
                      model_path)
        # Stop early if no improvement over early_stopping_epochs
        elif e + 1 > early_stopping_epochs:
            if best_val_loss not in baseline_net_val_loss[-early_stopping_epochs:]:
                print('[*] Stopping early at epoch', e + 1)
                break
    return baseline_net_train_loss, baseline_net_val_loss
    
if __name__ == '__main__':
    # Train baseline CNN on clean and noisy datasets
    print('[*] Loading data...')
    (train_loader, val_loader,
     train_clean_loader, val_clean_loader,
     test_loader) = load_cifar10()

    print('[*] Creating network...')
    baseline_net = BaselineCNN(in_channels=3, n_classes=10)
    lr = 0.001
    baseline_optimizer = optim.Adam(baseline_net.parameters(), lr=lr,
                                    betas=(0.9, 0.999))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_net = baseline_net.to(device)

    print('[*] Training on clean dataset...')
    baseline_clean_train_loss, baseline_clean_val_loss = baseline_train(
        baseline_net, baseline_optimizer, train_clean_loader, val_clean_loader,
        epochs=1000, model_name='clean', device=device)

    plt.figure()
    trained_epochs = len(baseline_clean_train_loss)
    plt.plot(range(1, trained_epochs + 1), baseline_clean_train_loss)
    plt.plot(range(1, trained_epochs + 1), baseline_clean_val_loss)
    plt.title('Baseline model losses on clean dataset')
    plt.legend(['Train', 'Validation'])

    plt.show()

