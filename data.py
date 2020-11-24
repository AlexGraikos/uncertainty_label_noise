import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def load_cifar10(plot_samples=False):
    # Load CIFAR-10 dataset
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./datasets/cifar',
                                             train=True, download=True,
                                             transform=transform)
    train_set_clean = torchvision.datasets.CIFAR10(root='./datasets/cifar',
                                                   train=True, download=True,
                                                   transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./datasets/cifar',
                                            train=False, download=True,
                                            transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')

    # Add synthetic label noise
    print('[*] Generating noisy dataset...')
    
    class_label_noise = [0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # Ratio of noisy images per class
    train_set_labels = np.array(train_set.targets)
    train_set_labels_noisy = train_set_labels.copy()
    
    for i in range(len(classes)):
        class_sample_idxs = np.argwhere(train_set_labels == i) 
        if class_label_noise[i] != 0:
            # Select samples to flip for each class
            samples_to_flip = np.random.choice(
                class_sample_idxs[:, 0], 
                int(class_label_noise[i] * len(class_sample_idxs)),
                replace=False)

            # Flip chosen samples to another random class
            flip_options = list(range(len(classes)))
            flip_options.remove(i)
            train_set_labels_noisy[samples_to_flip] = np.random.choice(
                flip_options, len(samples_to_flip), replace=True)
        
        # Check successful label flips
        class_sample_idxs = np.argwhere(train_set_labels == i)
        differences = np.sum(train_set_labels[class_sample_idxs] 
                             != train_set_labels_noisy[class_sample_idxs])
        print('[*] Ratio of noisy images in class {}:'.format(i + 1), 
              differences / len(class_sample_idxs))
    # Set train set labels to noisy
    train_set.targets = train_set_labels_noisy.tolist()

    # Split datasets set to train/validation
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [int(0.8 * len(train_set_labels)), 
                    int(0.2 * len(train_set_labels))])
    train_clean_subset, val_clean_subset = torch.utils.data.random_split(
        train_set_clean, [int(0.8 * len(train_set_labels)), 
                          int(0.2 * len(train_set_labels))])

    # Create data loaders
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(
        dataset=train_subset, shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_subset, shuffle=False, batch_size=batch_size)
    train_clean_loader = torch.utils.data.DataLoader(
        dataset=train_clean_subset, shuffle=True, batch_size=batch_size)
    val_clean_loader = torch.utils.data.DataLoader(
        dataset=val_clean_subset, shuffle=False, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=1)

    # Plot training clean/noisy batches
    images, labels = iter(train_loader).next()
    images_clean, labels_clean = iter(train_clean_loader).next()

    if plot_samples:
        plt.figure(figsize=(8, 2.5))
        for i in range(batch_size):
            plt.subplot(1, batch_size, i + 1)
            img_np = np.transpose(images_clean.numpy()[i, :], (1, 2, 0))
            plt.imshow(img_np / 2 + 0.5)
            plt.title(classes[labels_clean[i]])
        plt.suptitle('Clean Dataset')

        plt.figure(figsize=(8, 2.5))
        for i in range(batch_size):
            plt.subplot(1, batch_size, i + 1)
            img_np = np.transpose(images.numpy()[i, :], (1, 2, 0))
            plt.imshow(img_np / 2 + 0.5)
            plt.title(classes[labels[i]])
        plt.suptitle('Noisy Dataset')

    return (train_loader, val_loader,
            train_clean_loader, val_clean_loader,
            test_loader)
            

if __name__ == '__main__':
    (train_loader, val_loader,
     train_clean_loader, val_clean_loader,
     test_loader) = load_cifar10(True)
    plt.show()
