import numpy as np
import torch
import torch.optim as optim
import argparse
from data import *
from networks import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model.')
    parser.add_argument('model_path', type=str,
                        help='Path to model.',
                        metavar='path/to/model.pth')
    parser.add_argument('model', type=str,
                        help='Model.',
                        choices=['baseline', 'aleatoric'],
                        metavar='<baseline/aleatoric>')
    args = parser.parse_args()

    print('[*] Loading model', args.model, 'from', args.model_path)
    # Select model
    if args.model == 'baseline':
        model = BaselineCNN(in_channels=3, n_classes=10)
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'aleatoric':
        model = AleatoricCNN(in_channels=3, n_classes=10)
        criterion = nn.NLLLoss()

    # Load model state
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = model.to(device)
    model.eval()

    # Test model
    print('[*] Loading data...')
    _, _, _, _, test_loader = load_cifar10()

    print('[*] Evaluating on test set...')
    test_loss = 0.0
    correct_predictions = 0
    n_test_samples = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Compute loss/accuracy
        output = model(images)
        model_loss = criterion(output, labels)
        test_loss += model_loss.item()
        with torch.no_grad():
            correct_predictions += np.sum(
                np.argmax(output.cpu().numpy(), axis=1) 
                == labels.cpu().numpy())
        n_test_samples += labels.shape[0]

    accuracy = correct_predictions / n_test_samples
    test_loss = test_loss / n_test_samples
    print('[*] Test Loss: {:.3f}'.format(test_loss),
          'Accuracy: {:.3f}'.format(accuracy))

