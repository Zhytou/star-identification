import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import config_name, num_false_star
from generate import point_dataset_path, num_class, num_input
from dataset import StarPointDataset
from train_fnn import FeedforwardNeuralNetModel


def check_accuracy(model: nn.Module, loader: DataLoader, device=torch.device('cpu')):
    '''
        Check the accuracy of the model.
    Args:
        model: the model to be checked
        loader: the data loader for validation or testing
    '''
    # set the model into evaluation model
    model.eval()
    # initialize the number of correct predictions
    correct = 0
    total = 0
    # Iterate through test dataset
    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)
        # Forward pass only to get logits/output
        outputs = model(points)
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        # Total number of labels
        total += labels.size(0)
        # Total correct predictions
        correct += (predicted == labels).sum().item()
    return round(100.0 * correct / total, 2)


if __name__ == '__main__':
    batch_size = 100
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # define datasets
    pos_test_dataset, mv_test_dataset, fs_test_dataset = [StarPointDataset(os.path.join(point_dataset_path, name)) for name in ['positional_noise_test', 'magnitude_noise_test', f'false_star_test/{num_false_star}']]
    # print datasets' sizes
    print(f'Training set: {len(pos_test_dataset)}, Validation set: {len(mv_test_dataset)}, Test set: {len(fs_test_dataset)}')

    # define data loaders
    pos_test_loader = DataLoader(pos_test_dataset, batch_size, shuffle=True)
    mv_test_loader = DataLoader(mv_test_dataset, batch_size, shuffle=False)
    fs_test_loader = DataLoader(fs_test_dataset, batch_size, shuffle=False)

    # load best model
    for model_type in ['fnn']:
        best_model = FeedforwardNeuralNetModel(num_input, num_class)
        model_path = f'model/{config_name}/{model_type}/best_model.pth'
        if not os.path.exists(model_path):
            print(f'{model_path} does not exist!')
            continue
        best_model.load_state_dict(torch.load(model_path))
        best_model.to(device)

        pos_accuracy = check_accuracy(best_model, pos_test_loader, device)
        mv_accuracy = check_accuracy(best_model, mv_test_loader, device)
        fs_accuracy = check_accuracy(best_model, fs_test_loader, device)
        print(f'{model_type} model pos_accuracy: {pos_accuracy}% mv_accuracy: {mv_accuracy}% fs_accuracy: {fs_accuracy}%')
