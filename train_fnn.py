import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dataloader, random_split

from dataset import StarPointDataset
from generate import star_num_per_sample, num_classes


# training setting
batch_size = 10
n_iters = 6000
learning_rate = 0.1

# define datasets for training & validation
dataset = StarPointDataset('data/star_points/labels.csv')
train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])

# create data loaders for our datasets
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# print split sizes
print(f'Training set has {len(train_dataset)} instances')

class FeedforwardNeuralNetModel(nn.Module):
    # model structure refers to https://www.mdpi.com/1424-8220/20/13/3684
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = [300, 100]):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout1d(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout1d(0.2),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out
    

model = FeedforwardNeuralNetModel(star_num_per_sample * 2, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

iter = 0
for epoch in range(num_epochs):
    for points, labels in train_loader:
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        scores = model(points)
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(scores, labels)
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()
        iter += 1
        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for points, labels in test_loader:
                # Forward pass only to get logits/output
                outputs = model(points)
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                # Total number of labels
                total += batch_size
                # Total correct predictions
                correct += (predicted == labels).sum().item()
            accuracy = round(100.0 * correct / total, 2)
            # Print Loss
            print(f'Iteration: {iter}. Loss: {loss.item()}. Accuracy: {accuracy}%')