import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dataloader, random_split

from dataset import StarPointDataset
from generate import star_num_per_sample, num_classes


# training setting
batch_size = 4
n_iters = 3000
learning_rate = 0.1

# define datasets for training & validation
dataset = StarPointDataset('data/star_points/labels.csv')
train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])

# create data loaders for our datasets
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# print split sizes
print('Training set has {} instances'.format(len(train_dataset)))

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 

        # Non-linearity
        self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function
        out = self.fc1(x)

        # Non-linearity
        out = self.sigmoid(out)

        # Linear function (readout)
        out = self.fc2(out)
        return out
    
input_dim = star_num_per_sample * 2
hidden_dim = 100
output_dim = num_classes

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
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
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))