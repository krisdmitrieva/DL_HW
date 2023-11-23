import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(2023)

writer = SummaryWriter(comment='CNN')
train_indices = torch.arange(5000)
test_indices = torch.arange(1500)

train_cifar_dataset = Subset(datasets.CIFAR100(
    train=True,
    download=True,
    root='./',
    transform=transforms.ToTensor()),
    train_indices
)

test_cifar_dataset = Subset(datasets.CIFAR100(
    train=False,
    download=True,
    root='./',
    transform=transforms.ToTensor()),
    test_indices
)

train_dl = DataLoader(train_cifar_dataset, batch_size=1)
test_dl = DataLoader(test_cifar_dataset, batch_size=1)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        return F.softmax(x, dim=1)


model = ConvNet()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 20

for epoch in range(num_epochs):
    error = 0
    for x, y in train_dl:
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        error += loss

        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss', error/len(train_cifar_dataset), epoch)

    correct_guess = 0
    for x, y in test_dl:
        model.eval()
        prediction = model(x)
        predicted_indices = torch.argmax(prediction, dim=1)
        correct_guess += (predicted_indices == y).float().sum()

    writer.add_scalar('Accuracy', correct_guess/len(test_cifar_dataset), epoch)
    print('Accuracy: ', correct_guess/len(test_cifar_dataset))
