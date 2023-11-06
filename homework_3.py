from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs_cifar_full')

train_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=
    transforms.ToTensor(), train=True
)

test_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=
    transforms.ToTensor(), train=False
)

train_cifar_dataloader = DataLoader(dataset=train_cifar_dataset, batch_size=1000, shuffle=True)
test_cifar_dataloader = DataLoader(dataset=test_cifar_dataset, batch_size=1000, shuffle=True)


class CIFAR100PredictorPerceptron(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        x = self.softmax(x)

        return x


model = CIFAR100PredictorPerceptron(input_size=3072, hidden_size=180, output_size=100)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 25

for epoch in range(num_epochs):
    correct_guess_train = 0
    error_train = 0
    for x, y in train_cifar_dataloader:
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        error_train += loss
        predicted_indices = torch.argmax(prediction, dim=1)
        correct_guess_train += (predicted_indices == y).float().sum()

        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss/train', error_train/len(train_cifar_dataset), epoch)
    writer.add_scalar('Accuracy/train', correct_guess_train/len(train_cifar_dataset), epoch)

    correct_guess_test = 0
    error_test = 0
    for x, y in test_cifar_dataloader:
        model.eval()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        error_test += loss
        predicted_indices = torch.argmax(prediction, dim=1)
        correct_guess_test += (predicted_indices == y).float().sum()

    writer.add_scalar('Loss/test', error_test/len(test_cifar_dataset), epoch)
    writer.add_scalar('Accuracy/test', correct_guess_test/len(test_cifar_dataset), epoch)

