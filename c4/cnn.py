import torch
import torch.nn as nn
import matplotlib.pylab as plt
import torchvision.transforms as T

from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize

class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, hidden_dim):
              super(BasicBlock, self).__init__()

              self.conv1 = nn.Conv2d(in_channels, hidden_dim,
                                     kernel_size=3, padding=1)
              self.conv2 = nn.Conv2d(hidden_dim, out_channels,
                                     kernel_size=3, padding=1)
              self.relu = nn.ReLU()

              self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
               x = self.conv1(x)
               x = self.relu(x)
               x = self.conv2(x)
               x = self.relu(x)
               x = self.pool(x)

               return x

class CNN(nn.Module):
       def __init__(self, num_classes):
              super(CNN, self).__init__()

              self.block1 = BasicBlock(in_channels=3, out_channels=32, hidden_dim=16)
              self.block2 = BasicBlock(in_channels=32, out_channels=128, hidden_dim=64)
              self.block3 = BasicBlock(in_channels=128, out_channels=256, hidden_dim=128)

              self.fc1 = nn.Linear(in_features=4096, out_features=2048)
              self.fc2 = nn.Linear(in_features=2048, out_features=256)
              self.fc3 = nn.Linear(in_features=256, out_features=num_classes)

              self.relu = nn.ReLU()

       def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = torch.flatten(x, start_dim=1)

            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)

            return x
       
transforms = Compose([
    RandomCrop((32, 32), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),

    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
])

training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_sampler=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(num_classes=10)

model.to(device)

#학습률 정의
lr = 1e-3

optim = Adam(model.parameters(), lr=lr)

for epoch in range(100):
    for data, label in train_loader:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

    if epoch == 0 or epoch%10 == 9:
        print(f"epoch{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "CIFAR.pth")

model.load_state_dict(torch.load("CIFAR.pth", map_location = device))

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:

        output = model(data.to(device))

        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr
    
    print(f"Accuracy:{num_corr/len(test_data)}")