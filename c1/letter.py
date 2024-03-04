'''
손글씨 분류하기 : 다중분류
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim.adam import Adam

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from torch.utils.data.dataloader import DataLoader

training_data = MNIST(root="./", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="./", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

#평가용은 데이터를 섞을 필요가 없음
test_loader = DataLoader(training_data, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

model.to(device)

lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epoch in range(20):
    for data, label in train_loader:
        optim.zero_grad()

        data = torch.reshape(data, (-1, 784)).to(device)
        preds = model(data)

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

    print(f"epoch{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "MNIST.pth")