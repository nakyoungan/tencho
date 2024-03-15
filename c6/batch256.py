import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

from torch.utils.data.dataset import Dataset
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

data = pd.read_csv("train.csv")

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        #RNN 층 정의
        self.rnn = nn.RNN(input_size=3, hidden_size=8, num_layers=5, batch_first=True)

        #주가를 예측하는 MLP 층 정의
        self.fc1 = nn.Linear(in_features=240, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)

        self.relu = nn.ReLU()   #활성화 함수 정의

    def forward(self,x,h0):
        x, hn = self.rnn(x, h0)     #RNN층의 출력

        #MLP층의 입력으로 사용되게 모양 변경
        x = torch.reshape(x, (x.shape[0], -1))

        #MLP층을 이용해 종가 예측
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        #예측한 종가를 1차원 벡터로 표현
        x = torch.flatten(x)

        return x


class Netflix(Dataset):
    def __init__(self):
        self.csv = pd.read_csv("train.csv")

        #데이터 정규화
        self.data = self.csv.iloc[:, 1:4].values   #개장가, 최고가, 최저가 추가 (종가와 날짜를 제외한 데이터)
        self.data = self.data / np.max(self.data)   #데이터의 범위가 100부터 400까지 커서 0과 1 사이로 정규화

        #종가 데이터 정규화
        self.label = data["Close"].values      #정답 데이터가 들어가는 변수. 파일의 종가 데이터
        self.label = self.label / np.max(self.label)

    def __len__(self):
        return len(self.data) - 30  #사용 가능한 배치 개수
        
    def __getitem__(self, i):
        data = self.data[i:i+30]   #입력 데이터 30일치 읽기
        label = self.label[i+30]    #종가 데이터 30일치 읽기

        return data, label


device = "cuda" if torch.cuda.is_available() else "cpu"

model = RNN().to(device)
dataset = Netflix()

'''
모델 학습하기

loader = DataLoader(dataset, batch_size=256)
optim = Adam(params = model.parameters(), lr=0.0001)

for epoch in range(200):
    iterator = tqdm.tqdm(loader)
    for data, label in iterator:
        optim.zero_grad()

        #초기 은닉
        h0 = torch.zeros(5, data.shape[0], 8).to(device)

        #모델 예측값
        pred = model(data.type(torch.FloatTensor).to(device), h0)

        #손실 계산
        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))

        loss.backward()
        optim.step()

        iterator.set_description(f"epoch{epoch} loss:{loss.item()}")
torch.save(model.state_dict(), "./rnn256.pth")
'''


'''
모델 평가하기
'''
loader = DataLoader(dataset, batch_size = 1)

preds = []

total_loss256 = 0

with torch.no_grad():
    model.load_state_dict(torch.load("rnn256.pth", map_location=device))

    for data, label in loader:
        h0 = torch.zeros(5, data.shape[0], 8).to(device)

        #모델 예측값
        pred = model(data.type(torch.FloatTensor).to(device), h0)
        preds.append(pred.item())

        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))

        total_loss256 += loss/len(loader)


print("배치사이즈 256 일 때:",total_loss256.item())

plt.plot(preds, label="prediction")
plt.plot(dataset.label[30:], label="actual")
plt.legend()
