'''
보스턴 집값 예측하기 : 회귀분석
'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_boston
from torch.optim.adam import Adam


dataset = load_boston()
dataFrame = pd.DataFrame(dataset["data"])
dataFrame.columns = dataset["feature_names"]
dataFrame["target"] = dataset["target"]

#신경망 모델 정의 입력된 층들이 순서대로 계산됨
model = nn.Sequential(
    nn.Linear(13,100),      #MLP 모델 (입력차원, 출력차원) 13개 특징 입력받아 100개 특징 반환
    nn.ReLU(),              #ReLU 함수
    nn.Linear(100,1)        #은닉층 뉴런 100개에서 1개의 출력 
)

X = dataFrame.iloc[:, :13].values   #정답을 제외한 특징을 X에 입력
Y = dataFrame["target"].values      #데이터 프레임의 타겟 값을 추출

batch_size = 100
learning_rate = 0.001

optim = Adam(model.parameters(), lr = learning_rate)        #최적화 기법

#epoch 반복
for epoch in range(200):
        
    #batch 반복
    for i in range(len(X)//batch_size):
        start = i*batch_size
        end = start + batch_size

        #파이토치 실수형 텐서로 변환
        x = torch.FloatTensor(X[start:end])
        y = torch.FloatTensor(Y[start:end])

        optim.zero_grad()       #가중치의 기울기를 0으로 초기화 (이전 배치에서 계산된 기울기가 남아있기 때문에 배치마다 초기화)
        preds = model(x)        #모델의 예측값 계산
        loss = nn.MSELoss()(preds, y)   #MSE손실계산
        loss.backward()     #오차역전파 (모든 가중치에 대한 기울기 계산)
        optim.step()        #최적화 진행 (가중치 얼마나 수정할지 정함)

    if epoch % 20 == 0:
        print(f"epoch{epoch} loss:{loss.item()}")

prediction = model(torch.FloatTensor(X[0, :13]))
real = Y[0]
print(f"prediction = {prediction.item()} real:{real}")