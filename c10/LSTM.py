import torch
import torch.nn as nn
import numpy as np
import glob
import pandas as pd
import os
import string
import tqdm

from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch.utils.data.dataset import Dataset


class TextGeneration(Dataset):
    def clean_text(self, txt):
        # 모든 단어를 소문자로 바꾸고 특수문자를 제거
        txt = "".join(v for v in txt if v not in string.punctuation).lower()
        return txt
    def __init__(self):
        all_headlines = []

        # ❶ 모든 헤드라인의 텍스트를 불러옴
        for filename in glob.glob("./data/*.csv"):
            if 'Articles' in filename:
                article_df = pd.read_csv(filename)

                # 데이터셋의 headline의 값을 all_headlines에 추가
                all_headlines.extend(list(article_df.headline.values))
                break

        # ❷ headline 중 unknown 값은 제거
        all_headlines = [h for h in all_headlines if h != "Unknown"]

        # ❸ 구두점 제거 및 전처리가 된 문장들을 리스트로 반환
        self.corpus = [self.clean_text(x) for x in all_headlines]
        self.BOW = {}

        # ➍ 모든 문장의 단어를 추출해 고유번호 지정
        for line in self.corpus:
            for word in line.split():
                if word not in self.BOW.keys():
                    self.BOW[word] = len(self.BOW.keys())

        # 모델의 입력으로 사용할 데이터
        self.data = self.generate_sequence(self.corpus)
    def generate_sequence(self, txt):
        seq = []

        for line in txt:
            line = line.split()
            line_bow = [self.BOW[word] for word in line]

            # 단어 2개를 입력으로, 그다음 단어를 정답으로
            data = [([line_bow[i], line_bow[i+1]], line_bow[i+2])
            for i in range(len(line_bow)-2)]

            seq.extend(data)

        return seq
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        data = np.array(self.data[i][0])  # ❶ 입력 데이터
        label = np.array(self.data[i][1]).astype(np.float32)  # ❷ 출력 데이터

        return data, label
    
class LSTM(nn.Module):
   def __init__(self, num_embeddings):
       super(LSTM, self).__init__()

       # ❶ 밀집표현을 위한 임베딩층
       self.embed = nn.Embedding(
           num_embeddings=num_embeddings, embedding_dim=16)

       # LSTM을 5개층을 쌓음
       self.lstm = nn.LSTM(
           input_size=16,
           hidden_size=64,
           num_layers=5,
           batch_first=True)

       # 분류를 위한 MLP층
       self.fc1 = nn.Linear(128, num_embeddings)
       self.fc2 = nn.Linear(num_embeddings,num_embeddings)

       # 활성화 함수
       self.relu = nn.ReLU()

   def forward(self, x):
       x = self.embed(x)

       # ❷ LSTM 모델의 예측값
       x, _ = self.lstm(x)
       x = torch.reshape(x, (x.shape[0], -1))
       x = self.fc1(x)
       x = self.relu(x)
       x = self.fc2(x)

       return x
   

# 학습을 진행할 프로세서 정의
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TextGeneration()  # 데이터셋 정의
model = LSTM(num_embeddings=len(dataset.BOW)).to(device)  # 모델 정의
loader = DataLoader(dataset, batch_size=64)
optim = Adam(model.parameters(), lr=0.001)
'''
for epoch in range(100):
   iterator = tqdm.tqdm(loader)
   for data, label in iterator:
       # 기울기 초기화
       optim.zero_grad()

       # 모델의 예측값
       pred = model(torch.tensor(data, dtype=torch.long).to(device))

       # 정답 레이블은 long 텐서로 반환해야 함
       loss = nn.CrossEntropyLoss()(
           pred, torch.tensor(label, dtype=torch.long).to(device))

       # 오차 역전파
       loss.backward()
       optim.step()

       iterator.set_description(f"epoch{epoch} loss:{loss.item()}")

torch.save(model.state_dict(), "lstm.pth")
'''

def generate(model, BOW, string="he", strlen = 10):
   device = "cuda" if torch.cuda.is_available() else "cpu"

   print(f"input word: {string}")

   with torch.no_grad():
       for p in range(strlen):
           # 입력 문장을 텐서로 변경
           words = torch.tensor(
               [BOW[w] for w in string.split()], dtype=torch.long).to(device)

           # ❶
           input_tensor = torch.unsqueeze(words[-2:], dim=0)
           output = model(input_tensor)  # 모델을 이용해 예측
           output_word = (torch.argmax(output).cpu().numpy())
           string += list(BOW.keys())[output_word]  # 문장에 예측된 단어를 추가
           string += " "

   print(f"predicted sentence: {string}")

model.load_state_dict(torch.load("lstm.pth", map_location=device))
pred = generate(model, dataset.BOW)
