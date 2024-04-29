import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

#Resnet 기본 블록
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicBlock, self).__init__()

        #합성곱층 정의
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        #배치정규화층 정의
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

    #기본 블록의 순전파 정의
    def forward(self, x):
        #스킵 커넥션을 위해 초기 입력 저장
        x_skip = x

        #ResNet 기본 블록에서 F(x)부분
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)

        #합성곱의 결과와 입력의 채널수를 맞춤
        x_self = self.downsample(x_skip)

        #합성곱층의 결과와 저장해놨던 입력값을 더해줌(스킵커넥션)
        x = x + x_self
        x = self.relu(x)

        return x
    
#ResNet 모델 정의하기
class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        
        #기본 블록
        self.b1 = BasicBlock(in_channels=3, out_channels=64)
        self.b2 = BasicBlock(in_channels=64, out_channels=128)
        self.b3 = BasicBlock(in_channels=128, out_channels=256)

        #평균풀링 수행
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        #분류기
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

        self.relu = nn.ReLU()

    #ResNet 모델의 순천파 정의
    def forward(self, x):
        #기본 블록과 풀링층 통과
        x = self.b1(x)
        x = self.pool(x)
        x = self.b2(x)
        x = self.pool(x)
        x = self.b3(x)
        x = self.pool(x)

        #분류기의 입력으로 사용하기 위한 평탄화
        x = torch.flatten(x, start_dim=1)

        #분류기로 예측값 출력
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


#모델 학습시키기
transforms = Compose([
    RandomCrop((32, 32), padding = 4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

training_data = CIFAR100(root = "./", train=True, download=False, transform=transforms)
test_data = CIFAR100(root = "./", train=False, download=False, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

#모델 정의
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet(num_classes=100)
model.to(device)

lr = 1e-5
optim = Adam(model.parameters(), lr = lr)
'''
for epoch in range(70):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        #최적화를 위해 기울기를 초기화
        optim.zero_grad()

        #모델의 예측값
        preds = model(data.to(device))

        #손실 계산 및 역전파
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "ResNet70.pth")

'''
model.load_state_dict(torch.load("ResNet.pth", map_location=device))

num_corr = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = output.argmax(dim=1)
        corr = preds.eq(label.to(device)).sum().item()
        num_corr += corr
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

accuracy = num_corr / len(test_data)
print(f"Accuracy: {accuracy}")

f1 = f1_score(all_labels, all_preds, average='macro')  # 'macro' for multiclass classification
print(f"F1 Score: {f1}")

recall = recall_score(all_labels, all_preds, average='macro')
print(f"Recall: {recall}")

conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)

# Confusion Matrix 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=range(100), yticklabels=range(100))
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

'''
#성능평가
model.load_state_dict(torch.load("ResNet100.pth", map_location=device))

num_corr = 0
true_labels = []
predicted_labels = []

with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr
        true_labels.extend(preds.cpu().numpy())
        predicted_labels.extend(label.cpu().numpy())

    accuracy = num_corr / len(test_data)
    print(f"Accuracy: {accuracy}")

    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    print(f"Precision: {precision}")

    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    print(f"Recall: {recall}")

    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    print(f"F1 Score: {f1}")


# Confusion Matrix 그리기
cm = confusion_matrix(true_labels, predicted_labels, normalize='all')
print(cm)

# Confusion Matrix 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(100), yticklabels=range(100))
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()
'''