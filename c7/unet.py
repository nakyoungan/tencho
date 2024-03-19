import glob
import torch
import numpy as np
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Resize
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader


'''
데이터 관리
'''
path_to_annotation = "./annotations/trimaps/"
path_to_image = "./images/"

annotation = Image.open(path_to_annotation + "Abyssinian_1.png")
plt.subplot(1,2,1)
plt.title("annotation")
plt.imshow(annotation)

image = Image.open(path_to_image + "Abyssinian_1.jpg")
plt.subplot(1,2,2)
plt.title("image")
plt.imshow(image)
'''
plt.show()
'''

class Pets(Dataset):
    def __init__(self, path_to_img,
                 path_to_anno,
                 train=True,
                 transforms=None,
                 input_size=(128,128)):
            
        #정답과 이미지를 이름순으로 정력
        self.images = sorted(glob.glob(path_to_img+"/*.jpg"))
        self.annotations = sorted(glob.glob(path_to_anno+"/*.png"))

        #데이터셋을 학습과 평가로 나눔
        self.X_train = self.images[:int(0.8*len(self.images))]
        self.X_test = self.images[int(0.8*len(self.images)):]
        self.Y_train = self.annotations[:int(0.8*len(self.annotations))]
        self.Y_test = self.annotations[int(0.8*len(self.annotations)):]

        self.train = train              #학습용 데이터 평가용 데이터 결정 여부
        self.transforms = transforms    #사용할 데이터 증강
        self.input_size = input_size    #입력 이미지 크기

    #데이터 개수를 나타냄
    def __len__(self):
        if self.train:
            return len(self.X_train)    #학습용 데이터셋 길이
        else:
            return len(self.X_test)     #평가용 데이터셋 길이
        
    #정답을 반환하는 함수
    def preprocess_mask(self, mask):
        mask = mask.resize(self.input_size)
        mask = np.array(mask).astype(np.float32)
        mask[mask != 2.0] = 1.0
        mask[mask == 2.0] = 0.0
        mask = torch.tensor(mask)
        return mask
    

    
    def __getitem__(self, i):
        if self.train:  #학습용 데이터
            X_train = Image.open(self.X_train[i])
            X_train = self.transforms(X_train)
            Y_train = Image.open(self.Y_train[i])
            Y_train = self.preprocess_mask(Y_train)

            return X_train, Y_train
        
        else:       #평가용 데이터
            X_test = Image.open(self.X_test[i])
            X_test = self.transforms(X_test)
            Y_test = Image.open(self.Y_test[i])
            Y_test = self.preprocess_mask(Y_test)

            return X_test, Y_test


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        #U-Net의 인코더에 사용되는 은닉층

        #기본블록
        self.enc1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #기본블록
        self.enc2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #기본블록
        self.enc3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #기본블록
        self.enc4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #인코더 마지막 기본 블록
        self.enc5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.enc5_2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        #디코더에 사용되는 은닉층
        self.upsample4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec4_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec3_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec1_3 = nn.Conv2d(64, 1, kernel_size=1)

        #합성곱과 업샘플링층의 활성화 함수
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder의 forward 정의
        x = self.enc1_1(x)
        x = self.relu(x)
        e1 = self.enc1_2(x)
        e1 = self.relu(e1)
        x = self.pool1(e1)

        x = self.enc2_1(x)
        x = self.relu(x)
        e2 = self.enc2_2(x)
        e2 = self.relu(e2)
        x = self.pool2(e2)

        x = self.enc3_1(x)
        x = self.relu(x)
        e3 = self.enc3_2(x)
        e3 = self.relu(e3)
        x = self.pool3(e3)

        x = self.enc4_1(x)
        x = self.relu(x)
        e4 = self.enc4_2(x)
        e4 = self.relu(e4)
        x = self.pool4(e4)

        x = self.enc5_1(x)
        x = self.relu(x)
        x = self.enc5_2(x)
        x = self.relu(x)

    # decoder의 forward 정의
        x = self.upsample4(x)
        x = torch.cat([x, e4], dim=1)
        x = self.dec4_1(x)
        x = self.relu(x)
        x = self.dec4_2(x)
        x = self.relu(x)

        x = self.upsample3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3_1(x)
        x = self.relu(x)
        x = self.dec3_2(x)
        x = self.relu(x)

        x = self.upsample2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2_1(x)
        x = self.relu(x)
        x = self.dec2_2(x)
        x = self.relu(x)

        x = self.upsample1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1_1(x)
        x = self.relu(x)
        x = self.dec1_2(x)
        x = self.relu(x)
        x = self.dec1_3(x)
        x = self.relu(x)

        x = torch.squeeze(x)

        return x
    

'''
학습 시작
'''

device = "cuda" if torch.cuda.is_available() else "cpu"

#데이터 전처리 정의
transform = Compose([
    Resize((128,128)),
    ToTensor()
])

train_set = Pets(path_to_img = path_to_image,
                 path_to_anno= path_to_annotation,
                 transforms = transform)

test_set = Pets(path_to_img = path_to_image,
                 path_to_anno= path_to_annotation,
                 transforms = transform,
                train=False)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set)

model = UNet().to(device)
learning_rate = 0.0001
optim = Adam(params=model.parameters(), lr = learning_rate)

for epoch in range(200):
  iterator = tqdm.tqdm(train_loader)
  
  for data, label in iterator:
    optim.zero_grad()
    preds = model(data.to(device))
    loss = nn.BCEWithLogitsLoss()(
        preds, 
        label.type(torch.FloatTensor).to(device))
    loss.backward()
    optim.step()
    iterator.set_description(f"epoch{epoch+1} loss: {loss.item()}")

torch.save(model.state_dict(), "./UNet.pth")


'''
성능 평가


model.load_state_dic(torch.load("./UNet.pth", map_location="cpu"))
data, label = test_set[1]
pred = model(torch.unsqueeze(data.to(device), dim=0))>0.5

with torch.no_grad():
    plt.subplot(1,2,1)
    plt.title("Predicted")
    plt.imshow(pred)
    plt.subplot(1,2,2)
    plt.title("Real")
    plt.imshow(label)
    plt.show()

    '''