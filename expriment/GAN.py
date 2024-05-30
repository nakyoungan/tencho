import glob
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as tf
import torch.nn as nn
import tqdm
import numpy as np

from torchvision.models.vgg import vgg19
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch.utils.data.dataset import Dataset

from PIL import Image

class CelebA(Dataset):
    def __init__(self):
        self.imgs = glob.glob("./img/img_align_celeba/img_align_celeba/*.jpg")        

        self.grayscale = tf.Compose([
            tf.Resize((64,64)),
            tf.Grayscale(num_output_channels=1),
            tf.ToTensor(),
            tf.Normalize((0.5, ), (0.5, ))
        ])

        self.colorization = tf.Compose([
            tf.Resize((64, 64)),
            tf.ToTensor(),
            tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        img = Image.open(self.imgs[i])

        grayscale = self.grayscale(img)
        colorization = self.colorization(img)

        return [grayscale, colorization]
    
#생성자 기본 블록 (skip connection)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x_ = x
        x = self.layers(x)

        #합성곱층을 거친 후 원래의 입력 텐서와 더해줌
        x = x_+x

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(    #5층
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
        )

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x_ = x

        x = self.res_blocks(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_

        x = self.conv3(x)

        return x

#kernel_size=3, stride=1, padding=1
class DiscBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock1, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layers(x)

#kernel_size=3, stride=2, padding=1
class DiscBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock2, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        
        self.blocks1 = DiscBlock2(in_channels=64, out_channels=64)
        self.blocks2 = DiscBlock1(in_channels=64, out_channels=128)
        self.blocks3 = DiscBlock2(in_channels=128, out_channels=128)
        self.blocks4 = DiscBlock1(in_channels=128, out_channels=256)
        self.blocks5 = DiscBlock2(in_channels=256, out_channels=256)
        self.blocks6 = DiscBlock1(in_channels=256, out_channels=512)
        self.blocks7 = DiscBlock2(in_channels=512, out_channels=512)

        self.fc1 = nn.Linear(4 * 4 * 512, 1024)
        self.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        x = self.blocks7(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:9])

    def forward(self, img):
        return self.feature_extractor(img)

#학습
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CelebA()
batch_size = 128
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

G = Generator().to(device)
G.apply(weights_init)

D = Discriminator().to(device)
D.apply(weights_init)

feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()

G_optim = Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999))
D_optim = Adam(D.parameters(), lr = 0.0002, betas = (0.5, 0.999))

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

train_gen_losses, train_disc_losses, train_counter = [], [], []

for epoch in range(1):
    gen_loss, disc_loss = 0, 0
    iterator = tqdm.tqdm(loader)

    for i, (grayscale, colorization) in enumerate(iterator):
        G_optim.zero_grad()
        D_optim.zero_grad()

        label_true = torch.ones(batch_size, dtype=torch.float32).to(device)
        label_false = torch.zeros(batch_size, dtype=torch.float32).to(device)

        ### Train Generator
        # Generate a colorization image from grayscale input
        gen_color = G(grayscale.to(device))
        # Adversarial loss
        #GAN_loss = criterion_GAN(D(gen_color), label_true)
        GAN_Loss = nn.MSELoss()(D(gen_color), label_true)

        # Content loss
        fake_features = feature_extractor(gen_color)
        real_features = feature_extractor(colorization.to(device)) 
        #content_loss = criterion_content(fake_features, real_features)
        content_loss = nn.L1Loss()(fake_features, real_features)
        
        # Total loss
        loss_G = content_loss + 1e-3 * GAN_loss
        loss_G.backward()
        G_optim.step()      

        ### Train Discriminator
        # Loss of colorization and grayscale images
        # real_loss = criterion_GAN(D(colorization.to(device)), label_true)
        # fake_loss = criterion_GAN(D(gen_color), label_false)
        real_loss = nn.MSELoss()(D(colorization.to(device)), label_true)
        fake_loss = nn.MSELoss()(D(gen_color.detach()), label_false)
        
        # Total loss
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()
        D_optim.step()

        iterator.set_description(f"epoch:{epoch} G_loss:{loss_G} D_loss:{loss_D}")

        # gen_loss += loss_G.item()
        # train_gen_losses.append(loss_G.item())

        # disc_loss += loss_D.item()
        # train_disc_losses.append(loss_D.item())

        # train_counter.append(i*batch_size + gen_color.size(0) + epoch*len(loader.dataset))
        # tqdm_bar.set_postfix(gen_loss=gen_loss/(i+1), disc_loss=disc_loss/(i+1))

torch.save(G.state_dict(), "G.pth")
torch.save(D.state_dict(), "D.pth")


#평가
G.load_state_dict(torch.load("G.pth", map_location=device))

with torch.no_grad():
    low_res, high_res = dataset[0]
    
    input_tensor = torch.unsqueeze(low_res, dim=0).to(device)

    pred = G(input_tensor)
    pred = pred.squeeze()
    pred = pred.permute(1, 2, 0).cpu().numpy()

    low_res = low_res.permute(1,2,0).numpy()

    plt.subplot(1,2,1)
    plt.title("grayscale image")
    plt.imshow(low_res)
    plt.subplot(1,2,2)
    plt.imshow(pred)
    plt.title("colorization image")
    plt.show()
