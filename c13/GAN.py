import glob
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as tf
import torch.nn as nn
import tqdm

from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
#from tensorboardX import SummaryWriter
#writer = SummaryWriter()

from PIL import Image

#path to images
pth_to_imgs = "./img_align_celeba"
imgs = glob.glob(os.path.join(pth_to_imgs, "*"))

'''
#show 9 images
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = Image.open(imgs[i])
    plt.imshow(img)

plt.show()
'''

transforms = tf.Compose([
    tf.Resize(64),
    tf.CenterCrop(64),
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(
    root = "./img/",
    transform=transforms
)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# sample model. It has nn.ConvTranspose2d(1, 3, 4, 1, 0, bias = False)
# First parameter = Channels of input (=1)
# Second parameter = Channels of output (=3)
# Third parameter = Kernel size (=4)
# Fourth parameter = stride (=1)
# fifth parameter = padding (=0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),       

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),

            nn.Tanh()     
        )

    def forward(self, x):
        return self.gen(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid()   
        )
    
    def forward(self, x):
        return self.disc(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

#학습
device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
G.apply(weights_init)

D = Discriminator().to(device)
D.apply(weights_init)

G_optim = Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999))
D_optim = Adam(D.parameters(), lr = 0.0002, betas = (0.5, 0.999))

#D와 G의 비율을 5:1로
discriminator_steps = 1
generator_steps = 5

for epoch in range(100):
    iterator = tqdm.tqdm(enumerate(loader), total=len(loader))

    for i, (real_images, _) in iterator:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # 감별자 업데이트
        for _ in range(discriminator_steps):
            D_optim.zero_grad()
            
            # 실제 이미지
            real_predictions = D(real_images)
            D_loss_real = nn.BCELoss()(torch.squeeze(real_predictions), real_labels)
            D_loss_real.backward()
            
            # 가짜 이미지
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = G(noise)
            fake_predictions = D(fake_images.detach())
            D_loss_fake = nn.BCELoss()(torch.squeeze(fake_predictions), fake_labels)
            D_loss_fake.backward()

            D_optim.step()
        
        # 생성자 업데이트
        G_optim.zero_grad()
        fake_predictions = D(fake_images)
        G_loss = nn.BCELoss()(torch.squeeze(fake_predictions), real_labels)
        G_loss.backward()
        G_optim.step()

        if i % discriminator_steps == 0:  # 전체 감별자 업데이트 주기가 완료될 때마다 로그 기록
            iterator.set_description(f"Epoch: {epoch} Iteration: {i} D_loss: {D_loss_real + D_loss_fake} G_loss: {G_loss}")

'''
for epochs in range(50):
    iterator = tqdm.tqdm(enumerate(loader, 0), total=len(loader))

    for i, data in iterator:
        D_optim.zero_grad()

        label = torch.ones_like(data[1], dtype=torch.float32).to(device)
        label_fake = torch.zeros_like(data[1], dtype=torch.float32).to(device)

        real = D(data[0].to(device))

        Dloss_real = nn.BCELoss()(torch.squeeze(real), label)
        Dloss_real.backward()

        #Discriminator가 5번 학습될 때 Generator는 1번 학습되게

        #감별자 학습
        noise = torch.randn(label.shape[0], 100, 1, 1, device=device)
        fake = G(noise)

        output = D(fake.detach())

        Dloss_fake = nn.BCELoss()(torch.squeeze(output), label_fake)
        Dloss_fake.backward()
        
        Dloss = Dloss_real + Dloss_fake
        D_optim.step()

        #생성자 학습
        G_optim.zero_grad()
        output = D(fake)
        Gloss = nn.BCELoss()(torch.squeeze(output),label)
        Gloss.backward()

        G_optim.step()

        iterator.set_description(f"eopch:{epochs} iteration:{i} D_loss:{Dloss} G_loss:{Gloss}")
'''
torch.save(G.state_dict(), "Generator5(100).pth")
torch.save(D.state_dict(), "Discriminator1(100).pth")

#평가
with torch.no_grad():
    G.load_state_dict(
        torch.load("Generator.pth", map_location=device))
    
    feature_vector = torch.randn(1, 100, 1, 1).to(device)
    pred = G(feature_vector).squeeze()
    pred = pred.permute(1, 2, 0).cpu().numpy()

    plt.subplot(1,2,1)
    plt.imshow(pred)
    plt.title("D:G = 1:1 image")

    G.load_state_dict(
        torch.load("Generator1.pth", map_location=device))
    
    feature_vector = torch.randn(1, 100, 1, 1).to(device)
    pred = G(feature_vector).squeeze()
    pred = pred.permute(1, 2, 0).cpu().numpy()

    plt.subplot(1,2,2)
    plt.imshow(pred)
    plt.title("D:G = 1:5 image")


    plt.show()