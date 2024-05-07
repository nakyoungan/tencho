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
from tensorboardX import SummaryWriter
writer = SummaryWriter()

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

G_optim = Adam(G.parameters(), lr = 0.0001, betas = (0.5, 0.999))
D_optim = Adam(D.parameters(), lr = 0.0001, betas = (0.5, 0.999))
'''
for epochs in range(500):
    iterator = tqdm.tqdm(enumerate(loader, 0), total=len(loader))

    for i, data in iterator:
        D_optim.zero_grad()

        label = torch.ones_like(data[1], dtype=torch.float32).to(device)
        label_fake = torch.zeros_like(data[1], dtype=torch.float32).to(device)

        real = D(data[0].to(device))

        Dloss_real = nn.BCELoss()(torch.squeeze(real), label)
        Dloss_real.backward()

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
    
    if epochs % 10 == 0:    # 매 10 iteration마다 업데이트
        writer.add_scalar('D_loss', Dloss.item(), epochs)
        writer.add_scalar('G_loss', Gloss.item(), epochs)

torch.save(G.state_dict(), "Generator500.pth")
torch.save(D.state_dict(), "Discriminator500.pth")
'''

#평가
with torch.no_grad():
    G.load_state_dict(
        torch.load("Generator100.pth", map_location=device))
    
    feature_vector = torch.randn(1, 100, 1, 1).to(device)
    pred = G(feature_vector).squeeze()
    pred = pred.permute(1, 2, 0).cpu().numpy()

    plt.subplot(1,2,1)
    plt.imshow(pred)
    plt.title("epoch100 predicted image")


with torch.no_grad():
    G.load_state_dict(
        torch.load("Generator.pth", map_location=device))
    
    feature_vector = torch.randn(1, 100, 1, 1).to(device)
    pred = G(feature_vector).squeeze()
    pred = pred.permute(1, 2, 0).cpu().numpy()

    plt.subplot(1,2,2)
    plt.imshow(pred)
    plt.title("epoch30 predicted image")

plt.show()