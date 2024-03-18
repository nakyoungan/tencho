import glob
import torch
import numpy as np

from torch.utils.data.dataset import Dataset
from PIL import Image


path_to_anno = "./annotations/trimaps/"
path_to_img = "./images/"


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
        self.Y_test = self.annotations[:int(0.8*len(self.annotations))]

        self.train = train              #학습용 데이터 평가용 데이터 결정 여부
        self.transforms = transforms    #사용할 데이터 증강
        self.input_size = input_size    #입력 이미지 크기