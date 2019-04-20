import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):

    def __init__(self, root, transform=None, train=True, test=False):
        '''
        目标：获取所有图片地址，划分数据集
        '''
        self.test = test
        # 所有图片的绝对路径
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        if self.test:
            # 测试文件：root/123.jpg
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            # 训练文件：root/cat.232.jpg
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7*imgs_num):]

        if transform is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if train:
                self.transform = T.Compose([
                    T.Scale(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize,
                ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        '''
        :param item:
        :return:
        '''
        img_path = self.imgs[item]
        if self.test:
            label = int(self.imgs[item].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
