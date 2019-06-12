import os
import random

import numpy as np
from scipy import io
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(CUBDataset, self).__init__()
        splits = np.loadtxt(os.path.join(dataPath, "train_val_test_split.txt"), int)
        self.files = np.loadtxt(os.path.join(dataPath, "images.txt"), str)[:,1]
        if sets == 'train':
            self.files = self.files[splits[:,1] == 0]
        elif sets == 'val':
            self.files = self.files[splits[:,1] == 1]
        else:
            self.files = self.files[splits[:,1] == 2]
        self.transform = transform
        self.datapath = dataPath
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        filename = self.files[idx]
        img = self.transform(Image.open(os.path.join(self.datapath, "images", filename)))
        if img.size(0) == 1:
            img = img.expand(3, img.size(1), img.size(2))
        seg = self.transform(Image.open(os.path.join(self.datapath, "segmentations", filename[:-3] + 'png')))
        if seg.size(0) != 1:
            seg = seg[:1]
        seg = (seg > .5).float()
        return img * 2 - 1, seg

class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(FlowersDataset, self).__init__()
        self.files =  io.loadmat(os.path.join(dataPath, "setid.mat"))
        if sets == 'train':
            self.files = self.files.get('tstid')[0]
        elif sets == 'val':
            self.files = self.files.get('valid')[0]
        else:
            self.files = self.files.get('trnid')[0]
        self.transform = transform
        self.datapath = dataPath
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        imgname = "image_%05d.jpg" % self.files[idx]
        segname = "segmim_%05d.jpg" % self.files[idx]
        img = self.transform(Image.open(os.path.join(self.datapath, "jpg", imgname)))
        seg = np.array(Image.open(os.path.join(self.datapath, "segmim", segname)))
        seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 254))
        seg = (seg * 255).astype('uint8').repeat(3,axis=2)
        seg = self.transform(Image.fromarray(seg))
        return img * 2 - 1, seg

class LFWDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(LFWDataset, self).__init__()
        with open(os.path.join(dataPath,sets+'.txt'), 'r') as f:
            self.files = np.array([l[:-1].split() for l in f.readlines()])
        self.transform = transform
        self.datapath = dataPath
        self.sets = sets
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.datapath,
                                      "lfw_funneled",
                                      self.files[idx,0],
                                      self.files[idx,1]+'.jpg'))
        img = self.transform(img)
        if self.sets == 'test' or self.sets == 'val':
            seg = Image.open(os.path.join(self.datapath,
                                          "parts_lfw_funneled_gt_images",
                                          self.files[idx,1]+'.ppm'))
            seg = self.transform(seg)
            seg = 1 - seg[2:]
        else:
            seg = img[:1]
        if img.size(0) == 1:
            img = img.expand(3, img.size(1), img.size(2))
        return img * 2 - 1, seg

class CMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(CMNISTDataset, self).__init__()
        self.mnist = torchvision.datasets.MNIST(dataPath,
                                                train=(sets=='train'),
                                                download=True,
                                                transform=transforms.Compose([transforms.Resize(28+28, Image.NEAREST),
                                                                              transforms.ToTensor(),]))
        self.mnist0 = iter(self.mnist)
        self.mnist1 = iter(self.mnist)
        self.sets = sets
    def __len__(self):
        return 1000 # arbitrary number for eval
    def __getitem__(self, idx):
        background = torch.randint(2, (3,1,1)).float().repeat(1,128,128)
        background = torch.FloatTensor(3,1,1).uniform_(.33,.66).repeat(1,128,128)
        background[0] = background[1]
        color0 = torch.FloatTensor(3,1,1)
        color0.uniform_(0,.33)
        color0[1] = color0[2]
        color1 = torch.FloatTensor(3,1,1)
        color1.uniform_(.66,1)
        color1[0] = color1[2]
        try:
            obj0, label0 = next(self.mnist0)
        except:
            self.mnist0 = iter(self.mnist)
            obj0, label0 = next(self.mnist0)
        while label0 % 2 != 0:
            try:
                obj0, label0 = next(self.mnist0)
            except:
                self.mnist0 = iter(self.mnist)
                obj0, label0 = next(self.mnist0)
        obj0 = (obj0 > .5).float()
        obj0 = obj0.repeat(3,1,1)
        try:
            obj1, label1 = next(self.mnist1)
        except:
            self.mnist1 = iter(self.mnist)
            obj1, label1 = next(self.mnist1)
        while label1 % 2 != 1:
            try:
                obj1, label1 = next(self.mnist1)
            except:
                self.mnist1 = iter(self.mnist)
                obj1, label1 = next(self.mnist1)
        obj1 = (obj1 > .5).float()
        obj1 = obj1.repeat(3,1,1)
        bg = background.clone()
        px0 = random.randint(0,bg.size(1)-obj0.size(1)-1)
        py0 = random.randint(0,bg.size(2)-obj0.size(2)-1)
        px1 = random.randint(0,bg.size(1)-obj1.size(1)-1)
        py1 = random.randint(0,bg.size(2)-obj1.size(2)-1)
        seg = torch.zeros(3,128,128)
        seg[2].fill_(1)
        order = random.randint(0,1)
        if order == 0:
            bg[:,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] = (bg[:,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] * (1-obj0)) + obj0 * color0
            bg[:,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] = (bg[:,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] * (1-obj1)) + obj1 * color1
            seg[:,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] = seg[:,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] * (1-obj0)
            seg[0,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] = seg[0,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] * (1-obj0[0]) + obj0[0]
            seg[:,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] = seg[:,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] * (1-obj1)
            seg[1,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] = seg[1,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] * (1-obj1[0]) + obj1[0]
        else:
            bg[:,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] = (bg[:,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] * (1-obj1)) + obj1 * color1
            bg[:,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] = (bg[:,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] * (1-obj0)) + obj0 * color0
            seg[:,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] = seg[:,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] * (1-obj1)
            seg[1,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] = seg[1,px1:px1+obj1.size(1),py1:py1+obj1.size(2)] * (1-obj1[0]) + obj1[0]
            seg[:,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] = seg[:,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] * (1-obj0)
            seg[0,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] = seg[0,px0:px0+obj0.size(1),py0:py0+obj0.size(2)] * (1-obj0[0]) + obj0[0]
        return bg * 2 - 1, seg
