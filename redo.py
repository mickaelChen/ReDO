#############################################################################
# Import                                                                    #
#############################################################################
import os
import math
import random
import argparse

import itertools

import numpy as np
from scipy import io
from PIL import Image

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F

#############################################################################
# Arguments                                                                 #
#############################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default="flowers", help='flowers | cub | lfw')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--nx', type=int, default=3, help='number of canals of the input image')
parser.add_argument('--sizex', type=int, default=128, help='size of the input image')
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--nMasks', type=int, default=2, help='number of masks')
parser.add_argument('--nResM', type=int, default=3, help='number of residual blocs in netM')
parser.add_argument('--nf', type=int, default=64)
parser.add_argument('--nfD', type=int, default=None)
parser.add_argument('--nfX', type=int, default=None)
parser.add_argument('--nfM', type=int, default=None)
parser.add_argument('--nfZ', type=int, default=None)
parser.add_argument('--useSelfAttD', action='store_true', help='use self attention for D')
parser.add_argument('--useSelfAttG', action='store_true', help='use self attention for G')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--nTest', type=int, default=5, help='input batch size for visu')
parser.add_argument('--nIteration', type=int, default=1e5, help='number of iterations')
parser.add_argument('--initOrthoGain', type=float, default=.8)
parser.add_argument('--lrG', type=float, default=1e-4, help='learning rate for G, default=1e-4')
parser.add_argument('--lrM', type=float, default=1e-5, help='learning rate for M, default=1e-5')
parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate for D, default=1e-4')
parser.add_argument('--lrZ', type=float, default=1e-4, help='learning rate for Z, default=1e-4')
parser.add_argument('--gStepFreq', type=int, default=1, help='wait x steps for G updates')
parser.add_argument('--dStepFreq', type=int, default=1, help='wait x steps for D updates')
parser.add_argument('--temperature', type=float, default=1, help='softmax temperature')
parser.add_argument('--wdecay', type=float, default=1e-4, help='weight decay for M, default=1e-4')
parser.add_argument('--wrecZ', type=float, default=5, help='weight for z reconstruction')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--checkpointFreq', type=int, default=500, help='frequency of checkpoints')
parser.add_argument('--iteration', type=int, default=0, help="iteration to load (to resume training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume', action='store_true', help='resume from last save')
parser.add_argument('--clean', action='store_true', help='clean previous states')
parser.add_argument('--silent', action='store_true', help='silent execution')

opt = parser.parse_args()

if not opt.silent:
    from tqdm import tqdm

if opt.resume:
    try:
        opt2 = torch.load(os.path.join(opt.outf, "options.pth"))
        opt2.clean = opt.clean
        opt2.silent = opt.silent
        opt2.outf = opt.outf
        opt = opt2
    except:
        pass

if 'bestValIoU' not in opt:
    opt.bestValIoU = 0

if opt.nfD is None:
    opt.nfD = opt.nf
if opt.nfX is None:
    opt.nfX = opt.nf
if opt.nfM is None:
    opt.nfM = opt.nf
if opt.nfZ is None:
    opt.nfZ = opt.nf

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
if not opt.silent:
    print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.device = "cuda:0"
device = torch.device(opt.device)
cudnn.benchmark = True
    
#############################################################################
# Datasets                                                                  #
#############################################################################
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
    
if opt.dataset == 'lfw':
    trainset = LFWDataset(dataPath=opt.dataroot,
                          sets='train',
                          transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                        transforms.CenterCrop(opt.sizex),
                                                        transforms.ToTensor(),
                          ]),)
    testset = LFWDataset(dataPath=opt.dataroot,
                         sets='test',
                         transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                       transforms.CenterCrop(opt.sizex),
                                                       transforms.ToTensor(),
                         ]),)
    valset = LFWDataset(dataPath=opt.dataroot,
                        sets='val',
                        transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                      transforms.CenterCrop(opt.sizex),
                                                      transforms.ToTensor(),
                        ]),)
if opt.dataset == 'cub':
    trainset = CUBDataset(opt.dataroot,
                          "train",
                          transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                              transforms.CenterCrop(opt.sizex),
                                              transforms.ToTensor(),
                          ]))
    testset = CUBDataset(opt.dataroot,
                         "test",
                         transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                             transforms.CenterCrop(opt.sizex),
                                             transforms.ToTensor(),
                         ]))
    valset = CUBDataset(opt.dataroot,
                        "val",
                        transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                            transforms.CenterCrop(opt.sizex),
                                            transforms.ToTensor(),
                        ]))
if opt.dataset == 'flowers':
    trainset = FlowersDataset(opt.dataroot,
                              "train",
                              transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                  transforms.CenterCrop(opt.sizex),
                                                  transforms.ToTensor(),
                              ]))
    testset = FlowersDataset(opt.dataroot,
                             "test",
                             transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                 transforms.CenterCrop(opt.sizex),
                                                 transforms.ToTensor(),
                             ]))
    valset = FlowersDataset(opt.dataroot,
                            "val",
                            transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                transforms.CenterCrop(opt.sizex),
                                                transforms.ToTensor(),
                            ]))
if opt.dataset == 'cmnist':
    trainset = CMNISTDataset(dataPath=opt.dataroot,
                             sets='train')
    testset = CMNISTDataset(dataPath=opt.dataroot,
                            sets='test')
    valset = CMNISTDataset(dataPath=opt.dataroot,
                           sets='val')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=4, drop_last=True)

#############################################################################
# Modules                                                                   #
#############################################################################
class SelfAttentionNaive(nn.Module):
    def __init__(self, nf, nh=False):
        super(SelfAttentionNaive, self).__init__()
        if not nh:
            nh = max(nf//8, 1)
        self.f = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))
        self.g = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))
        self.h = spectral_norm(nn.Conv2d(nf, nf, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.nh = nh
        self.nf = nf
    def forward(self, x):
        fx = self.f(x).view(x.size(0), self.nh, x.size(2)*x.size(3))
        gx = self.g(x).view(x.size(0), self.nh, x.size(2)*x.size(3))
        hx = self.h(x).view(x.size(0), self.nf, x.size(2)*x.size(3))
        s = fx.transpose(-1,-2).matmul(gx)
        b = F.softmax(s, dim=1)
        o = hx.matmul(b)
        return o.view_as(x) * self.gamma + x

class SelfAttention(nn.Module):
    def __init__(self, nf, nh=False):
        super(SelfAttention, self).__init__()
        if not nh:
            nh = max(nf//8, 1)
        self.f = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))
        self.g = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))
        self.h = spectral_norm(nn.Conv2d(nf, nf//2, 1, bias=False))
        self.o = spectral_norm(nn.Conv2d(nf//2, nf, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.nh = nh
        self.nf = nf
    def forward(self, x):
        fx = self.f(x).view(x.size(0), self.nh, x.size(2)*x.size(3))
        gx = self.g(x)
        gx = F.max_pool2d(gx, kernel_size=2)
        gx = gx.view(x.size(0), self.nh, x.size(2)*x.size(3)//4)
        s = gx.transpose(-1,-2).matmul(fx)
        s = F.softmax(s, dim=1)
        hx = self.h(x)
        hx = F.max_pool2d(hx, kernel_size=2)
        hx = hx.view(x.size(0), self.nf//2, x.size(2)*x.size(3)//4)
        ox = hx.matmul(s).view(x.size(0), self.nf//2, x.size(2), x.size(3))
        ox = self.o(ox)
        return ox * self.gamma + x
    
class _resDiscriminator128(nn.Module):
    def __init__(self, nIn=3, nf=64, selfAtt=False):
        super(_resDiscriminator128, self).__init__()
        self.blocs = []
        self.sc = []
        # first bloc
        self.bloc0 = nn.Sequential(spectral_norm(nn.Conv2d(nIn, nf, 3, 1, 1, bias=True)),
                                   nn.ReLU(),
                                   spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                   nn.AvgPool2d(2),)
        self.sc0 = nn.Sequential(nn.AvgPool2d(2),
                                 spectral_norm(nn.Conv2d(nIn, nf, 1, bias=True)),)
        if selfAtt:
            self.selfAtt = SelfAttention(nf)
        else:
            self.selfAtt = nn.Sequential()
        # Down blocs
        for i in range(4):
            nfPrev = nf
            nf = nf*2
            self.blocs.append(nn.Sequential(nn.ReLU(),
                                            spectral_norm(nn.Conv2d(nfPrev, nf, 3, 1, 1, bias=True)),
                                            nn.ReLU(),
                                            spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                            nn.AvgPool2d(2),))
            self.sc.append(nn.Sequential(nn.AvgPool2d(2),
                                         spectral_norm(nn.Conv2d(nfPrev, nf, 1, bias=True)),))
        # Last Bloc
        self.blocs.append(nn.Sequential(nn.ReLU(),
                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                        nn.ReLU(),
                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))))
        self.sc.append(nn.Sequential())
        self.dense = nn.Linear(nf, 1)
        self.blocs = nn.ModuleList(self.blocs)
        self.sc = nn.ModuleList(self.sc)
    def forward(self, x):
        x = self.selfAtt(self.bloc0(x) + self.sc0(x))
        for k in range(len(self.blocs)):
            x = self.blocs[k](x) + self.sc[k](x)
        x = x.sum(3).sum(2)
        return self.dense(x)

class _resEncoder128(nn.Module):
    def __init__(self, nIn=3, nf=64, nOut=8):
        super(_resEncoder128, self).__init__()
        self.blocs = []
        self.sc = []
        # first bloc
        self.blocs.append(nn.Sequential(spectral_norm(nn.Conv2d(nIn, nf, 3, 1, 1, bias=True)),
                                        nn.ReLU(),
                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                        nn.AvgPool2d(2),))
        self.sc.append(nn.Sequential(nn.AvgPool2d(2),
                                     spectral_norm(nn.Conv2d(nIn, nf, 1, bias=True)),))
        # Down blocs
        for i in range(4):
            nfPrev = nf
            nf = nf*2
            self.blocs.append(nn.Sequential(nn.ReLU(),
                                            spectral_norm(nn.Conv2d(nfPrev, nf, 3, 1, 1, bias=True)),
                                            nn.ReLU(),
                                            spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                            nn.AvgPool2d(2),))
            self.sc.append(nn.Sequential(nn.AvgPool2d(2),
                                         spectral_norm(nn.Conv2d(nfPrev, nf, 1, bias=True)),))
        # Last Bloc
        self.blocs.append(nn.Sequential(nn.ReLU(),
                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                        nn.ReLU(),
                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))))
        self.sc.append(nn.Sequential())
        self.dense = nn.Linear(nf, nOut)
        self.blocs = nn.ModuleList(self.blocs)
        self.sc = nn.ModuleList(self.sc)
    def forward(self, x):
        for k in range(len(self.blocs)):
            x = self.blocs[k](x) + self.sc[k](x)
        x = x.sum(3).sum(2)
        return self.dense(x)
    
class _resMaskedGenerator128(nn.Module):
    def __init__(self, nf=64, nOut=3, nc=8, selfAtt=False):
        super(_resMaskedGenerator128, self).__init__()
        if selfAtt:
            self.selfAtt = SelfAttention(nf*2)
        else:
            self.selfAtt = nn.Sequential()
        self.dense = nn.Linear(nc, 4*4*nf*16)
        self.convA = []
        self.convB = []
        self.normA = []
        self.normB = []
        self.gammaA = []
        self.gammaB = []
        self.betaA = []
        self.betaB = []
        self.sc = []
        nfPrev = nf*16
        nfNext = nf*16
        for k in range(5):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev + 1, nfNext, 3, 1, 1, bias=False)),))
            self.convB.append(spectral_norm(nn.Conv2d(nfNext, nfNext, 3, 1, 1, bias=True )))
            self.normA.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.normB.append(nn.InstanceNorm2d(nfNext, affine=False))
            self.gammaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.gammaB.append(nn.Conv2d(nc, nfNext, 1, bias=True))
            self.betaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaB.append(nn.Conv2d(nc, nfNext, 1, bias=True))
            self.sc.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                         spectral_norm(nn.Conv2d(nfPrev, nfNext, 1, bias=True))))
            nfPrev = nfNext
            nfNext = nfNext // 2
        self.convA = nn.ModuleList(self.convA)
        self.convB = nn.ModuleList(self.convB)
        self.normA = nn.ModuleList(self.normA)
        self.normB = nn.ModuleList(self.normB)
        self.gammaA =nn.ModuleList(self.gammaA)
        self.gammaB =nn.ModuleList(self.gammaB)
        self.betaA = nn.ModuleList(self.betaA)
        self.betaB = nn.ModuleList(self.betaB)
        self.sc = nn.ModuleList(self.sc)
        self.normOut = nn.InstanceNorm2d(nf, affine=False)
        self.gammaOut = nn.Conv2d(nc, nf, 1, bias=True)
        self.betaOut = nn.Conv2d(nc, nf, 1, bias=True)
        self.convOut = spectral_norm(nn.Conv2d(nf, nOut, 3, 1, 1))
        self.convOut = spectral_norm(nn.Conv2d(nf + 1, nOut, 3, 1, 1))
        ##############################
    def forward(self, m, z, c):
        ######### Upsample ###########
        x = self.dense(z.view(z.size(0),z.size(1))).view(z.size(0), -1, 4, 4)
        mask_ratio = m.size(-1) // 4
        for k in range(5):
            if k == 4:
                x = self.selfAtt(x)
            h = self.convA[k](torch.cat((F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)),
                                         F.avg_pool2d(m, kernel_size=mask_ratio)), 1))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            mask_ratio = mask_ratio // 2
        x = self.convOut(torch.cat((F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)),
                                    m), 1))
        x = torch.tanh(x)
        return x * m
    
class _downConv(nn.Module):
    def __init__(self, nIn=3, nf=128, spectralNorm=False):
        super(_downConv, self).__init__()
        self.mods = nn.Sequential(nn.ReflectionPad2d(3),
                                  spectral_norm(nn.Conv2d(nIn, nf//4, 7, bias=False)) if spectralNorm else nn.Conv2d(nIn, nf//4, 7, bias=False),
                                  nn.InstanceNorm2d(nf//4, affine=True),
                                  nn.ReLU(),
                                  spectral_norm(nn.Conv2d(nf//4, nf//2, 3, 2, 1, bias=False)) if spectralNorm else nn.Conv2d(nf//4, nf//2, 3, 2, 1, bias=False),
                                  nn.InstanceNorm2d(nf//2, affine=True),
                                  nn.ReLU(),
                                  spectral_norm(nn.Conv2d(nf//2, nf, 3, 2, 1, bias=False)) if spectralNorm else nn.Conv2d(nf//2, nf, 3, 2, 1, bias=False),
                                  nn.InstanceNorm2d(nf, affine=True),
                                  nn.ReLU(),
        )
    def forward(self, x):
        return self.mods(x)
class _resBloc(nn.Module):
    def __init__(self, nf=128, spectralNorm=False):
        super(_resBloc, self).__init__()
        self.blocs = nn.Sequential(spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)) if spectralNorm else nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
                                   nn.InstanceNorm2d(nf, affine=True),
                                   nn.ReLU(),
                                   spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)) if spectralNorm else nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        )
        self.activationF = nn.Sequential(nn.InstanceNorm2d(nf, affine=True),
                                         nn.ReLU(),
        )
    def forward(self, x):
        return self.activationF(self.blocs(x) + x)
class _upConv(nn.Module):
    def __init__(self, nOut=3, nf=128, spectralNorm=False):
        super(_upConv, self).__init__()
        self.mods = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                  spectral_norm(nn.Conv2d(nf, nf//2, 3, 1, 1, bias=False)) if spectralNorm else nn.Conv2d(nf, nf//2, 3, 1, 1, bias=False),
                                  nn.InstanceNorm2d(nf//2, affine=True),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=2, mode='nearest'),
                                  spectral_norm(nn.Conv2d(nf//2, nf//4, 3, 1, 1, bias=False)) if spectralNorm else nn.Conv2d(nf//2, nf//4, 3, 1, 1, bias=False),
                                  nn.InstanceNorm2d(nf//4, affine=True),
                                  nn.ReLU(),
                                  nn.ReflectionPad2d(3),
                                  spectral_norm(nn.Conv2d(nf//4, nOut, 7, bias=True)) if spectralNorm else nn.Conv2d(nf//4, nOut, 7, bias=True),
        )
    def forward(self, x):
        return self.mods(x)
class _netEncM(nn.Module):
    def __init__(self, sizex=128, nIn=3, nMasks=2, nRes=5, nf=128, temperature=1):
        super(_netEncM, self).__init__()
        self.nMasks = nMasks
        sizex = sizex // 4 
        self.cnn = nn.Sequential(*([_downConv(nIn, nf)] +
                                   [_resBloc(nf=nf) for i in range(nRes)]))
        self.psp = nn.ModuleList([nn.Sequential(nn.AvgPool2d(sizex),
                                                nn.Conv2d(nf,1,1),
                                                nn.Upsample(size=sizex, mode='bilinear')),
                                  nn.Sequential(nn.AvgPool2d(sizex//2, sizex//2),
                                                nn.Conv2d(nf,1,1),
                                                nn.Upsample(size=sizex, mode='bilinear')),
                                  nn.Sequential(nn.AvgPool2d(sizex//3, sizex//3),
                                                nn.Conv2d(nf,1,1),
                                                nn.Upsample(size=sizex, mode='bilinear')),
                                  nn.Sequential(nn.AvgPool2d(sizex//6, sizex//6),
                                                nn.Conv2d(nf,1,1),
                                                nn.Upsample(size=sizex, mode='bilinear'))])
        self.out = _upConv(1 if nMasks == 2 else nMasks, nf+4)
        self.temperature = temperature
    def forward(self, x):
        f = self.cnn(x)
        m = self.out(torch.cat([f] + [pnet(f) for pnet in self.psp], 1))
        if self.nMasks == 2:
            m = torch.sigmoid(m / self.temperature)
            m = torch.cat((m, (1-m)), 1)
        else:
            m = F.softmax(m / self.temperature, dim=1)
        return m

class _netGenX(nn.Module):
    def __init__(self, sizex=128, nOut=3, nc=8, nf=64, nMasks=2, selfAtt=False):
        super(_netGenX, self).__init__()
        if sizex != 128:
            raise NotImplementedError
        self.net = nn.ModuleList([_resMaskedGenerator128(nf=nf, nOut=nOut, nc=nc, selfAtt=selfAtt) for k in range(nMasks)])
        self.nMasks = nMasks
    def forward(self, masks, c):
        masks = masks.unsqueeze(2)
        y = []
        for k in range(self.nMasks):
            y.append(self.net[k](masks[:,k], c[:,k], c[:,k]).unsqueeze(1))
        return torch.cat(y,1)
    
class _netRecZ(nn.Module):
    def __init__(self, sizex=128, nIn=3, nc=5, nf=64, nMasks=2):
        super(_netRecZ, self).__init__()
        if sizex == 128:
            self.net = _resEncoder128(nIn=nIn, nf=nf, nOut=nc*nMasks)
        elif sizex == 64:
            self.net = _resEncoder64(nIn=nIn, nf=nf, nOut=nc*nMasks)
        self.nc = nc
        self.nMasks = nMasks
    def forward(self, x):
        c = self.net(x)
        return c.view(c.size(0), self.nMasks, self.nc, 1 , 1)
       
def weights_init_ortho(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, opt.initOrthoGain)
        
netEncM = _netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(device)
netGenX = _netGenX(sizex=opt.sizex, nOut=opt.nx, nc=opt.nz, nf=opt.nfX, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG).to(device)
netDX = _resDiscriminator128(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(device)

netEncM.apply(weights_init_ortho)
netGenX.apply(weights_init_ortho)
netDX.apply(weights_init_ortho)

#############################################################################
# Optimizer                                                                 #
#############################################################################
optimizerEncM = torch.optim.Adam(netEncM.parameters(), lr=opt.lrM, betas=(0, 0.9), weight_decay=opt.wdecay, amsgrad=False)
optimizerGenX = torch.optim.Adam(netGenX.parameters(), lr=opt.lrG, betas=(0, 0.9), amsgrad=False)
optimizerDX = torch.optim.Adam(netDX.parameters(), lr=opt.lrD, betas=(0, 0.9), amsgrad=False)

if opt.wrecZ > 0:
    netRecZ = _netRecZ(sizex=opt.sizex, nIn=opt.nx, nc=opt.nz, nf=opt.nfZ, nMasks=opt.nMasks).to(device)
    netRecZ.apply(weights_init_ortho)
    optimizerRecZ = torch.optim.Adam(netRecZ.parameters(), lr=opt.lrZ, betas=(0, 0.9), amsgrad=False)

#############################################################################
# Load                                                                      #
#############################################################################
if opt.iteration > 0:
    state = torch.load(os.path.join(opt.outf, "state_%05d.pth" % opt.iteration))
    netEncM.load_state_dict(state["netEncM"])
    netGenX.load_state_dict(state["netGenX"])
    netDX.load_state_dict(state["netDX"])
    optimizerEncM.load_state_dict(state["optimizerEncM"])
    optimizerGenX.load_state_dict(state["optimizerGenX"])
    optimizerDX.load_state_dict(state["optimizerDX"])
    if opt.wrecZ > 0:
        netRecZ.load_state_dict(state["netRecZ"])
        optimizerRecZ.load_state_dict(state["optimizerRecZ"])
else:
    try:
        os.remove(os.path.join(opt.outf, "train.dat"))
    except:
        pass
    try:
        os.remove(os.path.join(opt.outf, "test.dat"))
    except:
        pass
    try:
        os.remove(os.path.join(opt.outf, "val.dat"))
    except:
        pass

#############################################################################
# Test                                                                      #
#############################################################################
def evaluate(netEncM, loader, device, nMasks=2):
    sumScoreAcc = 0
    sumScoreIoU = 0
    nbIter = 0
    if nMasks > 2:
        raise NotImplementedError
    for xLoad, mLoad in loader:
        xData = xLoad.to(device)
        mData = mLoad.to(device)
        mPred = netEncM(xData)
        sumScoreAcc += torch.max(((mPred[:,:1] >= .5).float() == mData).float().mean(-1).mean(-1),
                                 ((mPred[:,:1] <  .5).float() == mData).float().mean(-1).mean(-1)).mean().item()
        sumScoreIoU += torch.max(
            ((((mPred[:,:1] >= .5).float() + mData) == 2).float().sum(-1).sum(-1) /
             (((mPred[:,:1] >= .5).float() + mData) >= 1).float().sum(-1).sum(-1)),
            ((((mPred[:,:1] <  .5).float() + mData) == 2).float().sum(-1).sum(-1) /
             (((mPred[:,:1] <  .5).float() + mData) >= 1).float().sum(-1).sum(-1))).mean().item()
        nbIter += 1
    return sumScoreAcc / nbIter, sumScoreIoU / nbIter

x_test, m_test = next(iter(torch.utils.data.DataLoader(testset, batch_size=opt.nTest, shuffle=True, num_workers=4, drop_last=True)))

x_test = x_test.to(device)

z_test = torch.randn((opt.nTest, opt.nMasks, opt.nz, 1, 1), device=device)
zn_test = torch.randn((opt.nTest, opt.nz, 1, 1), device=device)

img_m_test = m_test[:,:1].float()
for n in range(opt.nTest):
    img_m_test[n] = (img_m_test[n] / img_m_test[n].max()) * 2 - 1

out_X = torch.full((opt.nMasks, opt.nTest+1, opt.nTest+5, opt.nx, opt.sizex, opt.sizex), -1).to(device)
out_X[:,1:,0] = x_test
out_X[:,1:,1] = img_m_test

valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batchSize, shuffle=True, num_workers=4, drop_last=True)

#############################################################################
# Train                                                                     #
#############################################################################
genData = iter(trainloader)
disData = iter(trainloader)
if not opt.silent:
    pbar = tqdm(total=opt.checkpointFreq)
while opt.iteration <= opt.nIteration:
    if not opt.silent:
        pbar.update(1)
    ########################## Get Batch #############################
    try:
        xLoadG, mLoadG = next(genData)
    except StopIteration:
        genData = iter(trainloader)
        xLoadG, mLoadG = next(genData)
    try:
        xLoadD, mLoadD = next(disData)
    except StopIteration:
        disData = iter(trainloader)
        xLoadD, mLoadD = next(disData)
    xData = xLoadG.to(device)
    mData = mLoadG.to(device)
    xReal = xLoadD.to(device)
    zData = torch.randn((xData.size(0), opt.nMasks, opt.nz, 1, 1), device=device)
    ########################## Reset Nets ############################
    netEncM.zero_grad()
    netGenX.zero_grad()
    netDX.zero_grad()
    netEncM.train()
    netGenX.train()
    netDX.train()
    if opt.wrecZ > 0:
        netRecZ.zero_grad()
        netRecZ.train()
    dStep = (opt.iteration % opt.dStepFreq == 0)
    gStep = (opt.iteration % opt.gStepFreq == 0)
    #########################  AutoEncode X #########################
    if gStep:
        mEnc = netEncM(xData)
        hGen = netGenX(mEnc, zData)
        xGen = (hGen + ((1 - mEnc.unsqueeze(2)) * xData.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))
        dGen = netDX(xGen)
        lossG = - dGen.mean()
        if opt.wrecZ > 0:
            zRec = netRecZ(hGen.sum(1))
            err_recZ = ((zData - zRec) * (zData - zRec)).mean()
            lossG += err_recZ * opt.wrecZ
        lossG.backward()
        optimizerEncM.step()
        optimizerGenX.step()
        optimizerRecZ.step()
    if dStep:
        netDX.zero_grad()
        with torch.no_grad():
            mEnc = netEncM(xData)
            hGen = netGenX(mEnc, zData)
            xGen = (hGen + ((1 - mEnc.unsqueeze(2)) * xData.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))
        dPosX = netDX(xReal)
        dNegX = netDX(xGen)
        err_dPosX = (-1 + dPosX)
        err_dNegX = (-1 - dNegX)
        err_dPosX = ((err_dPosX < 0).float() * err_dPosX).mean()
        err_dNegX = ((err_dNegX < 0).float() * err_dNegX).mean()
        (-err_dPosX - err_dNegX).backward()
        optimizerDX.step()
    opt.iteration += 1
    if opt.iteration % opt.checkpointFreq == 0:
        if not opt.silent:
            pbar.close()
        netEncM.eval()
        netGenX.eval()
        netDX.eval()
        if opt.wrecZ > 0:
            netRecZ.eval()
        with torch.no_grad():
            mEnc_test = netEncM(x_test)
            out_X[:,1:,3] = mEnc_test.transpose(0,1).unsqueeze(2)*2-1
            out_X[:,1:,2] = ((out_X[:,1:,3] < 0).float() * -1) + (out_X[:,1:,3] > 0).float()
            out_X[:,1:,4] = (netGenX(mEnc_test, z_test) + ((1 - mEnc_test.unsqueeze(2)) * x_test.unsqueeze(1))).transpose(0,1)
            for k in range(opt.nMasks):
                for i in range(opt.nTest):
                    zx_test = z_test.clone()
                    zx_test[:, k] = zn_test[i]
                    out_X[k, 1:, i+5] = netGenX(mEnc_test, zx_test)[:,k] + ((1 - mEnc_test[:,k:k+1]) * x_test)
            scoreAccTrain, scoreIoUTrain = evaluate(netEncM, trainloader, device, opt.nMasks)
            scoreAccVal, scoreIoUVal = evaluate(netEncM, valloader, device, opt.nMasks)
            if not opt.silent:
                print("train:", scoreAccTrain, scoreIoUTrain)
                print("val:", scoreAccVal, scoreIoUVal)
            try:
                with open(os.path.join(opt.outf, 'train.dat'), 'a') as f:
                    f.write(str(opt.iteration) + ' ' + str(scoreAccTrain) + ' ' + str(scoreIoUTrain) + '\n')
            except:
                print("Cannot save in train.dat")
            try:
                with open(os.path.join(opt.outf, 'val.dat'), 'a') as f:
                    f.write(str(opt.iteration) + ' ' + str(scoreAccVal) + ' ' + str(scoreIoUVal) + '\n')
            except:
                print("Cannot save in val.dat")
            try:
                vutils.save_image(out_X.view(-1,opt.nx,opt.sizex, opt.sizex), os.path.join(opt.outf, "out_%05d.png" % opt.iteration), normalize=True, range=(-1,1), nrow=opt.nTest+5)
            except:
                print("Cannot save output")
        netEncM.zero_grad()
        netGenX.zero_grad()
        netDX.zero_grad()
        stateDic = {
            'netEncM': netEncM.state_dict(),
            'netGenX': netGenX.state_dict(),
            'netDX': netDX.state_dict(),
            'optimizerEncM': optimizerEncM.state_dict(),
            'optimizerGenX': optimizerGenX.state_dict(),
            'optimizerDX': optimizerDX.state_dict(),
            'options': opt,
        }
        if opt.wrecZ > 0:
            netRecZ.zero_grad()
            stateDic['netRecZ'] = netRecZ.state_dict()
            stateDic['optimizerRecZ'] = optimizerRecZ.state_dict(),
        try:
            torch.save(stateDic, os.path.join(opt.outf, 'state_%05d.pth' % opt.iteration))
            torch.save(opt, os.path.join(opt.outf, "options.pth"))
        except:
            print("Cannot save checkpoint")
        if opt.bestValIoU < scoreIoUVal:
            opt.bestValIoU = scoreIoUVal
            try:
                torch.save(stateDic, os.path.join(opt.outf, 'best.pth'))
            except:
                print("Cannot save best")
        if opt.clean and opt.iteration > opt.checkpointFreq:
            try:
                os.remove(os.path.join(opt.outf, 'state_%05d.pth' % (opt.iteration - opt.checkpointFreq)))
            except:
                pass
        if not opt.silent:
            pbar = tqdm(total=opt.checkpointFreq)
        netEncM.train()
        netGenX.train()
        netDX.train()
        if opt.wrecZ > 0:
            netRecZ.train()
