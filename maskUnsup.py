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
# Arguments                                                  rint               #
#############################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default="flowers", help='flowers | cub')
parser.add_argument('--dataroot', default='./data', help='path to datasets')

parser.add_argument('--nx', type=int, default=3, help='number of canals of the input image')
parser.add_argument('--sizex', type=int, default=128, help='the height / width of the input image')
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')

parser.add_argument('--nMasks', type=int, default=2, help='number of masks')

parser.add_argument('--nResM', type=int, default=3, help='number of residual blocs in netM')
parser.add_argument('--nResX', type=int, default=3, help='number of residual blocs in netX')
parser.add_argument('--nf', type=int, default=64)
parser.add_argument('--nfD', type=int, default=None)
parser.add_argument('--nfX', type=int, default=None)
parser.add_argument('--nfM', type=int, default=None)
parser.add_argument('--nfZ', type=int, default=None)
parser.add_argument('--pooling', default="avg")

parser.add_argument('--useSelfAttD', action='store_true', help='use self attention for D')
parser.add_argument('--useSelfAttG', action='store_true', help='use self attention for G')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--nTest', type=int, default=5, help='input batch size for visu')

parser.add_argument('--nIteration', type=int, default=1e5, help='number of iterations')

parser.add_argument('--initOrthoGain', type=float, default=.8)
parser.add_argument('--initOrthoGainX', type=float, default=None)
parser.add_argument('--initOrthoGainZ', type=float, default=None)
parser.add_argument('--initOrthoGainM', type=float, default=None)
parser.add_argument('--initOrthoGainD', type=float, default=None)
parser.add_argument('--lrG', type=float, default=1e-4, help='learning rate for G, default=1e-4')
parser.add_argument('--lrM', type=float, default=1e-5, help='learning rate for M, default=1e-5')
parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate for D, default=1e-4')
parser.add_argument('--lrZ', type=float, default=1e-4, help='learning rate for Z, default=1e-4')
parser.add_argument('--gStepFreq', type=int, default=1, help='wait x steps for G updates')
parser.add_argument('--mStepFreq', type=int, default=1, help='wait x steps for M updates')
parser.add_argument('--dStepFreq', type=int, default=1, help='wait x steps for D updates')
parser.add_argument('--zStepFreq', type=int, default=1, help='wait x steps for Z updates')

parser.add_argument('--temperature', type=float, default=1, help='softmax temperature')
parser.add_argument('--wdecay', type=float, default=1e-4, help='weight decay for M, default=1e-4')
parser.add_argument('--wvar', type=float, default=0, help='weight for variance penalty')
parser.add_argument('--wmean', type=float, default=0, help='weight for area mean loss')
parser.add_argument('--maskmean0', type=float, default=-1, help='mean for mask0')
parser.add_argument('--maskmean1', type=float, default=-1, help='mean for mask1')

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

if opt.initOrthoGainX is None:
    opt.initOrthoGainX = opt.initOrthoGain
if opt.initOrthoGainD is None:
    opt.initOrthoGainD = opt.initOrthoGain
if opt.initOrthoGainZ is None:
    opt.initOrthoGainZ = opt.initOrthoGain
if opt.initOrthoGainM is None:
    opt.initOrthoGainM = opt.initOrthoGain

if opt.nfD is None:
    opt.nfD = opt.nf
if opt.nfX is None:
    opt.nfX = opt.nf
if opt.nfM is None:
    opt.nfM = opt.nf
if opt.nfZ is None:
    opt.nfZ = opt.nf

print(opt)
    
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
        if img.size(0) == 1:
            img = img.expand(3, img.size(1), img.size(2))
        seg = self.transform(Image.open(os.path.join(self.datapath, "segmask", segname)))
        if seg.size(0) != 1:
            seg = seg[:1]
        return img * 2 - 1, seg

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(CelebADataset, self).__init__()
        self.files = os.listdir(dataPath)
        if sets == 'train':
            self.files = self.files[:200000]
        else:
            self.files = self.files[200000:]
        self.transform = transform
        self.datapath = dataPath
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = self.transform(Image.open(os.path.join(self.datapath, self.files[idx])))
        if img.size(0) == 1:
            img = img.expand(3, img.size(1), img.size(2))
        return img * 2 - 1, 0

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

class LFW3Dataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(LFW3Dataset, self).__init__()
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
        if img.size(0) == 1:
            img = img.expand(3, img.size(1), img.size(2))
        if self.sets == 'test' or self.sets == 'val':
            seg = Image.open(os.path.join(self.datapath,
                                          "parts_lfw_funneled_gt_images",
                                          self.files[idx,1]+'.ppm'))
            seg = self.transform(seg)
        else:
            seg = img
        return img * 2 - 1, seg
    
if opt.dataset == 'lfw':
    if opt.nMasks == 3:
        trainset = LFW3Dataset(dataPath=os.path.join(opt.dataroot, "lfw-funneled"),
                               sets='train',
                               transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                             transforms.CenterCrop(opt.sizex),
                                                             transforms.ToTensor(),
                               ]),)
        testset = LFW3Dataset(dataPath=os.path.join(opt.dataroot, "lfw-funneled"),
                              sets='test',
                              transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                            transforms.CenterCrop(opt.sizex),
                                                            transforms.ToTensor(),
                              ]),)
        valset = LFW3Dataset(dataPath=os.path.join(opt.dataroot, "lfw-funneled"),
                             sets='val',
                             transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                           transforms.CenterCrop(opt.sizex),
                                                           transforms.ToTensor(),
                             ]),)    
    else:
        trainset = LFWDataset(dataPath=os.path.join(opt.dataroot, "lfw-funneled"),
                              sets='train',
                              transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                            transforms.CenterCrop(opt.sizex),
                                                            transforms.ToTensor(),
                              ]),)
        testset = LFWDataset(dataPath=os.path.join(opt.dataroot, "lfw-funneled"),
                             sets='test',
                             transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                           transforms.CenterCrop(opt.sizex),
                                                           transforms.ToTensor(),
                             ]),)
        valset = LFWDataset(dataPath=os.path.join(opt.dataroot, "lfw-funneled"),
                             sets='val',
                             transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                           transforms.CenterCrop(opt.sizex),
                                                           transforms.ToTensor(),
                             ]),)
if opt.dataset == 'celebA':
    trainset = CelebADataset(dataPath=os.path.join(opt.dataroot, "celebA", "aligned"),
                             sets='train',
                             transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                           transforms.CenterCrop(opt.sizex),
                                                           transforms.ToTensor(),
                             ]))
    testset = CelebADataset(dataPath=os.path.join(opt.dataroot, "celebA", "aligned"),
                            sets='test',
                            transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                          transforms.CenterCrop(opt.sizex),
                                                          transforms.ToTensor(),
                            ]))
if opt.dataset == 'cub':
    trainset = CUBDataset(os.path.join(opt.dataroot, "CUB_200_2011", "CUB_200_2011"),
                          "train",
                          transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                              transforms.CenterCrop(opt.sizex),
                                              transforms.ToTensor(),
                          ]))
    testset = CUBDataset(os.path.join(opt.dataroot, "CUB_200_2011", "CUB_200_2011"),
                         "test",
                         transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                             transforms.CenterCrop(opt.sizex),
                                             transforms.ToTensor(),
                         ]))
    valset = CUBDataset(os.path.join(opt.dataroot, "CUB_200_2011", "CUB_200_2011"),
                         "val",
                         transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                             transforms.CenterCrop(opt.sizex),
                                             transforms.ToTensor(),
                         ]))
    if opt.maskmean1 == -1:
        opt.maskmean1 = 0.825
        opt.maskmean0 = 1 - 0.825
if opt.dataset == 'flowers':
    trainset = FlowersDataset(os.path.join(opt.dataroot, "102flowers"),
                              "train",
                              transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                  transforms.CenterCrop(opt.sizex),
                                                  transforms.ToTensor(),
                              ]))
    testset = FlowersDataset(os.path.join(opt.dataroot, "102flowers"),
                             "test",
                             transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                 transforms.CenterCrop(opt.sizex),
                                                 transforms.ToTensor(),
                             ]))
    valset = FlowersDataset(os.path.join(opt.dataroot, "102flowers"),
                             "val",
                             transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                 transforms.CenterCrop(opt.sizex),
                                                 transforms.ToTensor(),
                             ]))
    if opt.maskmean1 == -1:
        opt.maskmean1 = 0.58
        opt.maskmean0 = 1 - 0.58

        
if opt.maskmean1 == -1:
    opt.maskmean1 = 0.5
    opt.maskmean0 = 0.5

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

class SelfAttentionMax(nn.Module):
    def __init__(self, nf, nh=False):
        super(SelfAttentionMax, self).__init__()
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

class SelfAttentionBigGan(nn.Module):
  def __init__(self, nf, name='attention'):
    super(SelfAttentionBigGan, self).__init__()
    self.nf = nf
    self.theta = spectral_norm(nn.Conv2d(self.nf, self.nf // 8, kernel_size=1, padding=0, bias=False))
    self.phi = spectral_norm(nn.Conv2d(self.nf, self.nf // 8, kernel_size=1, padding=0, bias=False))
    self.g = spectral_norm(nn.Conv2d(self.nf, self.nf // 2, kernel_size=1, padding=0, bias=False))
    self.o = spectral_norm(nn.Conv2d(self.nf // 2, self.nf, kernel_size=1, padding=0, bias=False))
    self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
  def forward(self, x):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])
    # Perform reshapes
    theta = theta.view(-1, self.nf // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self.nf // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self.nf // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.nf // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x

SelfAttention = SelfAttentionMax
    
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

class _resDiscriminator64(nn.Module):
    def __init__(self, nIn=3, nf=64, selfAtt=False):
        super(_resDiscriminator64, self).__init__()
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
        for i in range(3):
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


class _resnet(nn.Module):
    def __init__(self, nIn=3, nf=64, nOut=3, nBlocs=9):
        super(_resnet, self).__init__()
        # Res 0
        self.bloc0 = nn.Sequential(spectral_norm(nn.Conv2d(nIn, nf, 3, 1, 1, bias=False)),
                                   nn.BatchNorm2d(nf, affine=True),
                                   nn.ReLU(),
                                   spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                   nn.AvgPool2d(2))
        self.sc0 = nn.Sequential(nn.AvgPool2d(2),
                                 spectral_norm(nn.Conv2d(nIn, nf, 1, bias=True)))
        # Res 1
        self.bloc1 = nn.Sequential(nn.BatchNorm2d(nf, affine=True),
                                   nn.ReLU(),
                                   spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)),
                                   nn.BatchNorm2d(nf, affine=True),
                                   nn.ReLU(),
                                   spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                   nn.AvgPool2d(2))
        self.sc1 = nn.Sequential(nn.AvgPool2d(2),
                                 spectral_norm(nn.Conv2d(nf, nf, 1, bias=True)))
        self.blocs = []
        for i in range(nBlocs):
            self.blocs.append(nn.Sequential(nn.BatchNorm2d(nf, affine=True),
                                            nn.ReLU(),
                                            spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)),
                                            nn.BatchNorm2d(nf, affine=True),
                                            nn.ReLU(),
                                            spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))))
        self.blocs = nn.ModuleList(self.blocs)
        # Res 2
        self.bloc2 = nn.Sequential(nn.BatchNorm2d(nf, affine=True),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2),
                                   spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)),
                                   nn.BatchNorm2d(nf, affine=True),
                                   nn.ReLU(),
                                   spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)))
        self.sc2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                 spectral_norm(nn.Conv2d(nf, nf, 1, bias=True)))
        # Res 3
        self.bloc3 = nn.Sequential(nn.BatchNorm2d(nf, affine=True),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2),
                                   spectral_norm(nn.Conv2d(nf, nOut, 3, 1, 1, bias=False)),
                                   nn.BatchNorm2d(nOut, affine=True),
                                   nn.ReLU(),
                                   spectral_norm(nn.Conv2d(nOut, nOut, 3, 1, 1, bias=True)))
        self.sc3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                 spectral_norm(nn.Conv2d(nf, nOut, 1, bias=True)))
    def forward(self, x):
        x = self.bloc0(x) + self.sc0(x)
        x = self.bloc1(x) + self.sc1(x)
        for bloc in self.blocs:
            x = bloc(x) + x
        x = self.bloc2(x) + self.sc2(x)
        x = self.bloc3(x) + self.sc3(x)
        return x

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

class _resEncoder64(nn.Module):
    def __init__(self, nIn=3, nf=64, nOut=8):
        super(_resEncoder64, self).__init__()
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
        for i in range(3):
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
    
class _resGenerator(nn.Module):
    def __init__(self, nIn=1, nf=64, nOut=3, nc=8, nLayers=3):
        super(_resGenerator, self).__init__()
        # Res 0
        self.convA0 = spectral_norm(nn.Conv2d(nIn, nf, 3, 1, 1, bias=False))
        self.convB0 = nn.Sequential(spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                    nn.AvgPool2d(2))
        self.normB0 = nn.InstanceNorm2d(nf, affine=False)
        self.gammaB0 = nn.Conv2d(nc, nf, 1, bias=True)
        self.betaB0 = nn.Conv2d(nc, nf, 1, bias=True)
        self.sc0 = nn.Sequential(nn.AvgPool2d(2),
                                 spectral_norm(nn.Conv2d(nIn, nf, 1, bias=True)))
        # Res 1
        self.normA1 = nn.InstanceNorm2d(nf, affine=False)
        self.normB1 = nn.InstanceNorm2d(nf, affine=False)
        self.convA1 = spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        self.convB1 = nn.Sequential(spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                                    nn.AvgPool2d(2))
        self.gammaA1 = nn.Conv2d(nc, nf, 1, bias=True)
        self.gammaB1 = nn.Conv2d(nc, nf, 1, bias=True)
        self.betaA1 = nn.Conv2d(nc, nf, 1, bias=True)
        self.betaB1 = nn.Conv2d(nc, nf, 1, bias=True)
        self.sc1 = nn.Sequential(nn.AvgPool2d(2),
                                 spectral_norm(nn.Conv2d(nf, nf, 1, bias=True)))
        # Res blocs
        self.convA = []
        self.convB = []
        self.normA = []
        self.normB = []
        self.gammaA = []
        self.gammaB = []
        self.betaA = []
        self.betaB = []
        for k in range(nLayers):
            self.normA.append(nn.InstanceNorm2d(nf, affine=False))
            self.normB.append(nn.InstanceNorm2d(nf, affine=False))
            self.convA.append(spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)))
            self.convB.append(spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)))
            self.gammaA.append(nn.Conv2d(nc, nf, 1, bias=True))
            self.gammaB.append(nn.Conv2d(nc, nf, 1, bias=True))
            self.betaA.append(nn.Conv2d(nc, nf, 1, bias=True))
            self.betaB.append(nn.Conv2d(nc, nf, 1, bias=True))
        self.convA = nn.ModuleList(self.convA)
        self.convB = nn.ModuleList(self.convB)
        self.normA = nn.ModuleList(self.normA)
        self.normB = nn.ModuleList(self.normB)
        self.gammaA =nn.ModuleList(self.gammaA)
        self.gammaB =nn.ModuleList(self.gammaB)
        self.betaA = nn.ModuleList(self.betaA)
        self.betaB = nn.ModuleList(self.betaB)
        # Res 2
        self.normA2 = nn.InstanceNorm2d(nf, affine=False)
        self.normB2 = nn.InstanceNorm2d(nf, affine=False)
        self.convA2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                    spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)))
        self.convB2 = spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.gammaA2 = nn.Conv2d(nc, nf, 1, bias=True)
        self.gammaB2 = nn.Conv2d(nc, nf, 1, bias=True)
        self.betaA2 = nn.Conv2d(nc, nf, 1, bias=True)
        self.betaB2 = nn.Conv2d(nc, nf, 1, bias=True)
        self.sc2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                 spectral_norm(nn.Conv2d(nf, nf, 1, bias=True)))
        # Res 3
        self.normA3 = nn.InstanceNorm2d(nf, affine=False)
        self.normB3 = nn.InstanceNorm2d(nOut, affine=False)
        self.convA3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                    spectral_norm(nn.Conv2d(nf, nOut, 3, 1, 1, bias=False)))
        self.convB3 = spectral_norm(nn.Conv2d(nOut, nOut, 3, 1, 1, bias=True))
        self.gammaA3 = nn.Conv2d(nc, nf, 1, bias=True)
        self.gammaB3 = nn.Conv2d(nc, nOut, 1, bias=True)
        self.betaA3 = nn.Conv2d(nc, nf, 1, bias=True)
        self.betaB3 = nn.Conv2d(nc, nOut, 1, bias=True)
        self.sc3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                 spectral_norm(nn.Conv2d(nf, nOut, 1, bias=True)))
    def forward(self, x, c):
        # Res 0
        h = x
        h = self.convA0(h)
        h = self.convB0(F.relu(self.normB0(h) * self.gammaB0(c) + self.betaB0(c)))
        y = h + self.sc0(x)
        # Res 1
        h = y
        h = self.convA1(F.relu(self.normA1(h) * self.gammaA1(c) + self.betaA1(c)))
        h = self.convB1(F.relu(self.normB1(h) * self.gammaB1(c) + self.betaB1(c)))
        y = h + self.sc1(y)
        # Res blocs
        for k in range(len(self.convA)):
            h = y
            h = self.convA[k](F.relu(self.normA[k](h) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            y = h + y
        # Res 2
        h = y
        h = self.convA2(F.relu(self.normA2(h) * self.gammaA2(c) + self.betaA2(c)))
        h = self.convB2(F.relu(self.normB2(h) * self.gammaB2(c) + self.betaB2(c)))
        y = h + self.sc2(y)
        # Res 3
        h = y
        h = self.convA3(F.relu(self.normA3(h) * self.gammaA3(c) + self.betaA3(c)))
        h = self.convB3(F.relu(self.normB3(h) * self.gammaB3(c) + self.betaB3(c)))
        y = h + self.sc3(y)
        return torch.tanh(y)

class _maxMaskedGenerator128(nn.Module):
    def __init__(self, nIn=128, nf=64, nOut=3, nc=8, nLayers=3, selfAtt=False):
        super(_maxMaskedGenerator128, self).__init__()
        if selfAtt:
            self.selfAtt = SelfAttention(nf*2)
        else:
            self.selfAtt = nn.Sequential()
        self.nLayers = nLayers
        ######## upsample #############
        self.dense = nn.Linear(nIn, 4*4*nf*16)
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
        for k in range(3):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ### transitioning
        self.normIn = nn.InstanceNorm2d(nfPrev, affine=False)
        self.gammaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.betaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.convIn = spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1))
        ###
        # Res blocs
        for k in range(nLayers):
            self.convA.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=False)))
            self.convB.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=True)))
            self.normA.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.normB.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.gammaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.gammaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.sc.append(nn.Sequential())
        for k in range(2):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ##############################
    def forward(self, m, z, c):
        ######### Upsample ###########
        x = self.dense(z.view(z.size(0),z.size(1))).view(z.size(0), -1, 4, 4)
        mask_ratio = m.size(-1) // 4
        x = F.max_pool2d(m, kernel_size=mask_ratio) * x
        for k in range(3):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            mask_ratio = mask_ratio // 2
            x = F.max_pool2d(m, kernel_size=mask_ratio) * x
        x = self.convIn(F.relu(self.normIn(x) * self.gammaIn(c) + self.betaIn(c)))
        mask_4 = F.max_pool2d(m, kernel_size=4)
        x = mask_4 * x
        for k in range(3, 3 + self.nLayers):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            x = mask_4 * x
        for k in range(3+self.nLayers, 3+self.nLayers+2):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            if k == 3+self.nLayers:
                x = self.selfAtt(x)
            mask_ratio = mask_ratio // 2
            x = F.max_pool2d(m, kernel_size=mask_ratio) * x
        x = self.convOut(F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)))
        x = torch.tanh(x)
        return x * m

class _avgResMaskedGenerator128(nn.Module):
    def __init__(self, nIn=128, nf=64, nOut=3, nc=8, nLayers=3, selfAtt=False):
        super(_avgResMaskedGenerator128, self).__init__()
        if selfAtt:
            self.selfAtt = SelfAttention(nf*2)
        else:
            self.selfAtt = nn.Sequential()
        self.nLayers = nLayers
        ######## upsample #############
        self.dense = nn.Linear(nIn, 4*4*nf*16)
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
        for k in range(3):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ### transitioning
        self.normIn = nn.InstanceNorm2d(nfPrev, affine=False)
        self.gammaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.betaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.convIn = spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1))
        ###
        # Res blocs
        for k in range(nLayers):
            self.convA.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=False)))
            self.convB.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=True)))
            self.normA.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.normB.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.gammaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.gammaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.sc.append(nn.Sequential())
        for k in range(2):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ##############################
    def forward(self, m, z, c):
        ######### Upsample ###########
        x = self.dense(z.view(z.size(0),z.size(1))).view(z.size(0), -1, 4, 4)
        mask_ratio = m.size(-1) // 4
        # x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        for k in range(3):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            mask_ratio = mask_ratio // 2
            x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        x = self.convIn(F.relu(self.normIn(x) * self.gammaIn(c) + self.betaIn(c)))
        # mask_4 = F.avg_pool2d(m, kernel_size=4)
        # x = mask_4 * x
        for k in range(3, 3 + self.nLayers):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            # x = mask_4 * x
        for k in range(3+self.nLayers, 3+self.nLayers+2):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            if k == 3+self.nLayers:
                x = self.selfAtt(x)
            # mask_ratio = mask_ratio // 2
            # x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        x = self.convOut(F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)))
        x = torch.tanh(x)
        return x * m

class _avgMaskedGenerator128(nn.Module):
    def __init__(self, nIn=128, nf=64, nOut=3, nc=8, nLayers=3, maskLayers=3, selfAtt=False):
        super(_avgMaskedGenerator128, self).__init__()
        if selfAtt:
            self.selfAtt = SelfAttention(nf*2)
        else:
            self.selfAtt = nn.Sequential()
        self.mLayers = maskLayers
        ######## upsample #############
        self.dense = nn.Linear(nIn, 4*4*nf*16)
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
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ##############################
    def forward(self, m, z, c):
        ######### Upsample ###########
        x = self.dense(z.view(z.size(0),z.size(1))).view(z.size(0), -1, 4, 4)
        mask_ratio = m.size(-1) // 4
        # x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        for k in range(5):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            mask_ratio = mask_ratio // 2
            if self.minMaskLayers < k and k < self.maxMaskLayers:
                x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
            if k == 3:
                x = self.selfAtt(x)
        x = self.convOut(F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)))
        x = torch.tanh(x)
        return x * m
    
class _resMaskedGenerator128(nn.Module):
    def __init__(self, nIn=128, nf=64, nOut=3, nc=8, nLayers=3, selfAtt=False):
        super(_resMaskedGenerator128, self).__init__()
        if selfAtt:
            self.selfAtt = SelfAttention(nf*2)
        else:
            self.selfAtt = nn.Sequential()
        self.nLayers = nLayers
        self.dense = nn.Linear(nIn, 4*4*nf*16)
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
            # self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
            #                                 spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        # x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        for k in range(5):
            # if k == 3:
            #     x = x * F.avg_pool2d(m, kernel_size=mask_ratio)
            if k == 4:
                x = self.selfAtt(x)
            # h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convA[k](torch.cat((F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)),
                                         F.avg_pool2d(m, kernel_size=mask_ratio)), 1))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            mask_ratio = mask_ratio // 2
        # x = x * m
        x = self.convOut(torch.cat((F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)),
                                    m), 1))
        x = torch.tanh(x)
        return x * m

class _maskedGenerator128(nn.Module):
    def __init__(self, nIn=128, nf=64, nOut=3, nc=8, nLayers=3, selfAtt=False):
        super(_maskedGenerator128, self).__init__()
        if selfAtt:
            self.selfAtt = SelfAttention(nf*2)
        else:
            self.selfAtt = nn.Sequential()
        self.nLayers = nLayers
        ######## upsample #############
        self.dense = nn.Linear(nIn, 4*4*nf*16)
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
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ##############################
    def forward(self, m, z, c):
        ######### Upsample ###########
        x = self.dense(z.view(z.size(0),z.size(1))).view(z.size(0), -1, 4, 4)
        # mask_ratio = m.size(-1) // 4
        # x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        for k in range(5):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            if k == 2:
                x = F.avg_pool2d(m, kernel_size=4) * x ###
            # mask_ratio = mask_ratio // 2
            # x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
            if k == 3:
                x = self.selfAtt(x)
        x = self.convOut(F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)))
        x = torch.tanh(x)
        return x * m
    
class _maxMaskedGenerator64(nn.Module):
    def __init__(self, nIn=128, nf=64, nOut=3, nc=8, nLayers=3, selfAtt=False):
        super(_maxMaskedGenerator64, self).__init__()
        if selfAtt:
            self.selfAtt = SelfAttention(nf)
        else:
            self.selfAtt = nn.Sequential()
        self.nLayers = nLayers
        ######## upsample #############
        self.dense = nn.Linear(nIn, 4*4*nf*16)
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
        nfNext = nf*8
        for k in range(3):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ### transitioning
        self.normIn = nn.InstanceNorm2d(nfPrev, affine=False)
        self.gammaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.betaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.convIn = spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1))
        ###
        # Res blocs
        for k in range(nLayers):
            self.convA.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=False)))
            self.convB.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=True)))
            self.normA.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.normB.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.gammaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.gammaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.sc.append(nn.Sequential())
        for k in range(2):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ##############################
    def forward(self, m, z, c):
        ######### Upsample ###########
        x = self.dense(z.view(z.size(0),z.size(1))).view(z.size(0), -1, 4, 4)
        mask_ratio = m.size(-1) // 4
        x = F.max_pool2d(m, kernel_size=mask_ratio) * x
        for k in range(3):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            mask_ratio = mask_ratio // 2
            x = F.max_pool2d(m, kernel_size=mask_ratio) * x
        x = self.convIn(F.relu(self.normIn(x) * self.gammaIn(c) + self.betaIn(c)))
        mask_2 = F.max_pool2d(m, kernel_size=2)
        x = mask_2 * x
        for k in range(3, 3 + self.nLayers):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            x = mask_2 * x
        for k in range(3+self.nLayers, 3+self.nLayers+1):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            if k == 3+self.nLayers:
                x = self.selfAtt(x)
            mask_ratio = mask_ratio // 2
            x = F.max_pool2d(m, kernel_size=mask_ratio) * x
        x = self.convOut(F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)))
        x = torch.tanh(x)
        return x * m

class _avgMaskedGenerator64(nn.Module):
    def __init__(self, nIn=128, nf=64, nOut=3, nc=8, nLayers=3, selfAtt=False):
        super(_avgMaskedGenerator64, self).__init__()
        if selfAtt:
            self.selfAtt = SelfAttention(nf)
        else:
            self.selfAtt = nn.Sequential()
        self.nLayers = nLayers
        ######## upsample #############
        self.dense = nn.Linear(nIn, 4*4*nf*16)
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
        nfNext = nf*8
        for k in range(3):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ### transitioning
        self.normIn = nn.InstanceNorm2d(nfPrev, affine=False)
        self.gammaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.betaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.convIn = spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1))
        ###
        # Res blocs
        for k in range(nLayers):
            self.convA.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=False)))
            self.convB.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=True)))
            self.normA.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.normB.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.gammaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.gammaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.sc.append(nn.Sequential())
        for k in range(2):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ##############################
    def forward(self, m, z, c):
        ######### Upsample ###########
        x = self.dense(z.view(z.size(0),z.size(1))).view(z.size(0), -1, 4, 4)
        mask_ratio = m.size(-1) // 4
        x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        for k in range(3):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            mask_ratio = mask_ratio // 2
            x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        x = self.convIn(F.relu(self.normIn(x) * self.gammaIn(c) + self.betaIn(c)))
        mask_2 = F.avg_pool2d(m, kernel_size=2)
        x = mask_2 * x
        for k in range(3, 3 + self.nLayers):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            x = mask_2 * x
        for k in range(3+self.nLayers, 3+self.nLayers+1):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            if k == 3+self.nLayers:
                x = self.selfAtt(x)
            mask_ratio = mask_ratio // 2
            x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        x = self.convOut(F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)))
        x = torch.tanh(x)
        return x * m

class _resMaskedGenerator64(nn.Module):
    def __init__(self, nIn=128, nf=64, nOut=3, nc=8, nLayers=3, selfAtt=False):
        super(_resMaskedGenerator64, self).__init__()
        if selfAtt:
            self.selfAtt = SelfAttention(nf)
        else:
            self.selfAtt = nn.Sequential()
        self.nLayers = nLayers
        ######## upsample #############
        self.dense = nn.Linear(nIn, 4*4*nf*16)
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
        nfNext = nf*8
        for k in range(3):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ### transitioning
        self.normIn = nn.InstanceNorm2d(nfPrev, affine=False)
        self.gammaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.betaIn = nn.Conv2d(nc, nfPrev, 1, bias=True)
        self.convIn = spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1))
        ###
        # Res blocs
        for k in range(nLayers):
            self.convA.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=False)))
            self.convB.append(spectral_norm(nn.Conv2d(nfPrev, nfPrev, 3, 1, 1, bias=True)))
            self.normA.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.normB.append(nn.InstanceNorm2d(nfPrev, affine=False))
            self.gammaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.gammaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.betaB.append(nn.Conv2d(nc, nfPrev, 1, bias=True))
            self.sc.append(nn.Sequential())
        for k in range(2):
            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                            spectral_norm(nn.Conv2d(nfPrev, nfNext, 3, 1, 1, bias=False)),))
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
        ##############################
    def forward(self, m, z, c):
        ######### Upsample ###########
        x = self.dense(z.view(z.size(0),z.size(1))).view(z.size(0), -1, 4, 4)
        # mask_ratio = m.size(-1) // 4
        # x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        for k in range(3):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            # mask_ratio = mask_ratio // 2
            # x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        x = F.avg_pool2d(m, kernel_size=2) * x ###
        x = self.convIn(F.relu(self.normIn(x) * self.gammaIn(c) + self.betaIn(c)))
        # mask_2 = F.avg_pool2d(m, kernel_size=2)
        # x = mask_2 * x
        for k in range(3, 3 + self.nLayers):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            # x = mask_2 * x
        for k in range(3+self.nLayers, 3+self.nLayers+1):
            h = self.convA[k](F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)))
            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))
            x = h + self.sc[k](x)
            if k == 3+self.nLayers:
                x = self.selfAtt(x)
            # mask_ratio = mask_ratio // 2
            # x = F.avg_pool2d(m, kernel_size=mask_ratio) * x
        x = self.convOut(F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)))
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
                                  nn.InstanceNorm2d(nf//2, affine=True), # nn.InstanceNorm2d(nf),
                                  nn.ReLU(),
                                  spectral_norm(nn.Conv2d(nf//2, nf, 3, 2, 1, bias=False)) if spectralNorm else nn.Conv2d(nf//2, nf, 3, 2, 1, bias=False),
                                  nn.InstanceNorm2d(nf, affine=True), # nn.InstanceNorm2d(nf),
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
        # self.out = _upConv(nMasks, nf+4)
        self.out = _upConv(1 if nMasks == 2 else nMasks, nf+4)
        self.temperature = temperature
    def forward(self, x):
        f = self.cnn(x)
        m = self.out(torch.cat([f] + [pnet(f) for pnet in self.psp], 1))
        # m = F.softmax(m / self.temperature, dim=1)
        if self.nMasks == 2:
            m = torch.sigmoid(m / self.temperature)
            m = torch.cat((m, (1-m)), 1)
        else:
            m = F.softmax(m / self.temperature, dim=1)
        return m

class _netGenX(nn.Module):
    def __init__(self, sizex=128, nIn=1, nOut=3, nc=8, nRes=5, nf=128, nMasks=2, selfAtt=False, pooling="avg", maskLayers=5):
        super(_netGenX, self).__init__()
        if sizex == 128:
            if pooling == "max":
                self.net = nn.ModuleList([_maxMaskedGenerator128(nIn=nc, nf=nf, nOut=nOut, nc=nc, nLayers=nRes, selfAtt=selfAtt) for k in range(nMasks)])
            elif pooling == "avg":
                self.net = nn.ModuleList([_resMaskedGenerator128(nIn=nc, nf=nf, nOut=nOut, nc=nc, nLayers=nRes, selfAtt=selfAtt) for k in range(nMasks)])
        elif sizex == 64:
            if pooling == "max":
                self.net = nn.ModuleList([_maxMaskedGenerator64(nIn=nc, nf=nf, nOut=nOut, nc=nc, nLayers=nRes, selfAtt=selfAtt) for k in range(nMasks)])
            elif pooling == "avg":
                self.net = nn.ModuleList([_resMaskedGenerator64(nIn=nc, nf=nf, nOut=nOut, nc=nc, nLayers=nRes, selfAtt=selfAtt) for k in range(nMasks)])
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

class _netRecM(nn.Module):
    pass

#############################################################################################################################################################
# def weights_init_ortho(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight, math.sqrt(2))
# 
# def weights_init_ortho_pix2pix(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight, 0.02)

# def weights_init_ortho(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight, opt.initOrthoGain)
# 
# def weights_init_ortho_pix2pix(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight, opt.initOrthoGain)
        
def weights_init_ortho_M(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, opt.initOrthoGainM)
def weights_init_ortho_X(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, opt.initOrthoGainX)
def weights_init_ortho_Z(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, opt.initOrthoGainZ)
def weights_init_ortho_D(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, opt.initOrthoGainD)

        
netEncM = _netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(device)
netGenX = _netGenX(sizex=opt.sizex, nIn=1, nOut=opt.nx, nc=opt.nz, nRes=opt.nResX, nf=opt.nfX, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG, pooling=opt.pooling).to(device)
# netGenX = _netGenX(sizex=opt.sizex, nIn=1, nOut=opt.nx, nc=opt.nz, nRes=opt.nResX, nf=opt.nf//2, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG, pooling=opt.pooling).to(device)
# netRecM = _netRecM().to(device)

if opt.sizex == 64:
    netDX = _resDiscriminator64(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(device)
elif opt.sizex == 128:
    netDX = _resDiscriminator128(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(device)

netEncM.apply(weights_init_ortho_M)
netGenX.apply(weights_init_ortho_X)
netDX.apply(weights_init_ortho_D)
# netRecM.apply(weights_init_pix2pix)

#############################################################################
# Optimizer                                                                 #
#############################################################################
#optimizerEncM = optim.Adam(netEncM.parameters(), lr=opt.lrM, betas=(0.5, 0.999), amsgrad=False)
#optimizerGenX = optim.Adam(netGenX.parameters(), lr=opt.lrG, betas=(0.5, 0.999), amsgrad=False)
#optimizerDX = optim.Adam(netDX.parameters(), lr=opt.lrD, betas=(0.5, 0.999), amsgrad=False)

optimizerEncM = torch.optim.Adam(netEncM.parameters(), lr=opt.lrM, betas=(0, 0.9), weight_decay=opt.wdecay, amsgrad=False)
optimizerGenX = torch.optim.Adam(netGenX.parameters(), lr=opt.lrG, betas=(0, 0.9), amsgrad=False)
optimizerDX = torch.optim.Adam(netDX.parameters(), lr=opt.lrD, betas=(0, 0.9), amsgrad=False)
# optimizerRecM = torch.optim.Adam(netRecM.parameters(), lr=opt.lrZ, betas=(0, 0.9), amsgrad=False)
# schedulerEncM = optim.lr_scheduler.StepLR(optimizerEncM, step_size=100000, gamma=0.1)

if opt.wrecZ > 0:
    netRecZ = _netRecZ(sizex=opt.sizex, nIn=opt.nx, nc=opt.nz, nf=opt.nfZ, nMasks=opt.nMasks).to(device)
    netRecZ.apply(weights_init_ortho_Z)
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
def variation_of_information(x, ymaps):
    sigma = 0.0
    n = x.size(2) * x.size(3)
    for vx in range(x.size(1)):
        mx = x[:,vx:vx+1]
        log2px = mx.sum((2,3)) / n
        log2px[log2px > 0] = log2px[log2px > 0].log2()
        for vy in ymaps.unique():
            my = (ymaps == vy).float()
            log2py = my.sum((2,3)) / n
            log2py[log2py > 0] = log2py[log2py > 0].log2()
            r = (mx + my == 2).sum((2,3)).float() / n
            log2r = r.clone()
            log2r[r>0] = log2r[r>0].log2()
            sigma += r * (2 * log2r - log2px - log2py)
    return sigma.abs().mean()

def probabilistic_rand_index(x, ymaps):
    # reduce memory size approx ...
    x = x[:,:,::2,::2]
    ymaps = ymaps[:,:,::2,::2].long()
    _, xmap = x.max(1)
    xmap = xmap.flatten()
    idExample = torch.arange(0,x.size(0)).unsqueeze(1).expand(x.size(0), x.size(2)*x.size(3)).flatten()
    c = x[idExample,xmap].view(x.size(0), x.size(2)*x.size(3), x.size(2)*x.size(3))
    y = x.new_zeros(ymaps.size(0), ymaps.size(1), ymaps.max().item()+1, ymaps.size(2), ymaps.size(3)).scatter(2, ymaps.unsqueeze(2), 1)
    ymaps = ymaps.flatten()
    idEx = torch.arange(0,y.size(0)).unsqueeze(1).unsqueeze(2).expand(y.size(0), y.size(1), y.size(3)*y.size(4)).flatten()
    idGt = torch.arange(0,y.size(1)).unsqueeze(0).unsqueeze(2).expand(y.size(0), y.size(1), y.size(3)*y.size(4)).flatten()
    p = y[idEx, idGt, ymaps].view(y.size(0), y.size(1), y.size(3)*y.size(4), y.size(3)*y.size(4)).mean(1)
    return (((c*p) + (1-c)*(1-p)).sum() - c.size(0)*c.size(1)) / (c.size(1)*(c.size(1)-1) * c.size(0))

def segmentation_covering(x, ymaps):
    ymaps = ymaps.long()
    sigma = x.new_zeros(x.size(1))
    y = x.new_zeros(ymaps.size(0), ymaps.size(1), ymaps.max()+1, ymaps.size(2), ymaps.size(3)).scatter(2, ymaps.unsqueeze(2), 1)
    xv = x.unsqueeze(1).unsqueeze(3)
    yv = y.unsqueeze(2)
    inter = (xv + yv == 2).sum((4,5)).float()
    union = (xv + yv >= 1).sum((4,5)).float()
    overlap = inter/union
    overlap.masked_fill_(torch.isnan(overlap), 0)
    return (xv.sum((3,4,5)) * overlap.max(3)[0]).sum(2).mean() / (xv.size(4)*xv.size(5))

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

testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=True, num_workers=4, drop_last=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batchSize, shuffle=True, num_workers=4, drop_last=True)

#############################################################################
# Train                                                                     #
#############################################################################
log_dPosX = []
log_dNegX = []
log_encM = []
log_recZ = []

genData = iter(trainloader)
disData = iter(trainloader)
if not opt.silent:
    pbar = tqdm(total=opt.checkpointFreq)
while opt.iteration <= opt.nIteration:
    if not opt.silent:
        pbar.update(1)
    #schedulerEncM.step()
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
    # netRecM.zero_grad()
    netGenX.zero_grad()
    netDX.zero_grad()
    netEncM.train()
    # netRecM.train()
    netGenX.train()
    netDX.train()
    if opt.wrecZ > 0:
        netRecZ.zero_grad()
        netRecZ.train()
    # schedulerEncM.step()
    dStep = (opt.iteration % opt.dStepFreq == 0)
    mStep = (opt.iteration % opt.mStepFreq == 0)
    gStep = (opt.iteration % opt.gStepFreq == 0)
    zStep = (opt.iteration % opt.zStepFreq == 0)
    #########################  AutoEncode X #########################
    if gStep or mStep or zStep:
        mEnc = netEncM(xData)
        hGen = netGenX(mEnc, zData)
        xGen = (hGen + ((1 - mEnc.unsqueeze(2)) * xData.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))
        dGen = netDX(xGen)
        err_dGen = - dGen.mean()
        # mean_m = mEnc.mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        # var_m = ((mEnc - mean_m) * (mEnc - mean_m)).mean()
        # dmean0 = mean_m[:,0].mean()
        # dmean0 = dmean0 - opt.maskmean0
        # dmean0 = (dmean0 * dmean0).sqrt()
        # dmean1 = mean_m[:,1].mean()
        # dmean1 = dmean1 - opt.maskmean1
        # dmean1 = (dmean1 * dmean1).sqrt()
        # dmean = dmean0 + dmean1
        lossG = err_dGen # - var_m * opt.wvar + dmean * opt.wmean
        if opt.wrecZ > 0:
            if zStep:
                zRec = netRecZ(hGen.sum(1))
                err_recZ = ((zData - zRec) * (zData - zRec)).mean()
                lossG += err_recZ * opt.wrecZ
                log_recZ.append(err_recZ.item())
                # mRec = netRecM()
            else:
                log_recZ.append(log_recZ[-1])
        lossG.backward()
        if mStep:
            optimizerEncM.step()
        if gStep:
            optimizerGenX.step()
        if opt.wrecZ > 0 and zStep:
            optimizerRecZ.step()
        with torch.no_grad():
            err_encM = None
            if opt.nMasks <= 2:
                err_encM = torch.max(((mEnc[:,:1] >= .5).float() == mData).float().mean(-1).mean(-1),
                                     ((mEnc[:,:1] <  .5).float() == mData).float().mean(-1).mean(-1)).mean().item()
            else:
                for p in itertools.permutations(range(opt.nMasks)):
                    current_errM = 0
                    for m in range(opt.nMasks):
                        current_errM += ((mEnc[:,m:m+1] >= .5).float() * mData[:,p[m]:p[m]+1]).float().mean(-1).mean(-1)
                    if err_encM is None:
                        err_encM = current_errM
                    else:
                        err_encM = torch.max(err_encM, current_errM)
                err_encM = err_encM.mean().item()
            log_encM.append(err_encM)
    else:
        log_encM.append(log_encM[-1])
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
        log_dPosX.append(dPosX.detach().mean().item())
        log_dNegX.append(dNegX.detach().mean().item())
    else:
        log_dPosX.append(log_dPosX[-1])
        log_dNegX.append(log_dNegX[-1])
    opt.iteration += 1
    if opt.iteration % opt.checkpointFreq == 0:
        if not opt.silent:
            pbar.close()
        try:
            with open(os.path.join(opt.outf, 'training.dat'), 'ab') as f:
                np.savetxt(f, np.vstack((
                    np.arange(opt.iteration - opt.checkpointFreq, opt.iteration),
                    np.array(log_dPosX),
                    np.array(log_dNegX),
                    np.array(log_encM),
                    np.array(log_recZ),
                )).T)
        except:
            pass
        if not opt.silent:
            print(opt.iteration,
                  np.array(log_dPosX).mean(),
                  np.array(log_dNegX).mean(),
                  np.array(log_encM).mean(),
                  np.array(log_recZ).mean(),
            )
        log_dPosX = []
        log_dNegX = []
        log_encM = []
        log_recZ = []
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
            # TRAINSET #
            sumScoreAcc_train = 0
            sumScoreIoU_train = 0
            nbIterTrain = 0
            for xLoad_train, mLoad_train in trainloader:
                xData_train = xLoad_train.to(device)
                mData_train = mLoad_train.to(device)
                mPred_train = netEncM(xData_train)
                if opt.nMasks <= 2:
                    sumScoreAcc_train += torch.max(((mPred_train[:,:1] >= .5).float() == mData_train).float().mean(-1).mean(-1),
                                                 ((mPred_train[:,:1] <  .5).float() == mData_train).float().mean(-1).mean(-1)).mean().item()
                    sumScoreIoU_train += torch.max(
                        ((((mPred_train[:,:1] >= .5).float() + mData_train) == 2).float().sum(-1).sum(-1) /
                         (((mPred_train[:,:1] >= .5).float() + mData_train) >= 1).float().sum(-1).sum(-1)),
                        ((((mPred_train[:,:1] <  .5).float() + mData_train) == 2).float().sum(-1).sum(-1) /
                         (((mPred_train[:,:1] <  .5).float() + mData_train) >= 1).float().sum(-1).sum(-1))).mean().item()
                else:
                    scoreAcc = None
                    scoreIoU = None
                    for p in itertools.permutations(range(opt.nMasks)):
                        current_scoreAcc = 0
                        current_scoreIoU = 0
                        for m in range(opt.nMasks):
                            current_scoreAcc += ((mPred_train[:,m:m+1] >= .5).float() * mData_train[:,p[m]:p[m]+1]).mean(-1).mean(-1)
                            current_scoreIoU += ((((mPred_train[:,m:m+1] >= .5).float() + mData_train[:,p[m]:p[m]+1]) == 2).float().sum(-1).sum(-1) /
                                                 (((mPred_train[:,m:m+1] >= .5).float() + mData_train[:,p[m]:p[m]+1]) >= 1).float().sum(-1).sum(-1))
                        if scoreAcc is None:
                            scoreAcc = current_scoreAcc
                            scoreIoU = current_scoreIoU
                        else:
                            scoreAcc = torch.max(scoreAcc, current_scoreAcc)
                            scoreIoU = torch.max(scoreIoU, current_scoreIoU)
                    sumScoreAcc_train += scoreAcc.mean().item()
                    sumScoreIoU_train += scoreIoU.mean().item()
                nbIterTrain += 1
            if not opt.silent:
                print("train:",
                      sumScoreAcc_train / nbIterTrain,
                      sumScoreIoU_train / nbIterTrain,)
            try:
                with open(os.path.join(opt.outf, 'train.dat'), 'a') as f:
                    f.write(str(opt.iteration) + ' ' + str(sumScoreAcc_train/nbIterTrain) + ' ' + str(sumScoreIoU_train/nbIterTrain) + '\n')
            except:
                print("Cannot save in train.dat")
            # VAL #
            sumScoreAcc_val = 0
            sumScoreIoU_val = 0
            nbIterVal = 0
            for xLoad_val, mLoad_val in valloader:
                xData_val = xLoad_val.to(device)
                mData_val = mLoad_val.to(device)
                mPred_val = netEncM(xData_val)
                if opt.nMasks <= 2:
                    sumScoreAcc_val += torch.max(((mPred_val[:,:1] >= .5).float() == mData_val).float().mean(-1).mean(-1),
                                                 ((mPred_val[:,:1] <  .5).float() == mData_val).float().mean(-1).mean(-1)).mean().item()
                    sumScoreIoU_val += torch.max(
                        ((((mPred_val[:,:1] >= .5).float() + mData_val) == 2).float().sum(-1).sum(-1) /
                         (((mPred_val[:,:1] >= .5).float() + mData_val) >= 1).float().sum(-1).sum(-1)),
                        ((((mPred_val[:,:1] <  .5).float() + mData_val) == 2).float().sum(-1).sum(-1) /
                         (((mPred_val[:,:1] <  .5).float() + mData_val) >= 1).float().sum(-1).sum(-1))).mean().item()
                else:
                    scoreAcc = None
                    scoreIoU = None
                    for p in itertools.permutations(range(opt.nMasks)):
                        current_scoreAcc = 0
                        current_scoreIoU = 0
                        for m in range(opt.nMasks):
                            current_scoreAcc += ((mPred_val[:,m:m+1] >= .5).float() * mData_val[:,p[m]:p[m]+1]).mean(-1).mean(-1)
                            current_scoreIoU += ((((mPred_val[:,m:m+1] >= .5).float() + mData_val[:,p[m]:p[m]+1]) == 2).float().sum(-1).sum(-1) /
                                                 (((mPred_val[:,m:m+1] >= .5).float() + mData_val[:,p[m]:p[m]+1]) >= 1).float().sum(-1).sum(-1))
                        if scoreAcc is None:
                            scoreAcc = current_scoreAcc
                            scoreIoU = current_scoreIoU
                        else:
                            scoreAcc = torch.max(scoreAcc, current_scoreAcc)
                            scoreIoU = torch.max(scoreIoU, current_scoreIoU)
                    sumScoreAcc_val += scoreAcc.mean().item()
                    sumScoreIoU_val += scoreIoU.mean().item()
                nbIterVal += 1
            if not opt.silent:
                print("val:",
                      sumScoreAcc_val / nbIterVal,
                      sumScoreIoU_val / nbIterVal,)
            try:
                with open(os.path.join(opt.outf, 'val.dat'), 'a') as f:
                    f.write(str(opt.iteration) + ' ' + str(sumScoreAcc_val/nbIterVal) + ' ' + str(sumScoreIoU_val/nbIterVal) + '\n')
            except:
                print("Cannot save in val.dat")
            ### Test
            sumScoreAcc_test = 0
            sumScoreIoU_test = 0
            nbIterTest = 0            
            for xLoad_test, mLoad_test in testloader:
                xData_test = xLoad_test.to(device)
                mData_test = mLoad_test.to(device)
                mPred_test = netEncM(xData_test)
                if opt.nMasks <= 2:
                    sumScoreAcc_test += torch.max(((mPred_test[:,:1] >= .5).float() == mData_test).float().mean(-1).mean(-1),
                                                  ((mPred_test[:,:1] <  .5).float() == mData_test).float().mean(-1).mean(-1)).mean().item()
                    sumScoreIoU_test += torch.max(
                        ((((mPred_test[:,:1] >= .5).float() + mData_test) == 2).float().sum(-1).sum(-1) /
                         (((mPred_test[:,:1] >= .5).float() + mData_test) >= 1).float().sum(-1).sum(-1)),
                        ((((mPred_test[:,:1] <  .5).float() + mData_test) == 2).float().sum(-1).sum(-1) /
                         (((mPred_test[:,:1] <  .5).float() + mData_test) >= 1).float().sum(-1).sum(-1))).mean().item()
                else:
                    scoreAcc = None
                    scoreIoU = None
                    for p in itertools.permutations(range(opt.nMasks)):
                        current_scoreAcc = 0
                        current_scoreIoU = 0
                        for m in range(opt.nMasks):
                            current_scoreAcc += ((mPred_test[:,m:m+1] >= .5).float() * mData_test[:,p[m]:p[m]+1]).mean(-1).mean(-1)
                            current_scoreIoU += ((((mPred_test[:,m:m+1] >= .5).float() + mData_test[:,p[m]:p[m]+1]) == 2).float().sum(-1).sum(-1) /
                                                 (((mPred_test[:,m:m+1] >= .5).float() + mData_test[:,p[m]:p[m]+1]) >= 1).float().sum(-1).sum(-1))
                        if scoreAcc is None:
                            scoreAcc = current_scoreAcc
                            scoreIoU = current_scoreIoU
                        else:
                            scoreAcc = torch.max(scoreAcc, current_scoreAcc)
                            scoreIoU = torch.max(scoreIoU, current_scoreIoU)
                    sumScoreAcc_test += scoreAcc.mean().item()
                    sumScoreIoU_test += scoreIoU.mean().item()
                nbIterTest += 1
            if not opt.silent:
                print("Test:",
                      sumScoreAcc_test / nbIterTest,
                      sumScoreIoU_test / nbIterTest,)
            try:
                with open(os.path.join(opt.outf, 'test.dat'), 'a') as f:
                    f.write(str(opt.iteration) + ' ' + str(sumScoreAcc_test/nbIterTest) + ' ' + str(sumScoreIoU_test/nbIterTest) + '\n')
            except:
                print("Cannot save in test.dat")
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
        if opt.bestValIoU < sumScoreIoU_val / nbIterVal:
            opt.bestValIoU = sumScoreIoU_val / nbIterVal
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



