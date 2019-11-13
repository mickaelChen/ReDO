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

import datasets
import models

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
parser.add_argument('--nf', type=int, default=64, help='base number of filters for conv nets')
parser.add_argument('--nfD', type=int, default=None, help='specific nf for netD, default to nf')
parser.add_argument('--nfX', type=int, default=None, help='specific nf for netX, default to nf')
parser.add_argument('--nfM', type=int, default=None, help='specific nf for netM, default to nf')
parser.add_argument('--nfZ', type=int, default=None, help='specific nf for netZ, default to nf')
parser.add_argument('--useSelfAttD', action='store_true', help='use self attention for D')
parser.add_argument('--useSelfAttG', action='store_true', help='use self attention for G')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--nTest', type=int, default=5, help='input batch size for visu')
parser.add_argument('--nIteration', type=int, default=5e4, help='number of iterations')
parser.add_argument('--initOrthoGain', type=float, default=.8, help='gain for the initialization')
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
parser.add_argument('--autoRestart', type=float, default=0, help='restart training if the ratio "size of region" / "size of image" is stricly smaller than x (collapse detected)')

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
    
    
if opt.dataset == 'lfw':
    trainset = datasets.LFWDataset(dataPath=opt.dataroot,
                                   sets='train',
                                   transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                                 transforms.CenterCrop(opt.sizex),
                                                                 transforms.ToTensor(),
                                   ]),)
    testset = datasets.LFWDataset(dataPath=opt.dataroot,
                                  sets='test',
                                  transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                                transforms.CenterCrop(opt.sizex),
                                                                transforms.ToTensor(),
                                  ]),)
    valset = datasets.LFWDataset(dataPath=opt.dataroot,
                                 sets='val',
                                 transform=transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                               transforms.CenterCrop(opt.sizex),
                                                               transforms.ToTensor(),
                                 ]),)
if opt.dataset == 'cub':
    trainset = datasets.CUBDataset(opt.dataroot,
                                   "train",
                                   transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                       transforms.CenterCrop(opt.sizex),
                                                       transforms.ToTensor(),
                                   ]))
    testset = datasets.CUBDataset(opt.dataroot,
                                  "test",
                                  transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                      transforms.CenterCrop(opt.sizex),
                                                      transforms.ToTensor(),
                                  ]))
    valset = datasets.CUBDataset(opt.dataroot,
                                 "val",
                                 transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                     transforms.CenterCrop(opt.sizex),
                                                     transforms.ToTensor(),
                                 ]))
if opt.dataset == 'flowers':
    trainset = datasets.FlowersDataset(opt.dataroot,
                                       "train",
                                       transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                           transforms.CenterCrop(opt.sizex),
                                                           transforms.ToTensor(),
                                       ]))
    testset = datasets.FlowersDataset(opt.dataroot,
                                      "test",
                                      transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                          transforms.CenterCrop(opt.sizex),
                                                          transforms.ToTensor(),
                                      ]))
    valset = datasets.FlowersDataset(opt.dataroot,
                                     "val",
                                     transforms.Compose([transforms.Resize(opt.sizex, Image.NEAREST),
                                                         transforms.CenterCrop(opt.sizex),
                                                         transforms.ToTensor(),
                                     ]))
if opt.dataset == 'cmnist':
    trainset = datasets.CMNISTDataset(dataPath=opt.dataroot,
                                      sets='train')
    testset = datasets.CMNISTDataset(dataPath=opt.dataroot,
                                     sets='test')
    valset = datasets.CMNISTDataset(dataPath=opt.dataroot,
                                    sets='val')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=4, drop_last=True)
       
def weights_init_ortho(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, opt.initOrthoGain)


netEncM = models._netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(device)
netGenX = models._netGenX(sizex=opt.sizex, nOut=opt.nx, nc=opt.nz, nf=opt.nfX, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG).to(device)
netDX = models._resDiscriminator128(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(device)
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
    netRecZ = models._netRecZ(sizex=opt.sizex, nIn=opt.nx, nc=opt.nz, nf=opt.nfZ, nMasks=opt.nMasks).to(device)
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
    minRegionSize = min((mPred[:,:1] >= .5).float().mean().item(), (mPred[:,:1] < .5).float().mean().item())
    return sumScoreAcc / nbIter, sumScoreIoU / nbIter, minRegionSize

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
    '''
    Instead of sampling a region at each iteration, fake images for all regions are computed at each iteration.
    This allow to build an entirely generated image we can feed to the information conservation network instead of partially redrawn images.
    '''
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
        if opt.wrecZ > 0:
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
            scoreAccTrain, scoreIoUTrain, minRegionSizeTrain = evaluate(netEncM, trainloader, device, opt.nMasks)
            scoreAccVal, scoreIoUVal, minRegionSizeVal = evaluate(netEncM, valloader, device, opt.nMasks)
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
        if opt.wrecZ > 0:
            netRecZ.zero_grad()
        if minRegionSizeTrain < opt.autoRestart and minRegionSizeVal < opt.autoRestart:
            print("Training appear to have collapsed.")
            if opt.iteration <= 7000:
                print("Reinitializing training.")
                netEncM = models._netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(device)
                netGenX = models._netGenX(sizex=opt.sizex, nOut=opt.nx, nc=opt.nz, nf=opt.nfX, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG).to(device)
                netDX = models._resDiscriminator128(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(device)
                netEncM.apply(weights_init_ortho)
                netGenX.apply(weights_init_ortho)
                netDX.apply(weights_init_ortho)
                optimizerEncM = torch.optim.Adam(netEncM.parameters(), lr=opt.lrM, betas=(0, 0.9), weight_decay=opt.wdecay, amsgrad=False)
                optimizerGenX = torch.optim.Adam(netGenX.parameters(), lr=opt.lrG, betas=(0, 0.9), amsgrad=False)
                optimizerDX = torch.optim.Adam(netDX.parameters(), lr=opt.lrD, betas=(0, 0.9), amsgrad=False)
                if opt.wrecZ > 0:
                    netRecZ.apply(weights_init_ortho)
                    optimizerRecZ = torch.optim.Adam(netRecZ.parameters(), lr=opt.lrZ, betas=(0, 0.9), amsgrad=False)
                if not opt.silent:
                    pbar = tqdm(total=opt.checkpointFreq)
                opt.iteration = 0
                stateDic = {}
                continue
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
            stateDic['netRecZ'] = netRecZ.state_dict()
            stateDic['optimizerRecZ'] = optimizerRecZ.state_dict()
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
