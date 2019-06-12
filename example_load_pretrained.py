import argparse
from PIL import Image

import torch
import torchvision

import models
import datasets

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=40, help='number of images')
parser.add_argument('--device', default='cpu', help='cpu | cuda:n, device to be loaded on')
parser.add_argument('--statePath', default=None, help='path to pretrained weights')
parser.add_argument('--statePathM', default=None, help='path to pretrained weights for mask predictor')
parser.add_argument('--statePathX', default=None, help='path to pretrained weights for region generator')
parser.add_argument('--statePathZ', default=None, help='path to pretrained weights for noise reconstruction')
parser.add_argument('--statePathD', default=None, help='path to pretrained weights for discriminator')
parser.add_argument('--dataroot', default=None, help='path to data')

load_options = parser.parse_args()

device = torch.device(load_options.device)

if not load_options.statePath is None:
    states = torch.load(load_options.statePath, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    if "netEncM" in states:
        netEncM = models._netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(device)
        netEncM.load_state_dict(states["netEncM"])
        netEncM.eval()
    if "netGenX" in states:
        netGenX = models._netGenX(sizex=opt.sizex, nOut=opt.nx, nc=opt.nz, nf=opt.nfX, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG).to(device)
        netGenX.load_state_dict(states["netGenX"])
        netGenX.eval()
    if "netRecZ" in states:
        netRecZ = models._netRecZ(sizex=opt.sizex, nIn=opt.nx, nc=opt.nz, nf=opt.nfZ, nMasks=opt.nMasks).to(device)
        netRecZ.load_state_dict(states["netRecZ"])
        netRecZ.eval()
    if "netDX" in states:
        netDX = models._resDiscriminator128(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(device)
        netDX.load_state_dict(states["netDX"])
        netDX.eval()

if not load_options.statePathM is None:
    states = torch.load(load_options.statePathM, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    netEncM = models._netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(device)
    netEncM.load_state_dict(states["netEncM"])
    netEncM.eval()
    
if not load_options.statePathX is None:
    states = torch.load(load_options.statePathX, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    netGenX = models._netGenX(sizex=opt.sizex, nOut=opt.nx, nc=opt.nz, nf=opt.nfX, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG).to(device)
    netGenX.load_state_dict(states["netGenX"])
    netGenX.eval()
    
if not load_options.statePathZ is None:
    states = torch.load(load_options.statePathZ, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    netRecZ = models._netRecZ(sizex=opt.sizex, nIn=opt.nx, nc=opt.nz, nf=opt.nfZ, nMasks=opt.nMasks).to(device)
    netRecZ.load_state_dict(states["netRecZ"])
    netRecZ.eval()

if not load_options.statePathD is None:
    states = torch.load(load_options.statePathD, map_location={'cuda:0' : load_options.device})
    opt = states['options']
    netDX = models._resDiscriminator128(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(device)
    netDX.load_state_dict(states['netDX'])
    netDX.eval()

if opt.dataset == "lfw":
    dataset = datasets.LFWDataset(dataPath=load_options.dataroot,
                                  sets='test',
                                  transform=torchvision.transforms.Compose([torchvision.transforms.Resize(opt.sizex, Image.NEAREST),
                                                                            torchvision.transforms.CenterCrop(opt.sizex),
                                                                            torchvision.transforms.ToTensor(),
                                  ]),)
if opt.dataset == 'cub':
    dataset = datasets.CUBDataset(load_options.dataroot,
                                  "train",
                                  torchvision.transforms.Compose([torchvision.transforms.Resize(opt.sizex, Image.NEAREST),
                                                                  torchvision.transforms.CenterCrop(opt.sizex),
                                                                  torchvision.transforms.ToTensor(),
                                  ]))
if opt.dataset == 'flowers':
    dataset = datasets.FlowersDataset(load_options.dataroot,
                                      "train",
                                      torchvision.transforms.Compose([torchvision.transforms.Resize(opt.sizex, Image.NEAREST),
                                                                      torchvision.transforms.CenterCrop(opt.sizex),
                                                                      torchvision.transforms.ToTensor(),
                                      ]))
if opt.dataset == 'cmnist':
    dataset = datasets.CMNISTDataset(dataPath=load_options.dataroot,
                                     sets='train')

loader = torch.utils.data.DataLoader(dataset, batch_size=load_options.batch_size, shuffle=True)
xData, mData = next(iter(loader))
xData = xData.to(device)
mData = mData.to(device)

## Use the same z for all images in batch: ##
# z = torch.randn(1, opt.nMasks, opt.nz, 1, 1).repeat(batch_size, 1, 1, 1, 1).to(device)

## or use different z: ##
z = torch.randn(load_options.batch_size, opt.nMasks, opt.nz, 1, 1).to(device)

with torch.no_grad():
    # Using the mask predictor:
    mPred = netEncM(xData)

    # Redrawing using soft predictred masks:   
    xGen = netGenX(mPred, z) + (xData.unsqueeze(1) * (1-mPred.unsqueeze(2)))

    # or using binarized predictred masks:
    # xGen = netGenX((mPred >= .5).float(), z) + (xData.unsqueeze(1) * (mPred < .5).float().unsqueeze(2))

    # or using ground truth masks:
    # xGen = netGenX(torch.cat((mData, 1-mData),1), z) + (xData.unsqueeze(1) * torch.cat((1-mData, mData),1).unsqueeze(2))

    # Saving the images:
    out = torch.cat((xData*.5+.5,
                     mData.expand_as(xData),
                     mPred[:,0:1].expand_as(xData),
                     (mPred[:,1:2] >= .5).float().expand_as(xData),
                     xGen[:,0] *.5+.5,
                     xGen[:,1]*.5+.5),
                    1)
    torchvision.utils.save_image(out.view(-1,3,128,128), 'out.png', normalize=True, range=(0,1), nrow=6)
