import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

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
