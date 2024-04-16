import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, cfg):
        super(_netG, self).__init__()
        self.ngpu = cfg.ngpu
        # self.main = nn.Sequential(
        #     # input is (nc) x 128 x 128
        #     nn.Conv2d(cfg.nc,cfg.nef,4,2,1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size: (nef) x 64 x 64
        #     nn.Conv2d(cfg.nef,cfg.nef,4,2,1, bias=False),
        #     nn.BatchNorm2d(cfg.nef),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size: (nef) x 32 x 32
        #     nn.Conv2d(cfg.nef,cfg.nef*2,4,2,1, bias=False),
        #     nn.BatchNorm2d(cfg.nef*2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size: (nef*2) x 16 x 16
        #     nn.Conv2d(cfg.nef*2,cfg.nef*4,4,2,1, bias=False),
        #     nn.BatchNorm2d(cfg.nef*4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size: (nef*4) x 8 x 8
        #     nn.Conv2d(cfg.nef*4,cfg.nef*8,4,2,1, bias=False),
        #     nn.BatchNorm2d(cfg.nef*8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size: (nef*8) x 4 x 4
        #     nn.Conv2d(cfg.nef*8,cfg.nBottleneck,4, bias=False),
        #     # tate size: (nBottleneck) x 1 x 1
        #     nn.BatchNorm2d(cfg.nBottleneck),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # input is Bottleneck, going into a convolution
        #     nn.ConvTranspose2d(cfg.nBottleneck, cfg.ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(cfg.ngf * 8),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(cfg.ngf * 8, cfg.ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(cfg.ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d(cfg.ngf * 4, cfg.ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(cfg.ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.ConvTranspose2d(cfg.ngf * 2, cfg.ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(cfg.ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 32 x 32
        #     nn.ConvTranspose2d(cfg.ngf, cfg.nc, 4, 2, 1, bias=False),
        #     # add another conv layer to go to 128x128
        #     nn.Tanh()
        #     # state size. (nc) x 64 x 64
        # )
        layers = []
        layers.append(nn.Conv2d(cfg.nc, cfg.nef, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.nef, cfg.nef, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.nef))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.nef, cfg.nef * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.nef * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.nef * 2, cfg.nef * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.nef * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.nef * 4, cfg.nef * 8, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.nef * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.nef * 8, cfg.nBottleneck, 4, bias=False))
        layers.append(nn.BatchNorm2d(cfg.nBottleneck))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.ConvTranspose2d(cfg.nBottleneck, cfg.ngf * 8, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(cfg.ngf * 8))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(cfg.ngf * 8, cfg.ngf * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.ngf * 4))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(cfg.ngf * 4, cfg.ngf * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.ngf * 2))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(cfg.ngf * 2, cfg.ngf, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.ngf))
        layers.append(nn.ReLU(True))

        if cfg.masking == 'random-box' or cfg.masking == 'random-crop' or cfg.masking == 'stitch':
            layers.append(nn.ConvTranspose2d(cfg.ngf, cfg.ngf, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(cfg.ngf))
            layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(cfg.ngf, cfg.nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netlocalD(nn.Module):
    def __init__(self, cfg):
        super(_netlocalD, self).__init__()
        self.ngpu = cfg.ngpu
        # self.main = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv2d(cfg.nc, cfg.ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(cfg.ndf, cfg.ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(cfg.ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(cfg.ndf * 2, cfg.ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(cfg.ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(cfg.ndf * 4, cfg.ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(cfg.ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv2d(cfg.ndf * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )
        layers = []
        layers.append(nn.Conv2d(cfg.nc, cfg.ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if cfg.masking == 'random-box' or cfg.masking == 'random-crop' or cfg.masking == 'stitch':
            layers.append(nn.Conv2d(cfg.ndf, cfg.ndf, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(cfg.ndf)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.ndf, cfg.ndf * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.ndf * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.ndf * 2, cfg.ndf * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.ndf * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.ndf * 4, cfg.ndf * 8, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.ndf * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.ndf * 8, 1, 4, 1, 0, bias=False))

        if not cfg.WGAN:
            layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1)

class _netlocalD_WGANGP(nn.Module):
    def __init__(self, cfg):
        super(_netlocalD_WGANGP, self).__init__()
        self.ngpu = cfg.ngpu
        layers = []
        layers.append(nn.Conv2d(cfg.nc, cfg.ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if cfg.masking == 'random-box' or cfg.masking == 'random-crop' or cfg.masking == 'stitch':
            layers.append(nn.Conv2d(cfg.ndf, cfg.ndf, 4, 2, 1, bias=False))
            # layers.append(nn.BatchNorm2d(cfg.ndf))
            layers.append(nn.LayerNorm([cfg.ndf, 32, 32]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.ndf, cfg.ndf * 2, 4, 2, 1, bias=False))
        # layers.append(nn.BatchNorm2d(cfg.ndf * 2))
        layers.append(nn.LayerNorm([cfg.ndf * 2, 16, 16]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.ndf * 2, cfg.ndf * 4, 4, 2, 1, bias=False))
        # layers.append(nn.BatchNorm2d(cfg.ndf * 4))
        layers.append(nn.LayerNorm([cfg.ndf * 4, 8, 8]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.ndf * 4, cfg.ndf * 8, 4, 2, 1, bias=False))
        # layers.append(nn.BatchNorm2d(cfg.ndf * 8))
        layers.append(nn.LayerNorm([cfg.ndf * 8, 4, 4]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(cfg.ndf * 8, 1, 4, 1, 0, bias=False))

        if not cfg.WGAN:
            layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1)


