import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision import models

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            #init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=(), do_init=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if do_init: 
        init_weights(net, init_type, init_gain=init_gain)
    return net

def vgg_make_layers(cfg, in_channels = 3, batch_norm = False, dilation = False):
    if dilation: 
        d_rate = 2
    else: 
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = d_rate, dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)   

class VGGNet(nn.Module):
    def __init__(self, batch_norm=False, post_pool=False):
        super(VGGNet, self).__init__()
        # frontend
        if not post_pool:
            self.frontend_1_feat = [64, 64]
            self.frontend_2_feat = ['M', 128, 128]
            self.frontend_3_feat = ['M', 256, 256, 256]
            self.frontend_4_feat = ['M', 512, 512, 512]
            self.frontend_5_feat = ['M', 512, 512, 512]
        else:
            self.frontend_1_feat = [64, 64, 'M']
            self.frontend_2_feat = [128, 128, 'M']
            self.frontend_3_feat = [256, 256, 256, 'M']
            self.frontend_4_feat = [512, 512, 512, 'M']
            self.frontend_5_feat = [512, 512, 512, 'M']
        self.frontend_1 = vgg_make_layers(self.frontend_1_feat, in_channels = 3, batch_norm = batch_norm)
        self.frontend_2 = vgg_make_layers(self.frontend_2_feat, in_channels = 64, batch_norm = batch_norm)
        self.frontend_3 = vgg_make_layers(self.frontend_3_feat, in_channels = 128, batch_norm = batch_norm)
        self.frontend_4 = vgg_make_layers(self.frontend_4_feat, in_channels = 256, batch_norm = batch_norm)
        self.frontend_5 = vgg_make_layers(self.frontend_5_feat, in_channels = 512, batch_norm = batch_norm)

        self.batch_norm = batch_norm

        mod = models.vgg16_bn(pretrained = True) if self.batch_norm else models.vgg16(pretrained = True)
        mod_params = list(mod.state_dict().items())
        frontend_params = list(self.frontend_1.state_dict().items()) +\
                        list(self.frontend_2.state_dict().items()) + \
                        list(self.frontend_3.state_dict().items()) + \
                        list(self.frontend_4.state_dict().items()) + \
                        list(self.frontend_5.state_dict().items())

        for i in range(len(frontend_params)):
            if 'num_batches_tracked' in frontend_params[i][0]: 
                continue
            frontend_params[i][1].data[:] = mod_params[i][1].data[:]

    def forward(self,x):
        e1 = self.frontend_1(x)
        e2 = self.frontend_2(e1)
        e3 = self.frontend_3(e2)
        e4 = self.frontend_4(e3)
        e5 = self.frontend_5(e4)
       
        return e1, e2, e3, e4, e5

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d): 
                m.eval()
