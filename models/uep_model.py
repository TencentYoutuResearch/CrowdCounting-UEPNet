import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from . import networks
from .networks import VGGNet

class DoubleHead(nn.Module):
    def __init__(self, num_classes):
        super(DoubleHead, self).__init__()
        self.PredBackbone1 = nn.Sequential(nn.Conv2d(512, 512, (1, 1)), nn.ReLU()) 
        self.PredBackbone2 = nn.Sequential(nn.Conv2d(512, 512, (1, 1)), nn.ReLU()) 
        self.head1 = nn.Conv2d(512, num_classes, (1, 1))
        self.head2 = nn.Conv2d(512, num_classes + 1, (1, 1))

    def forward(self, x):
        x1 = self.PredBackbone1(x)
        x2 = self.PredBackbone2(x)

        return self.head1(x1), self.head2(x2)

class UEPModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--with_bn', action='store_true', help='wether use batch_norm in the backbone network.')
        parser.add_argument('--freeze_bn', action='store_true', help='wether freeze weights of batch_norm while training.')
        parser.add_argument('--vgg_post_pool', action='store_true', help='wether extract features after pooling layer.')
        parser.add_argument('--heatmap_multi', default=1.0, type=float, help='the scale ratio for heatmap')
        parser.add_argument('--psize', default=8, type=int, help='the max patch size begin to divide')
        parser.add_argument('--pstride', default=8, type=int, help='the max downsample stride')
        parser.add_argument('--folder', default=1, type=int, help='the folder index')
        return parser


    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['cls', 'total']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['data_img', 'data_gt', 'pred_heatmap']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['VggBackbone', 'DivideUpsample0', 'DivideUpsample1', 'ClsPred']
  
        if self.opt.dataset_mode == 'shtechparta':
            self.min_density = 1.6e-4 
            self.max_density = 8.746914863586426
            num_split = 9
            bin_size = (self.max_density - self.min_density) / num_split
            self.label_indices = (np.array(list(range(num_split)))) * bin_size + self.min_density

            self.label_indices = np.array([0.00016, 0.0048202634789049625, 0.01209819596260786, 0.02164922095835209, 0.03357841819524765, 0.04810526967048645, 0.06570728123188019, 0.08683456480503082, 0.11207923293113708, 0.1422334909439087, 0.17838051915168762, 0.22167329490184784, 0.2732916474342346, 0.33556100726127625, 0.41080838441848755, 0.5030269622802734, 0.6174761652946472, 0.762194037437439, 0.9506691694259644, 1.2056223154067993, 1.5706151723861694, 2.138580322265625, 3.233219861984253, 7.914860725402832]) #24
            self.indices_proxy = np.array([0, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624, 0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833, 0.1263212381117904, 0.15925543689080027, 0.19863706203617743, 0.24597249461239232, 0.3025175130111165, 0.3707221162631514, 0.4537206813235279, 0.5560940547912038, 0.6838185522926952, 0.8476390438597705, 1.0642417040590761, 1.3645639664610938, 1.8055319029995607, 2.541316177212592, 3.87642023839676, 8.247815291086832]) #25
            self.label_indices2 = np.array([0.00016, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624, 0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833, 0.1263212381117904, 0.15925543689080027, 0.19863706203617743, 0.24597249461239232, 0.3025175130111165, 0.3707221162631514, 0.4537206813235279, 0.5560940547912038, 0.6838185522926952, 0.8476390438597705, 1.0642417040590761, 1.3645639664610938, 1.8055319029995607, 2.541316177212592, 3.87642023839676, 8.247815291086832]) #24
            self.indices_proxy2 = np.array([0, 0.0008736941759623788, 0.00460105649110827, 0.011909992029514994, 0.021447560775165905, 0.03335742127399603, 0.04785158393927123, 0.06538952954794941, 0.08647975537451662, 0.11168024780931907, 0.14175821026385504, 0.17778540202168958, 0.22097960677712483, 0.2724192081348686, 0.3344926685808885, 0.40938709885499597, 0.5012436541947841, 0.6149288298909453, 0.7585325340575756, 0.9452185066011628, 1.1967563985336944, 1.5541906336372862, 2.0969205546489382, 2.9970217618726727, 4.51882041862729, 8.527834415435791]) #25
            
        self.num_class = self.label_indices.size + 1
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netVggBackbone = VGGNet(self.opt.with_bn, self.opt.vgg_post_pool)

        self.netDivideUpsample0 = Upsample(512, 256, 256 + 512, 512)
        self.netDivideUpsample1 = Upsample(512, 256, 256 + 256, 512)

        self.netClsPred = DoubleHead(self.num_class)
 
        self.netVggBackbone = networks.init_net(self.netVggBackbone, gpu_ids=self.gpu_ids, do_init=False)
        self.netDivideUpsample0 = networks.init_net(self.netDivideUpsample0, init_type='normal', init_gain=0.01, gpu_ids=self.gpu_ids)
        self.netDivideUpsample1 = networks.init_net(self.netDivideUpsample1, init_type='normal', init_gain=0.01, gpu_ids=self.gpu_ids)
        self.netClsPred = networks.init_net(self.netClsPred, init_type='normal', init_gain=0.01, gpu_ids=self.gpu_ids)
 
        if self.isTrain:  # only defined during training time
            if self.opt.freeze_bn: 
                self.netVggBackbone.freeze_bn()
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(self.device) 
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer = torch.optim.SGD(itertools.chain(self.netVggBackbone.parameters(), \
                                                            self.netDivideUpsample0.parameters(), \
                                                            self.netDivideUpsample1.parameters(), \
                                                            self.netClsPred.parameters()), \
                                                                lr=opt.lr)
            self.optimizers = [self.optimizer]

    def get_label_indices(self):
        return [0.] + self.label_indices.tolist() + [self.max_density]

    def to_eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def to_train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
        if self.opt.freeze_bn: 
            self.netVggBackbone.freeze_bn()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.data_img = input['img'].to(self.device) 
        self.data_gt = input['gt'].to(self.device) if 'gt' in input else None

    def get_gt_label(self, target):
        target = torch.from_numpy(target[np.newaxis, np.newaxis, :, :].copy())
        gt_densitymap = target * self.opt.heatmap_multi
        gt_count = get_local_count(gt_densitymap, self.opt.psize, self.opt.pstride)
        return count_to_class(gt_count, self.label_indices).squeeze_(dim=1), gt_count

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        conv1, conv2, conv3, conv4, conv5 = self.netVggBackbone(self.data_img)
        feat0 = conv5
        feat1 = self.netDivideUpsample0(feat0, conv4)
        feat2 = self.netDivideUpsample1(feat1, conv3)

        self.cls, self.cls2 = self.netClsPred(feat2)

        self.count1 = class_to_count(self.cls.max(dim=1, keepdim=True)[1], self.indices_proxy)
        self.count2 = class_to_count(self.cls2.max(dim=1, keepdim=True)[1], self.indices_proxy2)

        self.count = (self.count1 + self.count2) / 2.

        self.pred_heatmap = self.count / self.opt.heatmap_multi

    def predict(self, return_sum=True):
        self.forward()
        if return_sum:
            if self.pred_heatmap.device.type == 'cuda':
                return self.pred_heatmap.data.cpu().numpy().sum()
            return self.pred_heatmap.data.numpy().sum()
        else:
            if self.pred_heatmap.device.type == 'cuda':
                return self.pred_heatmap.data.cpu().numpy()
            return self.pred_heatmap.data.numpy()

    def predict_heatmap(self):
        self.forward()
        if self.pred_heatmap.device.type == 'cuda':
            return self.pred_heatmap.data.cpu().numpy(), self.cls.cpu()
        return self.pred_heatmap.data.numpy(), self.cls

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        pass

class Upsample(nn.Module):
    def __init__(self, up_in_ch, up_out_ch, cat_in_ch, cat_out_ch):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(*[nn.Conv2d(up_in_ch, up_out_ch, 3, 1, padding=1), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv2d(cat_in_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU(), \
                                    nn.Conv2d(cat_out_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU()])
        
    def forward(self, low, high):
        low = self.up(low)
        low = self.conv1(low)
            
        x = torch.cat([high, low], dim=1)

        x = self.conv2(x)
        return x

def get_local_count(density_map, psize, pstride):
    on_gpu = (density_map.device.type == 'cuda')
    psize, pstride = int(psize), int(pstride)
    density_map = density_map.cpu().type(torch.float32)
    conv_kernel = torch.ones(1, 1, psize, psize, dtype = torch.float32)
    if torch.cuda.is_available():
        density_map, conv_kernel = density_map.cuda(), conv_kernel.cuda()
    
    count_map = F.conv2d(density_map, conv_kernel, stride=pstride)
    if not on_gpu: 
        count_map = count_map.cpu()

    return count_map

def count_to_class(count_map, label_indice):
    if isinstance(label_indice, np.ndarray):
        label_indice = torch.from_numpy(label_indice) 
    
    on_gpu = (count_map.device.type == 'cuda')        
    label_indice = label_indice.cpu().type(torch.float32)
    cls_num = len(label_indice) + 1
    cls_map = torch.zeros(count_map.size()).type(torch.LongTensor) 
    count_map = count_map.type(torch.float32)
    if torch.cuda.is_available():
        count_map, label_indice, cls_map = count_map.cuda(), label_indice.cuda(), cls_map.cuda()
    
    for i in range(0, cls_num - 1):
        if torch.cuda.is_available():
            cls_map = cls_map + (count_map >= label_indice[i]).cpu().type(torch.LongTensor).cuda()
        else:
            cls_map = cls_map + (count_map >= label_indice[i]).cpu().type(torch.LongTensor)
    if not on_gpu: 
        cls_map = cls_map.cpu() 
    return cls_map

def class_to_count(pre_cls, indices_proxy):
    on_gpu = (pre_cls.device.type == 'cuda')  
    
    label2count = torch.tensor(indices_proxy)
    label2count = label2count.type(torch.FloatTensor)

    input_size = pre_cls.size()
    pre_cls = pre_cls.reshape(-1).cpu()

    pre_counts = torch.index_select(label2count, 0, pre_cls.cpu().type(torch.LongTensor))
    pre_counts = pre_counts.reshape(input_size)

    if on_gpu: 
        pre_counts = pre_counts.cuda()

    return pre_counts

if __name__ == "__main__":
    pass