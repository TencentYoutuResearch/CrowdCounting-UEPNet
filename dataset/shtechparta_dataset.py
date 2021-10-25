import os.path
from dataset.base_dataset import BaseDataset
from dataset.image_folder import make_dataset
from PIL import Image
import cv2
import random
import math
import os
from datetime import datetime
import numpy as np
from numpy import inf

from scipy.signal import gaussian
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utility.utils import mkdir_if_missing

class SHTechPartADataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.img_map = {}
        self.img_list = []
        self.gaussian_kernel_size = opt.gaussian_kernel_size
        for _, train_list in enumerate(opt.train_lists.split(',')):
            train_list = train_list.strip()
            with open(os.path.join(opt.dataroot, train_list)) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(opt.dataroot, line[0].strip())] = os.path.join(opt.dataroot, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--limit_shorter_side', default=256, type=int, help='the minimum size of shorter side after resize')
        parser.add_argument('--no_resize', action='store_true', help='if specified, apply no resize')
        return parser

    def get_ada_density_map_gaussian(self, img_shape, points):
        near_nn = 3
        leafsize = 2048
        img_density = np.zeros(img_shape, dtype=np.float32)
        gt_count = len(points)
        if gt_count == 0:
            return img_density

        points_arr = np.array(points)
        X_tree = KDTree(points_arr, leafsize=leafsize)
        nn_dis, _ = X_tree.query(points_arr, k=(near_nn + 1), p=2)
        nn_dis[nn_dis == inf] = 0
        
        avg_dis = np.sum(nn_dis, axis=1) / (np.count_nonzero(nn_dis, axis=1) + 1e-8)

        for ind, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                pt2d[int(pt[1]), int(pt[0])] = 1.
            else:
                continue
            if gt_count > 1:
                sigma = avg_dis[ind] * 0.3
            else:
                sigma = 0.1 * min(img_shape[0], img_shape[1]) #case: 1 point
            img_density += gaussian_filter(pt2d, sigma, mode='constant')
        return img_density

    def __getitem__(self, idx):
        while True:
            img = Image.open(self.img_list[idx])   
            if img is None:
                idx = random.randrange(0, len(self.img_list))
                continue
            img = img.convert('RGB')# Open and ensure in RGB mode - in case image is palettised
            
            img_width, img_height = img.size
            interp_mode = [
                Image.BILINEAR, Image.HAMMING, Image.NEAREST, Image.BICUBIC,
                Image.LANCZOS
            ]
            interp_indx = np.random.randint(0, 5)
            if self.opt.no_resize:
                resize_rate = 1.
            else:
                resize_rate = random.choice([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

            resize_h, resize_w = int(math.ceil(resize_rate * img_height)), int(math.ceil(resize_rate * img_width))
            if resize_h > resize_w:
                if resize_w < self.opt.limit_shorter_side:
                    resize_rate = self.opt.limit_shorter_side / img_width
                    resize_h, resize_w = int(math.ceil(resize_rate * img_height)), int(math.ceil(resize_rate * img_width))
            else:
                if resize_h < self.opt.limit_shorter_side:
                    resize_rate = self.opt.limit_shorter_side / img_height
                    resize_h, resize_w = int(math.ceil(resize_rate * img_height)), int(math.ceil(resize_rate * img_width))
    
            img = img.resize((resize_w, resize_h), resample=interp_mode[interp_indx])

            gt_list = self.img_map[self.img_list[idx]]
            points = []
            with open(gt_list) as f_label:
                for line in f_label:
                    x = float(line.strip().split(' ')[0]) * resize_rate
                    y = float(line.strip().split(' ')[1]) * resize_rate
                    points.append([x, y])
            
            target = self.get_ada_density_map_gaussian((img.size[1], img.size[0]), points)
      
            crop_ratio = self.opt.crop_ratio
            if crop_ratio > 0:
                img_width, img_height = img.size
                crop_size = (min(max(int(img_width * crop_ratio), 128), img_width) // 64 * 64, min(max(int(img_height * crop_ratio), 128), img_height) // 64 * 64)

                crop_patch_list = [ (0, 0),
                                    (img_width - crop_size[0], 0),
                                    (0, img_height - crop_size[1]),
                                    (img_width - crop_size[0], img_height - crop_size[1]),
                                    (random.randint(0, img_width - crop_size[0]), random.randint(0, img_height - crop_size[1])),
                                    (random.randint(0, img_width - crop_size[0]), random.randint(0, img_height - crop_size[1])),
                                    (random.randint(0, img_width - crop_size[0]), random.randint(0, img_height - crop_size[1])),
                                    (random.randint(0, img_width - crop_size[0]), random.randint(0, img_height - crop_size[1])),
                                    (random.randint(0, img_width - crop_size[0]), random.randint(0, img_height - crop_size[1]))] 
                dx, dy = random.choice(crop_patch_list)

                img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
                target = target[dy:(crop_size[1] + dy), dx:(crop_size[0] + dx)]

            img_width, _ = img.size
            if random.random() > 0.5:
                target = np.fliplr(target)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            break

        img = T.ToTensor()(img)
        img = img - torch.Tensor([0.4501, 0.4454, 0.4301]).view(3,1,1)           

        return {'img': img, 'gt': torch.from_numpy(target[np.newaxis, :, :].copy())}

    def get_preprocess_func_for_test(self):
        def preproceess_func(image_path):
            img = Image.open(image_path)
            img = img.convert('RGB')# Open and ensure in RGB mode - in case image is palettised
            img_w, img_h = img.size
            
            img = img.resize((img_w // 64 * 64, img_h // 64 * 64), resample=Image.BILINEAR)

            img = T.ToTensor()(img)
            img = img - torch.Tensor([0.4501, 0.4454, 0.4301]).view(3,1,1)           
            img = torch.unsqueeze(img, 0)

            return {'img': img}
        return preproceess_func

    def __len__(self):
        return len(self.img_list)

    def get_test_imgs(self, img_list):
        file_list = []
        file_map = {}
        with open(img_list, 'rt') as f_list:
            for line in f_list:
                file_path = os.path.join(self.opt.dataroot, line.strip().split()[0]).strip()
                file_list.append(file_path)
                file_map[file_path] = os.path.join(self.opt.dataroot, line.strip().split()[1]).strip()

        final_img_list = {}
        for image_path in file_list:
            try:
                img_show = Image.open(image_path)
            except OSError:
                continue
            if img_show is None:
                continue
            if img_show.size[0] == 0 or img_show.size[1]== 0:
                continue
            
            txtr = file_map[image_path]
            if os.path.exists(txtr):
                final_img_list[image_path] = file_map[image_path]

        return final_img_list