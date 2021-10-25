import time
import os
import numpy as np
from options.train_options import TrainOptions
from dataset import create_dataset
from models import create_model
from utility.visualizer import Visualizer
from utility.utils import mkdir_if_missing

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    model.load_networks('best')
    model.to_eval()
    if opt.test_lists:
        if len(opt.test_lists.strip().split(',')) > 0:
            process_func = dataset.get_preprocess_func_for_test()

            for test_list in opt.test_lists.strip().split(','):
                eval_img_list = dataset.get_test_imgs(test_list)
                all_imgs = list(eval_img_list.keys())
                pred_gt_array = np.zeros((len(all_imgs), 2))
                for img_ind, img_path in enumerate(all_imgs):
                    model.set_input(process_func(img_path))
                    pred_gt_array[img_ind, 0] = model.predict()
        
                    txtr = img_path[:-3] + 'txt'
                    txtfile = open(txtr)
                    gts = txtfile.readlines()
                    gt = len(gts)
                    txtfile.close()
                    pred_gt_array[img_ind, 1] = gt

                mae = np.mean(np.abs(pred_gt_array[:, 0] - pred_gt_array[:, 1]))
                mse = np.sqrt(np.mean(np.power(pred_gt_array[:, 0] - pred_gt_array[:, 1], 2)))
                print("Eval results for {}: MAE: {}, MSE: {}.".format(test_list, mae, mse))