from __future__ import print_function
import os
import sys
import torch
import numpy as np
from PIL import Image


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))
            
class AvgMeter(object):
    def __init__(self, max_avg_count=100):
        self.summary = 0
        self.avg_value = 0
        self.cnt = 0
        self.value_list = []
        self.max_avg_count = max_avg_count

    def __add__(self, other):
        if self.cnt == self.max_avg_count:
            self.summary -= self.value_list.pop(0)
            self.cnt -= 1
        
        self.value_list.append(other)
        self.summary += other
        self.cnt += 1
        self.avg_value = self.summary / self.cnt

        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __iaddr__(self, other):
        return self.__add__(other)