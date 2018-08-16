# -*- coding:utf-8 -*-
"""
this is the entrance of this project
"""

import os
import sys
from model import model
import tools

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# use command input specify GPU id
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

if __name__ == '__main__':
    batch_size = 32
    learning_rate = 0.01
    keep_prob = 1
    epoch = 100
    # path = '/data0/LUNA/cubic_normalization_npy'
    path = tools.npy_path

    # test_path = '../../data/cubic_normalization_test'
    test_size = 0.1
    seed = 121

    print(" begin...")
    model = model(learning_rate, keep_prob, batch_size, epoch)
    model.inference(path, 0, test_size, seed, True)
