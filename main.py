# -*- coding:utf-8 -*-
"""
this is the entrance of this project
"""

import os
import sys
from model import model

'''
Usage: python3 main.py [-g gpu_id] [-p base_dir] [-t]
eg.
$ python3 main.py -g 0  # 指定使用第一块GPU
$ python3 main.py -g 2 -p truncate_400  # 指定使用第三块GPU，文件目录为truncate_400，训练模型
$ python3 main.py -g 2 -p truncate_400 -t 80  # 指定使用第三块GPU，文件目录为truncate_400，使用ckpt-80测试模型
'''

gpu = '3'
base_path = ''
train_flag = True
test_iterator = 80
for i in range(len(sys.argv)):
    if '-g' == sys.argv[i]:
        gpu = sys.argv[i+1]
    elif '-p' == sys.argv[i]:
        base_path = sys.argv[i+1]
    elif '-t' == sys.argv[i]:
        train_flag = False
        test_iterator = int(sys.argv[i+1])

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

if __name__ == '__main__':
    batch_size = 32
    learning_rate = 0.01
    keep_prob = 0.7
    epoch = 80
    # path = '/data0/LUNA/cubic_normalization_npy'

    # test_path = '../../data/cubic_normalization_test'
    test_size = 0.1
    seed = 121

    print(" begin...")
    model = model(learning_rate, keep_prob, batch_size, epoch)
    model.inference(base_path, 0, test_size, seed, train_flag, test_iterator)
