'''
Created by Wang Qiu Li

7/4/2018

prepare data for malignancy model
'''

import os
from sklearn.model_selection import train_test_split
import numpy as np
import tools
import random

nodules = tools.get_nodules()
'''
id = 0
malignancy level = 29
'''

def get_train_and_test_filename(path,test_size,seed):
    all_file_list = os.listdir(path)
    origin_file_list = [f for f in all_file_list if 'high.npy' in f or 'low.npy' in f]
    random.seed(seed)
    origin_train_files, origin_test_files = train_test_split(origin_file_list, test_size=test_size, random_state=seed)
    # 经过旋转得到的新数据，必须和原数据位于同一集合
    for f in all_file_list:
        if 'high.npy' not in f and 'low.npy' not in f:
            origin_of_f = f[0:f.rindex('_')] + '.npy'
            if origin_of_f in origin_train_files:
                origin_train_files.append(f)
            # elif origin_of_f in origin_test_files:
            #     origin_test_files.append(f)
    random.shuffle(origin_train_files)
    random.shuffle(origin_test_files)
    return origin_train_files, origin_test_files

def get_high_data(path):
    filelist = os.listdir(path)
    returnlist = []
    for onefile in filelist:
        if 'low' in onefile:
            returnlist.append(onefile)
    return returnlist

def get_batch_withlabels_high(basedir, batch_filename):
    '''
    get batch
    return data and label
    '''
    batch_array = []
    batch_label = []

    sphercity = []
    margin = []
    lobulation =[]
    spiculation = []

    temp_filename = []

    for one in batch_filename:
        if 'high' in one:
            temp_filename.append(one)

    for onefile in temp_filename:
        try:
            # print(onefile)
            index = onefile[:onefile.find('_')]
            # print(index)
            chara_list = []
            for nodule in nodules:
                if nodule.id == index:
                    sphercity.append([nodule.sphercity, nodule.sphercity])
                    margin.append([nodule.margin, nodule.margin])
                    lobulation.append([nodule.lobulation, nodule.lobulation])
                    spiculation.append([nodule.spiculation, nodule.spiculation])
            arr = np.load(basedir + onefile)
            batch_array.append(arr)
            # id = re.findall("\d+",onefile)[0]
            if 'high' in onefile:
                batch_label.append([0, 1])
            elif 'low' in onefile:
                batch_label.append([1, 0])

        except Exception as e:
            print("file not exists! %s"%onefile)
            batch_array.append(batch_array[-1])

    return np.array(batch_array), np.array(sphercity), np.array(margin), np.array(lobulation), np.array(spiculation), np.array(batch_label)


def get_batch_withlabels(basedir, batch_filename):
    '''
    get batch
    return data and label
    '''
    batch_array = []
    batch_label = []
    batch_path = []

    sphercity = []
    margin = []
    lobulation =[]
    spiculation = []

    for onefile in batch_filename:
        try:
            # print(onefile)
            index = onefile[:onefile.find('_')]
            # print(index)


            # print(index)
            chara_list = []
            for nodule in nodules:
                if nodule.id == index:
                    sphercity.append([nodule.sphercity, nodule.sphercity])
                    margin.append([nodule.margin, nodule.margin])
                    lobulation.append([nodule.lobulation, nodule.lobulation])
                    spiculation.append([nodule.spiculation, nodule.spiculation])
            arr = np.load(basedir + onefile)
            batch_array.append(arr)
            # id = re.findall("\d+",onefile)[0]
            if 'high' in onefile:
                batch_label.append([0, 1])
            elif 'low' in onefile:
                batch_label.append([1, 0])
            batch_path.append({
                'id': int(index),
                'path': onefile}
            )

        except Exception as e:
            print("file not exists! %s"%onefile)
            batch_array.append(batch_array[-1]) 
    return np.array(batch_array), np.array(batch_label), batch_path



def get_selnet_batch(basedir, batch_filename):
    '''
    get batch for selnet, the label specify the type of the nodules [non-margin, margin]
    return data and label
    '''
    batch_array = []
    batch_label = []
    batch_path = []

    cnt_non_margin = 0
    cnt_margin = 0
    for onefile in batch_filename:
        try:
            index = onefile[:onefile.find('_')]
            
            nodule = None
            for n in nodules:
                if n.id == index:
                    nodule = n
            arr = np.load(basedir + onefile)
            batch_array.append(arr)
            # id = re.findall("\d+",onefile)[0]
            if nodule.spiculation > 2:
                batch_label.append([0, 1])
                cnt_margin += 1
            elif nodule.spiculation <= 2:
                batch_label.append([1, 0])
                cnt_non_margin += 1
            batch_path.append({
                'id': int(index),
                'path': onefile}
            )

        except Exception as e:
            print("file not exists! %s"%onefile)
            batch_array.append(batch_array[-1]) 
    # print(f'selnet dataset stat: non-margin:{cnt_non_margin}, margin:{cnt_margin}')
    return np.array(batch_array), np.array(batch_label), batch_path

def get_batch(basedir, batch_filename):
    '''
    get batch
    return data and label
    '''
    batch_array = []
    batch_label = []

    for onefile in batch_filename:
        try:
            arr = np.load(basedir + onefile)
            batch_array.append(arr)
            # id = re.findall("\d+",onefile)[0]
            if 'high' in onefile:
                batch_label.append([0, 1])
            elif 'low' in onefile:
                batch_label.append([1, 0])

        except Exception as e:
            print("file not exists! %s"%onefile)
            batch_array.append(batch_array[-1]) 
    return np.array(batch_array), np.array(batch_label)

