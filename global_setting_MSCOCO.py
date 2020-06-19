#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:07:15 2019

@author: war-machince
"""
#%%
docker_path = './'
NFS_path = docker_path
#%%
train_img_path = NFS_path + 'TFRecords/train_MSCOCO_img_ZLIB.tfrecords'
train_origin_img_path = NFS_path + 'TFRecords/train_MSCOCO_origin_img.tfrecords'
dic_img_path = NFS_path+'TFRecords/dic_10_MSCOCO_img_ZLIB.tfrecords'
print('Use test set as validation')
validation_img_path = NFS_path + 'TFRecords/test_MSCOCO_img_ZLIB.tfrecords'
test_img_path = NFS_path + 'TFRecords/test_MSCOCO_img_ZLIB.tfrecords'
label_graph_path = './label_graph/graph_label_wiki_MSCOCO_k_5.npz'
sparse_img_dict_path = NFS_path + 'TFRecords/'
#%%
batch_size = 32#32
learning_rate_base = 0.01
thresold_coeff = 1e-6
limit_learning_rate = 1e-4
decay_rate_cond = 0.8
signal_strength = 0.3
n_report=50
k = 3
n_cycles = 10000
regularizers = [0.0001]
early_stopping = True
lr_schedule = 'EMA'