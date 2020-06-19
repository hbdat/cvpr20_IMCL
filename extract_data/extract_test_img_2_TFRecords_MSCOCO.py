#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:18:58 2019

@author: war-machince
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd) 
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import tensorflow as tf
import pandas as pd
import os.path
import os
import pdb
from nets import resnet_v1
from preprocessing import preprocessing_factory
import numpy as np
import h5py
import json
import pickle
from shutil import copyfile
#%%
data_set = 'test'
print(data_set)
#nrows = None
docker_path = './'
path = docker_path+'data/MSCOCO_1k/'
project_path = docker_path
num_classes = 1000
batch_size = 32
print('dataset: {}'.format(data_set))
num_parallel_calls=1
is_save = True
os.environ["CUDA_VISIBLE_DEVICES"]="5"
#%%
print('loading data')
label_path_1k = path + 'coco1k_coco-valid2_label_counts.h5'

label_path_81 = path + 'coco_instancesGT_eval_valid2.h5'
img_partition = path + 'captions_valid22014.json'
data_path = path
data_path += 'val2014/'
img_tfrecord_filename = project_path+'TFRecords/'+data_set+'_MSCOCO_img_ZLIB.tfrecords'
#%%
def load_1k_name():
    path = '/home/project_amadeus/mnt/raptor/hbdat/data/MSCOCO_1k/meta/vocab_coco.pkl'
    with open(path,'rb') as f:
        vocab = pickle.load(f)
    return vocab['words']
#%%
with open(img_partition) as f:
    data = json.load(f)
files = [d['file_name'].split('.')[0] for d in data['images']]
f_1k = h5py.File(label_path_1k, 'r')
f_81 = h5py.File(label_path_81, 'r')
all_labels_1k = f_1k['gtLabel'].value
all_labels_81 = f_81['gtLabel'].value
assert all_labels_1k.shape[0] == len(files) and all_labels_81.shape[0] == len(files)
dict_labels = {}
for idx_f,file in enumerate(files):
    dict_labels[file] = (all_labels_1k[idx_f],all_labels_81[idx_f])
classes = load_1k_name()
#%%
image_size = resnet_v1.resnet_v1_101.default_image_size
def PreprocessImage(image, network='resnet_v1_101'):
      # If resolution is larger than 224 we need to adjust some internal resizing
      # parameters for vgg preprocessing.
      preprocessing_kwargs = {}
      preprocessing_fn = preprocessing_factory.get_preprocessing(name=network, is_training=False)
      height = image_size
      width = image_size
      image = preprocessing_fn(image, height, width, **preprocessing_kwargs)
      image.set_shape([height, width, 3])
      return image

def read_img(file):
    compressed_image = tf.read_file(data_path+file+'.jpg', 'rb')
    image = tf.image.decode_jpeg(compressed_image, channels=3)
    processed_image = PreprocessImage(image)
    return processed_image,file

def get_label(processed_image,file):
    img_id = file.decode('utf-8').split('.')[0]
    label_1k = np.squeeze(dict_labels[img_id][0]).astype(np.int32)
    label_81 = np.squeeze(dict_labels[img_id][1]).astype(np.int32)
    #print(partition_idx)
    return processed_image,img_id,label_1k,label_81
#%%
print('data size',len(files))
print('loading data:done')
dataset = tf.data.Dataset.from_tensor_slices(files)
dataset = dataset.map(read_img,num_parallel_calls)
dataset = dataset.map(lambda processed_image,file: tuple(tf.py_func(get_label, [processed_image,file], [tf.float32, tf.string, tf.int32,tf.int32])),num_parallel_calls)
dataset = dataset.batch(batch_size).prefetch(batch_size)#.map(PreprocessImage)
iterator=dataset.make_one_shot_iterator().get_next()
#%%
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
g = tf.get_default_graph()
df = pd.DataFrame()
df['classes'] = classes
#%%
#with tf.device('/device:GPU:{}'.format(gpu_idx)):
with g.as_default():
    if is_save:
        opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        img_writer = tf.python_io.TFRecordWriter(img_tfrecord_filename, options=opts)
    idx = 0
    while True:
        try:
            (processed_image,img_ids,labels_1k,labels_81)=sess.run(iterator)
#            pdb.set_trace()
            if idx%(1000//32)==0:
                imd_id = img_ids[0].decode('utf-8')
                label_1k = labels_1k[0]
                copyfile(data_path+imd_id+'.jpg',project_path+'sanity_check_MSCOCO_test_labels/'+imd_id+'.jpg')
                df[imd_id]=label_1k
                df.to_csv(project_path+'sanity_check_MSCOCO_test_labels/label_1k.csv')
            print('batch no. {}'.format(idx))
            for idx_s in range(processed_image.shape[0]):
                img = processed_image[idx_s,:,:,:].ravel()
                img_id = img_ids[idx_s]
                label_1k = labels_1k[idx_s,:]
                label_81 = labels_81[idx_s,:]
                example = tf.train.Example()
                example.features.feature["img"].bytes_list.value.append(tf.compat.as_bytes(img.tostring()))
                example.features.feature["img_id"].bytes_list.value.append(tf.compat.as_bytes(img_id))
                example.features.feature["label_81"].bytes_list.value.append(tf.compat.as_bytes(label_81.tostring()))
                example.features.feature["label_1k"].bytes_list.value.append(tf.compat.as_bytes(label_1k.tostring()))
                if is_save:
                    img_writer.write(example.SerializeToString())
                
            idx += 1
        except tf.errors.OutOfRangeError:
            print('end')
            break
    if is_save:
        img_writer.close()
sess.close()
tf.reset_default_graph()