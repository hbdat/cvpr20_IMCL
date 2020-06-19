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
#%%
data_set = 'train'
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
label_path = path + 'meta/'
data_path = path
label_path += 'labels_captions_coco_vocabS1k_train.h5'
data_path += 'train2014/'
img_tfrecord_filename = project_path+'TFRecords/'+data_set+'_MSCOCO_img_ZLIB.tfrecords'
#%%
f = h5py.File(label_path, 'r')
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
    label = np.squeeze(f['labels-'+img_id]).astype(np.int32)
    #print(partition_idx)
    return processed_image,img_id,label
#%%
files = [file.split('-')[-1] for file in list(f.keys())]
pdb.set_trace()
print('data size',len(files))
print('loading data:done')
dataset = tf.data.Dataset.from_tensor_slices(files)
dataset = dataset.map(read_img,num_parallel_calls)
dataset = dataset.map(lambda processed_image,file: tuple(tf.py_func(get_label, [processed_image,file], [tf.float32, tf.string, tf.int32])),num_parallel_calls)
dataset = dataset.batch(batch_size).prefetch(batch_size)#.map(PreprocessImage)
iterator=dataset.make_one_shot_iterator().get_next()
#%%
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
g = tf.get_default_graph()
#%%
#with tf.device('/device:GPU:{}'.format(gpu_idx)):
with g.as_default():
    if is_save:
        opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        img_writer = tf.python_io.TFRecordWriter(img_tfrecord_filename, options=opts)
    idx = 0
    while True:
        try:
            (processed_image,img_ids,labels)=sess.run(iterator)
#            pdb.set_trace()
            print('batch no. {}'.format(idx))
            for idx_s in range(processed_image.shape[0]):
                img = processed_image[idx_s,:,:,:].ravel()
                img_id = img_ids[idx_s]
                label = labels[idx_s,:]
                example = tf.train.Example()
                example.features.feature["img"].bytes_list.value.append(tf.compat.as_bytes(img.tostring()))
                example.features.feature["img_id"].bytes_list.value.append(tf.compat.as_bytes(img_id))
                example.features.feature["label"].bytes_list.value.append(tf.compat.as_bytes(label.tostring()))
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