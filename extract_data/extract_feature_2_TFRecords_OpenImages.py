# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:28:20 2018

@author: badat
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import os.path
import os
import numpy as np
import time
from tensorflow.contrib import slim
from preprocessing import preprocessing_factory
from nets import resnet_v1
import pdb
#%%
data_set = 'train'
print(data_set)
#nrows = None
path = './data/2017_11/'
num_classes = 5000
net_name = 'resnet_v1_101'
checkpoints_dir= './model/resnet/oidv2-resnet_v1_101.ckpt'
batch_size = 32
is_save = True
print('dataset: {}'.format(data_set))
num_parallel_calls=1
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#%%
image_size = resnet_v1.resnet_v1_101.default_image_size
height = image_size
width = image_size
def PreprocessImage(image, network):
      preprocessing_kwargs = {}
      preprocessing_fn = preprocessing_factory.get_preprocessing(name=network, is_training=False)
      height = image_size
      width = image_size
      image = preprocessing_fn(image, height, width, **preprocessing_kwargs)
      image.set_shape([height, width, 3])
      return image
#%%
print('loading data')
df_label = pd.read_csv(path+data_set+'/annotations-human.csv')
labelmap_path = path+'classes-trainable.txt'
dict_path =  path+'class-descriptions.csv'
feature_tfrecord_filename = './TFRecords/'+data_set+'_feature.tfrecords'
data_path= './image_data/OpenImages/'+data_set+'/'
#%% split dataframe
print('partitioning data')
capacity = 40000
partition_df = []
t = len(df_label)//capacity
for idx_cut in range(t):
    #print(idx_cut)
    partition_df.append(df_label.iloc[idx_cut*capacity:(idx_cut+1)*capacity])
partition_df.append(df_label.iloc[t*capacity:])
#%%
files=[]
partition_idxs = []
for idx_partition,partition in enumerate(partition_df):
    #print(idx_partition)
    file_partition=[ img_id+'.jpg' for img_id in partition['ImageID'].unique() if os.path.isfile(data_path+img_id+'.jpg')]
    files.extend(file_partition)
    partition_idxs.extend([idx_partition]*len(file_partition))
n_samples = len(files)
print('number of sample: {} dataset: {}'.format(n_samples,data_set))
#%%
def LoadLabelMap(labelmap_path, dict_path):
  """Load index->mid and mid->display name maps.

  Args:
    labelmap_path: path to the file with the list of mids, describing
        predictions.
    dict_path: path to the dict.csv that translates from mids to display names.
  Returns:
    labelmap: an index to mid list
    label_dict: mid to display name dictionary
  """
  labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path)]

  label_dict = {}
  for line in tf.gfile.GFile(dict_path):
    words = [word.strip(' "\n') for word in line.split(',', 1)]
    label_dict[words[0]] = words[1]

  return labelmap, label_dict

predictions_eval = 0
predictions_eval_resize = 0

labelmap, label_dict = LoadLabelMap(labelmap_path, dict_path)

#%%
def read_img(file,partition_idx):
    compressed_image = tf.read_file(data_path+file, 'rb')
    image = tf.image.decode_jpeg(compressed_image, channels=3)
    processed_image = PreprocessImage(image,net_name)
#    processed_image=tf.Print(processed_image,[tf.shape(image)])
    return processed_image,file,partition_idx

def get_label(processed_image,file,partition_idx):
    img_id = file.decode('utf-8').split('.')[0]
    df_img_label=partition_df[partition_idx].query('ImageID=="{}"'.format(img_id))
    label = np.zeros(num_classes,dtype=np.int32)
    #print(len(df_img_label))
    for index, row in df_img_label.iterrows():
        if row['LabelName'] not in labelmap:
            continue #not trainable classes
        idx=labelmap.index(row['LabelName'])
        label[idx] = 2*row['Confidence']-1
        
    #print(partition_idx)
    return processed_image,img_id,label
#%%
print('loading data:done')
dataset = tf.data.Dataset.from_tensor_slices((files,partition_idxs))
dataset = dataset.map(read_img,num_parallel_calls)
dataset = dataset.map(lambda processed_images,file,partition_idx: tuple(tf.py_func(get_label, [processed_images,file,partition_idx], [tf.float32, tf.string, tf.int32])),num_parallel_calls)
dataset = dataset.batch(batch_size).prefetch(batch_size)#.map(PreprocessImage)
processed_images,img_ids,labels=dataset.make_one_shot_iterator().get_next()
#%%
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
g = tf.get_default_graph()
#%%
with g.as_default():
    if is_save:
        feature_writer = tf.python_io.TFRecordWriter(feature_tfrecord_filename)
    
    img_input_ph = tf.placeholder(dtype=tf.float32,shape=[None,height,width,3])
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        _, _ = resnet_v1.resnet_v1_101(img_input_ph, num_classes=5000, is_training=False)
        init_fn = slim.assign_from_checkpoint_fn(checkpoints_dir,slim.get_model_variables())
        features = g.get_tensor_by_name('resnet_v1_101/pool5:0')
    
    idx = 0
    init_fn(sess)
    while True:#idx < 3125:
        try:
            processed_images_v,img_ids_v,labels_v=sess.run([processed_images,img_ids,labels])
            features_v = sess.run(features,{img_input_ph:processed_images_v})
            print('batch no. {}'.format(idx))
            for idx_s in range(features_v.shape[0]):
                feature = features_v[idx_s,:,:,:]
                feature = np.reshape(feature, [49, 2048])
                img_id = img_ids_v[idx_s]
                label = labels_v[idx_s,:]
                example = tf.train.Example()
                example.features.feature["feature"].bytes_list.value.append(tf.compat.as_bytes(feature.tostring()))
                example.features.feature["img_id"].bytes_list.value.append(tf.compat.as_bytes(img_id))
                example.features.feature["label"].bytes_list.value.append(tf.compat.as_bytes(label.tostring()))
                if is_save:
                    feature_writer.write(example.SerializeToString())
                
            idx += 1
        except tf.errors.OutOfRangeError:
            print('end')
            break
    if is_save:
        feature_writer.close()
sess.close()
tf.reset_default_graph()