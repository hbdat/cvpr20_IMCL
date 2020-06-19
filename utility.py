# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:34:41 2019

@author: badat
"""
from sklearn.metrics import average_precision_score
import numpy as np
import tensorflow as tf

#%%
def compute_AP(Prediction,Label,names):
    num_class = Prediction.shape[1]
    ap=np.zeros(num_class)
    for idx_cls in range(num_class):
        prediction = np.squeeze(Prediction[:,idx_cls])
        label = np.squeeze(Label[:,idx_cls])
        mask = np.abs(label)==1
        if np.sum(label>0)==0:
            continue
        binary_label=np.clip(label[mask],0,1)
        ap[idx_cls]=average_precision_score(binary_label,prediction[mask])#AP(prediction,label,names)
    return ap
#%%
def LoadLabelMap(labelmap_path, dict_path):
  labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path)]

  label_dict = {}
  for line in tf.gfile.GFile(dict_path):
    words = [word.strip(' "\n') for word in line.split(',', 1)]
    label_dict[words[0]] = words[1]

  return labelmap, label_dict

#%% Dataset
image_size = resnet_v1.resnet_v1_101.default_image_size
height = image_size
width = image_size
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

def read_img(img_id,data_path):
    compressed_image = tf.read_file(data_path+img_id+'.jpg', 'rb')
    image = tf.image.decode_jpeg(compressed_image, channels=3)
    processed_image = PreprocessImage(image)
    return processed_image

def parser_train(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id =  parsed['img_id']
    label = tf.decode_raw( parsed['label'],tf.int32)
    img = read_img(img_id,train_data_path)
    return img_id,img,label

def parser_validation(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id =  parsed['img_id']
    label = tf.decode_raw( parsed['label'],tf.int32)
    img = read_img(img_id,validation_data_path)
    return img_id,img,label
#%%
def construct_dictionary(batch_attribute,batch_id,sparse_dict_Attribute_f,sparse_dict_img,n_neighbour):
    
    similar_score=np.matmul(np.clip(batch_attribute,-1/c,1),np.clip(np.transpose(sparse_dict_Attribute_f),-1/c,1))
    m_similar_index=np.argsort(similar_score,axis=1)[:,0:n_neighbour]
    index_dict = m_similar_index.flatten()
    return sparse_dict_Attribute_f[index_dict,:],sparse_dict_img[index_dict,:,:]

def compute_feature_prediction_large_batch(img):
    prediction_l = []
    feature_l = []
    
    for idx_partition in range(img.shape[0]//partition_size+1):
        prediction_partition,feature_partition = sess.run([Prediction,features_concat],{img_input_ph:img[idx_partition*partition_size:(idx_partition+1)*partition_size,:,:,:]})
        prediction_l.append(prediction_partition)
        feature_l.append(feature_partition)
    prediction = np.concatenate(prediction_l)
    feature = np.concatenate(feature_l)
    return prediction,feature