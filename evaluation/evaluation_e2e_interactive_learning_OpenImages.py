# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 18:09:44 2018

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
from nets import resnet_v1
from measurement import apk,compute_number_misclassified
import colaborative_loss
import D_utility
import global_setting_OpenImage
import pdb
from tensorflow.contrib import slim
from sklearn.metrics import average_precision_score
from preprocessing import preprocessing_factory
#%%
####### CSSM #######
#path_pretrain = './result/e2e_asym_OpenImage_c_1_f_0.0_1e-08_20_0.8_signal_str_0.5_114292_GPU_7_thresCoeff_0.001_c_2.0_stamp_1548132082.4567087/'
#global_setting_OpenImage.saturated_Thetas_model  = path_pretrain+'model_ES.npz'
#global_setting_OpenImage.model_path = path_pretrain+'model_colaborative 0.002247315365821123 feature 0.0 regularizer 0_ES.ckpt'

####### self-training #######
#path_pretrain = './result/blackbox_self_training_diff_LR_OpenImage_1e-08_20_0.8_signal_str_0.5_182867_GPU_6_searchIntvl_1000_n_update_ST_1_ST_batch_20_stamp_1548107207.7237468/'
#global_setting_OpenImage.saturated_Thetas_model  = path_pretrain+'model_ES.npz'
#global_setting_OpenImage.model_path = path_pretrain+'model_regularizer 0_ES.ckpt'

####### latent noise #######
#path_pretrain = './result/blackbox_latent_noise_OpenImage_1e-08_20_0.8_signal_str_0.5_182867_GPU_4_searchIntvl_5000_c_2.0_stamp_1546202323.602185/'
#global_setting_OpenImage.saturated_Thetas_model  = path_pretrain+'model_ES.npz'
#global_setting_OpenImage.model_path = path_pretrain+'model_regularizer 0_ES.ckpt'
baseline = True
#%% data flag
#
is_G = True
is_nonzero_G = True
is_constrant_G = False
is_sum_1=True
is_optimize_all_G = True
#
is_use_batch_norm = True
capacity = -1
val_capacity = -1
dictionary_evaluation_interval=250
partition_size = 300
strength_identity = 1
idx_GPU=4
train_data_path= '/home/project_amadeus/mnt/cygnus/train/'
validation_data_path= '/home/project_amadeus/mnt/cygnus/test/'
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
template_name='e2e_asym_OpenImage_{}_{}_{}_signal_str_{}_{}_GPU_{}_thresCoeff_{}_c_{}_stamp_{}'
list_alphas_colaborative = [1]#
list_alphas_feature = [0.5]#0,0.5,1,2
global_step = tf.Variable(0, trainable=False,dtype=tf.float32)
learning_rate = tf.Variable(global_setting_OpenImage.e2e_learning_rate_base,trainable = False,dtype=tf.float32)
n_iters = 1
decay_rate_schedule = 1
schedule_wrt_report_interval = 80
#report_length = global_setting_OpenImage.e2e_n_cycles*n_iters//global_setting_OpenImage.report_interval +1 #in case that my lousy computation is wrong
#patient=report_length//100
c = 2.0

is_save = True
parallel_iterations = 1
#%%
print('number of cycles {}'.format(global_setting_OpenImage.n_cycles))
print('number partition_size ',partition_size)
#%%
def compute_AP(Prediction,Label):
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

#%% label mapping function
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
#%%
labelmap, label_dict = LoadLabelMap(global_setting_OpenImage.labelmap_path, global_setting_OpenImage.dict_path)
list_label = []
for id_name in labelmap:
    list_label.append(label_dict[id_name])
n_class = len(list_label)
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

def read_raw_img(img_id,data_path):
    return tf.read_file(data_path+img_id+'.jpg','rb')

def parser_train(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id =  parsed['img_id']
    label = tf.decode_raw( parsed['label'],tf.int32)
    img = read_raw_img(img_id,train_data_path)
    return img_id,img,label

def parser_validation(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id =  parsed['img_id']
    label = tf.decode_raw( parsed['label'],tf.int32)
    img = read_raw_img(img_id,validation_data_path)
    return img_id,img,label
#%%
def compute_feature_prediction_large_batch(img,is_silent = False):
    prediction_l = []
    feature_l = []
    tic = time.clock()
    for idx_partition in range(img.shape[0]//partition_size+1):
        if not is_silent:
            print('{}.'.format(idx_partition),end='')
        prediction_partition,feature_partition = sess.run([Prediction,features_concat],{img_input_ph:img[idx_partition*partition_size:(idx_partition+1)*partition_size]})
        prediction_l.append(prediction_partition)
        feature_l.append(feature_partition)
    if not is_silent:
        print('time: ',time.clock()-tic)
    prediction = np.concatenate(prediction_l)
    feature = np.concatenate(feature_l)
    print()
    return prediction,feature

def load_memory(iterator_next,size,capacity = -1):
    labels_l = []
    ids_l=[]
    imgs_l = []
    print('load memory')
    if capacity == -1:
        n_p = size//partition_size+1
    else:
        n_p = capacity
    for idx_partition in range(n_p):
        print('{}.'.format(idx_partition),end='')
        (img_ids_p,img_p,labels_p) = sess.run(iterator_next)
        labels_l.append(labels_p)
        ids_l.append(img_ids_p)
        imgs_l.append(img_p)
    print()
    labels = np.concatenate(labels_l)
    ids = np.concatenate(ids_l)
    imgs = np.concatenate(imgs_l)
    return ids,imgs,labels

def compute_feature_prediction_large_batch_iterator(iterator_next,size):
    prediction_l = []
    feature_l = []
    labels_l = []
    ids_l=[]
    print('compute large batch')
    for idx_partition in range(10):#range(size//partition_size+1):
        print('partition ',idx_partition)
        tic = time.clock()
        (img_ids_p,img_p,labels_p) = sess.run(iterator_next)
        print(time.clock()-tic)
        tic = time.clock()
        prediction_partition,feature_partition = sess.run([Prediction,features_concat],{img_input_ph:img_p})
        print(time.clock()-tic)
        prediction_l.append(prediction_partition)
        feature_l.append(feature_partition)
        labels_l.append(labels_p)
        ids_l.append(img_ids_p)
    prediction = np.concatenate(prediction_l)
    feature = np.concatenate(feature_l)
    labels = np.concatenate(labels_l)
    ids_l = np.concatenate(ids_l)
    return prediction,ids_l,feature,labels

def get_img_sparse_dict_support_v2(support_ids):
    imgs = []
    for s_id in support_ids:
        imgs.append(read_img(s_id.decode("utf-8"),train_data_path)[tf.newaxis,:,:,:])
    imgs = sess.run(imgs)
    return np.concatenate(imgs)
def get_img_sparse_dict_support(idx_support,iterator_next,size):
    imgs_l = []
    labels_l = []
    print('get img dict support')
    for idx_partition in range(size//partition_size+1):
        print('partition ',idx_partition)
        (img_ids_p,img_p,labels_p) = sess.run(iterator_next)
        min_idx = idx_partition*partition_size
        max_idx = min_idx+img_p.shape[0]
        selector = np.where((idx_support>=min_idx) & (idx_support<max_idx))
        imgs_l.append(img_p[selector])
        labels_l.append(labels_p[selector])
    imgs = np.concatenate(imgs_l)
    labels = np.concatenate(labels_l)
    return imgs,labels
#%% load in memory
sess = tf.InteractiveSession()#tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
g = tf.get_default_graph()
#%%
Theta = tf.get_variable('Theta',shape=[2049,n_class])
learning_rate_fh=tf.placeholder(dtype=tf.float32,shape=())
op_assign_learning_rate = learning_rate.assign(learning_rate_fh)
#%%
dataset_in_2 = tf.data.TFRecordDataset(global_setting_OpenImage.test_path)
dataset_in_2 = dataset_in_2.map(parser_validation).batch(partition_size)
val_iterator_next = dataset_in_2.make_one_shot_iterator().get_next()
#(img_val_ids,val_img_v,val_labels)=sess.run([img_val_ids,val_img,val_labels])
#%%
n_sparse_dict = D_utility.count_records(global_setting_OpenImage.sparse_dict_path)
n_val = D_utility.count_records(global_setting_OpenImage.test_path)
img_val_ids,val_imgs_v,val_labels = load_memory(val_iterator_next,n_val,val_capacity)
#%%
#with slim.arg_scope(resnet_v1.resnet_arg_scope()):
saver = tf.train.import_meta_graph('./model/resnet/oidv2-resnet_v1_101.ckpt.meta')
img_input_ph = g.get_tensor_by_name('input_values:0')
features_concat = g.get_tensor_by_name('resnet_v1_101/pool5:0')
features_concat = tf.squeeze(features_concat)
#%% normalize norm
#features_concat=D_utility.project_unit_norm(features_concat)
#%%
features_concat = tf.concat([features_concat,tf.ones([tf.shape(features_concat)[0],1])],axis = 1,name='feature_input_point')
index_point = tf.placeholder(dtype=tf.int32,shape=())
F = features_concat[:index_point,:]
sparse_dict = features_concat[index_point:,:]
F_concat_ph = g.get_tensor_by_name('feature_input_point:0')

#%%
tf.global_variables_initializer().run()
#%%
print('placeholder assignment')
#%%
Theta_fh = tf.placeholder(dtype=tf.float32, shape=[2049,n_class])
op_assign_Theta = Theta.assign(Theta_fh)

#%% compute normalizer
Prediction = tf.matmul(features_concat,Theta)

#%% computational graph
#writer = tf.summary.FileWriter(logdir='./logdir', graph=tf.get_default_graph())
#writer.close()
#%%
print('done placeholder assignment')
def experiment_cond_success():
    return True#(alpha_colaborative_o >0) or (alpha_colaborative_o + alpha_feature_o==0)

n_experiment= 0

for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            n_experiment += 1
print('Total number of experiment: {}'.format(n_experiment))
#%%

print('-'*30)
df_result = pd.DataFrame()
pos_idx = 0
print('hardcode position of Thetas={} and basline {}: '.format(pos_idx,baseline))
data=np.load(global_setting_OpenImage.saturated_Thetas_model)
if baseline:
    init_Theta = data['Theta']
else:
    init_Theta = data['Thetas'][:,:,pos_idx]

#pdb.set_trace()
tf.global_variables_initializer().run()
#%%
sess.run(op_assign_Theta,{Theta_fh:init_Theta})
saver.restore(sess, global_setting_OpenImage.model_path)
#%%
#%%
# absolute regularization
raitio_regularizer_grad_v=1

name = path_pretrain.split('/')[2]
#%% create dir
if not os.path.exists('./result/evaluation_test_set_without_asym/'+name) and is_save:
    os.makedirs('./result/evaluation_test_set_without_asym/'+name)
#%%
posfix = 'ES'
if 'ES' not in global_setting_OpenImage.saturated_Thetas_model:
    posfix='final'
ap_save_name = './result/evaluation_test_set_without_asym/'+name+'/mAP_'+posfix+'.csv'
print('save_path_name',ap_save_name)
Thetas = np.zeros((2049,n_class,n_experiment))
Gs = np.zeros((n_class,n_class,n_experiment))
idx_experiment = 0
for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):
            
            
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            
#            print('report length {}'.format(report_length))
#            res_mAP = np.zeros(report_length)
#            res_loss = np.zeros(report_length)
#            res_loss_logistic=np.zeros(report_length)
#            res_sum_num_miss_p=np.zeros(report_length)
#            res_sum_num_miss_n=np.zeros(report_length)
#            res_grad_logistic=np.zeros(report_length)
#            res_lr=np.zeros(report_length)
#            res_norm_f=np.zeros(report_length)
            #
            
#            alpha_colaborative = raitio_colaborative_grad_v*alpha_colaborative_o
#            alpha_feature = raitio_featrue_grad_v*alpha_feature_o
#            alpha_regularizer = raitio_regularizer_grad_v*alpha_regularizer_o
#            
            tf.global_variables_initializer().run()
            print('reset Theta')
            saver.restore(sess, global_setting_OpenImage.model_path)
            sess.run(op_assign_Theta,{Theta_fh:init_Theta})
#            sess.run(op_alpha_colaborative_var,{alpha_colaborative_var_fh:alpha_colaborative})
#            sess.run(op_alpha_feature_var,{alpha_feature_var_fh:alpha_feature})
#            sess.run(op_alpha_regularizer,{alpha_regularizer_var_fh:alpha_regularizer})
#            extension = 'colaborative {} feature {} regularizer {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer)
#    
            #exponential moving average
            expon_moving_avg_old = np.inf
            expon_moving_avg_new = 0
            #
            m = 0
            df_ap = pd.DataFrame()
            df_ap['label']=list_label
#            print('lambda colaborative: {} lambda_feature: {} regularizer: {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer))
#            n_nan = 0
#            
            validate_Prediction_val,_=compute_feature_prediction_large_batch(val_imgs_v)
            ap = compute_AP(validate_Prediction_val,val_labels)
            num_mis_p,num_mis_n=compute_number_misclassified(validate_Prediction_val,val_labels)
            df_ap['ap']=ap
            m_AP=np.mean(ap)
            print('mAP: ',m_AP)
            if is_save:
                
                
                df_ap.to_csv(ap_save_name)
#%%
sess.close()
tf.reset_default_graph()