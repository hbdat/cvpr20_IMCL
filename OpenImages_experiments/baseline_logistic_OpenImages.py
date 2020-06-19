# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 18:38:40 2018

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
from average_precision import apk
import global_setting_OpenImage 
import D_utility
#%% logging level
tf.logging.set_verbosity(tf.logging.INFO)
#%% override
#global_setting_OpenImage.learning_rate_base = 0.001
global_setting_OpenImage.batch_size=32
global_setting_OpenImage.n_cycles*=1#60*global_setting_OpenImage.report_interval
global_setting_OpenImage.report_interval = 100
global_setting_OpenImage.n_cycles = 3657355//global_setting_OpenImage.batch_size
#%% data flag
idx_GPU=7
beta = 0.02
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
df_image = pd.read_csv('./data/2017_11/train/images.csv')

list_alphas = [0,0.001,0.01,0.1,1]

global_step = tf.Variable(0, trainable=False,dtype=tf.float32)
learning_rate = 1.0/(tf.sqrt(global_step)+1.0)*global_setting_OpenImage.learning_rate_base#tf.Variable(global_setting_OpenImage.learning_rate_base,trainable = False,dtype=tf.float32)

n_iters = 1

schedule_wrt_report_interval = 80

name = 'baseline_logistic_OpenImages'
save_name = name+'.csv'
df_result = pd.DataFrame()
is_save = False
global_setting_OpenImage.report_interval = 100
#n_process_map = 4
#%%
print('number of cycles {}'.format(global_setting_OpenImage.n_cycles))
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

predictions_eval = 0
predictions_eval_resize = 0
#%%
labelmap, label_dict = LoadLabelMap(global_setting_OpenImage.labelmap_path, global_setting_OpenImage.dict_path)
list_label = []
for id_name in labelmap:
    list_label.append(label_dict[id_name])

#%% Dataset

def parser(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'feature': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id = parsed['img_id']
    feature = tf.decode_raw( parsed['feature'],tf.float32)
    label = tf.decode_raw( parsed['label'],tf.int32)
    return img_id,feature,label

#%% load in memory
sess = tf.InteractiveSession()#tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
g = tf.get_default_graph()
#with g.as_default():
saver = tf.train.import_meta_graph(global_setting_OpenImage.model_path+ '.meta')
saver.restore(sess, global_setting_OpenImage.model_path)
weight = tf.squeeze( g.get_tensor_by_name('resnet_v1_101/logits/weights:0'))
bias =  g.get_tensor_by_name('resnet_v1_101/logits/biases:0')[tf.newaxis,:]
init_Theta = tf.concat([weight,bias],name='Theta',axis = 0).eval()
#%%
Theta = tf.get_variable('Theta',shape=[2049,5000])
#%%
dataset = tf.data.TFRecordDataset(global_setting_OpenImage.record_path)
dataset = dataset.map(parser)
dataset = dataset.shuffle(20000)
dataset = dataset.batch(global_setting_OpenImage.batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
(img_ids,img_features,labels) = iterator.get_next()

dataset_in = tf.data.TFRecordDataset(global_setting_OpenImage.validation_path)
dataset_in = dataset_in.map(parser).batch(50000)
(img_val_ids,F_val,val_labels) = dataset_in.make_one_shot_iterator().get_next()
F_val = tf.concat([F_val,tf.ones([tf.shape(F_val)[0],1])],axis = 1)
(img_val_ids,F_val,val_labels)=sess.run([img_val_ids,F_val,val_labels])

#%%
def AP(prediction,label,names):
    mask = np.abs(label)==1
    if np.sum(label==1)==0:
        return 0.0
    groundtruth = names[label == 1]
    prediction = prediction[mask]
    retrieval = names[mask]
    sort_idx = np.argsort(prediction)[::-1]
    retrieval = retrieval[sort_idx]
    return apk(groundtruth,retrieval,len(prediction))

def compute_AP(Prediction,Label,names):
    num_class = Prediction.shape[1]
    ap=np.zeros(num_class)
    for idx_cls in range(num_class):
        prediction = Prediction[:,idx_cls]
        label = Label[:,idx_cls]
        ap[idx_cls]=AP(prediction,label,names)
    return ap
#%%
F = tf.squeeze(img_features)
F = tf.concat([F,tf.ones([tf.shape(F)[0],1])],axis = 1)

#%%
alpha_regularizer_var = tf.get_variable('alpha_regularizer',dtype=tf.float32,trainable=False, shape=())
alpha_regularizer_var_fh = tf.placeholder(dtype=tf.float32, shape=())
#%%
op_alpha_regularizer = alpha_regularizer_var.assign(alpha_regularizer_var_fh)
#%%
G = np.load('./label_graph/graph_label_naive.npy').astype(np.float32)
G=D_utility.preprocessing_graph(G)
G_empty_diag = G - np.diag(np.diag(G))
G_init=G_empty_diag[G_empty_diag!=0]
G_var = tf.get_variable("G_var", G_init.shape)
op_G_var=G_var.assign(G_init)
indices = []
#updates = []
shape = tf.constant(G.shape)
counter = 0

diag_G = tf.diag(np.diag(G))

for idx_row in range(G_empty_diag.shape[1]):
    idx_cols = np.where(G_empty_diag[idx_row,:]!=0)[0]
    for idx_col in idx_cols:
        if G[idx_row,idx_col]-G_init[counter] != 0:
            raise Exception('error relation construction')
        if idx_row != idx_col:
            indices.append([idx_row,idx_col])
        counter += 1
part_G_var = tf.scatter_nd(indices, G_var, shape)+diag_G
#%% disperse measurement
dispersion=tf.reduce_sum(tf.abs(part_G_var)) - tf.reduce_sum(tf.diag_part(tf.abs(part_G_var)))
#%%

with tf.variable_scope("logistic"):
    logits = tf.matmul(F,Theta)
    labels_binary = tf.div(labels+1,2)
    labels_weight = tf.abs(labels)
    loss_logistic = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=logits,weights=labels_weight)

with tf.variable_scope("regularizer"):
    loss_regularizer = tf.square(tf.norm(Theta))
#%% shared operation
grad_logistic = tf.gradients(loss_logistic, Theta)
grad_regularizer = tf.gradients(loss_regularizer,Theta)

norm_grad_logistic = tf.norm(grad_logistic)
norm_grad_regularizer = tf.norm(grad_regularizer)
norm_Theta = tf.norm(Theta)
raitio_regularizer_grad = norm_grad_logistic/norm_grad_regularizer

validate_Prediction = tf.matmul(F_val,Theta)
#%%
tf.global_variables_initializer().run()
sess.run(iterator.initializer)
#%%
def append_info(m_AP,loss_value,lr_v):
    
    res_mAP[index]=m_AP
    res_loss[index] = loss_value
    res_lr[index]=lr_v
    
    df_result['mAP: regularizer {}'.format(alpha_regularizer)]=res_mAP
    df_result['loss: regularizer {}'.format(alpha_regularizer)]=res_loss
    df_result['lr: regularizer {}'.format(alpha_regularizer)]=res_lr

#%%
Theta_fh = tf.placeholder(dtype=tf.float32, shape=[2049,5000])
op_assign_Theta = Theta.assign(Theta_fh)
global_step_fh=tf.placeholder(dtype=tf.float32,shape=())
op_assign_global_step = global_step.assign(global_step_fh)

#%%
tf.global_variables_initializer().run()
sess.run(op_G_var)
sess.run(op_assign_Theta,{Theta_fh:init_Theta})
sess.run(iterator.initializer)

#%%
#optimizer = tf.train.AdamOptimizer(learning_rate=global_setting_OpenImage.learning_rate_base)#tf.train.RMSPropOptimizer(learning_rate=learning_rate)#,momentum=0.9
optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
loss = loss_logistic
#%% hypergradient
grad_loss = tf.gradients(loss, Theta)
#%%
train = optimizer.minimize(loss,var_list=[Theta],global_step = global_step)
print('done placeholder assignment')
def experiment_cond_success():
    return alpha_colaborative_o == 0.1 and alpha_feature_o ==0

n_experiment= 0

for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            n_experiment += 1
print('Total number of experiment: {}'.format(n_experiment))

Thetas = np.zeros((2049,5000,n_experiment))
Gs = np.zeros((5000,5000,n_experiment))
idx_experiment = 0
expon_moving_avg_old = np.inf
expon_moving_avg_new = 0  
for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):
            index_column = idx_alpha_colaborative*len(list_alphas)*len(list_alphas)+idx_alpha_feature*len(list_alphas)+idx_alpha_regularizer
            
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            
            report_length = global_setting_OpenImage.n_cycles*n_iters//global_setting_OpenImage.report_interval +1 #in case that my lousy computation is wrong
            print('report length {}'.format(report_length))
            res_mAP = np.zeros(report_length)
            res_loss = np.zeros(report_length)
            res_loss_logistic=np.zeros(report_length)
            res_loss_R=np.zeros(report_length)
            res_loss_feature=np.zeros(report_length)
            res_grad_logistic=np.zeros(report_length)
            res_grad_R=np.zeros(report_length)
            res_lr=np.zeros(report_length)
            #loss_R#
            sess.run(iterator.initializer)
            alpha_regularizer = alpha_regularizer_o
            
            tf.global_variables_initializer().run()
            print('reset Theta')
            sess.run(op_G_var)
            sess.run(op_assign_Theta,{Theta_fh:init_Theta})
            sess.run(op_alpha_regularizer,{alpha_regularizer_var_fh:alpha_regularizer})
            
            df_ap = pd.DataFrame()
            df_ap['label']=list_label
            #%%
            tic = time.clock()
            for idx_cycle in range(global_setting_OpenImage.n_cycles):
                _,loss_value,lr_v  = sess.run([train,loss,learning_rate])
                
                index = (idx_cycle*n_iters)//global_setting_OpenImage.report_interval
                
                if (idx_cycle*n_iters) % global_setting_OpenImage.report_interval == 0 :#or idx_iter == n_iters-1:
                    print('Elapsed time udapte: {}'.format(time.clock()-tic))
                    tic = time.clock()
                    print('index {} -- compute mAP'.format(index))
                    print('Loss {} lr {}'.format(loss_value,lr_v))
                    validate_Prediction_val = validate_Prediction.eval()
                    ap = compute_AP(validate_Prediction_val,val_labels,img_val_ids)
                    df_ap['index {}'.format(index)]=ap
                    m_AP=np.mean(ap)
#                        
                    append_info(m_AP,loss_value,lr_v)
                    print('mAP {}'.format(m_AP))
                    if is_save:
                        Thetas[:,:,idx_experiment]=Theta.eval()
                        Gs[:,:,idx_experiment]=part_G_var.eval()
                        df_result.to_csv('./result/'+save_name)
                        ap_save_name = './result/baseline_ap_{}.csv'
                        df_ap.to_csv(ap_save_name.format(alpha_regularizer))
                        if index%(int(report_length/4)) == 0:
                            np.savez('./result/'+name, Thetas=Thetas, Gs=Gs)
            idx_experiment+=1
if is_save:
    np.savez('./result/'+name, Thetas=Thetas, Gs=Gs)
#%%
sess.close()
tf.reset_default_graph()