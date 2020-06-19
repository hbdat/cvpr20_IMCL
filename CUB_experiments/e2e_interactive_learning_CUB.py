# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 12:43:27 2018

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
import global_setting_CUB
import pdb
from tensorflow.contrib import slim
from sklearn.metrics import average_precision_score
#%% override
global_setting_CUB.batch_size=32
global_setting_CUB.learning_rate_base = 0.01
global_setting_CUB.e2e_n_cycles = 1000
#%% data flag
idx_GPU=7
is_feature_probagation = True
is_use_batch_norm = False
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
fractions = global_setting_CUB.fractions#[0.1,0.2,0.4,0.8,1]
list_alphas_colaborative = [2]#[1,0.5,5]
list_alphas_feature = [0.5]#[0.01,0.1,0.5]#[1]#[1,0.5,5]
global_step = tf.Variable(0, trainable=False,dtype=tf.float32)
if global_setting_CUB.lr_schedule == 'EMA':
    learning_rate = tf.Variable(global_setting_CUB.learning_rate_base,trainable = False,dtype=tf.float32)
else:
    learning_rate = 1.0/(tf.sqrt(global_step)+1.0)*global_setting_CUB.learning_rate_base
n_iters = 1
n_neighbour = 10
decay_rate_schedule = 1
schedule_wrt_report_interval = 80
c = 2.0
is_save = False
weight_class = 10
parallel_iterations = 1
partition_size = 200
template_name='e2e_interactie_learning_CUB'

#%%
print('number of cycles {}'.format(global_setting_CUB.n_cycles))
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
        prediction = np.squeeze(Prediction[:,idx_cls])
        label = np.squeeze(Label[:,idx_cls])
        mask = np.abs(label)==1
        if np.sum(label>0)==0:
            continue
        binary_label=np.clip(label[mask],0,1)
        ap[idx_cls]=average_precision_score(binary_label,prediction[mask])#AP(prediction,label,names)
    return ap
#%% label mapping function
def LoadLabelMap(attr_name_file, class_name_file):
    attr_name = []
    class_name = []
    with open(attr_name_file,"r") as f:
        lines=f.readlines()
        for line in lines:
            idx,name=line.rstrip('\n').split(' ')
            attr_name.append(name)
        
    with open(class_name_file,"r") as f:
        lines=f.readlines()
        for line in lines:
            idx,name=line.rstrip('\n').split(' ')
            class_name.append(name)
    return attr_name,class_name 
#%%
attr_name, class_name = LoadLabelMap(global_setting_CUB.attr_name_file, global_setting_CUB.class_name_file)
n_attr = len(attr_name)
n_class = len(class_name)
n_dim = n_attr+n_class
#%% Dataset
image_size = resnet_v1.resnet_v1_101.default_image_size
height = image_size
width = image_size
def parser(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'img': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string),
               'attribute':tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id = tf.decode_raw( parsed['img_id'],tf.int32)
    img = tf.reshape(tf.decode_raw( parsed['img'],tf.float32),[height, width, 3])
    Attribute = tf.reshape(tf.decode_raw( parsed['attribute'],tf.int32),[len(attr_name),-1])
    label = tf.decode_raw( parsed['label'],tf.int32)
    one_hot_label = tf.matmul(tf.one_hot(tf.squeeze(label), n_class,weight_class,-1,dtype=tf.int32)[:,tf.newaxis],tf.ones((1,tf.shape(Attribute)[1]),dtype=tf.int32))
    Attribute_c = tf.concat([Attribute,one_hot_label],axis=0)
    return img_id,img,label,Attribute_c
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
#%% load in memory
sess = tf.InteractiveSession()#tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
g = tf.get_default_graph()
#%%
Theta_trainable = tf.get_variable('Theta',shape=[2049,n_attr])
Theta = tf.concat([Theta_trainable,tf.zeros([2049,n_class])],axis = 1)
#%%
fraction_idx_var = tf.get_variable('fraction_idx_var',shape=(),dtype = tf.int32,trainable = False)
#%%
dataset = tf.data.TFRecordDataset(global_setting_CUB.train_img_path)
dataset = dataset.map(parser)
dataset = dataset.shuffle(20000)
dataset = dataset.batch(global_setting_CUB.batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
(img_ids,img,labels,Attributes) = iterator.get_next()
Attributes_f = Attributes[:,:,fraction_idx_var]
#in memory
num_dict_sample = global_setting_CUB.batch_size*n_neighbour#D_utility.count_records(global_setting_CUB.sparse_img_dict_path)
dataset_in_1 = tf.data.TFRecordDataset(global_setting_CUB.sparse_img_dict_path)
dataset_in_1 = dataset_in_1.map(parser).batch(50000)
(sparse_dict_img_id,sparse_dict_img,sparse_dict_label,sparse_dict_Attributes) = dataset_in_1.make_one_shot_iterator().get_next()
sparse_dict_img_id,sparse_dict_img,sparse_dict_label,sparse_dict_Attributes = sess.run([sparse_dict_img_id,sparse_dict_img,sparse_dict_label,sparse_dict_Attributes])

dataset_in_2 = tf.data.TFRecordDataset(global_setting_CUB.validation_img_path)
dataset_in_2 = dataset_in_2.map(parser).batch(50000)
(img_val_ids,val_img,val_labels,val_attributes) = dataset_in_2.make_one_shot_iterator().get_next()
(img_val_ids,val_img_v,val_labels,val_attributes)=sess.run([img_val_ids,val_img,val_labels,val_attributes])
val_attributes = val_attributes[:,:,0]

dataset_in_3 = tf.data.TFRecordDataset(global_setting_CUB.test_img_path)
dataset_in_3 = dataset_in_3.map(parser).batch(50000)
(img_test_ids,test_img,test_labels,test_attributes) = dataset_in_3.make_one_shot_iterator().get_next()
(img_test_ids,test_img_v,test_labels,test_attributes)=sess.run([img_test_ids,test_img,test_labels,test_attributes])
test_attributes = test_attributes[:,:,0]
#%%
#sparse_dict_img_id = tf.constant(sparse_dict_img_id)
#sparse_dict_img = tf.constant(sparse_dict_img)
#sparse_dict_label = tf.constant(sparse_dict_label)
#sparse_dict_Attributes = tf.constant(sparse_dict_Attributes)
#%%
image_size = resnet_v1.resnet_v1_101.default_image_size
height = image_size
width = image_size
img_input_ph = tf.placeholder(dtype=tf.float32,shape=[None,height,width,3])#tf.concat([img,sparse_dict_img],axis = 0,name='img_input_point')
#%%
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logit, end_points = resnet_v1.resnet_v1_101(img_input_ph, num_classes=1000, is_training=is_use_batch_norm,reuse=tf.AUTO_REUSE)
#    init_fn = slim.assign_from_checkpoint_fn(checkpoints_dir,slim.get_model_variables())
    features_concat = g.get_tensor_by_name('resnet_v1_101/pool5:0')

#%%
features_concat = tf.squeeze(features_concat)
features_concat = tf.concat([features_concat,tf.ones([tf.shape(features_concat)[0],1])],axis = 1,name='feature_input_point')
index_point = tf.placeholder(dtype=tf.int32,shape=())
F = features_concat[:index_point,:]
sparse_dict = features_concat[index_point:,:]
F_concat_ph = g.get_tensor_by_name('feature_input_point:0')
#%%
alpha_colaborative_var = tf.get_variable('alphha_colaborative',dtype=tf.float32,trainable=False, shape=())
alpha_colaborative_var_fh = tf.placeholder(dtype=tf.float32, shape=())

alpha_feature_var = tf.get_variable('alpha_feature',dtype=tf.float32,trainable=False, shape=())
alpha_feature_var_fh = tf.placeholder(dtype=tf.float32, shape=())

alpha_regularizer_var = tf.get_variable('alpha_regularizer',dtype=tf.float32,trainable=False, shape=())
alpha_regularizer_var_fh = tf.placeholder(dtype=tf.float32, shape=())

alpha_feature_norm_var = tf.get_variable('alpha_feature_norm',dtype=tf.float32,trainable=False, shape=())

#%%
op_alpha_colaborative_var = alpha_colaborative_var.assign(alpha_colaborative_var_fh)
op_alpha_feature_var = alpha_feature_var.assign(alpha_feature_var_fh)
op_alpha_regularizer = alpha_regularizer_var.assign(alpha_regularizer_var_fh)
#%%
G = np.eye(n_dim)
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
part_G_var = tf.eye(n_dim)#tf.scatter_nd(indices, G_var, shape)+diag_G
#%% disperse measurement
dispersion=tf.reduce_sum(tf.abs(part_G_var)) - tf.reduce_sum(tf.diag_part(tf.abs(part_G_var)))
#%%

attribute_f_ph = tf.placeholder(dtype=tf.float32, shape=(None,n_dim)) #Attributes[:,:,fraction_idx_var]
sparse_dict_Attributes_f_ph = tf.placeholder(dtype=tf.float32, shape=(None,n_dim))#sparse_dict_Attributes[:,:,fraction_idx_var]
size_minibatch = tf.cast(index_point,tf.float32)
with tf.variable_scope("sparse_coding_OMP"):
    A,P_L,P_F= colaborative_loss.e2e_OMP_asym_sigmoid_Feature_Graph(Theta,F,sparse_dict,attribute_f_ph,sparse_dict_Attributes_f_ph,
                                                         global_setting_CUB.k,part_G_var,alpha_colaborative_var,
                                                         alpha_feature_var,parallel_iterations,
                                                         c,global_setting_CUB.thresold_coeff,is_balance=False)
with tf.variable_scope("sparse_coding_colaborative_graph"):
    R_L,R_F=colaborative_loss.e2e_OMP_asym_sigmoid_loss_Feature_Graph(Theta,F,sparse_dict,attribute_f_ph,sparse_dict_Attributes_f_ph,A,P_L,P_F,part_G_var,parallel_iterations,c)
    loss_colaborative=tf.square(tf.norm(R_L))*1.0/size_minibatch
    
with tf.variable_scope("sparse_coding_feature"):
    loss_feature = tf.square(tf.norm(R_F))*1.0/size_minibatch

with tf.variable_scope("sparse_coding_norm_constraint"):
    feature_norm = tf.square(tf.norm(F))*1.0/size_minibatch
    
with tf.variable_scope("logistic"):
    logits = tf.matmul(F,Theta[:,:n_attr])
    labels_binary = tf.clip_by_value(attribute_f_ph,0,1)[:,:n_attr]
    labels_weight = tf.abs(tf.clip_by_value(attribute_f_ph,-1,1))[:,:n_attr]
    loss_logistic = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=logits,weights=labels_weight)

with tf.variable_scope("regularizer"):
    loss_regularizer = tf.square(tf.norm(Theta[:-1,:]))
#%%
tf.global_variables_initializer().run()
sess.run(iterator.initializer)
#%%
def append_info(m_AP,sum_num_miss_p,sum_num_miss_n,loss_value,loss_logistic_value,lr_v,norm_f,index):
    
    res_mAP[index]=m_AP
    res_loss[index] = loss_value
    res_loss_logistic[index]=loss_logistic_value
    res_lr[index]=lr_v
    res_sum_num_miss_p[index]=sum_num_miss_p
    res_sum_num_miss_n[index]=sum_num_miss_n
    res_norm_f[index]=norm_f
    
    df_result['mAP: colaborative {} feature {} regularizer {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer)]=res_mAP
    df_result['sum_num_miss_p: colaborative {} feature {} regularizer {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer)]=res_sum_num_miss_p
    df_result['sum_num_miss_n: colaborative {} feature {} regularizer {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer)]=res_sum_num_miss_n
    df_result['loss: colaborative {} feature {} regularizer {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer)]=res_loss
    df_result['logistic: colaborative {} feature {} regularizer {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer)]=res_loss_logistic
    df_result['lr: colaborative {} feature {} regularizer {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer)]=res_lr
    df_result['norm_f: colaborative {} feature {} regularizer {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer)]=res_norm_f
#%%
print('placeholder assignment')
#%%
Theta_trainable_fh = tf.placeholder(dtype=tf.float32, shape=[2049,n_attr])
op_assign_Theta_trainable = Theta_trainable.assign(Theta_trainable_fh)
if global_setting_CUB.lr_schedule == 'EMA':
    learning_rate_fh=tf.placeholder(dtype=tf.float32,shape=())
    op_assign_learning_rate = learning_rate.assign(learning_rate_fh)
#%% compute normalizer

trainable_vars = tf.trainable_variables()[1]#Theta_trainable#tf.trainable_variables()[:-3]
grad_logistic = tf.gradients(loss_logistic,trainable_vars)
grad_colaborative = tf.gradients(loss_colaborative,trainable_vars)
grad_feature = tf.gradients(loss_feature,trainable_vars)

norm_grad_logistic = tf.norm(grad_logistic)
norm_grad_colaborative = tf.norm(grad_colaborative)
norm_grad_feature = tf.norm(grad_feature)

ratio_loss = loss_logistic/loss_colaborative
Prediction = tf.matmul(features_concat,Theta)
#%%
optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
#%% loss
loss = loss_logistic
loss += alpha_colaborative_var*loss_colaborative
loss += alpha_regularizer_var*loss_regularizer
if is_feature_probagation:
    loss += alpha_feature_var*loss_feature
#%%
grad_vars = optimizer.compute_gradients(loss)
#%% computational graph
writer = tf.summary.FileWriter(logdir='./logdir', graph=tf.get_default_graph())
writer.close()
#%%
train = optimizer.apply_gradients(grad_vars,global_step=global_step)#optimizer.minimize(loss,var_list=[Theta,G_var])#
print('done placeholder assignment')
def experiment_cond_success():
    return True#(alpha_colaborative_o >0) or (alpha_colaborative_o + alpha_feature_o==0)

def experiment_cond_success_frac():
    return True#fraction>0.1

n_experiment= 0

for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            n_experiment += 1
print('Total number of experiment: {}'.format(n_experiment))

def test_set_evaluation():
    print('compute stats on test set')
    test_Prediction_val_l = []
    for idx_partition in range(test_img_v.shape[0]//partition_size+1):
        test_Prediction_partition_val = sess.run(Prediction,{img_input_ph:test_img_v[idx_partition*partition_size:(idx_partition+1)*partition_size,:,:,:]})
        test_Prediction_val_l.append(test_Prediction_partition_val)
    test_Prediction_truncate_v = np.concatenate(test_Prediction_val_l)[:,:n_attr]
    test_attributes_truncate = test_attributes[:,:n_attr]
    ap_test = compute_AP(test_Prediction_truncate_v,test_attributes_truncate,img_test_ids)
    m_AP_test=np.mean(ap_test)
    num_mis_p,num_mis_n=compute_number_misclassified(test_Prediction_truncate_v,test_attributes_truncate)
    sum_num_miss_p_test = np.sum(num_mis_p)
    sum_num_miss_n_test = np.sum(num_mis_n)
    
    if is_save:
        index=-1
        df_ap['index {}: ap'.format(index)]=ap_test
        append_info(m_AP_test,sum_num_miss_p_test,sum_num_miss_n_test,-1,-1,-1,-1,index)
        df_result.to_csv('./result/'+name+'/mAP.csv')
        ap_save_name = './result/'+name+'/ap_colaborative {} feature {} regularizer {}.csv'
        df_ap.to_csv(ap_save_name.format(alpha_colaborative,alpha_feature,alpha_regularizer))
#%%

for fraction_idx,fraction  in enumerate(fractions):
    if not experiment_cond_success_frac():
        print('skip')
        continue
    saver = tf.train.Saver()
    name = template_name.format(global_setting_CUB.learning_rate_base,global_setting_CUB.batch_size,global_setting_CUB.decay_rate_cond
                                               ,global_setting_CUB.signal_strength,global_setting_CUB.n_cycles,fraction,
                                               idx_GPU,global_setting_CUB.thresold_coeff,c,time.time())
    print('-'*30)
    print('fraction {}'.format(fraction))
    df_result = pd.DataFrame()
    data=np.load(global_setting_CUB.e2e_saturated_Thetas_model.format(fraction,fraction))
    init_Theta= data['Theta']#[:,:,fraction_idx]
    tf.global_variables_initializer().run()
    #%%
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        init_fn = slim.assign_from_checkpoint_fn(global_setting_CUB.e2e_checkpoints_dir.format(fraction,fraction),slim.get_model_variables())
    #%%
    sess.run(op_G_var)
    sess.run(op_assign_Theta_trainable,{Theta_trainable_fh:init_Theta})
    sess.run(iterator.initializer)
    init_fn(sess)
    sess.run(op_alpha_colaborative_var,{alpha_colaborative_var_fh:1e5})
    sess.run(op_alpha_feature_var,{alpha_feature_var_fh:1})
    sess.run(alpha_feature_norm_var.assign(0))
    sess.run(fraction_idx_var.assign(fraction_idx))
    
    sparse_dict_Attributes_f = sparse_dict_Attributes[:,:,fraction_idx]
    #%%
    img_ids_v,img_v,labels_v,Attributes_f_v=sess.run([img_ids,img,labels,Attributes_f])
    print('compute subset dict')
    _,sparse_dict_feature=compute_feature_prediction_large_batch(sparse_dict_img)
    _,mini_feature=compute_feature_prediction_large_batch(img_v)
    feature_concat = np.concatenate([mini_feature,sparse_dict_feature],axis=0)
    A_v=sess.run(A,{F_concat_ph:feature_concat,attribute_f_ph:Attributes_f_v,sparse_dict_Attributes_f_ph:sparse_dict_Attributes_f,index_point:img_v.shape[0]})
    A_v[A_v<0]==0
    index_dict = A_v.flatten()
    sub_sparse_dict_img = sparse_dict_img[index_dict,:]
    sub_sparse_dict_attr = sparse_dict_Attributes_f[index_dict,:]
    img_input = np.concatenate([img_v,sub_sparse_dict_img],axis=0)
    print('done')
    #
    print('compute thres_norm')
    norm_f = np.linalg.norm(sparse_dict_feature,ord='fro')**2*1/sparse_dict_img.shape[0]
    thres_norm_v = norm_f
    print('thres_norm {}'.format(thres_norm_v))
    #
    loss_colaborative_v,loss_feature_v,loss_logistic_v,norm_grad_logistic_v,norm_grad_colaborative_v,norm_grad_feature_v,feature_norm_v  = sess.run([loss_colaborative,loss_feature,loss_logistic,norm_grad_logistic,norm_grad_colaborative,norm_grad_feature,feature_norm],
                                             {img_input_ph:img_input,attribute_f_ph:Attributes_f_v,sparse_dict_Attributes_f_ph:sub_sparse_dict_attr,index_point:img_v.shape[0]})
    #
    raitio_colaborative_grad_v = norm_grad_logistic_v/norm_grad_colaborative_v if norm_grad_colaborative_v > 0 else 0 #loss_logistic_v/loss_colaborative_v#
    raitio_regularizer_grad_v=1
    raitio_featrue_grad_v = norm_grad_logistic_v/norm_grad_feature_v if norm_grad_feature_v > 0 else 0#raitio_colaborative_grad_v#
    raitio_norm_loss_v = 0#raitio_featrue_loss_v#np.abs(loss_logistic_v/loss_feature_constraint_v) if np.abs(loss_feature_constraint_v) > 0 else raitio_featrue_loss_v#loss_logistic_v/(norm_f*0.2)**2#
    print(raitio_colaborative_grad_v,raitio_regularizer_grad_v,raitio_featrue_grad_v,raitio_norm_loss_v)
    
    #%% create dir
    if not os.path.exists('./result/'+name) and is_save:
        os.makedirs('./result/'+name)
    #%%
    Thetas = np.zeros((2049,n_attr,n_experiment))
    Gs = np.zeros((n_dim,n_dim,n_experiment))
    idx_experiment = 0
    for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
        for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
            for idx_alpha_regularizer,alpha_regularizer_o in enumerate(global_setting_CUB.regularizers):
                
                if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                    print('skip')
                    continue
                
                report_length = global_setting_CUB.n_cycles*n_iters//global_setting_CUB.report_interval +1 #in case that my lousy computation is wrong
                print('report length {}'.format(report_length))
                res_mAP = np.zeros(report_length)
                res_loss = np.zeros(report_length)
                res_loss_logistic=np.zeros(report_length)
                res_sum_num_miss_p=np.zeros(report_length)
                res_sum_num_miss_n=np.zeros(report_length)
                res_grad_logistic=np.zeros(report_length)
                res_lr=np.zeros(report_length)
                res_norm_f=np.zeros(report_length)
                #
                
                alpha_colaborative = raitio_colaborative_grad_v*alpha_colaborative_o
                alpha_feature = raitio_featrue_grad_v*alpha_feature_o
                alpha_regularizer = raitio_regularizer_grad_v*alpha_regularizer_o
                alpha_feature_norm = raitio_norm_loss_v*0.5
                
                tf.global_variables_initializer().run()
                print('reset Theta')
                sess.run(iterator.initializer)
                sess.run(op_G_var)
                init_fn(sess)
                sess.run(op_assign_Theta_trainable,{Theta_trainable_fh:init_Theta})
                sess.run(op_alpha_colaborative_var,{alpha_colaborative_var_fh:alpha_colaborative})
                sess.run(op_alpha_feature_var,{alpha_feature_var_fh:alpha_feature})
                sess.run(op_alpha_regularizer,{alpha_regularizer_var_fh:alpha_regularizer})
                sess.run(alpha_feature_norm_var.assign(alpha_feature_norm))
                sess.run(fraction_idx_var.assign(fraction_idx))
                #exponential moving average
                expon_moving_avg_old = np.inf
                expon_moving_avg_new = 0
                #
                m = 0
                df_ap = pd.DataFrame()
                df_ap['label']=attr_name
                print('lambda colaborative: {} lambda_feature: {} regularizer: {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer))
                #%%
                for idx_cycle in range(global_setting_CUB.e2e_n_cycles):
                    index = (idx_cycle*n_iters)//global_setting_CUB.report_interval
                    tic = time.clock()
                    img_ids_v,img_v,labels_v,Attributes_f_v=sess.run([img_ids,img,labels,Attributes_f])
                    
                    print('compute subset dict')
                    _,sparse_dict_feature=compute_feature_prediction_large_batch(sparse_dict_img)
                    _,mini_feature=compute_feature_prediction_large_batch(img_v)
                    feature_concat = np.concatenate([mini_feature,sparse_dict_feature],axis=0)
                    norm_f = np.linalg.norm(feature_concat,ord='fro')**2*1/feature_concat.shape[0]
                    
                    A_v=sess.run(A,{F_concat_ph:feature_concat,attribute_f_ph:Attributes_f_v,sparse_dict_Attributes_f_ph:sparse_dict_Attributes_f,index_point:img_v.shape[0]})
                    A_v[A_v<0]==0
                    index_dict = A_v.flatten()
                    sub_sparse_dict_img = sparse_dict_img[index_dict,:]
                    sub_sparse_dict_attr = sparse_dict_Attributes_f[index_dict,:]
                    img_input = np.concatenate([img_v,sub_sparse_dict_img],axis=0)
                    print('done')
                    _,loss_value,logistic_v,lr_v,labels_binary_v,labels_weight_v  = sess.run([train,loss,loss_logistic,learning_rate,labels_binary,labels_weight],
                                                             {img_input_ph:img_input,attribute_f_ph:Attributes_f_v,sparse_dict_Attributes_f_ph:sub_sparse_dict_attr,index_point:img_v.shape[0]})
                    
                    print('Elapsed time udapte: {}'.format(time.clock()-tic))
                    print('Loss {} logistic {} norm_f {} lr {}'.format(loss_value,logistic_v,norm_f,lr_v))
                    if (idx_cycle*n_iters) % global_setting_CUB.report_interval == 0 :#or idx_iter == n_iters-1:
                        time_o = time.clock()
                        print('index {} -- compute mAP'.format(index))
                        print('{} alpha: colaborative {} feature {} regularizer {}'.format(name,alpha_colaborative,alpha_feature,alpha_regularizer))
                        
                        
                        validate_Prediction_val,_=compute_feature_prediction_large_batch(val_img_v)
                        validate_Prediction_val_truncate  = validate_Prediction_val[:,:n_attr]
                        val_attributes_truncate = val_attributes[:,:n_attr]
                        ap = compute_AP(validate_Prediction_val_truncate,val_attributes_truncate,img_val_ids)
                        num_mis_p,num_mis_n=compute_number_misclassified(validate_Prediction_val_truncate,val_attributes_truncate)
                        df_ap['index {}: ap'.format(index)]=ap
                        df_ap['index {}: num_mis_p'.format(index)]=num_mis_p
                        df_ap['index {}: num_mis_n'.format(index)]=num_mis_n
                        m_AP=np.mean(ap)
                        sum_num_miss_p = np.sum(num_mis_p)
                        sum_num_miss_n = np.sum(num_mis_n)
                        #exponential_moving_avg
                        if global_setting_CUB.lr_schedule == 'EMA':
                            expon_moving_avg_old=expon_moving_avg_new
                            expon_moving_avg_new = expon_moving_avg_new*(1-global_setting_CUB.signal_strength)+m_AP*global_setting_CUB.signal_strength
                            if expon_moving_avg_new<expon_moving_avg_old and learning_rate.eval() >= global_setting_CUB.limit_learning_rate and m <= 0:
                                print('Adjust learning rate')
                                sess.run(op_assign_learning_rate,{learning_rate_fh:learning_rate.eval()*global_setting_CUB.decay_rate_cond})
                                m = 2
                            m -= 1
                        append_info(m_AP,sum_num_miss_p,sum_num_miss_n,loss_value,logistic_v,lr_v,norm_f,index)
                        best_mAP = np.max(res_mAP[:-1])
                        print('mAP {} sum_num_miss_p {} sum_num_miss_n {} best_mAP {}'.format(m_AP,sum_num_miss_p,sum_num_miss_n,best_mAP))
                        if is_save:
                            df_result.to_csv('./result/'+name+'/mAP.csv')
                            ap_save_name = './result/'+name+'/ap_colaborative {} feature {} regularizer {}.csv'
                            df_ap.to_csv(ap_save_name.format(alpha_colaborative,alpha_feature,alpha_regularizer))
                            if not global_setting_CUB.early_stopping or (global_setting_CUB.early_stopping and m_AP >= best_mAP):
                                Thetas[:,:,idx_experiment]=Theta_trainable.eval()
                                Gs[:,:,idx_experiment]=part_G_var.eval()
                                print('-'*30)
                                print('Save best')
                                np.savez('./result/'+name+"/model", Thetas=Thetas, Gs=Gs)
                                model_name = 'model_frac_{}_{}.ckpt'.format(fraction,alpha_regularizer_o)
                                saver.save(sess, './result/'+name+"/"+model_name)
                                test_set_evaluation()
                idx_experiment+=1
#%%
sess.close()
tf.reset_default_graph()