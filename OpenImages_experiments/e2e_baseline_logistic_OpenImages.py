# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 18:23:01 2018

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
#%% override
global_setting_OpenImage.batch_size=32
global_setting_OpenImage.report_interval = 100
global_setting_OpenImage.e2e_n_cycles = 3657355//global_setting_OpenImage.batch_size
#%% data flag
#
is_G = True
is_nonzero_G = True
is_constrant_G = False
is_sum_1=True
is_optimize_all_G = True
#
#is_use_batch_norm = True
capacity = 20
dictionary_evaluation_interval=100
n_size_memory = 500
partition_size = 200
strength_identity = 1
idx_GPU=0
train_data_path= '/home/project_amadeus/mnt/cygnus/train/'
validation_data_path= '/home/project_amadeus/mnt/cygnus/validation/'
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
c = 2.0
template_name='blackbox_baseline_OpenImage_{}_{}_{}_signal_str_{}_{}_GPU_{}_thresCoeff_{}_c_{}_stamp_{}'
name = template_name.format(global_setting_OpenImage.e2e_learning_rate_base,global_setting_OpenImage.batch_size,global_setting_OpenImage.decay_rate_cond
                                               ,global_setting_OpenImage.signal_strength,global_setting_OpenImage.n_cycles,
                                               idx_GPU,global_setting_OpenImage.thresold_coeff,c,time.time())
list_alphas_colaborative = [1]#
list_alphas_feature = [1]#0,0.5,1,2
global_step = tf.Variable(0, trainable=False,dtype=tf.float32)
learning_rate = tf.Variable(global_setting_OpenImage.e2e_learning_rate_base,trainable = False,dtype=tf.float32)
n_iters = 1
decay_rate_schedule = 1
schedule_wrt_report_interval = 80
report_length = global_setting_OpenImage.e2e_n_cycles*n_iters//global_setting_OpenImage.report_interval +1 #in case that my lousy computation is wrong
patient=report_length//100

is_save = False
parallel_iterations = 1
#%%
print('number of cycles {}'.format(global_setting_OpenImage.n_cycles))
print('number n_size_memory ',n_size_memory)
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
def compute_feature_prediction_large_batch(img):
    prediction_l = []
    feature_l = []
    for idx_partition in range(img.shape[0]//partition_size+1):
        print('{}.'.format(idx_partition),end='')
        prediction_partition,feature_partition = sess.run([Prediction,features_concat],{img_input_ph:img[idx_partition*partition_size:(idx_partition+1)*partition_size]})
        prediction_l.append(prediction_partition)
        feature_l.append(feature_partition)
    print()
    prediction = np.concatenate(prediction_l)
    feature = np.concatenate(feature_l)
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
dataset = tf.data.TFRecordDataset(global_setting_OpenImage.record_path)
dataset = dataset.map(parser_train)
dataset = dataset.shuffle(2000)
dataset = dataset.batch(global_setting_OpenImage.batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
(img_ids,img,labels) = iterator.get_next()

#in memory
dataset_in_1 = tf.data.TFRecordDataset(global_setting_OpenImage.sparse_dict_path)
dataset_in_1 = dataset_in_1.map(parser_train).batch(partition_size).repeat()
sparse_dict_iterator_next = dataset_in_1.make_one_shot_iterator().get_next()
#sparse_dict_img_id,sparse_dict_img,sparse_dict_labels = sess.run([sparse_dict_img_id,sparse_dict_img,sparse_dict_labels])

dataset_in_2 = tf.data.TFRecordDataset(global_setting_OpenImage.validation_path)
dataset_in_2 = dataset_in_2.map(parser_validation).batch(partition_size).repeat()
val_iterator_next = dataset_in_2.make_one_shot_iterator().get_next()
#(img_val_ids,val_img_v,val_labels)=sess.run([img_val_ids,val_img,val_labels])
#%%
n_val = D_utility.count_records(global_setting_OpenImage.validation_path)
img_val_ids,val_imgs_v,val_labels = load_memory(val_iterator_next,n_val,capacity)
#%%
#with slim.arg_scope(resnet_v1.resnet_arg_scope()):
saver = tf.train.import_meta_graph('./model/resnet/oidv2-resnet_v1_101.ckpt.meta')
img_input_ph = g.get_tensor_by_name('input_values:0')
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
#%%
op_alpha_colaborative_var = alpha_colaborative_var.assign(alpha_colaborative_var_fh)
op_alpha_feature_var = alpha_feature_var.assign(alpha_feature_var_fh)
op_alpha_regularizer = alpha_regularizer_var.assign(alpha_regularizer_var_fh)
#%%
G = np.load(global_setting_OpenImage.label_graph_path).astype(np.float32)
#G = np.load(global_setting_OpenImage.label_graph_path)['label_graph'].astype(np.float32)
if is_sum_1:
    G = D_utility.preprocessing_graph(G)
else:
    np.fill_diagonal(G,strength_identity)

G_empty_diag = G - np.diag(np.diag(G))
if is_optimize_all_G:
    G_init=G[G!=0]
else:
    G_init=G_empty_diag[G_empty_diag!=0]
    
G_var = tf.get_variable("G_var", G_init.shape)
op_G_var=G_var.assign(G_init)
op_G_nonnegative = G_var.assign(tf.clip_by_value(G_var,0,1))
op_G_constraint = G_var.assign(tf.clip_by_value(G_var,-1,0.5))
indices = []
counter = 0
diag_G = tf.diag(np.diag(G))
#pdb.set_trace()
for idx_row in range(G_empty_diag.shape[1]):
    if is_optimize_all_G:
        idx_cols = np.where(G[idx_row,:]!=0)[0]
    else:
        idx_cols = np.where(G_empty_diag[idx_row,:]!=0)[0]
    for idx_col in idx_cols:
        if G[idx_row,idx_col]-G_init[counter] != 0:
            raise Exception('error relation construction')
        indices.append([idx_row,idx_col])
        counter += 1
if is_G:
    if is_optimize_all_G:
        part_G_var = tf.scatter_nd(indices, G_var, G.shape)
    else:
        part_G_var = diag_G+tf.scatter_nd(indices, G_var, G.shape)#tf.eye(5000) #
else:
    part_G_var = tf.eye(5000)
#%% disperse measurement
dispersion_diag = tf.reduce_sum(tf.diag_part(tf.abs(part_G_var)))
dispersion=tf.reduce_sum(tf.abs(part_G_var))-dispersion_diag
dispersion_neg = tf.reduce_sum(tf.abs(tf.clip_by_value(part_G_var,-10,0)))
#%%

labels_ph = tf.placeholder(dtype=tf.float32, shape=(None,n_class)) #Attributes[:,:,fraction_idx_var]
with tf.variable_scope("logistic"):
    logits = tf.matmul(F,Theta)
    labels_binary = tf.clip_by_value(labels_ph,0,1)
    labels_weight = tf.abs(tf.clip_by_value(labels_ph,-1,1))
    loss_logistic = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=logits,weights=labels_weight)

with tf.variable_scope("regularizer"):
    loss_regularizer = tf.square(tf.norm(Theta[:-1,:]))
#%%
tf.global_variables_initializer().run()
sess.run(iterator.initializer)
#%%
def append_info(m_AP,loss_value,loss_logistic_value,lr_v,norm_f):
    
    res_mAP[index]=m_AP
    res_loss[index] = loss_value
    res_loss_logistic[index]=loss_logistic_value
    res_lr[index]=lr_v
    res_norm_f[index]=norm_f
    
    df_result['mAP: '+extension]=res_mAP
    df_result['loss: '+extension]=res_loss
    df_result['logistic: '+extension]=res_loss_logistic
    df_result['lr: '+extension]=res_lr
    df_result['norm_f: '+extension]=res_norm_f
#%%
print('placeholder assignment')
#%%
Theta_fh = tf.placeholder(dtype=tf.float32, shape=[2049,n_class])
op_assign_Theta = Theta.assign(Theta_fh)

#%% compute normalizer
trainable_vars = Theta#tf.trainable_variables()[:-3]
grad_logistic = tf.gradients(loss_logistic,trainable_vars)
grad_regularizer = tf.gradients(loss_regularizer,trainable_vars)
#grad_feature = tf.gradients(loss_feature,trainable_vars)

norm_grad_logistic = tf.norm(grad_logistic)
norm_grad_regularizer = tf.norm(grad_regularizer)
Prediction = tf.matmul(features_concat,Theta)

#%%
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)#tf.train.AdamOptimizer(learning_rate=0.001),momentum=0.9
optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )

optimizer_G = tf.train.RMSPropOptimizer(
      learning_rate*500,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
loss = loss_logistic
#loss += alpha_regularizer_var*loss_regularizer
grad_var_all = optimizer.compute_gradients(loss)
Theta_grads = grad_var_all[:-1]
G_grads = [grad_var_all[-1]]
train = optimizer.apply_gradients(Theta_grads)
#train_G = optimizer_G.apply_gradients(G_grads)
reset_optimizer_op = tf.variables_initializer(optimizer.variables())
#reset_optimizer_G_op = tf.variables_initializer(optimizer_G.variables())
#%% computational graph
writer = tf.summary.FileWriter(logdir='./logdir', graph=tf.get_default_graph())
writer.close()
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

data=np.load(global_setting_OpenImage.saturated_Thetas_model)
init_Theta = data['Thetas'][:,:,0]
tf.global_variables_initializer().run()
#%%
#sess.run(op_G_var)
sess.run(op_assign_Theta,{Theta_fh:init_Theta})
sess.run(iterator.initializer)
saver.restore(sess, global_setting_OpenImage.model_path)
sess.run(op_alpha_colaborative_var,{alpha_colaborative_var_fh:1e5})
sess.run(op_alpha_feature_var,{alpha_feature_var_fh:1})
#%%
#pdb.set_trace()
img_ids_v,img_v,labels_v=sess.run([img_ids,img,labels])
loss_logistic_v,norm_grad_logistic_v,norm_grad_regularizer_v  = sess.run([loss_logistic,norm_grad_logistic,norm_grad_regularizer],
                                         {img_input_ph:img_v,labels_ph:labels_v,index_point:img_v.shape[0]})
raitio_regularizer_grad_v = norm_grad_logistic_v/norm_grad_regularizer_v if norm_grad_regularizer_v > 0 else 0
#%%
# absolute regularization
raitio_regularizer_grad_v=1
#
print(raitio_regularizer_grad_v)


#%% create dir
if not os.path.exists('./result/'+name) and is_save:
    os.makedirs('./result/'+name)
#%%
Thetas = np.zeros((2049,n_class,n_experiment))
Gs = np.zeros((n_class,n_class,n_experiment))
idx_experiment = 0
for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):
            
            
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            
            print('report length {}'.format(report_length))
            res_mAP = np.zeros(report_length)
            res_loss = np.zeros(report_length)
            res_loss_logistic=np.zeros(report_length)
            res_grad_logistic=np.zeros(report_length)
            res_lr=np.zeros(report_length)
            res_norm_f=np.zeros(report_length)
            #
            
            alpha_regularizer = raitio_regularizer_grad_v*alpha_regularizer_o
            
            tf.global_variables_initializer().run()
            print('reset Theta')
            sess.run(iterator.initializer)
            saver.restore(sess, global_setting_OpenImage.model_path)
            sess.run(op_assign_Theta,{Theta_fh:init_Theta})
            sess.run(op_alpha_regularizer,{alpha_regularizer_var_fh:alpha_regularizer})
            extension = 'regularizer {}'.format(alpha_regularizer)
    
            #exponential moving average
            expon_moving_avg_old = np.inf
            expon_moving_avg_new = 0
            #
            m = 0
            df_ap = pd.DataFrame()
            df_ap['label']=list_label
            print('regularizer: {}'.format(alpha_regularizer))
            n_nan = 0
            #%%
            validate_Prediction_val,_=compute_feature_prediction_large_batch(val_imgs_v)
            ap = compute_AP(validate_Prediction_val,val_labels)
            m_AP=np.mean(ap)
#            pdb.set_trace()
            print('Initialize mAP: ',m_AP)
            tic = time.clock()
            for idx_cycle in range(global_setting_OpenImage.e2e_n_cycles):
                index = (idx_cycle*n_iters)//global_setting_OpenImage.report_interval
                img_ids_v,img_v,labels_v=sess.run([img_ids,img,labels])
                
                _,loss_value,logistic_v,lr_v  = sess.run([train,loss,loss_logistic,learning_rate],
                                                         {img_input_ph:img_v,labels_ph:labels_v,index_point:img_v.shape[0]})
                
                print('.',end='')
                if (idx_cycle*n_iters) % global_setting_OpenImage.report_interval == 0 :#or idx_iter == n_iters-1:
#                    pdb.set_trace()
                    loss_regularizer_v = tf.norm(Theta).eval()
                    print('Elapsed time udapte: {}'.format(time.clock()-tic))
                    tic = time.clock()
                    print('index {} -- compute mAP'.format(index))
                    print('{} alpha:regularizer {}'.format(name,alpha_regularizer))
#                    validate_Prediction_val,_,_,val_labels=compute_feature_prediction_large_batch_iterator(val_iterator_next,n_val)
                    validate_Prediction_val,validate_feature_v=compute_feature_prediction_large_batch(val_imgs_v)
                    norm_f = np.linalg.norm(validate_feature_v,ord='fro')/validate_feature_v.shape[0]
                    ap = compute_AP(validate_Prediction_val,val_labels)
                    num_mis_p,num_mis_n=compute_number_misclassified(validate_Prediction_val,val_labels)
                    df_ap['index {}: ap'.format(index)]=ap
                    df_ap['index {}: num_mis_p'.format(index)]=num_mis_p
                    df_ap['index {}: num_mis_n'.format(index)]=num_mis_n
                    m_AP=np.mean(ap)
                    sum_num_miss_p = np.sum(num_mis_p)
                    sum_num_miss_n = np.sum(num_mis_n)
                    #exponential_moving_avg
                    expon_moving_avg_old=expon_moving_avg_new
                    expon_moving_avg_new = expon_moving_avg_new*(1-global_setting_OpenImage.signal_strength)+m_AP*global_setting_OpenImage.signal_strength
                    if expon_moving_avg_new<expon_moving_avg_old and learning_rate.eval() >= global_setting_OpenImage.e2e_limit_learning_rate and m <= 0:
                        print('Adjust learning rate')
                        sess.run(op_assign_learning_rate,{learning_rate_fh:learning_rate.eval()*global_setting_OpenImage.decay_rate_cond})
                        m = patient
                    m -= 1
                    append_info(m_AP,loss_value,logistic_v,lr_v,norm_f)
                    print('mAP {} sum_num_miss_p {} sum_num_miss_n {}'.format(m_AP,sum_num_miss_p,sum_num_miss_n))
                    print('Loss {} logistic {} regularizer {} lr {} norm_f {}'.format(loss_value,logistic_v,loss_regularizer_v,lr_v,norm_f))
                    if is_save:
                        Thetas[:,:,idx_experiment]=Theta.eval()
                        Gs[:,:,idx_experiment]=part_G_var.eval()
                        df_result.to_csv('./result/'+name+'/mAP.csv')
                        ap_save_name = './result/'+name+'/regularizer {}.csv'
                        df_ap.to_csv(ap_save_name.format(alpha_regularizer))
#                        if index%(int(report_length/4)) == 0:
#                            np.savez('./result/'+name, Thetas=Thetas, Gs=Gs)
#                        if global_setting_OpenImage.early_stopping and m_AP > np.max(res_mAP):
                        np.savez('./result/'+name+"/model", Thetas=Thetas, Gs=Gs)
                        model_name = 'model_'+extension+'.ckpt'
                        saver.save(sess, './result/'+name+"/"+model_name)
                
            idx_experiment+=1
#%%
sess.close()
tf.reset_default_graph()