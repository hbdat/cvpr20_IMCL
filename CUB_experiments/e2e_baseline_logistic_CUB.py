# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:10:30 2018

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
global_setting_CUB.e2e_n_cycles*=2
global_setting_CUB.batch_size=32
global_setting_CUB.learning_rate_base = 0.01
#%% data flag
print('V for Victory')
idx_GPU=1
is_use_batch_norm = False
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
template_name='e2e_baseline_logistic_CUB'
fractions = global_setting_CUB.fractions
global_step = tf.Variable(0, trainable=False,dtype=tf.float32)
learning_rate = tf.Variable(global_setting_CUB.learning_rate_base,trainable = False,dtype=tf.float32)

n_iters = 1
decay_rate_schedule = 1
schedule_wrt_report_interval = 80
c = 2.0
checkpoints_dir = './model/resnet_CUB/resnet_v1_101.ckpt'
is_save = True
parallel_iterations = 1
partition_size = 200
#%%
print('number of cycles {}'.format(global_setting_CUB.e2e_n_cycles))
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
    return img_id,img,label,Attribute
#%%
def construct_dictionary(batch_attribute,batch_id,sparse_dict_Attribute_f,sparse_dict_img,n_neighbour):
    similar_score=np.matmul(batch_attribute,np.transpose(sparse_dict_Attribute_f))
    m_similar_index=np.argsort(similar_score,axis=1)[:,0:n_neighbour]
    index_dict = m_similar_index.flatten()
    return sparse_dict_Attribute_f[index_dict,:],sparse_dict_img[index_dict,:,:]
#%% load in memory
sess = tf.InteractiveSession()#tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
g = tf.get_default_graph()
#%%
Theta = tf.get_variable('Theta',shape=[2049,n_attr])
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

img_input_ph = tf.placeholder(dtype=tf.float32,shape=[None,height,width,3])#tf.concat([img,sparse_dict_img],axis = 0,name='img_input_point')
#%%
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logit, end_points = resnet_v1.resnet_v1_101(img_input_ph, num_classes=1000, is_training=is_use_batch_norm,reuse=tf.AUTO_REUSE)
    init_fn = slim.assign_from_checkpoint_fn(checkpoints_dir,slim.get_model_variables())
    features_concat = g.get_tensor_by_name('resnet_v1_101/pool5:0')

#%%
features_concat = tf.squeeze(features_concat)
features_concat = tf.concat([features_concat,tf.ones([tf.shape(features_concat)[0],1])],axis = 1,name='feature_input_point')
#%%
alpha_regularizer_var = tf.get_variable('alpha_regularizer',dtype=tf.float32,trainable=False, shape=())
alpha_regularizer_var_fh = tf.placeholder(dtype=tf.float32, shape=())
#%%
op_alpha_regularizer = alpha_regularizer_var.assign(alpha_regularizer_var_fh)
#%%
G = np.eye(n_attr)
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
part_G_var = tf.eye(n_attr)#tf.scatter_nd(indices, G_var, shape)+diag_G
#%% disperse measurement
dispersion=tf.reduce_sum(tf.abs(part_G_var)) - tf.reduce_sum(tf.diag_part(tf.abs(part_G_var)))
#%%

attribute_f_ph = tf.placeholder(dtype=tf.float32, shape=(None,n_attr)) #Attributes[:,:,fraction_idx_var]
    
with tf.variable_scope("logistic"):
    logits = tf.matmul(features_concat,Theta)
    labels_binary = tf.div(attribute_f_ph+1,2)
    labels_weight = tf.abs(attribute_f_ph)
    loss_logistic = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=logits,weights=labels_weight)

with tf.variable_scope("regularizer"):
    loss_regularizer = tf.square(tf.norm(Theta[:-1,:]))
#%% shared operation
grad_logistic = tf.gradients(loss_logistic, Theta)
grad_regularizer = tf.gradients(loss_regularizer,Theta)

norm_grad_logistic = tf.norm(grad_logistic)
norm_grad_regularizer = tf.norm(grad_regularizer)
norm_Theta = tf.norm(Theta)
raitio_regularizer_grad = norm_grad_logistic/norm_grad_regularizer

Prediction = tf.matmul(features_concat,Theta)
#%%
tf.global_variables_initializer().run()
sess.run(iterator.initializer)
#%%
def append_info(m_AP,sum_num_miss_p,sum_num_miss_n,loss_value,loss_logistic_value,lr_v,dispersion_eval,index):
    
    res_mAP[index]=m_AP
    res_loss[index] = loss_value
    res_disperseion[index] = dispersion_eval
    res_loss_logistic[index]=loss_logistic_value
    res_lr[index]=lr_v
    res_sum_num_miss_p[index]=sum_num_miss_p
    res_sum_num_miss_n[index]=sum_num_miss_n
    df_result['mAP']=res_mAP
    df_result['sum_num_miss_p']=res_sum_num_miss_p
    df_result['sum_num_miss_n']=res_sum_num_miss_n
    df_result['loss']=res_loss
    df_result['logistic']=res_loss_logistic
    df_result['dispersion']=res_disperseion
    df_result['lr']=res_lr
    
#%%
print('placeholder assignment')
#%%
Theta_fh = tf.placeholder(dtype=tf.float32, shape=[2049,n_attr])
op_assign_Theta = Theta.assign(Theta_fh)
learning_rate_fh=tf.placeholder(dtype=tf.float32,shape=())
op_assign_learning_rate = learning_rate.assign(learning_rate_fh)

#%%
optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
loss = loss_logistic
loss += alpha_regularizer_var*loss_regularizer
#%%
grad_vars = optimizer.compute_gradients(loss)
writer = tf.summary.FileWriter(logdir='./logdir', graph=tf.get_default_graph())
writer.close()
#%%
train = optimizer.apply_gradients(grad_vars,global_step = global_step)#optimizer.minimize(loss,var_list=[Theta,G_var])#
print('done placeholder assignment')
#%%
name = template_name.format(global_setting_CUB.learning_rate_base,global_setting_CUB.batch_size,global_setting_CUB.decay_rate_cond
                                               ,global_setting_CUB.signal_strength,global_setting_CUB.e2e_n_cycles,
                                               idx_GPU,global_setting_CUB.thresold_coeff,c,time.time())
if not os.path.exists('./result/'+name) and is_save:
        os.makedirs('./result/'+name)
        
def test_set_evaluation():
    print('compute stats on test set')
    test_Prediction_val_l = []
    for idx_partition in range(test_img_v.shape[0]//partition_size+1):
        test_Prediction_partition_val = sess.run(Prediction,{img_input_ph:test_img_v[idx_partition*partition_size:(idx_partition+1)*partition_size,:,:,:]})
        test_Prediction_val_l.append(test_Prediction_partition_val)
    test_Prediction_val = np.concatenate(test_Prediction_val_l)
    ap_test = compute_AP(test_Prediction_val,test_attributes,img_test_ids)
    m_AP_test=np.mean(ap_test)
    num_mis_p,num_mis_n=compute_number_misclassified(test_Prediction_val,test_attributes)
    sum_num_miss_p_test = np.sum(num_mis_p)
    sum_num_miss_n_test = np.sum(num_mis_n)
    
    if is_save:
        index=-1
        df_ap['index {}: ap'.format(index)]=ap_test
        append_info(m_AP_test,sum_num_miss_p_test,sum_num_miss_n_test,-1,-1,-1,-1,index)
        df_result.to_csv(path_fraction+'/mAP_{}_{}.csv'.format(fraction,alpha_regularizer_o))
        ap_save_name = path_fraction+'/ap_baseline_{}_{}.csv'.format(fraction,alpha_regularizer_o)
        df_ap.to_csv(ap_save_name)
#%%
for fraction_idx,fraction  in enumerate(fractions):
    saver = tf.train.Saver()
    path_fraction = './result/'+name+'/fraction_'+str(fraction)
    if  is_save:
        os.makedirs(path_fraction)
    print('-'*30)
    print('fraction {}'.format(fraction))
    df_result = pd.DataFrame()
    
    tf.global_variables_initializer().run()
    #%%
    sess.run(op_G_var)
    sess.run(iterator.initializer)
    init_fn(sess)
    sess.run(fraction_idx_var.assign(fraction_idx))
    
    #%%
    
    idx_experiment = 0
    for idx_alpha_regularizer,alpha_regularizer_o in enumerate(global_setting_CUB.regularizers):
                
        report_length = global_setting_CUB.e2e_n_cycles*n_iters//global_setting_CUB.report_interval +2 #in case that my lousy computation is wrong
        print('report length {}'.format(report_length))
        res_mAP = np.zeros(report_length)
        res_loss = np.zeros(report_length)
        res_loss_logistic=np.zeros(report_length)
        res_sum_num_miss_p=np.zeros(report_length)
        res_sum_num_miss_n=np.zeros(report_length)
        res_grad_logistic=np.zeros(report_length)
        res_lr=np.zeros(report_length)
        res_disperseion=np.zeros(report_length)
        #
        
        tf.global_variables_initializer().run()
        print('reset Theta')
        sess.run(iterator.initializer)
        sess.run(op_G_var)
        init_fn(sess)
        sess.run(fraction_idx_var.assign(fraction_idx))
        sess.run(op_alpha_regularizer,{alpha_regularizer_var_fh:alpha_regularizer_o})
        #exponential moving average
        expon_moving_avg_old = np.inf
        expon_moving_avg_new = 0
        #
        m = 0
        df_ap = pd.DataFrame()
        df_ap['label']=attr_name
        #%%
        for idx_cycle in range(global_setting_CUB.e2e_n_cycles):
            index = (idx_cycle*n_iters)//global_setting_CUB.report_interval
            tic = time.clock()
            img_ids_v,img_v,labels_v,Attributes_f_v=sess.run([img_ids,img,labels,Attributes_f])
            
            _,loss_value,logistic_v,lr_v  = sess.run([train,loss,loss_logistic,learning_rate],
                                                     {img_input_ph:img_v,attribute_f_ph:Attributes_f_v})
            print('Elapsed time udapte: {}'.format(time.clock()-tic))
            print('Loss {} logistic {} lr {}'.format(loss_value,logistic_v,lr_v))
            if (idx_cycle*n_iters) % global_setting_CUB.report_interval == 0 :#or idx_iter == n_iters-1:
                time_o = time.clock()
                print('index {} -- compute mAP'.format(index))
                validate_Prediction_val_l = []
                
                for idx_partition in range(val_img_v.shape[0]//partition_size+1):
                    validate_Prediction_partition_val = sess.run(Prediction,{img_input_ph:val_img_v[idx_partition*partition_size:(idx_partition+1)*partition_size,:,:,:]})
                    validate_Prediction_val_l.append(validate_Prediction_partition_val)
                validate_Prediction_val = np.concatenate(validate_Prediction_val_l)
                
                dispersion_eval = dispersion.eval()
                ap = compute_AP(validate_Prediction_val,val_attributes,img_val_ids)
                num_mis_p,num_mis_n=compute_number_misclassified(validate_Prediction_val,val_attributes)
                df_ap['index {}: ap'.format(index)]=ap
                df_ap['index {}: num_mis_p'.format(index)]=num_mis_p
                df_ap['index {}: num_mis_n'.format(index)]=num_mis_n
                m_AP=np.mean(ap)
                sum_num_miss_p = np.sum(num_mis_p)
                sum_num_miss_n = np.sum(num_mis_n)
                #exponential_moving_avg
                expon_moving_avg_old=expon_moving_avg_new
                expon_moving_avg_new = expon_moving_avg_new*(1-global_setting_CUB.signal_strength)+m_AP*global_setting_CUB.signal_strength
                if expon_moving_avg_new<expon_moving_avg_old and learning_rate.eval() >= global_setting_CUB.limit_learning_rate and m <= 0:
                    print('Adjust learning rate')
                    sess.run(op_assign_learning_rate,{learning_rate_fh:learning_rate.eval()*global_setting_CUB.decay_rate_cond})
                    m = 2
                m -= 1
                best_map = np.max(res_mAP[:-1])
                append_info(m_AP,sum_num_miss_p,sum_num_miss_n,loss_value,logistic_v,lr_v,dispersion_eval,index)
                print('mAP {} sum_num_miss_p {} sum_num_miss_n {} dispersion {} regularizer {} best map {}'.format(m_AP,sum_num_miss_p,sum_num_miss_n,dispersion_eval,alpha_regularizer_o,best_map))
                
                
                
                if is_save:
                    Theta_v=Theta.eval()
                    G_v=part_G_var.eval()
                    df_result.to_csv(path_fraction+'/mAP_{}_{}.csv'.format(fraction,alpha_regularizer_o))
                    ap_save_name = path_fraction+'/ap_baseline_{}_{}.csv'.format(fraction,alpha_regularizer_o)
                    df_ap.to_csv(ap_save_name)
                    if global_setting_CUB.early_stopping and m_AP >= best_map:
                        test_set_evaluation()
                        print('-'*30)
                        print('Save best')
                        np.savez(path_fraction+"/model", Theta=Theta_v, G=G_v)
                        model_name = '/model_frac_{}_{}.ckpt'.format(fraction,alpha_regularizer_o)
                        saver.save(sess,path_fraction+model_name)
        
        
#%%
sess.close()
tf.reset_default_graph()
