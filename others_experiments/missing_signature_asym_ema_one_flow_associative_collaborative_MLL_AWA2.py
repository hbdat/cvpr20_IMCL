# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:11:14 2018

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
from measurement import apk,compute_number_misclassified,confusion_matrix
import colaborative_loss
import global_setting_AWA2
from D_utility import DAP,zeroshot_evaluation,LoadLabelMap,preprocessing_graph,signature_completion,evaluate_completion
import pdb
from global_setting_AWA2 import fractions,list_alphas_colaborative,list_alphas_feature
#%% logging level
#global_setting_AWA2.n_cycles//=3
global_setting_AWA2.learning_rate_base=0.01
global_setting_AWA2.batch_size=32
#%%
def experiment_cond_success_frac():
    return fraction!=0.2#
def experiment_cond_success():
    index = idx_alpha_colaborative*len(list_alphas_colaborative)+idx_alpha_feature*len(list_alphas_feature)+idx_alpha_regularizer*len(global_setting_AWA2.regularizers)
    return True#((alpha_colaborative_o >= 0.1) or (alpha_colaborative_o + alpha_feature_o==0))

n_experiment= 0

for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate(global_setting_AWA2.regularizers):
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            n_experiment += 1
print('Total number of experiment: {}'.format(n_experiment))
#%% data flag
idx_GPU=1
is_balance = True
if is_balance:
    global_setting_AWA2.saturated_Thetas_model='./result/missing_signature_balance_1_inv_time__AWA2_baseline_0.01_1000_0.8_signal_str_0.5_1000_1_GPU_1_thresCoeff_1e-06_stamp_1536002989.4956863/model_signatures.npz'
else:
    global_setting_AWA2.saturated_Thetas_model='./result/none/model_signatures.npz'

os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
name_template='AWA2_missing_signature_balance_'+str(is_balance)+'_{}_{}_{}_signal_str_{}_{}_frac_{}_GPU_{}_thresCoeff_{}_c_{}_stamp_{}'
df_image = pd.read_csv('./data/2017_11/train/images.csv')

global_step = tf.Variable(0, trainable=False,dtype=tf.float32)
if global_setting_AWA2.lr_schedule == 'EMA':
    learning_rate = tf.Variable(global_setting_AWA2.learning_rate_base,trainable = False,dtype=tf.float32)
else:
    learning_rate = 1.0/(tf.sqrt(global_step)+1.0)*global_setting_AWA2.learning_rate_base
n_iters = 1
decay_rate_schedule = 1
schedule_wrt_report_interval = 80
c = 2.0
quantization = False

is_save = True
parallel_iterations = 1
#%%
print('number of cycles {}'.format(global_setting_AWA2.n_cycles))

#%%
data = np.load(global_setting_AWA2.mask_signature_path)
Missing_signature_q=data['Missing_signature_q']
signature_q = Missing_signature_q[:,:,-1]
stat_label_pos = np.sum(Missing_signature_q==1,axis = 0)
stat_label_neg = np.sum(Missing_signature_q==-1,axis = 0)
ratio_balance_pos_neg = np.clip(np.divide(stat_label_neg,stat_label_pos),0,10)
n_class,n_attr,_ = Missing_signature_q.shape
#casting to numpy
with tf.device('/cpu:0'):
    Missing_signature_q = tf.cast(Missing_signature_q,tf.float32)
ratio_balance_pos_neg = tf.cast(ratio_balance_pos_neg,tf.float32)

#%% Dataset
def parser(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'feature': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string),
               'attribute':tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id = tf.decode_raw( parsed['img_id'],tf.int64)
    feature = tf.decode_raw( parsed['feature'],tf.float32)
    label = tf.squeeze(tf.decode_raw(parsed['label'],tf.int32))
    Attribute = tf.reshape(Missing_signature_q[label,:,:],[n_attr,-1])    
    return img_id,feature,label,Attribute
#%% load in memory
sess = tf.InteractiveSession()#tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
g = tf.get_default_graph()
#%%
Theta = tf.get_variable('Theta',shape=[2049,n_attr])
#%%
dataset = tf.data.TFRecordDataset(global_setting_AWA2.record_path)
dataset = dataset.map(parser)
dataset = dataset.shuffle(20000)
dataset = dataset.batch(global_setting_AWA2.batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
(img_ids,img_features,labels,Attributes) = iterator.get_next()

#in memory
dataset_in_1 = tf.data.TFRecordDataset(global_setting_AWA2.sparse_dict_path)
dataset_in_1 = dataset_in_1.map(parser).batch(50000)
(sparse_dict_img_id,sparse_dict,sparse_dict_label,sparse_dict_Attributes) = dataset_in_1.make_one_shot_iterator().get_next()

sparse_dict = tf.concat([sparse_dict,tf.ones([tf.shape(sparse_dict)[0],1])],axis = 1)
sparse_dict_img_id,sparse_dict,sparse_dict_label,sparse_dict_Attributes = sess.run([sparse_dict_img_id,sparse_dict,sparse_dict_label,sparse_dict_Attributes])

dataset_in_2 = tf.data.TFRecordDataset(global_setting_AWA2.validation_path)
dataset_in_2 = dataset_in_2.map(parser).batch(50000)
(img_val_ids,F_val,val_labels,val_attributes) = dataset_in_2.make_one_shot_iterator().get_next()
F_val = tf.concat([F_val,tf.ones([tf.shape(F_val)[0],1])],axis = 1)
(img_val_ids,F_val,val_labels,val_attributes)=sess.run([img_val_ids,F_val,val_labels,val_attributes])
val_attributes = val_attributes[:,:,0]
num_dict_sample = sparse_dict.shape[0]
#%%
seen = np.unique(sparse_dict_label)
unseen = np.array([l for l in range(n_class) if l not in seen])
#%%
#with tf.device('/gpu:0'):
sparse_dict_img_id = tf.constant(sparse_dict_img_id)
sparse_dict = tf.constant(sparse_dict)
sparse_dict_label = tf.constant(sparse_dict_label)
sparse_dict_Attributes = tf.constant(sparse_dict_Attributes)
#%%
fraction_idx_var = tf.get_variable('fraction_idx_var',shape=(),dtype = tf.int32)
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
G = np.eye(n_attr)
G=preprocessing_graph(G)
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
ratio_balance_f = ratio_balance_pos_neg[:,fraction_idx_var]
atttibute_f = Attributes[:,:,fraction_idx_var]
sparse_dict_Attributes_f = sparse_dict_Attributes[:,:,fraction_idx_var]
with tf.variable_scope("sparse_coding_OMP"):
    A,P,R_F= colaborative_loss.OMP_asym_sigmoid_Feature_Graph(Theta,F,sparse_dict,atttibute_f,sparse_dict_Attributes_f,
                                                         global_setting_AWA2.k,part_G_var,alpha_colaborative_var,
                                                         alpha_feature_var,parallel_iterations,
                                                         c,global_setting_AWA2.thresold_coeff)
#    A= tf.Print(A,[tf.norm(F)],'Place 1:')
with tf.variable_scope("sparse_coding_colaborative_graph"):
    R=colaborative_loss.OMP_asym_sigmoid_loss_Feature_Graph(Theta,F,sparse_dict,atttibute_f,sparse_dict_Attributes_f,A,P,part_G_var,parallel_iterations,c)
#    R= tf.Print(R,[tf.norm(F)],'Place 2:')
    loss_colaborative=tf.square(tf.norm(R))*1.0/global_setting_AWA2.batch_size
    
with tf.variable_scope("sparse_coding_feature"):
    loss_feature = tf.square(tf.norm(R_F))
    
with tf.variable_scope("logistic"):
    logits = tf.matmul(F,Theta)
    labels_binary = tf.clip_by_value(atttibute_f,0,1)
    if is_balance:
        labels_weight = tf.abs(tf.clip_by_value(atttibute_f,-1,0))+tf.multiply(tf.clip_by_value(atttibute_f,0,1),ratio_balance_f)
    else:
        labels_weight = tf.abs(atttibute_f)
    loss_logistic = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=logits,weights=labels_weight)

with tf.variable_scope("regularizer"):
    loss_regularizer = tf.square(tf.norm(Theta[:-1,:]))
#%% shared operation
grad_logistic = tf.gradients(loss_logistic, Theta)
grad_colaborative = tf.gradients(loss_colaborative, Theta)
grad_regularizer = tf.gradients(loss_regularizer,Theta)

norm_grad_logistic = tf.norm(grad_logistic)
norm_grad_colaborative = tf.norm(grad_colaborative)
norm_grad_regularizer = tf.norm(grad_regularizer)
norm_Theta = tf.norm(Theta)
raitio_colaborative_grad = norm_grad_logistic/norm_grad_colaborative
raitio_regularizer_grad = norm_grad_logistic/norm_grad_regularizer

ratio_loss = loss_logistic/loss_colaborative
validate_Prediction = tf.matmul(F_val,Theta)
sigmoid_validate_Prediction = tf.nn.sigmoid(tf.matmul(F_val,Theta))

#label completion
mask_label = 1.0-tf.abs(tf.clip_by_value(sparse_dict_Attributes_f,-1,1))
Label_completion = tf.multiply(tf.nn.tanh(tf.matmul(sparse_dict,Theta)),mask_label)+sparse_dict_Attributes_f
#%%
tf.global_variables_initializer().run()
sess.run(iterator.initializer)
#%%
def append_info(m_error,m_error_q,loss_value,loss_logistic_value,lr_v):
    
    res_loss[index] = loss_value
    res_loss_logistic[index]=loss_logistic_value
    res_lr[index]=lr_v
    res_error[index]=m_error
    res_error_q[index]=m_error_q
    
    df_result['error: '+extension]=res_error
    df_result['error_q: '+extension]=res_error_q
    df_result['loss: '+extension]=res_loss
    df_result['logistic: '+extension]=res_loss_logistic
    df_result['lr: '+extension]=res_lr

#%%
print('placeholder assignment')
#%%
Theta_fh = tf.placeholder(dtype=tf.float32, shape=[2049,n_attr])
op_assign_Theta = Theta.assign(Theta_fh)
if global_setting_AWA2.lr_schedule == 'EMA':
    learning_rate_fh=tf.placeholder(dtype=tf.float32,shape=())
    op_assign_learning_rate = learning_rate.assign(learning_rate_fh)

#%%
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)#tf.train.AdamOptimizer(learning_rate=0.001),momentum=0.9
optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
loss = loss_logistic
loss += alpha_colaborative_var*loss_colaborative
loss += alpha_regularizer_var*loss_regularizer
#%%
grad_vars = optimizer.compute_gradients(loss,[Theta,G_var])
#%%
train = optimizer.apply_gradients(grad_vars,global_step)#optimizer.minimize(loss,var_list=[Theta,G_var])#

#%%
fractions_sort=np.argsort(fractions)
ranks = np.empty_like(fractions_sort)
ranks[fractions_sort] = np.arange(len(fractions_sort))
for fraction_idx,fraction  in zip(ranks,fractions):
    name = name_template.format(global_setting_AWA2.learning_rate_base,global_setting_AWA2.batch_size,global_setting_AWA2.decay_rate_cond
                                               ,global_setting_AWA2.signal_strength,global_setting_AWA2.n_cycles,fraction,
                                               idx_GPU,global_setting_AWA2.thresold_coeff,c,time.time())
    if not experiment_cond_success_frac():
        print('skip')
        continue
    data=np.load(global_setting_AWA2.saturated_Thetas_model)
    init_Theta = data['Thetas'][:,:,fraction_idx]
    tf.global_variables_initializer().run()
    sess.run(op_G_var)
    #sess.run(op_assign_Theta,{Theta_fh:init_Theta})
    sess.run(iterator.initializer)
    df_result = pd.DataFrame()
    #%%
    sess.run(op_G_var)
    sess.run(op_assign_Theta,{Theta_fh:init_Theta})
    sess.run(op_alpha_colaborative_var,{alpha_colaborative_var_fh:1})
    sess.run(op_alpha_feature_var,{alpha_feature_var_fh:0})
    sess.run(op_alpha_regularizer,{alpha_regularizer_var_fh:0})
    sess.run(fraction_idx_var.assign(fraction_idx))
    #%%
    loss_feature_eval,loss_colaborative_eval= sess.run([loss_feature,loss_colaborative])
    raitio_colaborative_grad_value,raitio_regularizer_grad_value = sess.run([raitio_colaborative_grad,raitio_regularizer_grad])
    raitio_regularizer_grad_value=1
    raitio_loss_coloborative_feature = loss_colaborative_eval/loss_feature_eval
    print(raitio_colaborative_grad_value,raitio_regularizer_grad_value,raitio_loss_coloborative_feature)

    
    #%% create dir
    if not os.path.exists('./result/'+name) and is_save:
        os.makedirs('./result/'+name)
    #%%
    Thetas = np.zeros((2049,n_attr,n_experiment))
    Gs = np.zeros((n_attr,n_attr,n_experiment))
    Signature_comp = np.zeros((n_class,n_attr,n_experiment))
    Signature_comp_q = np.zeros((n_class,n_attr,n_experiment))
    idx_experiment = 0
    for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
        for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
            for idx_alpha_regularizer,alpha_regularizer_o in enumerate(global_setting_AWA2.regularizers):
                
                if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                    print('skip')
                    continue
                
                report_length = global_setting_AWA2.n_cycles*n_iters//global_setting_AWA2.report_interval +1 #in case that my lousy computation is wrong
                print('report length {}'.format(report_length))
                res_error = np.zeros(report_length)
                res_loss = np.zeros(report_length)
                res_loss_logistic=np.zeros(report_length)
                res_error_q=np.zeros(report_length)
                res_lr=np.zeros(report_length)
                res_disperseion=np.zeros(report_length)
                #
                sess.run(iterator.initializer)
                alpha_colaborative = raitio_colaborative_grad_value*alpha_colaborative_o
                alpha_feature = alpha_colaborative*alpha_feature_o
                alpha_regularizer = raitio_regularizer_grad_value*alpha_regularizer_o
                
                tf.global_variables_initializer().run()
                print('reset Theta')
                sess.run(op_G_var)
                sess.run(op_assign_Theta,{Theta_fh:init_Theta})
                sess.run(op_alpha_colaborative_var,{alpha_colaborative_var_fh:alpha_colaborative})
                sess.run(op_alpha_feature_var,{alpha_feature_var_fh:alpha_feature})
                sess.run(op_alpha_regularizer,{alpha_regularizer_var_fh:alpha_regularizer})
                sess.run(fraction_idx_var.assign(fraction_idx))
                #exponential moving average
                expon_moving_avg_old = np.inf
                expon_moving_avg_new = 0
                #
                m = 0
                df_ap = pd.DataFrame()
#                df_ap['label']=attr_name
                print('lambda colaborative: {} lambda_feature: {} regularizer: {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer))
                #%%
                for idx_cycle in range(global_setting_AWA2.n_cycles):
                    tic = time.clock()
                    _,loss_value,logistic_v,lr_v  = sess.run([train,loss,loss_logistic,learning_rate])
                    index = (idx_cycle*n_iters)//global_setting_AWA2.report_interval
                    print('Elapsed time udapte: {}'.format(time.clock()-tic))
                    print('Loss {} logistic {} lr {}'.format(loss_value,logistic_v,lr_v))
                    if (idx_cycle*n_iters) % global_setting_AWA2.report_interval == 0 :#or idx_iter == n_iters-1:
                        time_o = time.clock()
                        print('index {} -- compute mAP'.format(index))
                        print('{} alpha: colaborative {} feature {} regularizer {}'.format(name,alpha_colaborative,alpha_feature,alpha_regularizer))
                        Label_completion_v,sparse_dict_label_v = sess.run([Label_completion,sparse_dict_label])
                        signature_comp=signature_completion(Label_completion_v,sparse_dict_label_v,signature_q,quantization)
                        error=evaluate_completion(signature_comp,signature_q,quantization)
                        m_error = np.mean(error)
                        
                        signature_comp_q=signature_completion(Label_completion_v,sparse_dict_label_v,signature_q,True)
                        error_q=evaluate_completion(signature_comp_q,signature_q,True)
                        m_error_q = np.mean(error_q)
                        print(m_error,m_error_q)
                        
                        res_error[index]=m_error
                        res_error_q[index]=m_error_q
                        extension='frac {} collaborative {} feature {} regularizer {}'.format(fraction,alpha_colaborative,alpha_feature,alpha_regularizer)
#                        df_ap['index {}: ap'.format(index)]=ap
#                        df_ap['index {}: num_mis_p'.format(index)]=num_mis_p
#                        df_ap['index {}: num_mis_n'.format(index)]=num_mis_n
#                        m_AP=np.mean(ap)
#                        sum_num_miss_p = np.sum(num_mis_p)
#                        sum_num_miss_n = np.sum(num_mis_n)
#                        if global_setting_AWA2.lr_schedule == 'EMA':
#                        #exponential_moving_avg
#                            expon_moving_avg_old=expon_moving_avg_new
#                            expon_moving_avg_new = expon_moving_avg_new*(1-global_setting_AWA2.signal_strength)+m_AP*global_setting_AWA2.signal_strength
#                            if expon_moving_avg_new<expon_moving_avg_old and learning_rate.eval() >= global_setting_AWA2.limit_learning_rate and m <= 0:
#                                print('Adjust learning rate')
#                                sess.run(op_assign_learning_rate,{learning_rate_fh:learning_rate.eval()*global_setting_AWA2.decay_rate_cond})
#                                m = 2
#                            m -= 1
                        append_info(m_error,m_error_q,loss_value,logistic_v,lr_v)
#                        print('mAP {} zs_acc_u_u {} zs_acc_u_a {} zs_acc_s_s {} zs_acc_s_a {} dispersion {}'.format(m_AP,acc_u_u,acc_u_a,acc_s_s,acc_s_a,dispersion_eval))
                        if is_save:
                            Thetas[:,:,idx_experiment]=Theta.eval()
                            Gs[:,:,idx_experiment]=part_G_var.eval()
                            df_result.to_csv('./result/'+name+'/error.csv')
                            Signature_comp[:,:,idx_experiment]=signature_comp
                            Signature_comp_q[:,:,idx_experiment]=signature_comp_q
                            np.savez('./result/'+name+"/model_signatures", Thetas=Thetas,Signature_comp_q=Signature_comp_q,Signature_comp=Signature_comp)
#                            ap_save_name = './result/'+name+'/ap_colaborative {} feature {} regularizer {}.csv'
#                            df_ap.to_csv(ap_save_name.format(alpha_colaborative,alpha_feature,alpha_regularizer))
#    #                        if index%(int(report_length/4)) == 0:
#    #                            np.savez('./result/'+name, Thetas=Thetas, Gs=Gs)
##                            pdb.set_trace()
#                            if global_setting_AWA2.early_stopping and m_AP >= np.max(res_mAP):
#                                print('-'*30)
#                                print('Save best')
#                                np.savez('./result/'+name+"/model", Thetas=Thetas, Gs=Gs)
#                                model_name = 'model_frac_{}_{}.ckpt'.format(fraction,alpha_regularizer_o)
                idx_experiment+=1
#%%
sess.close()
tf.reset_default_graph()