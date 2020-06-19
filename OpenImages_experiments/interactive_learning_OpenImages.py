# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:37:08 2018

@author: badat
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
import pandas as pd
import os.path
import os
import numpy as np
import time
from measurement import apk
import colaborative_loss
import D_utility
import pdb
import global_setting_OpenImage
from sklearn.metrics import average_precision_score
#%% logging level
global_setting_OpenImage.batch_size=32
global_setting_OpenImage.n_cycles*=1#60*global_setting_OpenImage.report_interval
global_setting_OpenImage.report_interval = 100
global_setting_OpenImage.n_cycles = 3657355//global_setting_OpenImage.batch_size
#global_setting_OpenImage.learning_rate_base = 0.01
global_setting_OpenImage.k=1
#global_setting_OpenImage.thresold_coeff = 0.01
#%% data flag
#
is_G = True
is_nonzero_G = True
is_constrant_G = False
is_sum_1=True
is_optimize_all_G = True
#
#global_setting_OpenImage.label_graph_path='./label_graph/graph_label_mixture.npz'
idx_GPU=7
strength_identity = 1
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
list_alphas_colaborative = [1]#[0.5,1,2]
list_alphas_feature = [0.5]#0,0.5,1,2
learning_rate = tf.Variable(global_setting_OpenImage.learning_rate_base,trainable = False,dtype=tf.float32)
n_iters = 1
#decay_rate_schedule = global_setting_OpenImage.n_cycles//(global_setting_OpenImage.report_interval*4)
c = 2.0
if is_sum_1:
    part_name = '_sum1'
else:
    part_name = '_str_i_'+str(strength_identity)
name = 'interactive_learning_OpenImages'
save_name = name+'.csv'
df_result = pd.DataFrame()
is_save = False
report_length = global_setting_OpenImage.n_cycles*n_iters//global_setting_OpenImage.report_interval +1 #in case that my lousy computation is wrong
patient=report_length//100
parallel_iterations = 1
#%%
#print('decay schedule {}'.format(decay_rate_schedule))
print('number of cycles {}'.format(global_setting_OpenImage.n_cycles))
#%% create dir
if not os.path.exists('./result/'+name) and is_save:
    os.makedirs('./result/'+name)
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
#%%
n_1 = 2048
#n_2 = 512
Theta_1 = tf.get_variable('Theta_1',shape=[2049,n_1])
#Theta_2 = tf.get_variable('Theta_2',shape=[n_1+1,n_2])
Theta_f = tf.get_variable('Theta_f',shape=[n_1+1,5000])

def padding_one(F):
    return tf.concat([F,tf.ones([tf.shape(F)[0],1])],axis = 1)

def deep_augmented_mapping(F):
    F = padding_one(F)
    F_f = tf.matmul(F,Theta_1)#tf.nn.relu(tf.matmul(F,Theta_1))
    
#    F_2 = padding_one(F_2)
#    F_f = tf.nn.relu(tf.matmul(F_2,Theta_2))
    
    F_f = padding_one(F_f)
    return F_f

def bias_initializer():
    op_bias_1=Theta_1[-1,:].assign(tf.zeros(n_1))
#    op_bias_2=Theta_2[-1,:].assign(tf.zeros(n_2))
    op_bias_f=Theta_f[-1,:].assign(tf.zeros(5000))
    sess.run([op_bias_1,op_bias_f])#op_bias_2

data=np.load(global_setting_OpenImage.saturated_Thetas_model)   
def init_Theta(): 
    init_Theta = data['Thetas'][:,:,0]
    sess.run([Theta_f.assign(init_Theta),Theta_1.assign(np.eye(n_1+1,n_1))])
#%%
dataset = tf.data.TFRecordDataset(global_setting_OpenImage.record_path)
dataset = dataset.map(parser)
dataset = dataset.batch(global_setting_OpenImage.batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
(img_ids,F,labels) = iterator.get_next()

#in memory
dataset_in_1 = tf.data.TFRecordDataset(global_setting_OpenImage.sparse_dict_path)
dataset_in_1 = dataset_in_1.map(parser).batch(50000)
(sparse_dict_img_id,sparse_dict,sparse_dict_label) = dataset_in_1.make_one_shot_iterator().get_next()

sparse_dict_img_id,sparse_dict,sparse_dict_label = sess.run([sparse_dict_img_id,sparse_dict,sparse_dict_label])

dataset_in_2 = tf.data.TFRecordDataset(global_setting_OpenImage.validation_path)
dataset_in_2 = dataset_in_2.map(parser).batch(50000)
(img_val_ids,F_val,val_labels) = dataset_in_2.make_one_shot_iterator().get_next()
(img_val_ids,F_val,val_labels)=sess.run([img_val_ids,F_val,val_labels])
num_dict_sample = sparse_dict.shape[0]
#%%
#with tf.device('/gpu:0'):
sparse_dict_img_id = tf.constant(sparse_dict_img_id)
sparse_dict = tf.constant(sparse_dict)
sparse_dict_label = tf.constant(sparse_dict_label)

F_val = tf.constant(F_val)
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
#%%
F_f = deep_augmented_mapping(F)
sparse_dict_f = deep_augmented_mapping(sparse_dict)
F_val_f = deep_augmented_mapping(F_val)
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
if global_setting_OpenImage.label_graph_path.split('.')[-1]=='npz':
    G = np.load(global_setting_OpenImage.label_graph_path)['arr_0'].astype(np.float32)   
else:
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
op_G_constraint = G_var.assign(tf.clip_by_value(G_var,-1,1))
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

with tf.variable_scope("sparse_coding_OMP"):
    A,P_L,P_F= colaborative_loss.e2e_OMP_asym_sigmoid_Feature_Graph(Theta_f,F_f,sparse_dict_f,labels,sparse_dict_label,
                                                         global_setting_OpenImage.k,part_G_var,alpha_colaborative_var,
                                                         alpha_feature_var,parallel_iterations,
                                                         c,global_setting_OpenImage.thresold_coeff,False)
#    A= tf.Print(A,[A],'Place 1:')
with tf.variable_scope("sparse_coding_colaborative_graph"):
    R_L,R_F=colaborative_loss.e2e_OMP_asym_sigmoid_loss_Feature_Graph(Theta_f,F_f,sparse_dict_f,labels,sparse_dict_label,A,P_L,P_F,part_G_var,parallel_iterations,c)
#    R= tf.Print(R,[tf.norm(F)],'Place 2:')
    loss_colaborative=tf.square(tf.norm(R_L))*1.0/global_setting_OpenImage.batch_size
    
with tf.variable_scope("sparse_coding_feature"):
    loss_feature = tf.square(tf.norm(R_F))*1.0/global_setting_OpenImage.batch_size
    
with tf.variable_scope("logistic"):
    logits = tf.matmul(F_f,Theta_f)
    labels_binary = tf.clip_by_value(labels,0,1)
    labels_weight = tf.abs(labels)
    loss_logistic = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=logits,weights=labels_weight)

#%% shared operation
norm_Theta = tf.norm(Theta_f)

ratio_loss = loss_logistic/loss_colaborative
validate_Prediction = tf.matmul(F_val_f,Theta_f)
validate_Prediction_G = tf.matmul(colaborative_loss.asym_sigmoid(validate_Prediction,c),part_G_var)
#%%
tf.global_variables_initializer().run()
sess.run(iterator.initializer)
#%%
def append_info(m_AP,m_AP_G,loss_value,loss_logistic_value,loss_feature_v,lr_v,dispersion_eval,dispersion_neg_v,dispersion_diag_v):
    
    res_mAP[index]=m_AP
    res_mAP_G[index]=m_AP_G
    res_loss[index] = loss_value
    res_dispersion[index] = dispersion_eval
    res_loss_logistic[index]=loss_logistic_value
    res_loss_feature[index] = loss_feature_v
    res_lr[index]=lr_v
    res_dispersion_neg[index]=dispersion_neg_v
    res_dispersion_diag[index]=dispersion_diag_v
    
    extension = 'colaborative {} feature {} regularizer {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer)
    df_result['mAP: '+extension]=res_mAP
    df_result['mAP_G: '+extension]=res_mAP_G
    df_result['loss: '+extension]=res_loss
    df_result['logistic: '+extension]=res_loss_logistic
    df_result['feature: '+extension]=res_loss_feature
    df_result['dispersion: '+extension]=res_dispersion
    df_result['dispersion_neg: '+extension]=res_dispersion_neg
    df_result['dispersion_diag: '+extension]=res_dispersion_diag
    df_result['lr: '+extension]=res_lr
#%%
print('placeholder assignment')
#%%
learning_rate_fh=tf.placeholder(dtype=tf.float32,shape=())
op_assign_learning_rate = learning_rate.assign(learning_rate_fh)

#%%
tf.global_variables_initializer().run()
sess.run(op_G_var)
sess.run(iterator.initializer)
init_Theta()
sess.run(op_alpha_colaborative_var,{alpha_colaborative_var_fh:1e5})
sess.run(op_alpha_feature_var,{alpha_feature_var_fh:1})
#%%
loss_logistic_v,loss_feature_v,loss_colaborative_v= sess.run([loss_logistic,loss_feature,loss_colaborative])
ratio_loss_coloborative = loss_logistic_v/loss_colaborative_v
ratio_loss_feature = loss_logistic_v/loss_feature_v
print(ratio_loss_coloborative,ratio_loss_feature)
#%%
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)#tf.train.AdamOptimizer(learning_rate=0.001),momentum=0.9
optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
scale_lr_G = 0.5/global_setting_OpenImage.learning_rate_base
optimizer_G = tf.train.RMSPropOptimizer(
      learning_rate*scale_lr_G,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )

loss = loss_logistic
loss += alpha_colaborative_var*loss_colaborative
loss += alpha_feature_var*loss_feature
grad_var_all = optimizer.compute_gradients(loss,[Theta_1,Theta_f,G_var])#Theta_2
Theta_grads = grad_var_all[:-1]
G_grads = grad_var_all[-1]
train = optimizer.apply_gradients(Theta_grads)#optimizer.minimize(loss,var_list=[Theta,G_var])#
train_G = optimizer_G.apply_gradients([G_grads])
reset_optimizer_op = tf.variables_initializer(optimizer.variables())
reset_optimizer_G_op = tf.variables_initializer(optimizer_G.variables())
#%%
print('done placeholder assignment')
def experiment_cond_success():
    return True#alpha_colaborative_o > 0 and alpha_feature_o ==0

n_experiment= 0

for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            n_experiment += 1
print('Total number of experiment: {}'.format(n_experiment))

Thetas_1 = np.zeros((2049,n_1,n_experiment))
Thetas_f = np.zeros((n_1+1,5000,n_experiment))
Gs = np.zeros((5000,5000,n_experiment))

Thetas_1_final = np.zeros((2049,n_1,n_experiment))
Thetas_f_final = np.zeros((n_1+1,5000,n_experiment))
Gs_final = np.zeros((5000,5000,n_experiment))

idx_experiment = 0

for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):
            
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            
            print('report length {} patient {}'.format(report_length,patient))
            res_mAP = np.zeros(report_length)
            res_loss = np.zeros(report_length)
            res_loss_logistic=np.zeros(report_length)
            res_mAP_G=np.zeros(report_length)
            res_loss_feature=np.zeros(report_length)
            res_dispersion_neg=np.zeros(report_length)
            res_lr=np.zeros(report_length)
            res_dispersion = np.zeros(report_length)
            res_dispersion_diag = np.zeros(report_length)
            #
            sess.run(iterator.initializer)
            alpha_colaborative = ratio_loss_coloborative *alpha_colaborative_o
            alpha_feature = ratio_loss_feature*alpha_feature_o
            alpha_regularizer = 0
            
            tf.global_variables_initializer().run()
            init_Theta()
            print('reset Theta')
            sess.run(op_G_var)
            sess.run(op_alpha_colaborative_var,{alpha_colaborative_var_fh:alpha_colaborative})
            sess.run(op_alpha_feature_var,{alpha_feature_var_fh:alpha_feature})
            
            #exponential moving average
            expon_moving_avg_old = np.inf
            expon_moving_avg_new = 0
            #
            m = 0
            n_nan = 0
            df_ap = pd.DataFrame()
            df_ap['label']=list_label
            print('lambda colaborative: {} lambda_feature: {} regularizer: {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer))
            #%%
            tic = time.clock()
            for idx_cycle in range(global_setting_OpenImage.n_cycles):
#                try:
                _,_,loss_value,logistic_v,loss_feature_v,lr_v,lr_G_v=sess.run([train,train_G,loss,loss_logistic,loss_feature,learning_rate,optimizer_G._learning_rate])
                if is_constrant_G:
                    sess.run(op_G_constraint)
                if is_nonzero_G:
                    sess.run(op_G_nonnegative)
                if np.isnan(loss_value):
                    print('nan encounter')
                    n_nan += 1
                    sess.run(reset_optimizer_op)
                    sess.run(reset_optimizer_G_op)
                    sess.run(op_G_var)
                    sess.run(op_assign_learning_rate,{learning_rate_fh:lr_v})
                    m = patient
                print('.',end='')
                index = (idx_cycle*n_iters)//global_setting_OpenImage.report_interval      
                if (idx_cycle*n_iters) % global_setting_OpenImage.report_interval == 0 :#or idx_iter == n_iters-1:
                    print('Elapsed time udapte: {}'.format(time.clock()-tic))
                    tic = time.clock()
                    print('index {} -- alpha_c {} alpha_f {}'.format(index,alpha_colaborative_o,alpha_feature_o))
                    print('{} alpha: colaborative {} feature {} regularizer {}'.format(name,alpha_colaborative,alpha_feature,alpha_regularizer))
                    tic_p = time.clock()
                    validate_Prediction_v,validate_Prediction_G_v,dispersion_v,dispersion_neg_v,dispersion_diag_v = sess.run([validate_Prediction,validate_Prediction_G,
                                                                              dispersion,dispersion_neg,dispersion_diag])
                    print('evaluation time:',time.clock()-tic_p,validate_Prediction_v.shape)
                    ap = compute_AP(validate_Prediction_v,val_labels,img_val_ids)
                    ap_G = compute_AP(validate_Prediction_G_v,val_labels,img_val_ids)
                    df_ap['index {}'.format(index)]=ap
                    m_AP=np.mean(ap)
                    m_AP_G=np.mean(ap_G)
                    #exponential_moving_avg
                    expon_moving_avg_old=expon_moving_avg_new
                    expon_moving_avg_new = expon_moving_avg_new*(1-global_setting_OpenImage.signal_strength)+m_AP*global_setting_OpenImage.signal_strength
                    if expon_moving_avg_new<expon_moving_avg_old and learning_rate.eval() >= global_setting_OpenImage.limit_learning_rate and m <= 0:
                        print('Adjust learning rate')
                        sess.run(op_assign_learning_rate,{learning_rate_fh:learning_rate.eval()*global_setting_OpenImage.decay_rate_cond})
                        m = patient
                    m -= 1
                    # decay schedule
#                    if index % decay_rate_schedule == 0 and index > 0:
#                        sess.run(op_assign_learning_rate,{learning_rate_fh:learning_rate.eval()*global_setting_OpenImage.decay_rate_cond})
                    #
                    append_info(m_AP,m_AP_G,loss_value,logistic_v,loss_feature_v,lr_v,dispersion_v,dispersion_neg_v,dispersion_diag_v)
                    print('mAP {} mAP_G {} dispersion {} dispersion_neg {} dispersion_diag {} n_nan {}'.format(m_AP,m_AP_G,dispersion_v,dispersion_neg_v,dispersion_diag_v,n_nan))
                    print('Loss {} logistic {} feature {} lr {} lr_G {}'.format(loss_value,logistic_v,loss_feature_v,lr_v,lr_G_v))

                    if is_save:
                        df_result.to_csv('./result/'+name+'/mAP.csv')
                        ap_save_name = './result/'+name+'/ap_colaborative {} feature {} regularizer {}.csv'
                        df_ap.to_csv(ap_save_name.format(alpha_colaborative,alpha_feature,alpha_regularizer))
                        if not global_setting_OpenImage.early_stopping or (global_setting_OpenImage.early_stopping and m_AP >= np.max(res_mAP)):
                            Thetas_1[:,:,idx_experiment]=Theta_1.eval()
                            Thetas_f[:,:,idx_experiment]=Theta_f.eval()
                            Gs[:,:,idx_experiment]=part_G_var.eval()
                        if index %20 == 0: #occationality saving model
                            np.savez('./result/'+name+"/models_ES", Thetas_1=Thetas_1,Thetas_f=Thetas_f,Gs=Gs)#, Thetas_2=Thetas_2
            
            if is_save:
                Thetas_1_final[:,:,idx_experiment]=Theta_1.eval()
                Thetas_f_final[:,:,idx_experiment]=Theta_f.eval()
                Gs_final[:,:,idx_experiment]=part_G_var.eval()
                np.savez('./result/'+name+"/models_final", Thetas_1=Thetas_1_final,Thetas_f=Thetas_f_final,Gs=Gs_final)#, Thetas_2=Thetas_2
            idx_experiment+=1
#%%
sess.close()
tf.reset_default_graph()