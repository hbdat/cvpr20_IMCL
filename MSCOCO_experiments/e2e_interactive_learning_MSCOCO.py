# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 05:20:30 2019

@author: badat
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

import pandas as pd
import os.path
import numpy as np
import time
import colaborative_loss
from nets import vgg
from D_utility import evaluate,Logger,LearningRate,get_compress_type,preprocessing_graph
from global_setting_MSCOCO import NFS_path,train_img_path,test_img_path,n_report,n_cycles,dic_img_path,label_graph_path
import pdb
import pickle
from tensorflow.contrib import slim
import tensorflow as tf
#%% data flag
idx_GPU=3
is_save = True
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
name='e2e_interactive_learning_MSCOCO'
save_path = NFS_path+'results/'+name
learning_rate_base = 0.01
batch_size = 1
n_cycles *= 32
#n_report *= 3
k = 3
c = 1
n_dict = 10000
thresold_coeff = 1e-20
partition_size = 200
alpha_colaborative_o = 0.5
alpha_feature_o = 0.25
alpha_regularizer_o = 0
is_feature_propagate = False
dic_eval_interval = 3000
#%%
description = 'SSML\n'
description += 'dic_eval_interval {}\n'.format(dic_eval_interval)
description += 'optimize G {}\n'.format(label_graph_path)
description += 'c {}\n'.format(c)
description += 'adapt learning rate \n'
description += 'alpha_colaborative_o {}\n'.format(alpha_colaborative_o)
description += 'alpha_feature_o {}\n'.format(alpha_feature_o)
description += 'alpha_regularizer_o {}\n'.format(alpha_regularizer_o)
description += 'learning_rate_base {} \n'.format(learning_rate_base)
description += 'batch_size {} \n'.format(batch_size)
description += 'signal_strength {} \n'.format(-1)
description += 'n_cycles {} \n'.format(n_cycles)
description += 'is_feature_propagate {}\n'.format(is_feature_propagate)
description += 'vgg'
#%%
print(description)
#%%
checkpoints_dir = './model/vgg_ImageNet/vgg_16.ckpt'
is_train = tf.Variable(True,trainable=False,name='is_train')
#%%
print('number of cycles {}'.format(n_cycles))
#%% Dataset
image_size = vgg.vgg_16.default_image_size
height = image_size
width = image_size
def parser(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'img': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id = parsed['img_id']
    img = tf.reshape(tf.decode_raw( parsed['img'],tf.float32),[height, width, 3])
    label = tf.clip_by_value(tf.decode_raw( parsed['label'],tf.int32),-1,1)
    return img_id,img,label

def parser_test(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'img': tf.FixedLenFeature([], tf.string),
               'label_1k': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id = parsed['img_id']
    img = tf.reshape(tf.decode_raw( parsed['img'],tf.float32),[height, width, 3])
    label = tf.clip_by_value(tf.decode_raw( parsed['label_1k'],tf.int32),-1,1)
    return img_id,img,label
#%%
def load_1k_name():
    path = '/home/project_amadeus/mnt/raptor/hbdat/data/MSCOCO_1k/meta/vocab_coco.pkl'
    with open(path,'rb') as f:
        vocab = pickle.load(f)
    return vocab['words']
classes = load_1k_name()
n_classes = len(classes)
#%% load in memory
sess = tf.InteractiveSession()#tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
g = tf.get_default_graph()
#%%
Theta = tf.get_variable('Theta',shape=[4096+1,n_classes])
#%%
fraction_idx_var = tf.get_variable('fraction_idx_var',shape=(),dtype = tf.int32,trainable = False)
#%%
dataset_tr = tf.data.TFRecordDataset(train_img_path,compression_type=get_compress_type(train_img_path))
dataset_tr = dataset_tr.map(parser)
dataset_tr = dataset_tr.batch(batch_size)
dataset_tr = dataset_tr.prefetch(32)
dataset_tr = dataset_tr.repeat()
iterator_tr = dataset_tr.make_initializable_iterator()
(img_ids_tr,img_tr,labels_tr) = iterator_tr.get_next()

dataset_dic = tf.data.TFRecordDataset(dic_img_path,compression_type=get_compress_type(dic_img_path))
dataset_dic = dataset_dic.map(parser).batch(100)
dataset_dic = dataset_dic.prefetch(32)
iterator_dic = dataset_dic.make_initializable_iterator()
(img_ids_dic,img_dic,labels_dic) = iterator_dic.get_next()


dataset_tst = tf.data.TFRecordDataset(test_img_path,compression_type=get_compress_type(test_img_path))
dataset_tst = dataset_tst.map(parser_test).batch(100)
dataset_tst = dataset_tst.prefetch(32)
iterator_tst = dataset_tst.make_initializable_iterator()
(img_ids_tst,img_tst,labels_tst) = iterator_tst.get_next()
#%% ResNet
img_input_ph = tf.placeholder(dtype=tf.float32,shape=[None,height,width,3])
with slim.arg_scope(vgg.vgg_arg_scope()):
    print('-'*30)
    print('no dropout')
    logit, end_points = vgg.vgg_16(img_input_ph, num_classes=1000, is_training=False)
    features_concat = end_points['vgg_16/fc7']
    
features_concat = features_concat[:,0,0,:]#tf.squeeze(features_concat)
features_concat = tf.concat([features_concat,tf.ones([tf.shape(features_concat)[0],1])],axis = 1,name='feature_input_point')
index_point = tf.placeholder(dtype=tf.int32,shape=())
F = features_concat[:index_point,:]
sparse_dict = features_concat[index_point:,:]
#%%
def process_G(is_sum_1=True):
    G = np.load(label_graph_path).astype(np.float32)
    G = np.clip(G,0,1)
    if is_sum_1:
        G = preprocessing_graph(G)
    

    G_empty_diag = G - np.diag(np.diag(G))
    G_init=G[G!=0]
        
    G_var = tf.get_variable("G_var", G_init.shape,initializer=tf.constant_initializer(G_init))
    indices = []
    counter = 0
    
    for idx_row in range(G_empty_diag.shape[1]):
        idx_cols = np.where(G[idx_row,:]!=0)[0]
        for idx_col in idx_cols:
            if G[idx_row,idx_col]-G_init[counter] != 0:
                raise Exception('error relation construction')
            indices.append([idx_row,idx_col])
            counter += 1
    
    part_G_var = tf.scatter_nd(indices, G_var, G.shape)
    
    return part_G_var
part_G_var = process_G(is_sum_1=True)
#%%
alpha_colaborative_var = tf.Variable(1.0,name='alphha_colaborative',dtype=tf.float32,trainable=False)

alpha_feature_var = tf.Variable(1000.0/4096,name='alpha_feature',dtype=tf.float32,trainable=False)

alpha_regularizer_var = tf.Variable(0,name='alpha_regularizer',dtype=tf.float32,trainable=False)
#%%
labels_ph = tf.placeholder(dtype=tf.float32, shape=(None,n_classes))
sparse_dict_labels_ph = tf.placeholder(dtype=tf.float32, shape=(None,n_classes))

### modification for faster runtime ###
features_ph = tf.placeholder(dtype=tf.float32, shape=(None,4096+1))
sparse_dict_c = tf.get_variable('sparse_dict_c',shape=(n_dict,4096+1),dtype = tf.float32,trainable = False)
sparse_dict_labels_c = tf.get_variable('sparse_dict_labels_c',shape=(n_dict,n_classes),dtype = tf.int32,trainable = False)
idx_exclude = tf.placeholder(shape=(),name='idx_exclude',dtype=tf.int32)
mask = tf.one_hot(idx_exclude,n_dict,on_value=False,off_value=True)
idx_all = tf.range(n_dict)
with tf.variable_scope("sparse_coding_OMP_caches"):
    sparse_dict_c_m = tf.boolean_mask(sparse_dict_c,mask)
    sparse_dict_labels_c_m = tf.boolean_mask(sparse_dict_labels_c,mask)
    A_c,P_L_c,P_F_c= colaborative_loss.e2e_OMP_asym_sigmoid_Feature_Graph(Theta,features_ph,sparse_dict_c_m,labels_ph,sparse_dict_labels_c_m,
                                                         k,part_G_var,alpha_colaborative_var,
                                                         alpha_feature_var,1,
                                                         c,thresold_coeff,is_balance=False)
    A_flaten = tf.reshape(A_c,[-1])
    idx_sim= tf.boolean_mask(idx_all,mask)
    idx_sim=tf.gather(idx_sim,A_flaten)
### modification for faster runtime ###


with tf.variable_scope("sparse_coding_OMP"):
    A,P_L,P_F= colaborative_loss.e2e_OMP_asym_sigmoid_Feature_Graph(Theta,F,sparse_dict,labels_ph,sparse_dict_labels_ph,
                                                         k,part_G_var,alpha_colaborative_var,
                                                         alpha_feature_var,1,
                                                         c,thresold_coeff,is_balance=False)

with tf.variable_scope("sparse_coding_colaborative_graph"):
    R_L,R_F=colaborative_loss.e2e_OMP_asym_sigmoid_loss_Feature_Graph(Theta,F,sparse_dict,labels_ph,sparse_dict_labels_ph,A,P_L,P_F,part_G_var,1,c)
    loss_colaborative=tf.square(tf.norm(R_L))*1.0/batch_size
    
with tf.variable_scope("sparse_coding_feature"):
    loss_feature = tf.square(tf.norm(R_F))*1.0/batch_size

with tf.variable_scope("sparse_coding_norm_constraint"):
    feature_norm = tf.square(tf.norm(F))*1.0/batch_size
    
with tf.variable_scope("logistic"):
    logits_p = tf.matmul(F,Theta)
    logits = tf.matmul(features_concat,Theta)
    loss_logistic = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_ph, logits=logits_p)

with tf.variable_scope("regularizer"):
    loss_regularizer = tf.square(tf.norm(Theta[:-1,:]))
#%% shared operation
grad_logistic = tf.gradients(loss_logistic, Theta)
grad_regularizer = tf.gradients(loss_regularizer,Theta)

norm_grad_logistic = tf.norm(grad_logistic)
norm_grad_regularizer = tf.norm(grad_regularizer)
norm_Theta = tf.norm(Theta)
raitio_regularizer_grad = norm_grad_logistic/norm_grad_regularizer
#%%
tf.global_variables_initializer().run()
sess.run(iterator_tr.initializer)
#%%
loss = loss_logistic
loss += alpha_colaborative_var*loss_colaborative
loss += alpha_regularizer_var*loss_regularizer
if is_feature_propagate:
    loss += alpha_feature_var*loss_feature
else:
    print('no feature propagate')
#%%
lr = LearningRate(learning_rate_base,sess,decay_rate=0.1,patient=5)
optimizer = tf.train.RMSPropOptimizer(
      lr.get_lr(),
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
train_vars = [var for var in tf.trainable_variables() if 'fc' in var.name or 'Theta' in var.name or 'G_var' in var.name]
grad_vars = optimizer.compute_gradients(loss,train_vars)
print('-'*30)
print('Decompose update ops')
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.apply_gradients(grad_vars)
print('-'*30)
###
sparse_dict_c_ph = tf.placeholder(dtype=tf.float32, shape=(n_dict,4096+1))
op_sparse_dict_c = sparse_dict_c.assign(sparse_dict_c_ph)
sparse_dict_labels_c_ph = tf.placeholder(dtype=tf.int32, shape=(n_dict,n_classes))
op_sparse_dict_labels_c_ph = sparse_dict_labels_c.assign(sparse_dict_labels_c_ph)
###
def update_cache(full_features_dic,full_labels_dic):
    sess.run(op_sparse_dict_c,{sparse_dict_c_ph:full_features_dic})
    sess.run(op_sparse_dict_labels_c_ph,{sparse_dict_labels_c_ph:full_labels_dic})

def compute_dic(img_ids_dic,img_dic,labels_dic,iterator_dic):
    print('Compute dic')
    tic = time.clock()
    sess.run(iterator_dic.initializer)
    full_features_dic = []
    full_labels_dic = []
    full_ids_dic = []
    full_imgs_dic = []
    while True:
        try:
            img_dic_v,labels_dic_v,img_ids_dic_v = sess.run([img_dic,labels_dic,img_ids_dic])
            feed_dict = {img_input_ph:img_dic_v}
            features_dic_v = sess.run(features_concat, feed_dict)
            full_features_dic.append(features_dic_v)
            full_labels_dic.append(labels_dic_v)
            full_ids_dic.append(img_ids_dic_v)
            full_imgs_dic.append(img_dic_v)
        except tf.errors.OutOfRangeError:
            print('end')
            break
    full_features_dic = np.concatenate(full_features_dic)
    full_labels_dic = np.concatenate(full_labels_dic)
    full_ids_dic = np.concatenate(full_ids_dic)
    full_imgs_dic = np.concatenate(full_imgs_dic)
    print('Time elapse: ',time.clock()-tic)
    print('save cache')
    tic = time.clock()
    update_cache(full_features_dic,full_labels_dic)
    print('Time elapse: ',time.clock()-tic)
    return full_features_dic,full_labels_dic,full_ids_dic,full_imgs_dic
    
def get_similarity_sample(img_tr_v,labels_tr_v,img_ids_tr_v,full_ids_dic,full_imgs_dic):
    n_dict = full_ids_dic.shape[0]
    idx_e = [idx for idx in range(n_dict) if full_ids_dic[idx] in img_ids_tr_v]
    if len(idx_e)==0:
        idx_e = [-1]
    assert len(idx_e)==1,'quick fix' 
    idx_e=idx_e[0]
    
    feed_dict = {img_input_ph:img_tr_v}
    features_tr_v = sess.run(features_concat, feed_dict)
    
    feed_dict = {features_ph:features_tr_v,labels_ph:labels_tr_v,idx_exclude:idx_e}
    
    try:
        idx_sim_v,A_c_v,P_L_c_v,P_F_c_v=sess.run([idx_sim,A_c,P_L_c,P_F_c],feed_dict)
    except tf.errors.InvalidArgumentError as e:
        print(e)
    
    if np.sum(A_c_v>0)==0:
        print('Error')
        pdb.set_trace()
        
    sub_img_dic = full_imgs_dic[idx_sim_v,:]
    sub_labels_dic = full_labels_dic[idx_sim_v,:]
    img_input = np.concatenate([img_tr_v,sub_img_dic],axis=0)
#    print('Time elapse: ',time.clock()-tic)
    return img_input,sub_labels_dic,idx_sim_v
#%%
#saver = tf.train.Saver()
print('-'*30)
print('save only trainable para : only for no batch norm')
print('-'*30)
init_var = [var for var in tf.trainable_variables() if 'G_var' not in var.name]
saver_init = tf.train.Saver(init_var)
tf.global_variables_initializer().run()
sess.run(iterator_tr.initializer)
#init_fn(sess)
#%%
print('restore Logistic')
saver_init.restore(sess,NFS_path+'results/cont_ss_OpenImage_log_GPU_6_1552076726d777008/model_ES.ckpt')
sess.run(lr.op_reset)
#%%
if is_save:
    os.makedirs(save_path)
    os.makedirs(save_path+'/plots')
    summary_writer = tf.summary.FileWriter(save_path, graph=tf.get_default_graph())
    with open(save_path+'/description.txt','w') as f:
        f.write(description)
    ##### test #####
    saver_init.save(sess, save_path+'/model.ckpt')
    ##### test #####
#%%
full_features_dic,full_labels_dic,full_ids_dic,full_imgs_dic=compute_dic(img_ids_dic,img_dic,labels_dic,iterator_dic)
l_logistic_acc = 0
l_colaborative_acc = 0
l_feature_v = 0
n_trial = 10
for i in range(n_trial):
    img_ids_tr_v,img_tr_v,labels_tr_v = sess.run([img_ids_tr,img_tr,labels_tr])
#    pdb.set_trace()
    img_input,sub_labels_dic,index_dict=get_similarity_sample(img_tr_v,labels_tr_v,img_ids_tr_v,full_ids_dic,full_imgs_dic)
    feed_dict = {img_input_ph:img_input,labels_ph:labels_tr_v,sparse_dict_labels_ph:sub_labels_dic,index_point:img_tr_v.shape[0]}
    loss_colaborative_v,loss_feature_v,loss_logistic_v,feature_norm_v  = sess.run([loss_colaborative,loss_feature,loss_logistic,feature_norm],feed_dict)
    l_logistic_acc += loss_logistic_v/n_trial
    l_colaborative_acc += loss_colaborative_v/n_trial
    l_feature_v += loss_feature_v/n_trial

raitio_colaborative_loss = l_logistic_acc/l_colaborative_acc if loss_colaborative_v > 0 else 0 
raitio_regularizer_loss=1
raitio_featrue_loss = l_logistic_acc/l_feature_v if loss_feature_v > 0 else 0
raitio_norm_loss = 0

alpha_colaborative = raitio_colaborative_loss*alpha_colaborative_o
alpha_feature = raitio_featrue_loss*alpha_feature_o
alpha_regularizer = raitio_regularizer_loss*alpha_regularizer_o


sess.run(alpha_colaborative_var.assign(alpha_colaborative))
sess.run(alpha_feature_var.assign(alpha_feature))
sess.run(alpha_regularizer_var.assign(alpha_regularizer))
print(alpha_colaborative,alpha_feature,alpha_regularizer)
#%%
accum_l = 0
alpha = 0.9
sess.run(iterator_tr.initializer)
logger = Logger(cols=['index','l','l_logistic','l_colaborative','l_feature','mAP','mAP_pascal','lr'],filename=save_path+'/log.csv',
                is_save=is_save)
eval_interval = max((n_cycles//n_report),500)
tic = time.clock()
for i in range(n_cycles):
    img_ids_tr_v,img_tr_v,labels_tr_v = sess.run([img_ids_tr,img_tr,labels_tr])
    ### reparations ###
    if (i+1) % dic_eval_interval == 0:
        full_features_dic,full_labels_dic,full_ids_dic,full_imgs_dic=compute_dic(img_ids_dic,img_dic,labels_dic,iterator_dic)
    img_input,sub_labels_dic,index_dict=get_similarity_sample(img_tr_v,labels_tr_v,img_ids_tr_v,full_ids_dic,full_imgs_dic)
    ### reparations ###
    _,l,l_logistic,l_colaborative,l_feature  = sess.run([train,loss,loss_logistic,loss_colaborative,loss_feature],
                               {img_input_ph:img_input,labels_ph:labels_tr_v,sparse_dict_labels_ph:sub_labels_dic,index_point:img_tr_v.shape[0]})
    accum_l = l*(1-alpha)+alpha*accum_l
    if np.isnan(l):
        pdb.set_trace()
    learning_rate = lr.learning_rate.eval()
    
    
    
    if i % eval_interval == 0 or i == n_cycles-1:
        print('Time elapse: ',time.clock()-tic)
        tic = time.clock()
        ap_tst,ap_pascal_tst=evaluate(iterator_tst,[img_ids_tst,img_tst,labels_tst],img_input_ph,logits,sess,is_train)
        mAP_tst = np.mean(ap_tst)
        mAP_pascal_tst=np.mean(ap_pascal_tst)
        learning_rate=lr.adapt(mAP_tst)
        values = [i,l,l_logistic,l_colaborative,l_feature,mAP_tst,mAP_pascal_tst,learning_rate]

        logger.add(values)
        print('{} loss: {} l_logistic: {} l_colaborative {} l_feature {} mAP: {} mAP_pascal: {} lr: {}'.format(*values))
        logger.save()

        print('learning rate',learning_rate)
        if is_save and mAP_tst >= logger.get_max('mAP'):
            saver_init.save(sess, save_path+'/model_ES.ckpt')
            
def evaluate_df():
    ap_tst,ap_pascal_tst=evaluate(iterator_tst,[img_ids_tst,img_tst,labels_tst],img_input_ph,logits,sess,is_train)
    print(np.mean(ap_tst))
    df = pd.DataFrame()
    df['classes']=classes
    df['ap']=ap_tst
    df['ap_pascal']=ap_pascal_tst
    return df

print('final model')
if is_save:
    saver_init.save(sess, save_path+'/model.ckpt')
df_f=evaluate_df()

print('ES model')
saver_init.restore(sess, save_path+'/model_ES.ckpt')
df_ES=evaluate_df()

if is_save:
    df_f.to_csv(save_path+'/F1_f_test.csv')
    df_ES.to_csv(save_path+'/F1_ES_test.csv')
#%%
sess.close()