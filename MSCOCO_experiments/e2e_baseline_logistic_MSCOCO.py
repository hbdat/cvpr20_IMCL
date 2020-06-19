# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:05:29 2019

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
from nets import vgg
from D_utility import evaluate,Logger,LearningRate,get_compress_type
from global_setting_MSCOCO import NFS_path,train_img_path,test_img_path,n_report,n_cycles
import pdb
import pickle
from tensorflow.contrib import slim
import tensorflow as tf
#%% data flag
idx_GPU=7
is_save = False
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
name='e2e_baseline_logistic_MSCOCO'
save_path = NFS_path+'results/'+name
learning_rate_base = 0.001
batch_size = 1
n_cycles *= 32
#%%
description = ''
description += 'learning_rate_base {} \n'.format(learning_rate_base)
description += 'batch_size {} \n'.format(batch_size)
description += 'signal_strength {} \n'.format(-1)
description += 'n_cycles {} \n'.format(n_cycles)
description += 'vgg'
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
    path = './data/MSCOCO_1k/vocab_coco.pkl'
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
#dataset_tr = dataset_tr.shuffle(20000)
dataset_tr = dataset_tr.batch(batch_size)
dataset_tr = dataset_tr.repeat()
iterator_tr = dataset_tr.make_initializable_iterator()
(img_ids_tr,img_tr,labels_tr) = iterator_tr.get_next()

dataset_tst = tf.data.TFRecordDataset(test_img_path,compression_type=get_compress_type(test_img_path))
dataset_tst = dataset_tst.map(parser_test).batch(100)
iterator_tst = dataset_tst.make_initializable_iterator()
(img_ids_tst,img_tst,labels_tst) = iterator_tst.get_next()
#%% ResNet
img_input_ph = tf.placeholder(dtype=tf.float32,shape=[None,height,width,3])
with slim.arg_scope(vgg.vgg_arg_scope()):
    logit, end_points = vgg.vgg_16(img_input_ph, num_classes=1000, is_training=is_train)
    features_concat = end_points['vgg_16/fc7']#g.get_tensor_by_name('resnet_v1_101/pool5:0')
    
features_concat = features_concat[:,0,0,:]#tf.squeeze(features_concat)
features_concat = tf.concat([features_concat,tf.ones([tf.shape(features_concat)[0],1])],axis = 1,name='feature_input_point')
labels_ph = tf.placeholder(dtype=tf.float32, shape=(None,n_classes)) #Attributes[:,:,fraction_idx_var]
#%%
with tf.variable_scope("logistic"):
    logits = tf.matmul(features_concat,Theta)
    labels_binary = labels_ph#tf.div(labels_ph+1,2)
    loss_logistic = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=logits) #,weights=labels_weight

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
lr = LearningRate(learning_rate_base,sess)
optimizer = tf.train.RMSPropOptimizer(
      lr.get_lr(),
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
loss = loss_logistic
loss_regularizer = tf.losses.get_regularization_loss()
#loss += tf.losses.get_regularization_loss()
grad_vars = optimizer.compute_gradients(loss)
print('-'*30)
print('Decompose update ops')
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.apply_gradients(grad_vars)
print('-'*30)
#%%
saver = tf.train.Saver()
init_var = tf.trainable_variables()
pdb.set_trace()
saver_init = tf.train.Saver(init_var)
df_result = pd.DataFrame()
tf.global_variables_initializer().run()
sess.run(iterator_tr.initializer)
if is_save:
    os.makedirs(save_path)
    os.makedirs(save_path+'/plots')
    summary_writer = tf.summary.FileWriter(save_path, graph=tf.get_default_graph())
    with open(save_path+'/description.txt','w') as f:
        f.write(description)
    ##### test #####
    saver.save(sess, save_path+'/model.ckpt')
    ##### test #####
#%%
accum_l = 0
alpha = 0.9
sess.run(iterator_tr.initializer)
logger = Logger(cols=['index','l','l_r','mAP','mAP_pascal','lr'],filename=save_path+'/log.csv',
                is_save=is_save)
eval_interval = max((n_cycles//n_report),500)
tic = time.clock()
for i in range(n_cycles):
    img_ids_tr_v,img_tr_v,labels_tr_v = sess.run([img_ids_tr,img_tr,labels_tr])
    feed_dict = {img_input_ph:img_tr_v,labels_ph:labels_tr_v}
    _, l,l_r = sess.run([train, loss,loss_regularizer], feed_dict)
    accum_l = l*(1-alpha)+alpha*accum_l
    if np.isnan(l):
        pdb.set_trace()
    learning_rate = lr.learning_rate.eval()
    if (i+1) % (n_cycles//4)==0:
        print('-'*30)
        new_lr=sess.run(lr.op_assign_learning_rate,{lr.learning_rate_fh:learning_rate*0.5})
        print(new_lr)
        print('-'*30)
    if i % eval_interval == 0 or i == n_cycles-1:
        print('Time elapse: ',time.clock()-tic)
        tic = time.clock()
        ap_tst,ap_pascal_tst=evaluate(iterator_tst,[img_ids_tst,img_tst,labels_tst],img_input_ph,logits,sess,is_train)
        pdb.set_trace()
        mAP_tst = np.mean(ap_tst)
        mAP_pascal_tst=np.mean(ap_pascal_tst)
        print('no learning rate adaptation')
        learning_rate=lr.adapt(0)
        values = [i,l,l_r,mAP_tst,mAP_pascal_tst,learning_rate]

        logger.add(values)
        print('{} loss: {} loss_regularize: {} mAP: {} mAP_pascal: {} lr: {}'.format(*values))
        logger.save()

        print('learning rate',learning_rate)
        if is_save and mAP_tst >= logger.get_max('mAP'):
            saver.save(sess, save_path+'/model_ES.ckpt')
            
def evaluate_df():
    ap_tst=evaluate(iterator_tst,[img_ids_tst,img_tst,labels_tst],img_input_ph,logits,sess,is_train)
    print(np.mean(ap_tst))
    df = pd.DataFrame()
    df['classes']=classes
    df['ap']=ap_tst
    return df

print('final model')
if is_save:
    saver.save(sess, save_path+'/model.ckpt')
df_f=evaluate_df()

print('ES model')
saver.restore(sess, save_path+'/model_ES.ckpt')
df_ES=evaluate_df()

if is_save:
    df_f.to_csv(save_path+'/F1_f_test.csv')
    df_ES.to_csv(save_path+'/F1_ES_test.csv')
#%%
sess.close()