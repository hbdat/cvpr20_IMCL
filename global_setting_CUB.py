# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:10:18 2018

@author: badat
"""
import D_utility
#%%
path = './'
attr_name_file = path+'data/CUB/CUB_200_2011/attributes/attributes.txt'
class_name_file =  path+'data/CUB/CUB_200_2011/classes.txt'
class_signature_file = path+'data/CUB/CUB_200_2011/attributes/class_attribute_labels_continuous.txt'
#%%
record_path = path+'TFRecord/zs_mask_train_CUB_feature.tfrecords'
trainval_path= path+'TFRecord/zs_mask_trainval_CUB_feature.tfrecords'
validation_path = path+'TFRecord/zs_validation_CUB_feature.tfrecords'
test_path = path+'TFRecord/zs_test_CUB_feature.tfrecords'
sparse_dict_path = path+'TFRecord/zs_mask_train_CUB_feature.tfrecords'
#%%
train_img_path = path+'TFRecord/zs_mask_train_CUB_img.tfrecords'
validation_img_path = path+'TFRecord/zs_validation_CUB_img.tfrecords'
test_img_path = path+'TFRecord/zs_test_CUB_img.tfrecords'
sparse_img_dict_path = path+'TFRecord/zs_mask_train_CUB_img.tfrecords'
#%%
mask_signature_path = './CUB_mask/missing_signature_v2.npz'
batch_size = 32#32
learning_rate_base = 0.001
thresold_coeff = 1e-6
limit_learning_rate = 1e-4
decay_rate_cond = 0.8
signal_strength = 0.3
report_interval=10
k = 3
n_cycles = 1000#D_utility.count_records(record_path)//batch_size
e2e_n_cycles = 1000
e2e_checkpoint_folder = path+'result/e2e_baseline_logistic_CUB'
e2e_saturated_Thetas_model = e2e_checkpoint_folder+'/fraction_{}/model.npz'
e2e_checkpoints_dir = e2e_checkpoint_folder+'/fraction_{}/model_frac_{}_0.0001.ckpt'
regularizers = [0.0001]
early_stopping = True
lr_schedule = 'EMA'

fractions = [0.1,0.2,0.4,0.6,0.8,1]