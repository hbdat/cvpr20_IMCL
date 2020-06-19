# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:10:18 2018

@author: badat
"""
import D_utility
#%%
path = './'
labelmap_path = path+'data/2017_11/classes-trainable.txt'
dict_path =  path+'data/2017_11/class-descriptions.csv'
model_path = path+'model/resnet/oidv2-resnet_v1_101.ckpt'
record_path = path+'TFRecord/train_feature.tfrecords'
validation_path = path+'TFRecord/validation_feature.tfrecords'
test_path = path+'TFRecord/test_feature.tfrecords'
sparse_dict_path = path+'TFRecord/full_sparse_dict_feature_with_label.tfrecords'
label_graph_path = path+'label_graph/graph_label_naive.npy'#'./label_graph/graph_label_tf_idf.npz'
saturated_Thetas_model = path+'result/baseline_logistic_OpenImages.npz'
indicator_path = path+'indicator/Indicator_1539792368.07521.csv'
#%%
batch_size = 32#1000#
learning_rate_base = 0.001
e2e_learning_rate_base = 1e-5
e2e_limit_learning_rate = 1.25e-9
thresold_coeff = 1e-3
limit_learning_rate = 1.25e-4
decay_rate_cond = 0.8
signal_strength = 0.1
report_interval=1000
early_stopping = True
k = 3
n_cycles = 3657355//batch_size#D_utility.count_records(record_path)//batch_size
best_alphas_colaborative = [1]
best_alphas_feature = [0.5]