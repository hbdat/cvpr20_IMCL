# coding: utf-8
# In[4]:
#import random
import numpy as np
import pandas as pd
import pdb
import tensorflow as tf
import os
from sklearn.metrics import average_precision_score
from global_setting_OpenImage import test_path
idx_GPU=6
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)   
#%%
print('Open Image V3 local')
path = './data/2017_11/'#'./model/2016_08_inception/'#
#test_path = './TFRecord/test_feature.tfrecords'
is_save = False
print('is_save: {}'.format(is_save))
c = 2.0
folder_name = 'evaluation_test_set_without_asym'
#%% load prediction
print('load prediction')

model_sources = ['./result/ablation_linearlayer_fixedG_1_sum1_OpenImage_0.001_32_0.8_signal_str_0.1_114292_GPU_7_thresCoeff_0.001_c_2.0_stamp_1560538474.4739165/models_ES.npz']#global_setting_OpenImage.saturated_Thetas_model
is_augmenteds = [True]

#model_sources = ['./result/linearlayer_noG_OpenImage_0.001_32_0.8_signal_str_0.1_114292_GPU_4_thresCoeff_0.001_c_2.0_stamp_1539746284.1236885/models_ES.npz',
#                 './result/best_augmentedLayerLinear_G_1_sum1_OpenImage_0.001_32_0.8_signal_str_0.1_114292_GPU_1_thresCoeff_0.001_c_2.0_stamp_1539792368.07521/models_ES.npz',
#                 './result/linearlayer_fixedG_1_sum1_OpenImage_0.001_32_0.8_signal_str_0.1_114292_GPU_3_thresCoeff_0.001_c_2.0_stamp_1539786366.9074607/models_ES.npz',
#                 './result/vec_lambda_augLinear_G_1_sum1_OpenImage_0.001_32_0.8_signal_str_0.1_114292_GPU_4_thresCoeff_0.001_c_2.0_stamp_1539983029.6347945/models_ES.npz',
#                 './result/noclamp_linearlayer_G_1_sum1_OpenImage_0.001_32_0.8_signal_str_0.1_114292_GPU_6_thresCoeff_0.001_c_2.0_stamp_1539792878.1971025/models_ES.npz',
#                 './result/self_training_OpenImage_0.001_32_0.8_signal_str_0.1_114292_1_GPU_7_thresCoeff_0.001_stamp_1539906163.067778/model.npz',
#                 './result/inv_sqrt_baseline_0.25_32_0.8_signal_str_0.1_457168_1_GPU_3_thresCoeff_1e-06_stamp_1533970053.34.npz']#global_setting_OpenImage.saturated_Thetas_model
#is_augmenteds = [True,True,True,True,True,False,False]
#%%
def parser(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'feature': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id = parsed['img_id']
    feature = tf.decode_raw( parsed['feature'],tf.float32)
    label = tf.decode_raw( parsed['label'],tf.int32)
    return img_id,feature,label
#%%
sess = tf.InteractiveSession()
#%%
dataset_in = tf.data.TFRecordDataset(test_path)
dataset_in = dataset_in.map(parser).batch(200000)
(img_test_ids,F_test,test_labels) = dataset_in.make_one_shot_iterator().get_next()
F_test = tf.concat([F_test,tf.ones([tf.shape(F_test)[0],1])],axis = 1)

(img_test_ids,F_test,test_labels)=sess.run([img_test_ids,F_test,test_labels])
pdb.set_trace()
#%%
labelmap_path=path+'classes-trainable.txt'#'labelmap.txt'#
dict_path=path+'class-descriptions.csv'#'dict.csv'#
def LoadLabelMap():
    labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path)]
    label_dict = {}
    for line in tf.gfile.GFile(dict_path):
        words = [word.strip(' "\n') for word in line.split(',', 1)]
        label_dict[words[0]] = words[1]
    return labelmap, label_dict

labelmap, label_dict = LoadLabelMap()
list_label = []
for id_name in labelmap:
    list_label.append(label_dict[id_name])
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
#%%
def asym_sigmoid(x,c):
    y = x+0.5*np.log(1/c)
    return (np.exp(y)-1/c*np.exp(-y))/(np.exp(y)+np.exp(-y))    
#%% check path exist
for model_name in model_sources:
    np.load(model_name)
    print(model_name,'-->success')
print('-'*30)
def collapse_Theta(data):
    Thetas_1 = data['Thetas_1']
    Thetas_f = data['Thetas_f']
    Theta_1 = Thetas_1[:,:,-1]
#        pdb.set_trace()
    # reserving perspective transformation
    theta_1_n_row =Theta_1.shape[0]
    Theta_1=np.concatenate((Theta_1,np.zeros((theta_1_n_row,1))),axis=1)
    Theta_1[-1,-1]=1
    #
    
    Theta_f = Thetas_f[:,:,-1]
    Theta = np.matmul(Theta_1,Theta_f)
    return Theta
#%%
for idx_model in range(len(model_sources)):
    model_name = model_sources[idx_model]
    
        
    components = model_name.split('/')
    if len(components) == 3:
        name = folder_name+'/'+components[-1]
    else:
        name = folder_name+'/'+components[-2]
    
    print(name)
    
    if not os.path.exists('./result/'+name) and is_save:
        os.makedirs('./result/'+name)
    
    data = np.load(model_name)
    if is_augmenteds[idx_model]:
        Theta=collapse_Theta(data)
    else:
        Thetas = data['Thetas']
        dims = Thetas.shape
        if len(dims) == 2:
            Thetas = Thetas[:,:,np.newaxis]
        Theta = Thetas[:,:,-1]
    df_ap = pd.DataFrame()
    df_ap['Label'] = list_label
    res_mAP = []
    
    Prediction = np.matmul(F_test,Theta)#asym_sigmoid(,c)
    ap=compute_AP(Prediction,test_labels)
    weight_ap = np.sum(np.abs(test_labels),axis = 0)
    v_mAP = np.mean(ap)
    v_w_mAP = np.inner(weight_ap,ap) / np.sum(weight_ap)
    df_ap['ap']=ap
    res_mAP.append([v_mAP,v_w_mAP])
    print('weighted ap: {} mean ap: {}'.format(v_w_mAP,v_mAP))
    if is_save:
        df_ap.to_csv('./result/'+name+'/mAP.csv')