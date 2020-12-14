from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd) 
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import tensorflow as tf
import pandas as pd
import os.path
import numpy as np
import time
#%%
data_set = 'train'
print(data_set)
#nrows = None
path = './data/2017_11/'
num_classes = 5000
net_name = 'resnet_v1_101'
model_path= './model/resnet/oidv2-resnet_v1_101.ckpt'
batch_size = 32
print('dataset: {}'.format(data_set))
num_parallel_calls=2
os.environ["CUDA_VISIBLE_DEVICES"]="3"
is_save = True 
#%%
print('loading data')
df_label = pd.read_csv(path+data_set+'/annotations-human.csv')
labelmap_path = path+'classes-trainable.txt'
dict_path =  path+'class-descriptions.csv'
feature_tfrecord_filename = './TFRecords/'+data_set+'_feature.tfrecords' 
data_path= './image_data/OpenImages/'+data_set+'/' 

#%% split dataframe
print('partitioning data')
capacity = 40000
partition_df = []
t = len(df_label)//capacity
for idx_cut in range(t):
    #print(idx_cut)
    partition_df.append(df_label.iloc[idx_cut*capacity:(idx_cut+1)*capacity])
partition_df.append(df_label.iloc[t*capacity:])
#%%
files=[]
partition_idxs = []
for idx_partition,partition in enumerate(partition_df):
    #print(idx_partition)
    file_partition=[ img_id+'.jpg' for img_id in partition['ImageID'].unique() if os.path.isfile(data_path+img_id+'.jpg')]
    files.extend(file_partition)
    partition_idxs.extend([idx_partition]*len(file_partition))
n_samples = len(files)
print('number of sample: {} dataset: {}'.format(n_samples,data_set))
#%%
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

labelmap, label_dict = LoadLabelMap(labelmap_path, dict_path)

#%%
def read_img(file,partition_idx):
    compressed_image = tf.read_file(data_path+file, 'rb')
    return compressed_image,file,partition_idx

def get_label(compressed_image,file,partition_idx):
    img_id = file.decode('utf-8').split('.')[0]
    df_img_label=partition_df[partition_idx].query('ImageID=="{}"'.format(img_id))
    label = np.zeros(num_classes,dtype=np.int32)
    #print(len(df_img_label))
    for index, row in df_img_label.iterrows():
        try:
            idx=labelmap.index(row['LabelName'])
            label[idx] = 2*row['Confidence']-1
        except:
            pass
    #print(partition_idx)
    return compressed_image,img_id,label
#%%
print('loading data:done')
dataset = tf.data.Dataset.from_tensor_slices((files,partition_idxs))
dataset = dataset.map(read_img,num_parallel_calls)
dataset = dataset.map(lambda compressed_image,file,partition_idx: tuple(tf.py_func(get_label, [compressed_image,file,partition_idx], [tf.string, tf.string, tf.int32])),num_parallel_calls)
dataset = dataset.batch(batch_size).prefetch(batch_size)#.map(PreprocessImage)
iterator=dataset.make_one_shot_iterator().get_next()
#%%
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
g = tf.get_default_graph()
#%%
#with tf.device('/device:GPU:{}'.format(gpu_idx)):
with g.as_default():
    feature_writer = tf.python_io.TFRecordWriter(feature_tfrecord_filename)
    
    saver = tf.train.import_meta_graph(model_path+ '.meta')
    saver.restore(sess, model_path)
    
    input_values = g.get_tensor_by_name('input_values:0')
    logits = g.get_tensor_by_name('resnet_v1_101/pool5:0')
    predictions = g.get_tensor_by_name('multi_predictions:0')
    idx = 0
    while True:
        try:
            (compressed_images,img_ids,labels)=sess.run(iterator)
            print('batch no. {}'.format(idx))
            extract_feature_eval= sess.run(logits,feed_dict={input_values:compressed_images})
            for idx_s in range(extract_feature_eval.shape[0]):
                feature = extract_feature_eval[idx_s,:,:,:].ravel()
                img_id = img_ids[idx_s]
                label = labels[idx_s,:]
                example = tf.train.Example()
                example.features.feature["feature"].bytes_list.value.append(tf.compat.as_bytes(feature.tostring()))
                example.features.feature["img_id"].bytes_list.value.append(tf.compat.as_bytes(img_id))
                example.features.feature["label"].bytes_list.value.append(tf.compat.as_bytes(label.tostring()))
                if is_save:
                    feature_writer.write(example.SerializeToString())
                
            idx += 1
        except tf.errors.OutOfRangeError:
            print('end')
            break
    
    if is_save:
        feature_writer.close()

sess.close()
tf.reset_default_graph()