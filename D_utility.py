# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:28:53 2018

@author: badat
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score
import pandas as pd
import pdb
#%%
def get_compress_type(file_name):
    compression_type = ''
    if 'ZLIB' in file_name:
        compression_type = 'ZLIB'
    elif 'GZIP' in file_name:
        compression_type = 'GZIP'
    return compression_type

def compute_AP(predictions,labels):
    num_class = predictions.shape[1]
    ap=np.zeros(num_class)
    ap_pascal = np.zeros(num_class)
    for idx_cls in range(num_class):
        prediction = np.squeeze(predictions[:,idx_cls])
        label = np.squeeze(labels[:,idx_cls])
#        mask = np.abs(label)==1
#        if np.sum(label>0)==0:
#            continue
        binary_label=np.clip(label,0,1)
        ap[idx_cls]=average_precision_score(binary_label,prediction)#average_precision_score(binary_label,prediction[mask])
        ap_pascal[idx_cls]=calc_pr_ovr_noref(binary_label,prediction)[-1]
    return ap,ap_pascal

def evaluate(iterator,tensors,features,logits,sess,is_train=None):
    if is_train is not None:    
        print('switch to inference model')
        sess.run(is_train.assign(False))
    sess.run(iterator.initializer)
    predictions = []
    labels = []
    while True:
        try:
            img_ids_v,features_v,labels_v = sess.run(tensors)
            feed_dict = {features:features_v}
            logits_v = sess.run(logits, feed_dict)
            predictions.append(logits_v)
            labels.append(labels_v)
        except tf.errors.OutOfRangeError:
            print('end')
            break
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    assert predictions.shape==labels.shape,'invalid shape'
    if is_train is not None:
        sess.run(is_train.assign(True))
    return compute_AP(predictions,labels)

def evaluate_latent_noise(iterator,tensors,features,logits,sess,is_train=None):
    if is_train is not None:    
        print('switch to inference model')
        sess.run(is_train.assign(False))
    sess.run(iterator.initializer)
    v_predictions = []
    h_predictions = []
    labels = []
    while True:
        try:
            img_ids_v,features_v,labels_v = sess.run(tensors)
            feed_dict = {features:features_v}
            logits_v = sess.run(logits, feed_dict)
            v_predictions.append(logits_v[0])
            h_predictions.append(logits_v[1])
            labels.append(labels_v)
        except tf.errors.OutOfRangeError:
            print('end')
            break
    v_predictions = np.concatenate(v_predictions)
    h_predictions = np.concatenate(h_predictions)
    labels = np.concatenate(labels)
    assert v_predictions.shape==labels.shape,'invalid shape'
    if is_train is not None:
        sess.run(is_train.assign(True))
    ap_v,ap_v_pascal = compute_AP(v_predictions,labels)
    ap_h,ap_h_pascal = compute_AP(h_predictions,labels)
    return ap_v,ap_v_pascal,ap_h,ap_h_pascal
#%%
def calc_pr_ovr_noref(counts, out):
  """
  [P, R, score, ap] = calc_pr_ovr(counts, out, K)
  Input    :
    counts : number of occurrences of this word in the ith image
    out    : score for this image
    K      : number of references
  Output   :
    P, R   : precision and recall
    score  : score which corresponds to the particular precision and recall
    ap     : average precision
  """ 
  #binarize counts
  counts = np.array(counts > 0, dtype=np.float32);
  tog = np.hstack((counts[:,np.newaxis].astype(np.float64), out[:, np.newaxis].astype(np.float64)))
  ind = np.argsort(out)
  ind = ind[::-1]
  score = np.array([tog[i,1] for i in ind])
  sortcounts = np.array([tog[i,0] for i in ind])

  tp = sortcounts;
  fp = sortcounts.copy();
  for i in range(sortcounts.shape[0]):
    if sortcounts[i] >= 1:
      fp[i] = 0.;
    elif sortcounts[i] < 1:
      fp[i] = 1.;
  P = np.cumsum(tp)/(np.cumsum(tp) + np.cumsum(fp));

  numinst = np.sum(counts);

  R = np.cumsum(tp)/numinst

  ap = voc_ap(R,P)
  return P, R, score, ap


def voc_ap(rec, prec):
  """
  ap = voc_ap(rec, prec)
  Computes the AP under the precision recall curve.
  """

  rec = rec.reshape(rec.size,1); prec = prec.reshape(prec.size,1)
  z = np.zeros((1,1)); o = np.ones((1,1));
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap
#%%
def count_records(file_name):
    c = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        c += 1
    return c

def preprocessing_graph(G):
    np.fill_diagonal(G,1)
    for idx_col in range(G.shape[1]):
        normalizer = np.sum(G[:,idx_col])
        G[:,idx_col] = G[:,idx_col]*1.0/normalizer
    return G

def generate_missing_signature(attributes,fractions):
    n_attr = attributes.shape[1]
    n_sample = attributes.shape[0]
    Masks = np.zeros((attributes.shape[0],attributes.shape[1],len(fractions)),dtype=np.float32)
    for idx_f in range(len(fractions)):
        select_fraction = fractions[idx_f]
        for idx_a in range(n_attr):
            sub_l = np.zeros(n_sample)
            attr = attributes[:,idx_a]
            pos_idx=np.where(attr>0)[0]
            n_pos  =len(pos_idx)
            if n_pos > 0:
                n_sub_pos = max(int(n_pos*select_fraction),1)
                sub_pos_idx = np.random.choice(pos_idx,n_sub_pos,False)
                sub_l[sub_pos_idx]=attr[sub_pos_idx]
            
            neg_idx=np.where(attr<0)[0]
            n_neg = len(neg_idx)
            if n_neg > 0:
                n_sub_neg = max(int(n_neg*select_fraction),1)
                sub_neg_idx = np.random.choice(neg_idx,n_sub_neg,False)
                sub_l[sub_neg_idx]=attr[sub_neg_idx]
            
            Masks[:,idx_a,idx_f]=sub_l
    return Masks

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
def LoadClassSignature(class_signature_file):
    signatures = []
    with open(class_signature_file,"r") as f:
        lines=f.readlines()
        for line in lines:
            attrs=line.rstrip('\n').split(' ')
            signatures.append(attrs)
    return np.array(signatures).astype(np.float32)/100

def quantizeSignature_mean(signatures):
    signatures_q = np.ones(signatures.shape)*-1
    signatures_m = np.mean(signatures,axis=0)
    signatures_s_m = signatures-signatures_m[np.newaxis,:]
    signatures_q[signatures_s_m>=0]=1
    signatures_q[signatures_s_m<0]=0
    return signatures_q

def quantizeSignature(signatures):
    signatures_q = np.ones(signatures.shape)*-1
    signatures_q[signatures>=0.5]=1
    signatures_q[signatures<0.5]=0
    return signatures_q

def quantizeSignature_0(signatures):
    signatures_q = np.ones(signatures.shape)*-1
    signatures_q[signatures>0]=1
    signatures_q[signatures<=0]=0
    return signatures_q

def DAP(sigmoid_Predictions,signatures_q,signatures):
    n = sigmoid_Predictions.shape[0]
    T = signatures_q[:,:,np.newaxis]*np.ones((1,1,n))
    prior = np.mean(signatures_q,0)
    # eliminate degenerative prior
    prior[prior==0]=0.5
    prior[prior==1]=0.5
    #
    clss_prior = np.multiply(signatures_q,prior)+np.multiply(1-signatures_q,1-prior)
    log_clss_prior = np.sum(np.log(clss_prior),1)
    #
    P_T = sigmoid_Predictions[:,:,np.newaxis].T
    Inter = np.multiply(T,P_T)+np.multiply(1-T,1-P_T)
    Score=np.sum(np.log(Inter),axis=1)
    # calibrate prior
    Score_calibrate = Score - log_clss_prior[:,np.newaxis]
    
    return Score_calibrate

#def DAP_sum(Predictions,signatures_q,signatures):
#    n = Predictions.shape[0]
#    T = signatures_q[:,:,np.newaxis]*np.ones((1,1,n))
#    prior = np.mean(signatures_q,0)
#    # eliminate degenerative prior
#    prior[prior==0]=0.5
#    prior[prior==1]=0.5
#    #
#    clss_prior = np.multiply(signatures_q,prior)+np.multiply(1-signatures_q,1-prior)
#    clss_prior = np.sum(clss_prior,1)
#    #
#    P_T = Predictions[:,:,np.newaxis].T
#    Inter = np.multiply(T,P_T)+np.multiply(1-T,1-P_T)
#    Score=np.sum(Inter,axis=1)
#    # calibrate prior
#    Score_calibrate = Score / clss_prior[:,np.newaxis]
#    
#    return Score_calibrate
def mean_acc(pred,true):
    clss = np.unique(true)
    acc = np.ones(len(clss))*-1
    for idx_c,c in enumerate(clss):
        pred_clss = pred[true==c]
        acc[idx_c] = np.sum(pred_clss==c)*1.0/len(pred_clss)
    return acc

def zeroshot_evaluation(Score_calibrate,t_labels,seen,unseen,mode = 'mean_acc'):
    t_labels = np.squeeze(t_labels)
    seen_tst_set = np.array([idx_s  for idx_s in range(len(t_labels)) if t_labels[idx_s] in seen])
    unseen_tst_set = np.array([idx_s  for idx_s in range(len(t_labels)) if t_labels[idx_s] in unseen])
    
    unconsider_class = np.array([idx_c  for idx_c in range(Score_calibrate.shape[1]) if (idx_c not in seen) and (idx_c not in unseen)])
    acc_u_u=-1
    acc_u_a=-1
    acc_s_s=-1
    acc_s_a=-1
#    for idx_c in unseen:
#        print(np.sum(t_labels==idx_c))
#    pdb.set_trace()
    if len(unconsider_class)>0:
        print('-'*30)
        print('detect unconsider class: ',unconsider_class.shape)
        print('-'*30)
        Score_calibrate[:,unconsider_class] = -1000
    
    if len(unseen_tst_set)>0:
        #u->a
        Score_calibrate_u_a = Score_calibrate[unseen_tst_set,:]
        clss_u_a = np.argmax(Score_calibrate_u_a,1)
        if mode == 'mean_acc':
            acc_u_a = np.mean(mean_acc(clss_u_a,t_labels[unseen_tst_set]))#
        else:
            acc_u_a = np.sum((clss_u_a-t_labels[unseen_tst_set])==0)*1.0/len(unseen_tst_set)
        #u->u
        Score_calibrate_u_u = Score_calibrate_u_a[:,unseen]
        clss_u_u = np.argmax(Score_calibrate_u_u,1)
        clss_u_u=np.array([unseen[l] for l in clss_u_u])
        if mode == 'mean_acc':
            acc_u_u = np.mean(mean_acc(clss_u_u,t_labels[unseen_tst_set]))##
        else:
            acc_u_u = np.sum((clss_u_u-t_labels[unseen_tst_set])==0)*1.0/len(unseen_tst_set)
    if len(seen_tst_set)>0:
        #s->a
        Score_calibrate_s_a = Score_calibrate[seen_tst_set,:]
        clss_s_a = np.argmax(Score_calibrate_s_a,1)
        if mode == 'mean_acc':
            acc_s_a = np.mean(mean_acc(clss_s_a,t_labels[seen_tst_set]))#
        else:
            acc_s_a = np.sum((clss_s_a-t_labels[seen_tst_set])==0)*1.0/len(seen_tst_set)
        #s->s
        Score_calibrate_s_s = Score_calibrate_s_a[:,seen]
        clss_s_s = np.argmax(Score_calibrate_s_s,1)
        clss_s_s=np.array([seen[l] for l in clss_s_s])
        if mode == 'mean_acc':
            acc_s_s = np.mean(mean_acc(clss_s_s,t_labels[seen_tst_set]))#
        else:
            acc_s_s =np.sum((clss_s_s-t_labels[seen_tst_set])==0)*1.0/len(seen_tst_set)
    return acc_u_u,acc_u_a,acc_s_s,acc_s_a
#%%
def signature_completion(Label_completion_v,sparse_dict_label_v,signature_q,quantization):
    unique_labels = np.unique(sparse_dict_label_v)
    signature_comp=np.zeros(signature_q.shape)
    for l in unique_labels:
        mask_l = sparse_dict_label_v == l
        signature_comp[l,:]=np.mean(Label_completion_v[mask_l,:],0)
    if quantization:
        raise Exception('not implemented')
#        signature_comp[signature_comp>0]=1
#        signature_comp[signature_comp<0]=-1
    return signature_comp
def evaluate_completion(signature_comp,signature_q,quantization):
    mask_comp = np.sum(np.abs(signature_comp),1)!=0
    if quantization:
        return np.sum((signature_comp!=signature_q)[mask_comp,:],1)
    else:
        return np.sum(np.abs((signature_comp-signature_q))[mask_comp,:],1)
def sparse_coding_signature(target_o,dic_o,k=3,thresold_coeff = 0.01,bias = False,c=2.0):
    if bias == False:
        c = 1.0
    target_b = np.clip(target_o,-1/c,1.0)
    dic_b = np.clip(dic_o,-1/c,1.0)
    n_target = target_o.shape[0]
    reconstruct = np.zeros(target_o.shape,dtype=np.float32)
    Active_set = np.ones((n_target,k),dtype=np.int32)*-1
    
    residual = target_b.copy()
    for idx_k in range(k):
        inner = np.matmul(residual,dic_b.T)
        idx_max_all = np.argmax(inner,axis=1)
        for idx_t in range(n_target):
            idx_max = idx_max_all[idx_t]
            if inner[idx_t,idx_max] < thresold_coeff :
                continue
            Active_set[idx_t,idx_k]=idx_max
            A = dic_b[Active_set[idx_t,:idx_k+1],:]
            pharse = np.linalg.inv(np.matmul(A,np.transpose(A)))
            pharse = np.matmul(np.transpose(A),pharse)
            pharse = np.matmul(residual[idx_t,:],pharse)
            residual[idx_t,:] -= np.matmul(pharse,A)
    #reconstruct
    for idx_t in range(n_target):
        active_set = Active_set[idx_t,:]
        active_set = active_set[active_set!=-1]
        A = dic_o[active_set,:]
        pharse = np.linalg.inv(np.matmul(A,np.transpose(A)))
        pharse = np.matmul(np.transpose(A),pharse)
        pharse = np.matmul(target_o[idx_t,:],pharse)
        reconstruct[idx_t,:] = np.matmul(pharse,A)
    return reconstruct,Active_set

def normalize_signature(signature):
    eps = 1e-6
    return signature/(np.linalg.norm(signature,axis=1)[:,np.newaxis]+eps)

def project_signature(signature_pred,signature_miss,proj_cols,unknown_signal,strengh_known):
    signature_proj = signature_pred.copy()
    signature_miss_sub = signature_miss[proj_cols,:]
    signature_proj_sub = signature_proj[proj_cols,:].copy()
    mask = signature_miss_sub!=unknown_signal
    signature_proj_sub[mask]=signature_miss_sub[mask]*strengh_known
    signature_proj[proj_cols,:]=signature_proj_sub
    return signature_proj
#
#def split_signature(signature_pred,signature_miss,unknown_signal):
#    mask_unknown = signature_miss == unknown_signal
#    signature_comp_M = signature_pred.copy()
#    signature_comp_M[mask_unknown]=0
#    signature_miss_M = signature_miss.copy()
#    signature_miss_M[!mask_unknown]=0
#    return signature_comp_M,signature_miss_M

def project_signature_sigmoid(signature_pred,signature_miss,proj_cols):
    signature_proj = signature_pred.copy()
    signature_miss_sub = signature_miss[proj_cols,:]
    signature_proj_sub = signature_proj[proj_cols,:].copy()
    mask = signature_miss_sub!=0.5
    signature_proj_sub[mask]=signature_miss_sub[mask]
    signature_proj[proj_cols,:]=signature_proj_sub
    return signature_proj

def project_signature_tanh(signature_pred,signature_miss,proj_cols):
    signature_proj = signature_pred.copy()
    signature_miss_sub = signature_miss[proj_cols,:]
    signature_proj_sub = signature_proj[proj_cols,:].copy()
    mask = signature_miss_sub!=0
    signature_proj_sub[mask]=signature_miss_sub[mask]
    signature_proj[proj_cols,:]=signature_proj_sub
    return signature_proj

def project_unit_norm(F):
    f_dim = tf.shape(F)[1]
    norm = tf.matmul(tf.norm(F,ord=2,axis=1)[:,tf.newaxis],tf.ones((1,f_dim))) + 1e-6
    return tf.divide(F,norm)

class Logger:
    def __init__(self,filename,cols,is_save=True):
        self.df = pd.DataFrame()
        self.cols = cols
        self.filename=filename
        self.is_save=is_save
    def add(self,values):
        self.df=self.df.append(pd.DataFrame([values],columns=self.cols),ignore_index=True)
    def save(self):
        if self.is_save:
            self.df.to_csv(self.filename)
    def get_max(self,col):
        return np.max(self.df[col])
    
class LearningRate:
    def __init__(self,lr,sess,signal_strength=0.3,limit_lr_scale=1e-3,decay_rate=0.8,patient=2):
        self.learning_rate = tf.Variable(lr,trainable = False,dtype=tf.float32)
        self.exp_moving_avg_old = 0
        self.exp_moving_avg_new = 0
        self.signal_strength = signal_strength
        self.limit_lr_scale = 1e-3
        self.decay_rate = 0.8
        self.patient = patient
        self.op_reset = self.learning_rate.assign(lr)
        self.limit_learning_rate = lr*limit_lr_scale
        self.m = 0
        self.sess = sess
        self.learning_rate_fh=tf.placeholder(dtype=tf.float32,shape=())
        self.op_assign_learning_rate = self.learning_rate.assign(self.learning_rate_fh)
        self.is_reset = False
        
    def get_lr(self):
        return self.learning_rate
    
    def adapt(self,mAP):
        cur_lr = self.learning_rate.eval()
        new_lr = cur_lr
        if self.is_reset:
            self.exp_moving_avg_old=self.exp_moving_avg_new=mAP
            self.is_reset = False
        else:
            self.exp_moving_avg_old=self.exp_moving_avg_new
            self.exp_moving_avg_new = self.exp_moving_avg_new*(1-self.signal_strength)+mAP*self.signal_strength
            
            if self.exp_moving_avg_new<self.exp_moving_avg_old and cur_lr >= self.limit_learning_rate and self.m <= 0:
                print('Adjust learning rate')
                new_lr=self.sess.run(self.op_assign_learning_rate,{self.learning_rate_fh:cur_lr*self.decay_rate})
                self.m = self.patient
            self.m -= 1
        return new_lr