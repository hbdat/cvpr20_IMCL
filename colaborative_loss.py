# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 23:15:47 2018

@author: badat
"""

import tensorflow as tf
#print('YES balance between feature and Theta')
#%%
upper_bound = 10
#%% asymmetric sigmoid curve
def asym_sigmoid(x,c):
    y = tf.clip_by_value(x+0.5*tf.log(1/c),-80,80)
    return (tf.exp(y)-1/c*tf.exp(-y))/(tf.exp(y)+tf.exp(-y))

#%%
def e2e_OMP_asym_sigmoid_loss_Feature_Graph(Theta,F,F_dict,Label,Label_dict,A,P_L,P_F,G,parallel_iterations=1,c = 2,is_clamp_label=True):
    Label = tf.cast(Label,tf.float32)
    Label_dict = tf.cast(Label_dict,tf.float32)
    def OMP_loss(package):
        f=package[0][tf.newaxis,:]
        label = package[1][tf.newaxis,:]
        binary_label = tf.clip_by_value(label,-1/c,upper_bound)
        active_set = package[2]
        mask = tf.not_equal(active_set,-1)
        support=tf.boolean_mask(active_set,mask)
        pharse_label = package[3]
        pharse_label=tf.boolean_mask(pharse_label,mask)[tf.newaxis,:]
        pharse_feature = package[4]
        pharse_feature=tf.boolean_mask(pharse_feature,mask)[tf.newaxis,:]
        
        
        y = asym_sigmoid(tf.matmul(f,Theta),c)
        # new label filling
        mask_label = 1.0-tf.abs(tf.clip_by_value(label,-1,1))
        if is_clamp_label:
            y=tf.multiply(mask_label,y) + binary_label
        F_active = tf.gather(F_dict,support)
        Label_active = tf.gather(Label_dict,support)
        binary_Label_active = tf.clip_by_value(Label_active,-1/c,upper_bound)
        Y_active = tf.matmul(asym_sigmoid(tf.matmul(F_active,Theta),c),G)
        mask_Label = 1.0-tf.abs(tf.clip_by_value(Label_active,-1,1))
        if is_clamp_label:
            Y_active = tf.multiply(mask_Label,Y_active) + binary_Label_active
        residual_label = y-tf.matmul(pharse_label,Y_active)
        residual_feature = f-tf.matmul(pharse_feature,F_active)
        return residual_label,residual_feature
    R_L,R_F = tf.map_fn(OMP_loss,(F,Label,A,P_L,P_F),parallel_iterations=parallel_iterations,dtype=(tf.float32,tf.float32))
    return R_L,R_F

def e2e_OMP_asym_sigmoid_Feature_Graph(Theta,F,F_dict,Label,Label_dict,k,G,lamb_label,lamb_feature,parallel_iterations=1,c = 2,thresold_coeff = 0.01,is_balance=True,is_clamp_label=True):
    denominator = tf.maximum(lamb_label,lamb_feature)+0.0001
    lamb_label = lamb_label/denominator
    lamb_feature = lamb_feature/denominator
    Label = tf.cast(Label,tf.float32)
    Label_dict = tf.cast(Label_dict,tf.float32)
    g = tf.get_default_graph()
    idx_dict = tf.range(tf.shape(Label_dict)[0])
    with g.as_default():
        Y_dict = tf.matmul(asym_sigmoid(tf.matmul(F_dict,Theta),c),G)
        binary_Label_dict = tf.clip_by_value(Label_dict,-1/c,upper_bound)
        mask_Label = 1.0-tf.abs(tf.clip_by_value(Label_dict,-1,1))
        #new label filling 
        if is_clamp_label:
            Y_dict = tf.multiply(mask_Label,Y_dict)+binary_Label_dict
        def OMP(package):
            f = package[0][tf.newaxis,:]
            label = package[1][tf.newaxis,:]
            binary_label = tf.clip_by_value(label,-1/c,upper_bound)
            mask_label = 1.0-tf.abs(tf.clip_by_value(label,-1,1))
            y = asym_sigmoid(tf.matmul(f,Theta),c)
            # new label filling
            if is_clamp_label:
                y=tf.multiply(mask_label,y) + binary_label
            
            idx_non_identical = tf.boolean_mask(idx_dict,tf.norm(F_dict - f,axis=1)>=0.001)#tf.reshape(tf.where(tf.norm(F_dict - f,axis=1)>=0.0001),[-1])
            Y_s = tf.gather(Y_dict,idx_non_identical)
            F_s = tf.gather(F_dict,idx_non_identical)
            names_s = idx_non_identical
            idx_k = tf.constant(0,name = 'counter_k')
            residual_label = y
            residual_feature = f
            active_set = tf.zeros(k,dtype = tf.int64,name='active_set')
            norm_Y_s = tf.square(tf.norm(Y_s,axis = 1))[:,tf.newaxis]
            norm_F_s = tf.square(tf.norm(F_s,axis = 1))[:,tf.newaxis]
            
            pharse_label = tf.get_variable('pharse_label',shape=[1,0],trainable = False)
            pharse_feature = tf.get_variable('pharse_feature',shape=[1,0],trainable = False)
            #non gradient
            # whatever it takes in, it has to return
            def body(idx_k,active_set,residual_label,residual_feature,pharse_label,pharse_feature):
                
                inner_label = tf.matmul(Y_s,tf.transpose(residual_label))
                inner_feature = tf.matmul(F_s,tf.transpose(residual_feature))
                score_label = tf.div(tf.square(inner_label),norm_Y_s)
                score_feature = tf.div(tf.square(inner_feature),norm_F_s)
                
                if is_balance:
                    score_label=tf.divide(score_label,tf.reduce_max(score_label))
                    score_feature = tf.divide(score_feature,tf.reduce_max(score_feature))
                
                balance_term = 1
                
                mask = tf.sign(inner_label)*tf.sign(lamb_label)+tf.sign(inner_feature)*tf.sign(lamb_feature) #maybe the error is here
                score = lamb_label*score_label+lamb_feature*balance_term*score_feature
                score = tf.multiply(score,mask)
                
                idx_max=tf.argmax(score)[0]
                max_value = score[idx_max,0]
                def true_fn(active_set=active_set,residual_label=residual_label,residual_feature=residual_feature,pharse_label=pharse_label,pharse_feature=pharse_feature):
                    active_set = active_set+tf.one_hot(idx_k,depth=k,on_value=idx_max,dtype=tf.int64)
                    support = active_set[:idx_k+1]
                    Y_active = tf.gather(Y_s,support)
                    
                    pharse_label = tf.matrix_inverse(tf.matmul(Y_active,tf.transpose(Y_active)))
                    pharse_label = tf.matmul(tf.transpose(Y_active),pharse_label)
                    pharse_label = tf.matmul(y,pharse_label)
                    
                    F_active = tf.gather(F_s,support)
                    pharse_feature = tf.matrix_inverse(tf.matmul(F_active,tf.transpose(F_active)))
                    pharse_feature = tf.matmul(tf.transpose(F_active),pharse_feature)
                    pharse_feature = tf.matmul(f,pharse_feature)
                    
                    residual_label = y-tf.matmul(pharse_label,Y_active)
                    residual_feature = f-tf.matmul(pharse_feature,F_active)
                    
                    return active_set,residual_label,residual_feature,pharse_label,pharse_feature
                
                def false_fn(active_set=active_set,residual_label=residual_label,residual_feature=residual_feature,pharse_label=pharse_label,pharse_feature=pharse_feature):
                    active_set = active_set+tf.one_hot(idx_k,depth=k,on_value=tf.cast(-1,tf.int64),dtype=tf.int64)
                    return active_set,residual_label,residual_feature,pharse_label,pharse_feature
                
                cond_non_rend = tf.equal(tf.count_nonzero(tf.equal(active_set,idx_max)),0)
                cond_is_begin = tf.equal(idx_k,0)
                non_redundant =  tf.logical_or(cond_non_rend,cond_is_begin)
                active_set,residual_label,residual_feature,pharse_label,pharse_feature=tf.cond(tf.logical_and(max_value>=thresold_coeff,non_redundant),true_fn,false_fn)
                idx_k = idx_k + 1
                
                return idx_k,active_set,residual_label,residual_feature,pharse_label,pharse_feature
            
            def cond(idx_k,active_set,residual_label,residual_feature,pharse_label,pharse_feature):
                return tf.less(idx_k,k)
            
            idx_k,active_set,residual_label,residual_feature,pharse_label,pharse_feature=tf.while_loop(cond,body,[idx_k,active_set,residual_label,residual_feature,pharse_label,pharse_feature],shape_invariants=[idx_k.get_shape(),active_set.get_shape(),residual_label.get_shape(),residual_feature.get_shape(),tf.TensorShape([1,None]),tf.TensorShape([1,None])],back_prop=False)
            num_support = tf.shape(pharse_label)[1]
            support = active_set[:num_support]
            names_active = tf.gather(names_s,support)
            names_active=tf.pad(names_active, [[0,k-num_support]], "CONSTANT",constant_values=0)
            pharse_label_p = tf.pad(tf.reshape(pharse_label,[-1]), [[0,k-num_support]], "CONSTANT")
            pharse_feature_p = tf.pad(tf.reshape(pharse_feature,[-1]), [[0,k-num_support]], "CONSTANT")
            return names_active,pharse_label_p,pharse_feature_p
    
        A,P_L,P_F = tf.map_fn(OMP,(F,Label),parallel_iterations=parallel_iterations,dtype=(tf.int32,tf.float32,tf.float32),back_prop=False)
        return A,P_L,P_F
#%%