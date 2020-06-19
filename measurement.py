import numpy as np
from sklearn.metrics import average_precision_score

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

def confusion_matrix(p_class,labels,n_label = 200):
    p_class = np.squeeze(p_class)
    labels = np.squeeze(labels)
    M = np.zeros((n_label,n_label))
    for idx_l in range(n_label):
        p_class_l=p_class[labels == idx_l]
        for idx_l_2 in range(n_label):
            M[idx_l,idx_l_2] = np.sum(p_class_l==idx_l_2)
    return M

def compute_number_misclassified(Prediction,Label):
    binary_Prediction = Prediction.copy()
    binary_Prediction[Prediction>=0]=1
    binary_Prediction[Prediction<0]=-1
    Res=np.abs(binary_Prediction-Label)/2
    Res_p = np.zeros(Label.shape)
    Res_n = np.zeros(Label.shape)
    Res_p[(Res>0)&(Label==1)]=1
    Res_n[(Res>0)&(Label==-1)]=1
    return np.sum(Res_p,0),np.sum(Res_n,0)

def apk(actual, predicted, k=10):

    """

    Computes the average precision at k.



    This function computes the average prescision at k between two lists of

    items.



    Parameters

    ----------

    actual : list

             A list of elements that are to be predicted (order doesn't matter)

    predicted : list

                A list of predicted elements (order does matter)

    k : int, optional

        The maximum number of predicted elements



    Returns

    -------

    score : double

            The average precision at k over the input lists



    """
    #print('precision at '+str(k))
    if len(predicted)>k:

        predicted = predicted[:k]



    score = 0.0

    num_hits = 0.0



    for i,p in enumerate(predicted):

        if p in actual and p not in predicted[:i]:

            num_hits += 1.0

            score += num_hits / (i+1.0)


#
#    if not actual:
#
#        return 0.0



    return score / min(len(actual), k)



def mapk(actual, predicted, k=10):

    """

    Computes the mean average precision at k.



    This function computes the mean average prescision at k between two lists

    of lists of items.



    Parameters

    ----------

    actual : list

             A list of lists of elements that are to be predicted 

             (order doesn't matter in the lists)

    predicted : list

                A list of lists of predicted elements

                (order matters in the lists)

    k : int, optional

        The maximum number of predicted elements



    Returns

    -------

    score : double

            The mean average precision at k over the input lists



    """

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])