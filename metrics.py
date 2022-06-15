# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/10 15:51:44
@Author  :   Chu Xiaokai
@Contact :   xiaokaichu@gmail.com
'''
import numpy as np

def get_dcg(ordered_labels):
    return np.sum((2 ** ordered_labels - 1) / np.log2(np.arange(ordered_labels.shape[0]) + 2))

def get_idcg(complete_labels, max_len):
    return get_dcg(np.sort(complete_labels)[:-1 - max_len:-1])

def get_err_k(ranked_labels, K):
    err = 0.0
    R = (np.exp2(ranked_labels) - 1 ) / (2**4)
    for k in range(1, K+1):
        tmp = 1. / k
        for i in range(1, k):
            tmp *= (1 - R[i-1]) 
        tmp *= R[k-1]
        err += tmp
    return err

def calc_err(query_list, K=[1,3,5,10], prefix=''):
    """ expected reciprocal rank """
    errs = [[], [], [], []]
    for item in query_list:
        pred, label = zip(*item)
        label = np.array(label)
        ranking = np.argsort(pred)[::-1]
        
        for i, k in enumerate(K):
            if len(pred) >= k:
                ranked_labels = label[ranking[:k]]
                this_err = get_err_k(ranked_labels, k)
                errs[i].append(this_err)
    
    return {prefix +'_err@'+str(k): np.mean(errs[i]) for i, k in enumerate(K)}

def calc_dcg(query_list, K=[1,3,5,10], prefix=''):
    """ discounted cumulative gain """
    dcgs = [[], [], [], []]
    for item in query_list:
        pred, label = zip(*item)
        label = np.array(label)
        ranking = np.argsort(pred)[::-1]
        for i, k in enumerate(K):
            if len(pred) >= k:
                topk_rankings = ranking[:k]
            else:
                topk_rankings = ranking
            ordered_label = label[topk_rankings]
            dcgs[i].append(get_dcg(ordered_label)) 
    
    return  {prefix +'_dcg@'+str(k): np.mean(dcgs[i]) for i, k in enumerate(K)}


def calc_ndcg(query_list, K=[1,3,5,10], prefix=''):
    """  normalized discounted cumulative gain   """
    ndcgs = [[], [], [], []]
    for item in query_list:
        pred, label = zip(*item)
        label = np.array(label)
        ranking = np.argsort(pred)[::-1]
        
        for i, k in enumerate(K):
            if len(pred) >= k:
                dcg = get_dcg(label[ranking[:k]])
                idcg = get_idcg(label, max_len=k) + 10e-9
                ndcgs[i].append( (dcg/idcg) )

    return {prefix +'_ndcg@'+str(k): np.mean(ndcgs[i])  for i, k in enumerate(K)}

def calc_pnr(query_list):
    """ positive negative rate 
        = positive pairs / negative pairs
    """
    pos_pair = 0.0
    neg_pair = 10e-9
    fair_pair = 0
    for item in query_list:
        for i in range(len(item)):
            for j in range(i+1, len(item)):
                if (item[i][0] > item[j][0] and item[i][1] > item[j][1]) or \
                    (item[i][0] < item[j][0] and item[i][1] < item[j][1]):
                    pos_pair += 1
                elif (item[i][0] > item[j][0] and item[i][1] < item[j][1]) or \
                    (item[i][0] < item[j][0] and item[i][1] > item[j][1]):
                    neg_pair += 1
                else:
                    fair_pair += 1
    return {'pnr': pos_pair / neg_pair}



def evaluate_all_metric(qid_list, label_list, score_list, freq_list=None):
    cur_qid = qid_list[0]
    all_query = []
    tmp = []
    results_dict = {}
    for i in range(len(qid_list)):
        if qid_list[i] != cur_qid: 
            all_query.append(tmp)
            cur_qid = qid_list[i]
            tmp = []
        tmp.append([score_list[i], label_list[i]])
    
    dcg_all = calc_dcg(all_query, prefix='all')
    ndcg_all = calc_ndcg(all_query, prefix='all')
    pnr = calc_pnr(all_query)
    err_all = calc_err(all_query, prefix='all')

    if not freq_list:
        result_list = [dcg_all, ndcg_all,  pnr, err_all]
        for item in result_list:
            results_dict.update(item)
        return results_dict
 
    # evaluate on different frequency data
    cur_qid = qid_list[0]
    cur_freq = int(freq_list[0])
    high_freq_query = []
    mid_freq_query = []
    low_freq_query = []
    tmp = []
    for i in range(len(qid_list)):
        if qid_list[i] != cur_qid: 
            if cur_freq == 0:
                high_freq_query.append(tmp)
            elif cur_freq == 1:
                mid_freq_query.append(tmp)
            elif cur_freq == 2:
                low_freq_query.append(tmp)
            # init
            cur_qid = qid_list[i]
            cur_freq = int(freq_list[i])
            tmp = []
        tmp.append([score_list[i], label_list[i]])
    
    if len(tmp) > 0:
        if cur_freq == 0:
            high_freq_query.append(tmp)
        elif cur_freq == 1:
            mid_freq_query.append(tmp)
        elif cur_freq == 2:
            low_freq_query.append(tmp)
    
    dcg_high_freq = calc_dcg(high_freq_query, prefix='high')
    dcg_mid_freq = calc_dcg(mid_freq_query, prefix='mid')
    dcg_low_freq = calc_dcg(low_freq_query, prefix='low')
    ndcg_high_freq = calc_ndcg(high_freq_query, prefix='high')
    ndcg_mid_freq = calc_ndcg(mid_freq_query, prefix='mid')
    ndcg_low_freq = calc_ndcg(low_freq_query, prefix='low')
    err_high_freq = calc_err(high_freq_query, prefix='high')
    err_mid_freq = calc_err(mid_freq_query, prefix='mid')
    err_low_freq = calc_err(low_freq_query, prefix='low')

    result_list = [dcg_all, dcg_high_freq, dcg_mid_freq, dcg_low_freq, ndcg_all, ndcg_high_freq, ndcg_mid_freq, ndcg_low_freq, pnr, err_all, err_high_freq, err_mid_freq, err_low_freq]
    for item in result_list:
        results_dict.update(item)
    return results_dict