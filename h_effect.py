# -*- coding: utf-8 -*-
"""
effect of varying support thresholds
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, chi2
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


#setup
P = 50
d = 5


######simulate categorical data
#IC mean and sigma
def sigma_create(P, rho):
    Sigma = np.zeros((P,P))
    for i in np.arange(P):
        for j in np.arange(P):
            Sigma[i,j] = rho**(np.abs(i-j))
    return Sigma

Mu = np.zeros(P)
rho = 0.5
Sigma = sigma_create(P, rho)

#threshold
np.random.seed(0)
Interval_size = np.random.dirichlet(np.ones(d), size=P).T
Cum_size = np.cumsum(Interval_size, axis=0)

Th_prob = np.zeros((d+1, P))
Th_prob[1:d,:] = Cum_size[0:(d-1),:]
Th_prob[d,:] = 1
Th = norm.ppf(Th_prob, loc=0, scale=1)

#map to categorical data
def cont_to_cate(Data, Th, P, d):
    X_cat = np.zeros_like(Data)
    for j in np.arange(P):
        x_cat = X_cat[:,j]
        for i in np.arange(d):
            x_cat = x_cat + ((Data[:,j]>=Th[i,j]) & (Data[:,j]<Th[i+1,j]))*(i+1)
        X_cat[:,j] = x_cat
    return X_cat.astype(int)

#categorical data to set
def cate_to_set(X_cat, N, P):
    X_set = []
    for i in np.arange(N):
        row_set = []
        for j in np.arange(P):
            row_set.append('x'+str(j+1)+'-'+str(X_cat[i,j]))
        X_set.append(row_set)
    return X_set


#example
N = 1000
np.random.seed(0)
Data_raw = multivariate_normal.rvs(mean=Mu, cov=Sigma, size=N)
X_cat = cont_to_cate(Data_raw, Th, P, d)
X_set = cate_to_set(X_cat, N, P)


######extract frequent patterns
#frequent itemset representation
def fim(X_set, sup_level):
    te = TransactionEncoder()
    te_ary = te.fit(X_set).transform(X_set)
    X_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    FIM_df = apriori(X_df, min_support=sup_level, use_colnames=True)
    FIM_df['length'] = FIM_df['itemsets'].apply(lambda x: len(x))

    FI = FIM_df['itemsets'].values
    FS = FIM_df['support'].values
    FL = FIM_df['length'].values
    FI_num = len(FIM_df)
    return FI, FS, FL, FI_num

def fi_prune(FI, FS, FL, FI_num, h_max, h_min):
    K = np.max(FL)
    H_val = np.linspace(h_max, h_min, num=K)
    H_all = np.zeros(FI_num)
    for j in np.arange(FI_num):
        H_all[j] = H_val[FL[j]-1]
    Keep_idx = (FS>=H_all)
    FI_new = FI[Keep_idx]
    FS_new = FS[Keep_idx]
    FL_new = FL[Keep_idx]
    FI_num_new = len(FI_new)
    return FI_new, FS_new, FL_new, FI_num_new

def fi_group_index(FL, FI_num):
    K = np.max(FL)
    FI_group_idx = np.zeros(K+1)
    for k in np.arange(K):
        FI_group_idx[k] = np.argmax(FL==(k+1))
    FI_group_idx[K] = FI_num
    return K, FI_group_idx.astype(int)

def set_to_fi(X_set, FI, N, FI_num):
    X_fi = np.zeros((N, FI_num))
    for i in np.arange(N):
        for j in np.arange(FI_num):
            X_fi[i,j] = FI[j].issubset(X_set[i])
    return X_fi * 1


#settings
h_max = 0.30
h_min = 0.20 #h_min can take different values to get different curves in Figure 10

#first step: mining using h_min
FI, FS, FL, FI_num = fim(X_set, h_min)

#second step: pruning 
FI, FS, FL, FI_num = fi_prune(FI, FS, FL, FI_num, h_max, h_min)
FI
FS
FL

K, FI_group_idx = fi_group_index(FL, FI_num)

X_fi = set_to_fi(X_set, FI, N, FI_num)


######phase I analysis
#IC: determine alpha limit
def X_fi_resample(X_fi, N_rep):
    Idx = np.random.randint(low=0, high=len(X_fi), size=N_rep)
    return X_fi[Idx,:]

def p_val_vec(X_fi, FS, Weight, const_a, const_c):
    P_ic = FS
    P_hat = np.sum(X_fi*Weight, axis=0) / const_a
    LRT = 2*P_hat*np.log(P_hat/P_ic) + 2*(1-P_hat)*np.log((1-P_hat)/(1-P_ic))
    P_val_vec = 1 - chi2.cdf(const_c*LRT, df=1)
    return P_val_vec

def p_val_mat(X_fi, FS, Weight, N, FI_num, win_size, const_a, const_c): 
    P_val_mat = np.zeros((N, FI_num))
    for i in np.arange(N):
        P_val_mat[i,:] = p_val_vec(X_fi[i:(i+win_size),:], FS, Weight, const_a, const_c)
    return P_val_mat

def p_val_mat_ascend(P_val_mat, FI_group_idx, K, method):
    P_val_mat_acd = np.zeros_like(P_val_mat)
    for k in np.arange(K):
        P_val_mat_acd_k = np.sort(P_val_mat[:,FI_group_idx[k]:FI_group_idx[k+1]], axis=1)
        P_val_mat_acd[:,FI_group_idx[k]:FI_group_idx[k+1]] = P_val_mat_acd_k
    if method == 0:
        P_val_mat_acd = np.sort(P_val_mat, axis=1)
    return P_val_mat_acd

def alpha_group_ascend(FI_group_idx, FI_num, K, alpha, method):
    Information = FI_group_idx[1:K+1] / FI_num
    Alpha_cum = np.zeros(K)
    if method == 1:
        Alpha_cum = 2 - 2*norm.cdf(norm.ppf(1-alpha/2)/np.sqrt(Information))
    if method == 2:
        Alpha_cum = alpha * np.log(1 + (np.exp(1)-1) * Information)
    if method == 3:
        Alpha_cum = alpha * Information
    if method == 4:
        Alpha_cum = alpha * np.power(Information, 2)
    if method == 5:
        Alpha_cum = alpha * np.power(Information, 0.5)    
    Alpha_group = np.zeros(K)
    Alpha_group[0] = Alpha_cum[0]
    for k in np.arange(1,K):
        Alpha_group[k] = Alpha_cum[k] - Alpha_cum[k-1]
    Alpha_group_acd = np.zeros(FI_num)
    for k in np.arange(K):
        alpha_k = Alpha_group[k]
        fi_num_k = FI_group_idx[k+1] - FI_group_idx[k]
        Alpha_group_acd[FI_group_idx[k]:FI_group_idx[k+1]] = (np.arange(fi_num_k)+1)/fi_num_k * alpha_k 
    if method == 0:
        Alpha_group_acd = (np.arange(FI_num)+1) / FI_num * alpha      
    return Alpha_group_acd.reshape((1,FI_num))

def arl_ic(X_fi, FS, FI_group_idx, FI_num, K, Weight, win_size, 
           const_a, const_c, alpha, n_rl, method):
    Alpha = alpha_group_ascend(FI_group_idx, FI_num, K, alpha, method)
    
    RL = np.zeros(n_rl)
    for rep_rl in np.arange(n_rl):
        np.random.seed(rep_rl*6886)
        X_fi_rep = X_fi_resample(X_fi, 800+win_size)
        P_val_rep = p_val_mat(X_fi_rep, FS, Weight, 800, FI_num, win_size, const_a, const_c)
        
        P_val_rep_acd = p_val_mat_ascend(P_val_rep, FI_group_idx, K, method)
        
        Alarm_rep = np.any(P_val_rep_acd<=Alpha, axis=1)
        
        if np.sum(Alarm_rep)==0:
            RL[rep_rl] = 800
        else:
            RL[rep_rl] = np.argmax(Alarm_rep) + 1
    return RL

def alpha_limit(X_fi, FS, FI_group_idx, FI_num, K, Weight, win_size, 
                const_a, const_c, n_rl, method,
                alpha_min, alpha_max, arl_target):
    alpha_l = alpha_min
    alpha_u = alpha_max
    alpha_hat = 0
    alpha_ite_num = 0
    
    while True:
        print(alpha_ite_num)
        alpha_ite_num = alpha_ite_num + 1
        alpha_m = (alpha_l+alpha_u)/2
        RL = arl_ic(X_fi, FS, FI_group_idx, FI_num, K, Weight, win_size, 
                    const_a, const_c, alpha_m, n_rl, method)
        arl = np.mean(RL)
        
        if np.abs(arl-arl_target)<=1e-0 or alpha_ite_num>=20:
            alpha_hat = alpha_m
            break
        if arl<arl_target:
            alpha_u = alpha_m
        if arl>arl_target:
            alpha_l = alpha_m
    return alpha_hat


#settings
win_size = 100
lamb = 0.10
Weight = np.zeros((win_size,1))
for i in np.arange(win_size):
    Weight[i,] = (1-lamb)**(win_size-1-i)
const_a = np.sum(Weight)
const_b = np.sum(Weight**2)
const_c = const_a**2 / const_b

alpha_target = 0.01
arl_target = 1/alpha_target

n_rl = 1000

method_idx = 3 #alpha function: linear with theta = 1.0


#determine alpha limit
alpha_min = 0
alpha_max = 1

alpha_hat = alpha_limit(X_fi, FS, FI_group_idx, FI_num, K, Weight, win_size, 
                        const_a, const_c, n_rl, method_idx,
                        alpha_min, alpha_max, arl_target)

alpha_hat


#oc arl
def arl_oc(Mu, Sigma, FS, FI_group_idx, FI_num, K, Weight, win_size, 
           const_a, const_c, alpha, n_rl, method):
    Alpha = alpha_group_ascend(FI_group_idx, FI_num, K, alpha, method)
    
    RL = np.zeros(n_rl)
    for rep_rl in np.arange(n_rl):
        np.random.seed(rep_rl*6886)
        Data_rep = multivariate_normal.rvs(mean=Mu, cov=Sigma, size=800+win_size)
        
        X_cat_rep = cont_to_cate(Data_rep, Th, P, d)
        X_set_rep = cate_to_set(X_cat_rep, 800+win_size, P)
        X_fi_rep = set_to_fi(X_set_rep, FI, 800+win_size, FI_num)
        
        P_val_rep = p_val_mat(X_fi_rep, FS, Weight, 800, FI_num, win_size, const_a, const_c)
        
        P_val_rep_acd = p_val_mat_ascend(P_val_rep, FI_group_idx, K, method)
        
        Alarm_rep = np.any(P_val_rep_acd<=Alpha, axis=1)
        
        if np.sum(Alarm_rep)==0:
            RL[rep_rl] = 800
        else:
            RL[rep_rl] = np.argmax(Alarm_rep) + 1
    return RL
           

#Mu_oc and Sigma_oc can be set according to OC cases 1 and 3 to get the results in Figure 10
Mu_oc = np.zeros(P)
#Mu_oc[0:5] = 0.9

Sigma_oc = np.zeros((P,P))
for i in np.arange(P):
    for j in np.arange(P):
        Sigma_oc[i,j] = rho**(np.abs(i-j))
        
#for i in np.arange(35,50):
#    for j in np.arange(35,50):
#        Sigma_oc[i,j] = (rho+0.25)**(np.abs(i-j))

RL_oc = arl_oc(Mu_oc, Sigma_oc, FS, FI_group_idx, FI_num, K, Weight, win_size, 
               const_a, const_c, alpha_hat, n_rl, method_idx)

print(np.mean(RL_oc, axis=0))


   
