# -*- coding: utf-8 -*-
"""
real case study on the air quality monitoring in Beijing city
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import matplotlib.pyplot as plt


#load categorical dataset
path = 'C:/Users/kwangai/Desktop/'
Data_cat = pd.read_csv(path+'beijing_air.csv', header=0, index_col=[0,1,2])
P = Data_cat.shape[1]
D = np.array([6, 6, 6, 6, 6, 6, 4, 4, 4, 2, 4, 8])


#training data 
X_cat_ic = np.array(Data_cat.iloc[:184,:])
N_ic = X_cat_ic.shape[0]

X_cat_ic = X_cat_ic[np.arange(0,N_ic,step=3)] #sampel every 3 days to mitigate autocorrelation
N_ic = X_cat_ic.shape[0]

#future data
X_cat_oc = np.array(Data_cat.iloc[184:,:])
N_oc = X_cat_oc.shape[0]

X_cat_oc = X_cat_oc[np.arange(0,N_oc,step=3)] #sampel every 3 days to mitigate autocorrelation
N_oc = X_cat_oc.shape[0]


#cross and autocorrelation analysis based on the cramer's v statistic
def cramer_v(X1, X2):
    Pi = np.array(pd.crosstab(X1, X2, normalize=True))
    d1, d2 = Pi.shape
            
    Pi_col = np.sum(Pi, axis=0, keepdims=True)
    Pi_row = np.sum(Pi, axis=1, keepdims=True)
    Pi_prodt = np.dot(Pi_row, Pi_col)
    idx = (Pi != Pi_prodt)
    cv = np.sum((Pi[idx]-Pi_prodt[idx])**2 / Pi_prodt[idx]) / (np.min([d1, d2])-1)  
    return cv

def cross_cramer_v(X, P):
    CV_mat = np.zeros((P,P))
    CV_vec = []
    for j1 in np.arange(P-1):
        for j2 in np.arange(j1+1,P):
            X1 = X[:,j1]
            X2 = X[:,j2]
            CV_mat[j1,j2] = cramer_v(X1, X2)
            CV_vec.append(CV_mat[j1,j2])
    return CV_mat, CV_vec

def auto_cramer_v(X, N, P, Lag):
    CV = np.zeros((P, Lag))
    for j in np.arange(P):
        for lag in np.arange(Lag):
            X1 = X[0:(N-lag-1), j]
            X2 = X[(lag+1):N, j]
            CV[j,lag] = cramer_v(X1, X2)
    return CV


CV_cross_mat, CV_cross_vec = cross_cramer_v(X_cat_ic, P)
pd.DataFrame(CV_cross_vec).describe()

Lag = 1
CV_auto = auto_cramer_v(X_cat_ic, N_ic, P, Lag)
pd.DataFrame(CV_auto).describe()
#Table 2 in the manuscript


######extract frequent patterns
#categorical data to set
def cate_to_set(X_cat, N, P):
    X_set = []
    for i in np.arange(N):
        row_set = []
        for j in np.arange(P):
            row_set.append('x'+str(j+1)+'-'+str(X_cat[i,j]))
        X_set.append(row_set)
    return X_set

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


X_set_ic = cate_to_set(X_cat_ic, N_ic, P)

sup_level = 0.30 #support threshold
FI, FS, FL, FI_num = fim(X_set_ic, sup_level)


FI[FL==1]
FI[FL==2]
FI[FL==3]
FI[FL==4]
FI[FL==5]
[len(FI[FL==i]) for i in np.arange(1,6)]

K, FI_group_idx = fi_group_index(FL, FI_num)
X_fi_ic = set_to_fi(X_set_ic, FI, N_ic, FI_num)


######phase I analysis
#IC: determine alpha limit by resampling
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
    if method == 1: #alpha spending function by normal distribution
        Alpha_cum = 2 - 2*norm.cdf(norm.ppf(1-alpha/2)/np.sqrt(Information))
    if method == 2: #alpha spending function by logarithmic function
        Alpha_cum = alpha * np.log(1 + (np.exp(1)-1) * Information)
    if method == 3: #alpha spending function by power function with theta = 1.0
        Alpha_cum = alpha * Information
    if method == 4: #alpha spending function by power function with theta = 2.0
        Alpha_cum = alpha * np.power(Information, 2)
    if method == 5: #alpha spending function by power function with theta = 0.5
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
    if method == 0: #no sequential testing
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
        
        if np.abs(arl-arl_target)<=1e-1 or alpha_ite_num>=20:
            alpha_hat = alpha_m
            break
        if arl<arl_target:
            alpha_u = alpha_m
        if arl>arl_target:
            alpha_l = alpha_m
    return alpha_hat


#settings
win_size = 20 #moving window size
lamb = 0.10 #smoothing parameter
Weight = np.zeros((win_size,1))
for i in np.arange(win_size):
    Weight[i,] = (1-lamb)**(win_size-1-i)
const_a = np.sum(Weight)
const_b = np.sum(Weight**2)
const_c = const_a**2 / const_b

alpha_target = 0.01
arl_target = 1/alpha_target

n_rl = 100


#determine alpha limit by bisection search
alpha_min = 0.05
alpha_max = 0.10

Alpha_hat = np.zeros(6)
for method_idx in np.arange(6):
    print(method_idx)
    alpha_hat = alpha_limit(X_fi_ic, FS, FI_group_idx, FI_num, K, Weight, win_size, 
                            const_a, const_c, n_rl, method_idx,
                            alpha_min, alpha_max, arl_target)
    Alpha_hat[method_idx] = alpha_hat

Alpha_hat


######phase II analysis
#control chart
def chart_point(X_fi, FS, FI_group_idx, FI_num, K, Weight, win_size, const_a, const_c, 
                Alpha, method_idx):
    Alpha_acd = alpha_group_ascend(FI_group_idx, FI_num, K, 
                                   Alpha[method_idx], method_idx)
    P_val = p_val_mat(X_fi, FS, Weight, len(X_fi)-win_size, FI_num, 
                      win_size, const_a, const_c)
    P_val_acd = p_val_mat_ascend(P_val, FI_group_idx, K, method_idx)
    
    P_val_ratio = P_val_acd / Alpha_acd
    Alarm_idx = np.argmax(P_val_ratio<1, axis=1)
    P_val_vec = np.zeros(len(X_fi)-win_size)
    for i in np.arange(len(P_val_vec)):
        P_val_vec[i] = P_val_ratio[i,Alarm_idx[i]]
    return P_val_ratio, P_val_vec, Alarm_idx


#oc chart
X_cat_oc1 = X_cat_ic[(N_ic-win_size):N_ic,:]
X_cat_oc2 = X_cat_oc
X_cat_oc_all = np.concatenate((X_cat_oc1, X_cat_oc2), axis=0)

X_set_oc = cate_to_set(X_cat_oc_all, N_oc+win_size, P)
X_fi_oc = set_to_fi(X_set_oc, FI, N_oc+win_size, FI_num)

P_val_ratio, P_val_vec, Alarm_idx = chart_point(X_fi_oc, FS, FI_group_idx, FI_num, K, 
                                                Weight, win_size, const_a, const_c, 
                                                Alpha_hat, 5)

np.mean(P_val_vec<1)





