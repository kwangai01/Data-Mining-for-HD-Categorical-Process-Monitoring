# -*- coding: utf-8 -*-
"""
real case study on the activity monitoring at smarthome

"""


import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import seaborn as sns
import matplotlib.pyplot as plt


#read data
#31 days, each day with 18 hours, ech hour with 3 observations (20-minute time interval)
path = 'C:/Users/kwangai/Desktop/'
X_cat = np.loadtxt(path+'smart_home.txt').astype(int)
N, P = X_cat.shape

t_seg = 20
Var_name = np.array(['time', 's7', 's8', 's9', 's13', 's14', 's15', 's18', 's19', 's20',
                     's21'])
#inactive sensors are removed; only 10 sensors are kept.


#activity level
def active_mat(X_cat):
    Active = np.zeros((31,54))
    for i in np.arange(31):
        for j in np.arange(54):
            Active[i,j] = np.sum(X_cat[54*i+j,1:])>0
    return Active

def active_mat_plot(Active_mat, l, path, fname):
    Active_mat_ic = Active_mat[0:26,(-l):]
    IC_idx = np.any(Active_mat_ic==0, axis=1)
    Active_mat_ic = Active_mat_ic[IC_idx,:]
    Active_mat_all = np.concatenate([Active_mat_ic, Active_mat[30:31,(-l):]], axis=0)
    
    sns.set()
    fig, ax = plt.subplots(1, 1, figsize=(5,10))
    
    sns.heatmap(Active_mat_all, linewidth=0.3, 
                xticklabels=False, yticklabels=False, cmap='YlGnBu',
                ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    
    plt.subplots_adjust(left=0.06, right=0.90, bottom=0.02, top=0.98)
    plt.savefig(path+fname+'.png', dpi=300)
    plt.close(fig)


Active_mat = active_mat(X_cat)
  
fname = 'home_heatmap'
active_mat_plot(Active_mat, 6, path, fname)
#Figure 12(b) in the manuscript


#training data and test data
D_ic = 26 #26 training days
N_ic = int(D_ic*18*60/t_seg)
X_cat_ic = X_cat[0:N_ic,:]

D_oc = 5 #5 test days
N_oc = int(D_oc*18*60/t_seg)
X_cat_oc = X_cat[N_ic:N,:]


#mutual correlation
def marg_dist(X_cat, j):
    X_cat_j = X_cat[:,j]
    d = np.max(X_cat_j) + 1
    Dist = np.zeros((d,1))
    for c in np.arange(d):
        Dist[c,0] = np.mean(X_cat_j==c)
    return Dist

def pair_dist(X_cat, j1, j2):
    X_cat_j1 = X_cat[:,j1]
    X_cat_j2 = X_cat[:,j2]
    d1 = np.max(X_cat_j1) + 1 
    d2 = np.max(X_cat_j2) + 1
    Dist = np.zeros((d1,d2))
    for c1 in np.arange(d1):
        for c2 in np.arange(d2):
            Dist[c1,c2] = np.mean((X_cat_j1==c1) & (X_cat_j2==c2))
    return Dist

def pair_dist_plot(Dist12, Dist12_hat, d1, d2, vl, vh, path, pname):
    sns.set()
    fig, ax = plt.subplots(1, 2, figsize=(12,8), gridspec_kw={'width_ratios':[1,1]})
    cbar_ax = fig.add_axes([.92, 0.3, 0.02, 0.4])
    
    sns.heatmap(Dist12, linewidth=0.3, linecolor='k', vmin=vl, vmax=vh, 
                annot=True, fmt='.2f', annot_kws={'size':30},
                yticklabels=np.arange(d1), xticklabels=np.arange(d2), 
                cmap='YlGnBu', cbar_ax=cbar_ax,
                ax=ax[0])
    ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0, fontsize=28)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0, fontsize=28)
    ax[0].set_ylabel('X1', fontsize=32, labelpad=5, rotation=0)
    ax[0].set_xlabel('X2', fontsize=32, labelpad=1, rotation=0)
    cbar = ax[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)
        
    sns.heatmap(Dist12_hat, linewidth=0.3, linecolor='k', vmin=vl, vmax=vh, 
                annot=True, fmt='.2f', annot_kws={'size':30},
                yticklabels=np.arange(d1), xticklabels=np.arange(d2), 
                cmap='YlGnBu', cbar=False,
                ax=ax[1])
    ax[1].set_yticklabels(ax[0].get_yticklabels(), rotation=0, fontsize=28)
    ax[1].set_xticklabels(ax[0].get_xticklabels(), rotation=0, fontsize=28)
    ax[1].set_ylabel('X1', fontsize=32, labelpad=5, rotation=0)
    ax[1].set_xlabel('X2', fontsize=32, labelpad=1, rotation=0)
    
    plt.subplots_adjust(left=0.06, right=0.90, bottom=0.32, top=0.72, wspace=0.18, hspace=0.15)
    plt.savefig(path+pname+'.png', dpi=200)
    plt.close(fig)
        

Dist1 = marg_dist(X_cat_ic, 1)
Dist2 = marg_dist(X_cat_ic, 4)
Dist12 = pair_dist(X_cat_ic, 1, 4)
Dist12_hat = Dist1.dot(Dist2.T)

vl = 0.01
vh = 0.53
d1, d2 = np.shape(Dist12)
pair_dist_plot(Dist12, Dist12_hat, d1, d2, vl, vh, path, 'home_pair_dist')
#Figure 1(a) in the manuscript


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
    CV = np.zeros((P-1, Lag))
    for j in np.arange(P-1):
        for lag in np.arange(Lag):
            X1 = X[0:(N-lag-1), j+1]
            X2 = X[(lag+1):N, j+1]
            CV[j,lag] = cramer_v(X1, X2)
    return CV


CV_cross_mat, CV_cross_vec = cross_cramer_v(X_cat_ic, P)
pd.DataFrame(CV_cross_vec).describe()
#Table 2 in the manuscript

Lag = 1
CV_auto = auto_cramer_v(X_cat_ic, N_ic, P, Lag)
pd.DataFrame(CV_auto).describe()
#Table 2 in the manuscript


######extract frequent patterns
#categorical data to set
def cate_to_set(X_cat, Var_name, N, P):
    X_set = []
    for i in np.arange(N):
        row_set = []
        for j in np.arange(P):
            row_set.append(Var_name[j]+'-'+str(X_cat[i,j]))
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


X_set_ic = cate_to_set(X_cat_ic, Var_name, N_ic, P)

sup_level = 0.20 #support threshold
FI, FS, FL, FI_num = fim(X_set_ic, sup_level)

FI = FI[FL<=4]
FS = FS[FL<=4]
FL = FL[FL<=4]
FI_num = len(FI)
FI
#results shown in Figure 12(a) in the manuscript

K, FI_group_idx = fi_group_index(FL, FI_num)
X_fi_ic = set_to_fi(X_set_ic, FI, N_ic, FI_num)


#phase I analysis: determine control limit (i.e., alpha) by resampling
def X_fi_resample(X_fi, D, N):
    D_idx = np.random.randint(low=0, high=D, size=D)
    Idx = np.zeros(N)
    for d in np.arange(D):
        Idx[(54*d):(54*d+54)] = np.arange(54*D_idx[d], 54*D_idx[d]+54)
    return X_fi[Idx.astype(int),:]

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
    if method == 0: #theta=1.0
        Alpha_cum = alpha * Information
    if method == 1: #theta=2.0
        Alpha_cum = alpha * np.power(Information, 2)
    if method == 2: #theta=0.5
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
    return Alpha_group_acd.reshape((1,FI_num))

def arl_ic(X_fi, D, N, FS, FI_group_idx, FI_num, K, Weight, win_size, 
           const_a, const_c, alpha, n_rl, method):
    Alpha = alpha_group_ascend(FI_group_idx, FI_num, K, alpha, method)
    
    RL = np.zeros(n_rl)
    for rep_rl in np.arange(n_rl):
        np.random.seed(rep_rl*6886)
        X_fi_rep = X_fi_resample(X_fi, D, N)
        P_val_rep = p_val_mat(X_fi_rep, FS, Weight, N-win_size, FI_num, win_size, const_a, const_c)
        
        P_val_rep_acd = p_val_mat_ascend(P_val_rep, FI_group_idx, K, method)
        
        Alarm_rep = np.any(P_val_rep_acd<=Alpha, axis=1)
        
        if np.sum(Alarm_rep)==0:
            RL[rep_rl] = N-win_size
        else:
            RL[rep_rl] = np.argmax(Alarm_rep) + 1
    return RL

def alpha_limit(X_fi, D, N, FS, FI_group_idx, FI_num, K, Weight, win_size, 
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
        RL = arl_ic(X_fi, D, N, FS, FI_group_idx, FI_num, K, Weight, win_size, 
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
win_size = int(18*60/t_seg) * 2 #size of moving window
lamb = 0.10 #EWMA smoothing parameter
Weight = np.zeros((win_size,1))
for i in np.arange(win_size):
    Weight[i,] = (1-lamb)**(win_size-1-i)
const_a = np.sum(Weight)
const_b = np.sum(Weight**2)
const_c = const_a**2 / const_b

alpha_target = 0.01
arl_target = 1/alpha_target

n_rl = 100


#determine alpha limit by bi-section search
alpha_min = 1e-5
alpha_max = 1e-4

Alpha_hat = np.zeros(6)
for method_idx in np.arange(6):
    print(method_idx)
    alpha_hat = alpha_limit(X_fi_ic, D_ic, N_ic, FS, FI_group_idx, FI_num, K, Weight, win_size, 
                            const_a, const_c, n_rl, method_idx,
                            alpha_min, alpha_max, arl_target)
    Alpha_hat[method_idx] = alpha_hat

Alpha_hat

Alpha_hat = np.array([4.37500000e-05, 3.25000000e-05, 3.21484375e-05])


#phase II analysis: control chart
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



X_cat_oc1 = X_cat_ic[(N_ic-win_size):N_ic,:]
X_cat_oc2 = X_cat_oc
X_cat_oc_all = np.concatenate((X_cat_oc1, X_cat_oc2), axis=0)

X_set_oc = cate_to_set(X_cat_oc_all, Var_name, N_oc+win_size, P)
X_fi_oc = set_to_fi(X_set_oc, FI, N_oc+win_size, FI_num)


#theta = 0.5 (method=2)
P_val_ratio, P_val_vec, Alarm_idx = chart_point(X_fi_oc, FS, FI_group_idx, FI_num, K, 
                                                Weight, win_size, const_a, const_c, 
                                                Alpha_hat, 2)

np.mean(P_val_vec<1)







