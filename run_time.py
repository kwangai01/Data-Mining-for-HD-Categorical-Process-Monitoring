# -*- coding: utf-8 -*-
"""
computing time 
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import time


######data simulation functions
#mean and sigma
def sigma_create(P, rho):
    Sigma = np.zeros((P,P))
    for i in np.arange(P):
        for j in np.arange(P):
            Sigma[i,j] = rho**(np.abs(i-j))
    return Sigma

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


######tensor decomposition model
def cp_e_step(X, G, U, N, P, h):
    Z = np.zeros((N,h))
    for i in np.arange(N):
        for k in np.arange(h):
            prodt = G[k]
            for j in np.arange(P):
                idx_level = X[i,j]-1
                prodt = prodt * U[j,k,idx_level]
            Z[i,k] = prodt
    return Z / np.sum(Z, axis=1, keepdims=True)
    
def cp_m_step_g(Z, N):
    G = np.sum(Z, axis=0) / N
    return G

def cp_m_step_u(X, Z, N, P, d, h):
    U = np.zeros((P, h, d))
    for j in np.arange(P):
        for k in np.arange(h):
            for c in np.arange(d):
                U[j,k,c] = np.sum(Z[:,k] * (X[:,j]==(c+1)))
    return U / np.sum(U, axis=2, keepdims=True)

def log_likelihood(X, G, U, N, P, h):
    l = 0
    for i in np.arange(N):
        l_ind = 0
        for k in np.arange(h):
            prodt = G[k]
            for j in np.arange(P):
                idx_level = X[i,j]-1
                prodt = prodt * U[j,k,idx_level]
            l_ind = l_ind + prodt
        l = l + np.log(l_ind)
    return l
    
def cp(X, G_ini, U_ini, N, P, d, h):
    G_old = G_ini
    U_old = U_ini
    LK = []
    LK.append(log_likelihood(X, G_old, U_old, N, P, h))
    
    ite = 0
    err = 10
    while err>1e-6 and ite<100:
        #e step
        Z = cp_e_step(X, G_old, U_old, N, P, h)
        #m step
        G_new = cp_m_step_g(Z, N)
        U_new = cp_m_step_u(X, Z, N, P, d, h)

        err = np.mean(np.abs(U_new - U_old))
        ite = ite + 1

        G_old = G_new
        U_old = U_new

        LK.append(log_likelihood(X, G_old, U_old, N, P, h))
        
    return G_old, U_old, LK

def post_u(U):
    U_new = U
    if np.any(U_new<1e-10):
        small_idx = U_new<1e-10
        U_new[small_idx] = 1e-10
        U_new = U_new / np.sum(U_new, axis=2, keepdims=True)
    return U_new


#######frequent pattern mining
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


#setup
d = 5
rho = 0.5

h = 8 #tensor decomposition model
sup_level = 0.30 #data mining model


######running time with n
P_set = np.array([50])
N_set = np.arange(200,4200,step=200)

rep_num = 10

Time = np.zeros((2, len(P_set), len(N_set), rep_num))

for p_idx in np.arange(len(P_set)):
    for n_idx in np.arange(len(N_set)):
        print(n_idx)
        P = P_set[p_idx]
        N = N_set[n_idx]
        
        #mean and cov
        Mu = np.zeros(P)
        Sigma = sigma_create(P, rho)

        #threshold
        np.random.seed(0)
        Interval_size = np.random.dirichlet(np.ones(d), size=P).T
        Cum_size = np.cumsum(Interval_size, axis=0)
        Th_prob = np.zeros((d+1, P))
        Th_prob[1:d,:] = Cum_size[0:(d-1),:]
        Th_prob[d,:] = 1
        Th = norm.ppf(Th_prob, loc=0, scale=1)
        
        for rep in np.arange(rep_num):
            #print(rep)
            np.random.seed(rep*6886)
            Data_raw = multivariate_normal.rvs(mean=Mu, cov=Sigma, size=N)
            X = cont_to_cate(Data_raw, Th, P, d)
            
            #tensor decomposition model
            start_time = time.time()
            G_ini = np.random.dirichlet(np.ones(h), size=1)[0,:]
            U_ini = np.random.dirichlet(np.ones(d), size=(P,h))
            G_cp, U_cp, LK_cp = cp(X, G_ini, U_ini, N, P, d, h)
            U_cp = post_u(U_cp)
            end_time = time.time()
            Time[0, p_idx, n_idx, rep] = end_time-start_time
            
            #data mining model
            X_set = cate_to_set(X, N, P)
            start_time = time.time()
            FI, FS, FL, FI_num = fim(X_set, sup_level)
            stop_time = time.time()
            Time[1, p_idx, n_idx, rep] = stop_time-start_time



######running time with p
P_set = np.arange(5, 205, step=5)
N_set = np.array([2000])

rep_num = 10

Time = np.zeros((2, len(P_set), len(N_set), rep_num))

for p_idx in np.arange(len(P_set)):
    print(p_idx)
    for n_idx in np.arange(len(N_set)):
        P = P_set[p_idx]
        N = N_set[n_idx]
        
        #mean and cov
        Mu = np.zeros(P)
        Sigma = sigma_create(P, rho)

        #threshold
        np.random.seed(0)
        Interval_size = np.random.dirichlet(np.ones(d), size=P).T
        Cum_size = np.cumsum(Interval_size, axis=0)
        Th_prob = np.zeros((d+1, P))
        Th_prob[1:d,:] = Cum_size[0:(d-1),:]
        Th_prob[d,:] = 1
        Th = norm.ppf(Th_prob, loc=0, scale=1)
        
        for rep in np.arange(rep_num):
            #print(rep)
            np.random.seed(rep*6886)
            Data_raw = multivariate_normal.rvs(mean=Mu, cov=Sigma, size=N)
            X = cont_to_cate(Data_raw, Th, P, d)
            
            #tensor decomposition model
            start_time = time.time()
            G_ini = np.random.dirichlet(np.ones(h), size=1)[0,:]
            U_ini = np.random.dirichlet(np.ones(d), size=(P,h))
            G_cp, U_cp, LK_cp = cp(X, G_ini, U_ini, N, P, d, h)
            U_cp = post_u(U_cp)
            end_time = time.time()
            Time[0, p_idx, n_idx, rep] = end_time-start_time
            
            #data mining model
            X_set = cate_to_set(X, N, P)
            start_time = time.time()
            FI, FS, FL, FI_num = fim(X_set, sup_level)
            stop_time = time.time()
            Time[1, p_idx, n_idx, rep] = stop_time-start_time


