import random,copy,math,time,os,csv,sys

import numpy as np
import pandas as pd
sys.path.append('../')
import evolutionary_dynamics as ed
import models as md




#CV_range = np.linspace(0.1,1,4)

def run_rgt_gamma(cv):
    Lr= 1.
    reps = 635000
    nmax = 700
    pfix = []
    srange = np.linspace(0.002,0.02,10)
    alpha = 1.0/cv**2*np.ones(len(srange))
    beta = (2**cv**2-1)/(1.+srange)
    params_range = np.array([alpha,beta]).T
    tau0_dist = lambda params: np.random.gamma(params[0],params[1])
    divide_func = md.divide_rgt_gamma
    pfix = ed.fixation_probs(params_range,divide_func,reps,tau0_dist,nmax=nmax)

    labels = ['params{}'.format(k) for k in range(len(params_range[0]))]
    labels = labels + ['pfix']
    data = np.hstack([params_range,pfix[:,None]])
    df = pd.DataFrame(data,columns=labels)

    code_path = os.path.dirname(os.path.realpath(__file__))
    out_path = code_path+"/output/fixation"
    os.makedirs(out_path, exist_ok=True)

    df.to_csv(out_path+"/rgt_gamma_cv{}_reps{}_nmax{}.csv".format(cv,reps,nmax))

#run_rgt_gamma(0.1)
#run_rgt_gamma(0.5)
run_rgt_gamma(1.0)
