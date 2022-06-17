import numpy as np


def divide_markov(gt,tau_avg,type):
    gt1 = np.random.exponential(tau_avg)
    gt2 = np.random.exponential(tau_avg)
    return gt1,gt2

# random generation time model ------------------------------------------------
# in this model generation times are uncorrelated and exponentially distributed
def divide_rgt(gt,params,type): # in this model generation times are uncorrelated and exponentially distributed
    tau_avg = params[0]
    sigma_tau = params[1]
    cmd = params[2]
    sigma = sigma_tau*np.sqrt(1-cmd**2) #
    gt1 = gt*cmd + tau_avg*(1-cmd) + np.random.normal(0.,sigma)
    gt2 = gt*cmd + tau_avg*(1-cmd) + np.random.normal(0.,sigma)
    return gt1,gt2

def divide_rgt_gamma(gt,params,type): # in this model generation times are uncorrelated and exponentially distributed
    alpha = params[0]
    beta = params[1]
    gt1 =  np.random.gamma(alpha, beta)
    gt2 =  np.random.gamma(alpha, beta)
    return gt1,gt2


def pop_growth_rate_rgt(tau_avg,sigma_tau,cmd):
    return 2*np.log(2.)/tau_avg/(1.+np.sqrt(1.-2.*np.log(2.)*sigma_tau**2/tau_avg*(1+cmd)/(1-cmd)))

def pop_growth_rate_rgt_gamma(alpha,beta):
    return (2**(1/alpha)-1.0)/beta


# random growth rate model
def divide_rgr(gr,params,type):
    gr_avg = params[0]
    sigma = params[1]
    gr1 = gr_avg +  np.random.normal(0.,sigma)
    gr2 = gr_avg +  np.random.normal(0.,sigma)
    gt1 = np.log(2)/gr1
    gt2 = np.log(2)/gr2
    return gt1,gt2

def pop_growth_rate_rgr(gr_avg,sigma_gr):
    return gr_avg - (1.0-np.log(2)/2.0)*sigma_gr**2/gr_avg


# bimodal models
def divide_bimodal(gt,params,type): # in this model generation times are uncorrelated and exponentially distributed
    t1 = params[0]
    t2 = params[1]
    q = params[2]
    r = np.random.rand()
    if r>q:
        gt1 = t1
        gt2 = t1
    else:
        gt1 = t2
        gt2 = t2
    return gt1,gt2

def divide_rgt_bimodal(gt,params,type): # in this model generation times are uncorrelated and exponentially distributed
    t1 = params[0]
    t2 = params[1]
    q = params[2]
    sigma1 = params[3]
    sigma2 = params[4]
    r = np.random.rand()
    if r>q:
        gt1 = t1+np.random.normal(0.,sigma1)
        gt2 = t1+np.random.normal(0.,sigma1)
    else:
        gt1 = t2+np.random.normal(0.,sigma2)
        gt2 = t2+np.random.normal(0.,sigma2)
    return gt1,gt2
