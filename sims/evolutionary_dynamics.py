import random,copy,math
from tqdm import tqdm_notebook
from tqdm import tqdm
import numpy as np
import csv
import datetime
import sys
from scipy import optimize


def make_cells_init(tinit,ntot,divide,div_params,*,frac=2,dt_sample=0.1):
    """
    generate an initial population of two species
    """
    gt = np.ones(ntot)
    dt = gt
    types = np.ones(ntot)
    cells_init = np.transpose(np.array([gt,dt,types]))
    t,n1,cells = moran_rgt(tinit,divide,cells_init,div_params=div_params,dt_sample=dt_sample)

    # set types of cells
    cells[:,2] = np.concatenate((np.ones(ntot//frac),np.zeros(ntot-ntot//frac)))

    # make generation times relative to 0.
    cells[:,0] = cells[:,0]-tinit
    return cells



def moran_rgt(tmax,divide,\
    cells_init,*,max_steps = 10**7,\
    save_all = False,\
    dt_sample = 0.1,\
    div_params = None,\
    stop_cond = lambda x: False):
    """
    Simulates neutral competition between two species

    Input
    ---------
    tmax - time to simulate
    divide - function specifying how division times are generated
    cells_init - initial state of system. Each cell is represented by a
            numpy array of the form [time of division,gen. time,species].
            species is 0 or 1 for a two species model. This means
            cells_init[5,1] would give the 6th cell's generation time

    optional inputs:
    max_steps - max divisions to record
    save_all  - if False save
    dt_sample - time between points to sample, because
                we usually don't want to save all point.
                ignotes if save_all = True
    div_params - paramaters to pass to divide function

    Ouput
    ---------
    t  - numpy array of division times
    n1 - numpy array of the number of cells of species 1
    cells - the list of cells at the time when the simulation ended
    """

    n1  = np.zeros(max_steps)
    t = np.zeros(max_steps)
    steps = 1
    t_last = 0. # time of last division
    cells = copy.deepcopy(cells_init)
    ntot = len(cells[:,0])
    n1[0] = np.sum([cells[:,2]])


    while t_last<tmax and steps-1<max_steps and stop_cond(n1[steps-1])==False:
        ind = np.argmin(cells[:,0]) # can this be made more efficient?
        mother_dt   = cells[ind,0] # division time of dividing cell
        mother_gt   = cells[ind,1] # generation time of dividing cell
        mother_type = cells[ind,2] # type of dividing cell

        t_last = mother_dt # current time is now division time

        gtL,gtR = divide(mother_gt,div_params,mother_type) # get daughter cell generation times

        cellL = np.array([t_last+gtL,gtL,mother_type]) # make "left" daughter
        cellR = np.array([t_last+gtR,gtR,mother_type]) # make "right" daughter
        cells[ind] = cellL # one of the daughters goes where the mother was

        ind2 = np.random.choice(ntot) #other daughter goes at random index
        cells[ind2] = cellR

        # we sample the state at time intervals dt_sample, this saves memory
        # and also makes it easier to process data later on
        if save_all == False:
            while t_last - t[steps-1]>dt_sample:
                t[steps] = t[steps-1] + dt_sample
                n1[steps] = np.sum([cells[:,2]]) # number of cells of type 1
                steps+=1
        else:
            t[steps] = t_last
            n1[steps] = np.sum([cells[:,2]])
            steps+=1

    return t[0:steps],n1[0:steps],cells


def neutral_diffusion(t,L,ntot,phi0):
    nt = len(t)
    phi = np.zeros(nt)
    phi[0] = phi0
    for k in range(1,nt):
        dt = t[k]-t[k-1]
        noise = np.random.normal(0.,np.sqrt(dt))
        phi[k] = phi[k-1] + np.sqrt(2*L/ntot*phi[k-1]*(1.-phi[k-1]))*noise
    return t,phi


def fixation(N,divide,tol,*,tmax = 10e9,tinit=4.,div_params = None,min_k = 100):

    num_fix = 0.
    k = 0
    err = 1000.
    q = 0.
    while err > tol or k < min_k:
        cells_init = make_cells_init(tinit,N,divide,div_params,frac = N);

        t,n,cells = moran_rgt(tmax,divide,cells_init,div_params=div_params,stop_cond = lambda x : (x == 0) or (x == N))
        k += 1
        if n[-1]>0:
            num_fix += 1
        q = num_fix/k
        err = np.sqrt(q*(1-q)/k)
    return q,err,k




########################################################################
# approximation using infinite population
########################################################################


def extinction(nmax,death_rate,divide,cell_init,*,max_steps = 10**7,\
    save_all = False,\
    dt_sample = 0.1,\
    div_params = None):
    """
    simulate dynamics of a growing population with constant death rate
    until extinction or clone size reaches nmax

    """

    n  = np.zeros(max_steps)
    t = np.zeros(max_steps)
    steps = 1
    t_last = 0. # time of last division
    n[0] = 1
    n_cells = 1
    dtype = [('dt', float), ('gt', float), ('type', float)]
    cells = np.zeros(nmax,dtype = dtype)
    cells[0] = cell_init

    while n_cells<nmax and n_cells>0 and steps-1<max_steps:
        cells[0:n_cells].sort(order=['dt'])
        #ind = np.argmin(cells[0:n_cells,0]) # can this be made more efficient?
        mother_dt   = cells[0]['dt'] # division time of dividing cell
        mother_gt   = cells[0]['gt'] # generation time of dividing cell
        mother_type = cells[0]['type'] # type of dividing cell

        p_death = 1.0-np.exp(-death_rate*mother_gt)
        roll = np.random.rand()
        if roll>p_death:
            t_last = mother_dt # current time is now division time

            gtL,gtR = divide(mother_gt,div_params,type) # get daughter cell generation times

            cellL = (t_last+gtL,gtL,mother_type) # make "left" daughter
            cellR = (t_last+gtR,gtR,mother_type) # make "right" daughter
            cells[0] = cellL # one of the daughters goes where the mother was
            cells[n_cells] = cellR
            n_cells +=1
        else:
            cells[:-1] = cells[1:]
            n_cells -=1

        # we sample the state at time intervals dt_sample, this saves memory
        # and also makes it easier to process data later on
        if save_all == False:
            while t_last - t[steps-1]>dt_sample:
                t[steps] = t[steps-1] + dt_sample
                n[steps] = n_cells # number of cells of type 1
                steps+=1
        else:
            t[steps] = t_last
            n[steps] = n_cells
            steps+=1

    return t[0:steps],n[0:steps],cells

def fixation_probs_extinction(param_range,divide_func,reps,tau0_dist,*,nmax=200,Lr=1.):
    pfix = []
    for params in param_range:
        fixed = 0
        print("computing fixation probability for params = ",params)
        for k in tqdm(range(reps)):
            # generate initial cell's generation time
            taud0 = tau0_dist(params)
            cells_init = (taud0,taud0,1)

            # run simulation
            t,n,cells = extinction(nmax,Lr,divide_func,cells_init,div_params=params,save_all=True)

            # check if fixed
            if n[-1]>0:
                fixed += 1
        pfix.append(fixed/reps)

    return np.array(pfix)
