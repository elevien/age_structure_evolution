import random
import copy
import time
import numpy as np
import os
import csv
import sys
sys.path.insert(1, '/Users/E/Dropbox/RESEARCH/shared/age_structure_fluctuations/code')
import models as MD
import two_species_neutral as TS

# define model paramaters
tau_avg = np.log(2.)
sigma_tau = 0.1
cmd = 0.
#params = [tau_avg,sigma_tau,cmd]
params = tau_avg
# define simulation paramaters
n_range = np.logspace(1,3,10,dtype=int)
n_monte = 1
tmax = 3*10**4.

# setup output folder
code_path = os.path.dirname(os.path.realpath(__file__))
out_path = code_path+"/output/interdivision_time_scaling/markov"
os.makedirs(out_path, exist_ok=True)
#
# w = csv.writer(open(out_path+"/params.csv", "w"))
# w.writerow([datetime.datetime.now()])
# w.writerow(["output",id])
# for key, val in params.items():
#    w.writerow([key, val])



# run simulations
L_dil =[] # the diluation rates
tau_var =[]
for n10 in n_range:
    tinit= 2*n10
    ntot = 2*n10
    #L_dil_avg = 0.
    cells_init = TS.make_cells_init(tinit,ntot,MD.divide_markov,params)
    t,n1,cells = TS.runsim_neutral(tmax,MD.divide_markov,cells_init,div_params=params,save_all=True)
    tau = t[1:]-t[:-1] # these are the times between division events in the population
       # L_dil_avg += 1/(ntot*np.mean(tau))
    #L_dil1.append(1/(ntot*np.mean(tau)))
    L_dil.append(len(t)/(ntot*t[-1]))
    tau_var.append(np.var(tau))
    np.savetxt(out_path+"/taus_n%i.txt"%ntot,tau)
    print(ntot)

np.savetxt(out_path+"/L_dil.txt",L_dil)
np.savetxt(out_path+"/tau_var.txt",tau_var)
np.savetxt(out_path+"/n_range.txt",n_range*2)
