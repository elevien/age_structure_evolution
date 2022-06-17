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


#params = [1.5,0.1,0.8]
params = [1.5,0.1,0.8,0.05,0.1]

output=[]
L = 0.
tinit = 100
n_monte = 100
ntot = 500
tmax = 4.
dt_sample =0.1

for k in range(n_monte):
    cells_init = TS.make_cells_init(tinit,ntot,MD.divide_rgt_bimodal,params)
    t,n1,cells = TS.runsim_neutral(tmax,MD.divide_rgt_bimodal,cells_init,\
    div_params=params,\
    dt_sample=dt_sample, save_all=False)
    output += [[t,n1,cells,cells_init]]
    print(k)

# process output
nt_min = np.min([len(out[1]) for out in output])
n1s = np.zeros((nt_min,n_monte))
for k in range(n_monte):
    n1s[:,k] = output[k][1][0:nt_min]


code_path = os.path.dirname(os.path.realpath(__file__))
out_path = code_path+"/output/growth_of_variance_rgt_bimodal"
os.makedirs(out_path, exist_ok=True)

np.savetxt(out_path+"/n1s.csv",n1s, delimiter=",")
np.savetxt(out_path+"/t.csv",t, delimiter=",")
