#!/home/farshidj/.conda/envs/farshid/bin/python
import subprocess
import shlex
import numpy as np
import time

def get_num_queued():
    p1 = subprocess.Popen(shlex.split("squeue -u farshidj"), stdout=subprocess.PIPE)
    p2 = subprocess.Popen(shlex.split("wc -l"), stdin=p1.stdout, stdout=subprocess.PIPE)
    out, err = p2.communicate()
    return int(out.decode("utf-8"))-1

def submit(fname,argdict):
    f = open(fname, 'w')
    f.write(
        """\
#!/bin/bash
#SBATCH --job-name="N{n:d}-c{c:.3g}"
#SBATCH --error="%j.err"
#SBATCH --output="%j.out"
#SBATCH --partition=long
#SBATCH --time="3-00:00:00"
#SBATCH --array=1-{num_seeds:d}
#SBATCH -A ajliu

/gpfs/home/farshidj/projects/competition/competition.py -t {t:d} -c {c:f} -n {n:d} -s $SLURM_ARRAY_TASK_ID -r {r:d}
""".format(**argdict)
        )
    f.close()
    subprocess.call(["sbatch", fname])
    return


if __name__ == "__main__":
    MAX_IN_QUEUE = 500

    cvs = [0.001, 0.1, .2, 0.3, 0.4, 0.6, 0.8, 1]
    num_cells = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    num_repeats = 1000000

    argdict = {}

    for n in num_cells[:4]:
        for c in cvs:
            while get_num_queued() >= MAX_IN_QUEUE:
                time.sleep(1800)

            argdict["t"] = n//10
            argdict["n"] = n
            argdict["c"] = c
            argdict["num_seeds"] = n//1
            argdict["r"] = (1*num_repeats)//n

            submit('C.sh',argdict)
