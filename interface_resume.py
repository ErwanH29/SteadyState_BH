import glob
import numpy as np
import os
import pickle as pkl
import time as cpu_time

from amuse.lab import nbody_system, Particles, units

from src.parti_initialiser import IMBH_init, MW_SMBH
from src.evol import evolve_system


#Set global variables
SMBH_code = MW_SMBH(mass=4e6 | units.MSun,
                    rvir=0.2 | units.pc)
eta  = 1e-5
tend = 100 | units.Myr
int_string = 'GRX'

final_snapshot = (glob.glob('/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7_paused/particle_trajectory/*'))

with open(final_snapshot[0], 'rb') as file_name:
    sim_snap = pkl.load(file_name)
print('Resuming from file: ', final_snapshot[0])

cluster_pop = np.shape(sim_snap)[0]-1
final_dt = (np.shape(sim_snap)[1]-1)/10**3
init_time = cpu_time.time()
iter = 0

for Nsim in range(10):
    iter += 1
    print('Running Simulation #'+str(iter))
    IMBH_code = IMBH_init()
    IMBH_parti, rhmass = IMBH_code.IMBH_first(cluster_pop, True, 
                                              sim_snap, SMBH_code.mass,
                                              SMBH_code.rvir)
    rvir = IMBH_parti.virial_radius()
    code_conv = nbody_system.nbody_to_si(IMBH_parti.mass.sum(), rvir)
    
    if int_string == 'GRX':
        SMBH = Particles(1)
        SMBH.mass = IMBH_parti[0].mass
        SMBH.velocity = IMBH_parti[0].velocity
        SMBH.position = IMBH_parti[0].position
        SMBH.key_tracker = IMBH_parti[0].key_tracker
        SMBH.collision_radius = 2*IMBH_parti[0].collision_radius
        SMBH.radius = IMBH_parti[0].radius
        SMBH.ejection = 0
        SMBH.collision_events = 0
        failed_simul = evolve_system(IMBH_parti, tend, eta, rvir, code_conv, 
                                     int_string, SMBH, Nsim, init_time, 
                                     final_dt | units.Myr)
        os.remove(final_snapshot[0])

    else:
        failed_simul = evolve_system(IMBH_parti, tend, eta, rvir, code_conv, 
                                     int_string, SMBH, Nsim, init_time, 
                                     final_dt | units.Myr)
