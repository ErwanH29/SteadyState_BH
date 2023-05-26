from parti_initialiser import *
from file_logistics import *
from evol import *

import os
import pickle as pkl
import time as cpu_time

SMBH_code = MW_SMBH()

eta  = 1e-5
tend = 100 | units.Myr
int_string = 'GRX'

seeds = [888888] 

final_snapshot = (glob.glob('/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7_paused/particle_trajectory/*'))

with open(final_snapshot[0], 'rb') as file_name:
    sim_snap = pkl.load(file_name)
print('Resuming from file: ', final_snapshot[0])

pops = np.shape(sim_snap)[0]
init_time = cpu_time.time()
iter = 0

for seed_ in seeds:
    for k in range(10):
        iter += 1
        print('=========== Simulation '+str(iter)+'/'+str(len(seeds))+' Running ===========')
        IMBH_code = IMBH_init()
        code_conv = nbody_system.nbody_to_si((pops*IMBH_code.mass + SMBH_code.mass), SMBH_code.distance)
        IMBH_parti, rhmass = IMBH_code.IMBH_first(pops, seed_, False, sim_snap)
        
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
            os.remove(final_snapshot[0])
            failed_simul = evolve_system(IMBH_parti, tend, eta, SMBH_code.distance,
                                         code_conv, int_string, SMBH, k, init_time, 
                                         (np.shape(sim_snap)[1]-1)/10**3 | units.Myr)

        else:
            failed_simul = evolve_system(IMBH_parti, tend, eta, SMBH_code.distance, 
                                         code_conv, int_string, None)
