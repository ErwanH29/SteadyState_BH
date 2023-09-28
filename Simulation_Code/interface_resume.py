from src.parti_initialiser import *
from src.file_logistics import *
from src.evol import *

import os
import pickle as pkl
import time as cpu_time

SMBH_code = MW_SMBH()

eta  = 1e-5
tend = 100 | units.Myr
int_string = 'GRX'

seeds = 888888

final_snapshot = (glob.glob('/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7_paused/particle_trajectory/*'))

with open(final_snapshot[0], 'rb') as file_name:
    sim_snap = pkl.load(file_name)
print('Resuming from file: ', final_snapshot[0])

no_sims = 10
pops = 10
init_time = cpu_time.time()
seeds = [888888] 

iter = 0
for k in range(no_sims):
    iter += 1
    print('=========== Simulation '+str(iter)+'/'+str(no_sims)+' Running ===========')
    IMBH_code = IMBH_init()
    code_conv = nbody_system.nbody_to_si((pops*IMBH_code.mass + SMBH_code.mass), 
                                         SMBH_code.distance)
    IMBH_parti, rhmass = IMBH_code.IMBH_first(pops, seeds, False, sim_snap)
    
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
        evolve_system = EvolveSystem(IMBH_parti, tend, eta, SMBH_code.distance,
                                     code_conv, int_string, SMBH, k, init_time, 
                                     (np.shape(sim_snap)[1]-1)/10**3 | units.Myr, 4)
        evolve_system.initialise_gravity()
        failed_simul = evolve_system.run_code()

    else:
        evolve_system = EvolveSystem(IMBH_parti, tend, eta, SMBH_code.distance, 
                                     code_conv, int_string, None, 18)
        evolve_system.initialise_gravity()
        failed_simul = evolve_system.run_code()
