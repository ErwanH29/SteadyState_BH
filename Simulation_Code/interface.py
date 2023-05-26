from parti_initialiser import *
from file_logistics import *
from evol import *

import time as cpu_time

SMBH_code = MW_SMBH()

eta  = 1e-5
tend = 100 | units.Myr
int_string = 'GRX'

pops = [10]
init_time = cpu_time.time()
seeds = [888888] 

for ipop_ in pops:
    iter = 0
    for seed_ in seeds:
        for k in range(10):
            iter += 1
            print('=========== Simulation '+str(iter)+'/'+str(len(seeds))+' Running ===========')
            IMBH_code = IMBH_init()
            code_conv = nbody_system.nbody_to_si((ipop_*IMBH_code.mass + SMBH_code.mass), SMBH_code.distance)
            IMBH_parti, rhmass = IMBH_code.IMBH_first(ipop_, seed_, True, None)
        
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
                failed_simul = evolve_system(IMBH_parti, tend, eta, SMBH_code.distance,
                                             code_conv, int_string, SMBH, k, init_time, 
                                             0 | units.Myr)

            else:
                failed_simul = evolve_system(IMBH_parti, tend, eta, SMBH_code.distance, 
                                             code_conv, int_string, None)
