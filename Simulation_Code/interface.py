from parti_initialiser import *
from evol import EvolveSystem

import time as cpu_time

SMBH_code = MW_SMBH()

eta  = 1e-5
tend = 100 | units.Myr
int_string = 'GRX'

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
    IMBH_parti, rhmass = IMBH_code.IMBH_first(pops, seeds, True, None)

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
        evolve_system = EvolveSystem(IMBH_parti, tend, eta, SMBH_code.distance,
                                     code_conv, int_string, SMBH, k, init_time, 
                                     0 | units.Myr, 4)
        evolve_system.initialise_gravity()
        failed_simul = evolve_system.run_code()

    else:
        evolve_system = EvolveSystem(IMBH_parti, tend, eta, SMBH_code.distance, 
                                     code_conv, int_string, None, 18)
        evolve_system.initialise_gravity()
        failed_simul = evolve_system.run_code()
