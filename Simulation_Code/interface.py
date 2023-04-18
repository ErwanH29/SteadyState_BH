from parti_initialiser import *
from file_logistics import *
from evol import *

SMBH_code = MW_SMBH()

eta  = 1e-5
tend = 100 | units.Myr
int_string = 'GRX'

pop = 10
seed = 888888 
no_sims = 30

for k in range(no_sims):
    print('=========== Simulation '+str(k)+'/'+str(no_sims)+' Running ===========')
    IMBH_code = IMBH_init()
    code_conv = nbody_system.nbody_to_si((pop*IMBH_code.mass + SMBH_code.mass), SMBH_code.distance)
    IMBH_parti, rhmass = IMBH_code.IMBH_first(pop, seed)

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
                                        code_conv, int_string, SMBH, k)

    else:
        failed_simul = evolve_system(IMBH_parti, tend, eta, SMBH_code.distance, 
                                        code_conv, int_string, None)
