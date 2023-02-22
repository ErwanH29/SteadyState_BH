from parti_initialiser import *
from file_logistics import *
from evol import *

SMBH_code = MW_SMBH()
IMBH_code = IMBH_init()

eta  = 1e-5
tend = 100 | units.Myr
int_string = 'GRX'

pops = [10]
seeds = [888888]
code_conv = nbody_system.nbody_to_si((pops[0]*IMBH_code.mass + SMBH_code.mass), SMBH_code.distance)
iter = -1
for k in range(20):
    iter += 1
    print('=========== Simulation '+str(iter)+'/20 Running ===========')
    IMBH_parti, rhmass = IMBH_code.IMBH_first(pops[0], seeds[0])
    
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
                                        code_conv, int_string, SMBH)

    else:
        failed_simul = evolve_system(IMBH_parti, tend, eta, SMBH_code.distance, 
                                        code_conv, int_string, SMBH)