import time as cpu_time
import warnings

from amuse.lab import units, nbody_system, Particles

from src.parti_initialiser import MW_SMBH, IMBH_init
from src.evol import evolve_system

warnings.simplefilter(action='ignore', category=FutureWarning)

#Set global variables
SMBH_code = MW_SMBH(mass=4e6 | units.MSun,
                    rvir=0.2 | units.pc)
tend = 100 | units.Myr
eta = 1e-5
cluster_pop = 10
int_string = 'Hermite'

init_time = cpu_time.time()

for Nsim in range(10):
    print('=========== Simulation #'+str(Nsim))
    IMBH_code = IMBH_init()
    IMBH_parti  = IMBH_code.IMBH_first(cluster_pop, 
                                       False, None, 
                                       SMBH_code.mass, 
                                       SMBH_code.rvir)
    code_conv = nbody_system.nbody_to_si(IMBH_parti.mass.sum(), 
                                         SMBH_code.rvir)

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
        failed_simul = evolve_system(IMBH_parti, tend, eta, SMBH_code.rvir,
                                     code_conv, int_string, SMBH, Nsim, 
                                     init_time, 0 | units.Myr)

    else:
        failed_simul = evolve_system(IMBH_parti, tend, eta, SMBH_code.rvir, 
                                     code_conv, int_string, None, Nsim,
                                     init_time, 0 | units.Myr)
