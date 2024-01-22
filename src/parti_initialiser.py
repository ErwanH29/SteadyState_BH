import numpy as np

from amuse.ic.plummer import new_plummer_model
from amuse.lab import nbody_system
from amuse.units import units, constants


class MW_SMBH(object):
    """
    Define SMBH mass and virial radius
    """
    def __init__(self, mass, rvir):

        self.mass = mass
        self.rvir = rvir
        self.bh_rad = (2*constants.G*mass)/(constants.c**2)


class IMBH_init(object):
    """
    Initialise the IMBH particles 
    """
    def __init__(self):
        self.mass = 1000 | units.MSun
    
    def IMBH_radius(self, mass):
        """
        Set the IMBH radius (Schwarzschild radius)
        """
        return (2*constants.G*mass)/(constants.c**2)

    def coll_radius(self, radius):
        """
        Set the collision radius (rISCO)
        """
        return 3*radius

    def IMBH_first(self, cluster_pop, prev_run, 
                   prev_data, SMBH_mass, rvir):
        """
        Initialise cluster. The first particle is the SMBH.

        Inputs:
        cluster_pop: Population of cluster
        prev_run:    Boolean to see if using data from previous ALICE run
        prev_data:   Previous ALICE run data file
        SMBH_mass:   SMBH mass
        rvir:        Initial cluster virial radius
        """
        
        self.N = cluster_pop+1
        IMBH_mass = cluster_pop*self.mass
        self.code_conv = nbody_system.nbody_to_si(IMBH_mass, rvir)

        if not (prev_run):
            particles = new_plummer_model(self.N, convert_nbody = self.code_conv)
            particles.ejection = 0
            particles.coll_events = 0
            particles.z *= 0.1
            particles.key_tracker = particles.key
            particles[0].position = [0,0,0] | units.km
            particles[0].velocity = [0,0,0] | units.kms
            particles[1:].mass = self.mass
            particles.radius = self.IMBH_radius(particles.mass)
            particles.collision_radius = self.coll_radius(particles.radius)

            min_dist = 0.15 | units.parsec
            max_dist = 0.25 | units.parsec
            
            for parti_ in particles[1:]:
                if parti_.position.length() < min_dist:
                    cfactor = np.random.uniform(1,1.67,1)
                    parti_.position *= cfactor*min_dist/parti_.position.length()
                if parti_.position.length() > max_dist:
                    cfactor = np.random.uniform(0.6,1,1)
                    parti_.position *= cfactor*max_dist/parti_.position.length()
            particles.scale_to_standard(convert_nbody=self.code_conv)

            for parti_ in particles[1:]:
                parti_.vx += (constants.G*SMBH_mass * (abs(parti_.y)/parti_.position.length()**2)).sqrt()
                parti_.vy += (constants.G*SMBH_mass * (abs(parti_.x)/parti_.position.length()**2)).sqrt()
                parti_.vz += (constants.G*SMBH_mass * (abs(parti_.z)/parti_.position.length()**2)).sqrt()
        else:
            self.N -= 1
            particles = new_plummer_model(self.N, convert_nbody=self.code_conv)
            for parti_ in range(np.shape(prev_data)[0]):
                part_params = prev_data.iloc[parti_][-1]

                particles[parti_].vx = part_params[3][0]
                particles[parti_].vy = part_params[3][1]
                particles[parti_].vz = part_params[3][2]
                particles[parti_].x = part_params[2][0]
                particles[parti_].y = part_params[2][1]
                particles[parti_].z = part_params[2][2]
                particles[parti_].mass = part_params[1]

        particles[1:].mass = self.mass
        particles[0].mass = SMBH_mass
        particles.radius = self.IMBH_radius(particles.mass)
        particles.collision_radius = self.coll_radius(particles.radius)
        
        return particles