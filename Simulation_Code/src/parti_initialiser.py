from amuse.lab import *
from amuse.units import (units, constants)
from amuse.ic.plummer import new_plummer_model
from random import choices
import numpy as np

class MW_SMBH(object):
    """Class which defines the central SMBH"""
    def __init__(self, mass = 4.e7 | units.MSun,
                 position = [0, 0, 0] | units.parsec,
                 velocity = [0, 0, 0] | (units.AU/units.yr),
                 distance = 0.2 | units.parsec):

        self.distance = distance
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.bh_rad = (2*constants.G*mass)/(constants.c**2)

class IMBH_init(object):
    """Class to initialise the IMBH particles"""
    def __init__(self):
        self.N = 0
        self.mass = 1000 | units.MSun
    
    def IMBH_radius(self, mass):
        """Function which sets the IMBH radius based on the Schwarzschild radius"""
        return (2*constants.G*mass)/(constants.c**2)

    def coll_radius(self, radius):
        """Function which sets the collision radius. 
           Based on the rISCO."""
        return 3*radius

    def ProbFunc(self, vel):
        """
        Function which initialises the velocity distribution [Maxwell distribution]
        
        Inputs:
        vel:    The velocity range for which to sample the weights from
        """

        sigmaV = 150

        return np.sqrt(2/np.pi)*(vel**2/sigmaV**3)*np.exp(-vel**2/(2*sigmaV**2))

    def velocityList(self, vseed):
        """Function to give a velocity for an initialised particle"""

        np.random.seed(vseed)

        vrange = np.linspace(0, 700)
        r = [-1,1]
        w = self.ProbFunc(vrange)
        scale = [np.random.choice(r), [np.random.choice(r)], [np.random.choice(r)]]

        vx = np.array(choices(vrange, weights=w, k = 1))*scale[0]
        vy = np.array(choices(vrange, weights=w, k = 1))*scale[1]
        vz = np.array(choices(vrange, weights=w, k = 1))*scale[2]
 
        return np.concatenate((vx,vy,vz))

    def kroupa_mass(self, pset):
        """Function to set particle masses based on the Kroupa function"""

        return new_kroupa_mass_distribution(pset, 50 | units.MSun, 10**5 | units.MSun)

    def plummer_distr(self, N):
        """Function to initialise the particles position based on the Plummer model"""

        distr = new_plummer_model(N, convert_nbody = self.code_conv)
        return distr

    def IMBH_first(self, init_parti, bool, ptracker):
        """
        Function to initialise the first IMBH population.
        The first particle forms the center of the cluster

        Inputs:
        init_parti: The number of IMBH particles you wish to simulate
        seed:       Seed which defines the initial configuration of the system
        bool:       Boolean to see if using data from previous ALICE run
        ptracker:   Previous ALICE run data file
        """
        
        SMBH_parti = MW_SMBH()
        self.N = init_parti+1
        self.code_conv = nbody_system.nbody_to_si(self.N*self.mass, 
                                                  SMBH_parti.distance)

        if (bool):
            particles = self.plummer_distr(self.N)
            vseed = 0
            for parti_ in particles:
                vseed += 1
                parti_.velocity = self.velocityList(vseed) * (1 | units.kms)
            particles.z *= 0.1
            particles[0].position = SMBH_parti.position
            particles[0].velocity = SMBH_parti.velocity
            particles[1:].mass = self.mass

            min_dist = 0.15 | units.parsec
            max_dist = 0.25 | units.parsec
            
            for parti_ in particles[1:]:
                if parti_.position.length() < min_dist:
                    parti_.position *= min_dist/parti_.position.length()
                if parti_.position.length() > max_dist:
                    parti_.position *= max_dist/parti_.position.length()
            particles.scale_to_standard(convert_nbody=self.code_conv)

            for parti_ in particles[1:]:
                parti_.vx += (constants.G*SMBH_parti.mass * (abs(parti_.y)/parti_.position.length()**2)).sqrt()
                parti_.vy += (constants.G*SMBH_parti.mass * (abs(parti_.x)/parti_.position.length()**2)).sqrt()

            particles[1:].mass = self.mass
            particles[0].mass = SMBH_parti.mass

        else: #If using data from previous simulation run
            self.N -= 1
            particles = self.plummer_distr(self.N)
            for parti_ in range(np.shape(ptracker)[0]):
                part_params = ptracker.iloc[parti_][-1]

                particles[parti_].mass = part_params[1]
                particles[parti_].position = part_params[2]
                particles[parti_].velocity = part_params[3]
                
        particles.radius = self.IMBH_radius(particles.mass)
        particles.collision_radius = self.coll_radius(particles.radius)
        particles.ejection = 0
        particles.coll_events = 0
        particles.key_tracker = particles.key
        
        return particles
