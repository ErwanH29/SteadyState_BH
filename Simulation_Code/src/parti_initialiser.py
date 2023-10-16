from amuse.lab import *
from amuse.units import (units, constants)
from amuse.ic.plummer import new_plummer_model
from amuse.ext.orbital_elements import new_binary_from_orbital_elements

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
        self.mass = 1e3 | units.MSun

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

    def velocityList(self):
        """Function to give a velocity for an initialised particle"""

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

    def IMBH_first(self, init_parti, mass_profile, flag, ptracker):
        """
        Function to initialise the first IMBH population.
        The first particle forms the center of the cluster

        Inputs:
        init_parti:   The number of IMBH particles you wish to simulate
        mass_profile: Mass profile of IMBH
        flag:         Flag to see if using data from previous ALICE run
        ptracker:     Previous ALICE run data file
        """
        
        SMBH_parti = MW_SMBH()
        isolated = False
        self.N = init_parti+1
        if mass_profile.lower() == "stellar" and not (isolated):
            self.N *= 2
        self.code_conv = nbody_system.nbody_to_si(self.N * self.mass, 
                                                  0.2 | units.pc)
        if (flag):
            particles = self.plummer_distr(self.N)
            for parti_ in particles:
                parti_.velocity = self.velocityList() * (1 | units.kms)
            if mass_profile.lower() != "uniform":
                mass = (np.random.uniform(100, 10000, 1)) | units.MSun
                prob = np.random.power(0.5, len(particles[1:]))
                mass_val = mass*prob
                for mass_ in range(len(mass_val)):
                    if mass_val[mass_] < (1 | units.MSun):
                        mass_val[mass_] = (np.random.uniform(100, 10000, 1)) | units.MSun
                particles[1:].mass = mass_val
                particles[1:].type = "imbh"
            elif mass_profile.lower() == "constant":
                particles[1:].mass = self.mass
                particles[1:].type = "imbh"
            if mass_profile.lower() == "stellar":
                stellar_masses = new_kroupa_mass_distribution(self.N, 10 | units.MSun, 100 | units.MSun)
                choice = np.random.choice([0, 1], (self.N-2))
                if (isolated):
                    parti_ = 0 
                    for choice_ in choice:
                        parti_ += 1
                        if choice_ == 1:
                            particles[parti_+1].mass = 10*stellar_masses[parti_+1]
                            particles[parti_+1].type = "imbh"
                    particles[particles.type != "imbh"].type = "stellar"
                else:
                    make_binary = binary_system()
                    incl_mean = 10 | units.deg
                    incl_std = 2 | units.deg

                    ecc = make_binary.ecc_distr("Jeans")
                    incl = make_binary.incl_distr("uniform", incl_mean, incl_std) * (1 | units.rad)
                    no_syst = int((self.N-1)/2)
                    for parti_ in range(no_syst):   
                        arg_of_peri = np.random.uniform(0, 2*np.pi, 1)[0] | units.deg
                        semi = np.random.uniform(10, 500, 1)[0] | units.au
                        true_anomaly = np.random.uniform(0, 2*np.pi, 1)[0] | units.deg
                        long_of_asc_node = np.random.uniform(0, 2*np.pi, 1)[0] | units.deg

                        particles[2*parti_+2].mass = 10*stellar_masses[parti_+2] 
                        binary = new_binary_from_orbital_elements(particles[2*parti_+1].mass, 
                                                        particles[2*parti_+2].mass,
                                                        abs(semi), ecc, true_anomaly, incl,
                                                        long_of_asc_node, arg_of_peri,
                                                        constants.G)
                        binary[0].position += particles[2*parti_+2].position
                        binary[1].position += particles[2*parti_+2].position
                        binary[0].type = "stellar"
                        binary[1].type = "imbh"
                        particles.remove_particle(particles[2*parti_+1])
                        particles.remove_particle(particles[2*parti_+2])
                        particles.add_particle(binary[0])
                        particles.add_particle(binary[1])
                        
            particles.z *= 0.1
            particles[0].position = SMBH_parti.position
            particles[0].velocity = SMBH_parti.velocity

            for parti_ in particles[1:]:
                parti_.vx += (constants.G*SMBH_parti.mass * (abs(parti_.y)/parti_.position.length()**2)).sqrt()
                parti_.vy += (constants.G*SMBH_parti.mass * (abs(parti_.x)/parti_.position.length()**2)).sqrt()

            min_dist = 0.15 | units.parsec
            max_dist = 0.25 | units.parsec
            for parti_ in particles[1:]:
                if parti_.position.length() < min_dist:
                    parti_.position *= min_dist/parti_.position.length()
                if parti_.position.length() > max_dist:
                    parti_.position *= max_dist/parti_.position.length()
            particles.scale_to_standard(convert_nbody=self.code_conv)

            parti_ = 0
            particles[0].mass = SMBH_parti.mass
            particles[0].type = "smbh"

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

        import matplotlib.pyplot as plt
        plt.hist(particles[1:].mass.value_in(units.MSun))
        plt.show()
        
        return particles


class binary_system(object):
  def ecc_distr(self, distr_string):
    if distr_string.lower() == "jeans":
      return np.random.uniform(0, 0.99, 1)[0]
  
  def incl_distr(self, incl_string, incl_mean, incl_std):
    if incl_string.lower() == "gaussian":
      return np.random.normal(incl_mean.value_in(units.deg), 
                              incl_std.value_in(units.deg), 1)[0]
    if incl_string.lower() == "uniform":
      return np.random.uniform(0, 90, 1)[0]
    
  def sem_distr(self, sem_string, sem_mean, sem_std):
    if sem_string.lower() == "gaussian":
      return np.random.normal(sem_mean.value_in(units.au), 
                              sem_std.value_in(units.au), 1)[0]
    if sem_string.lower() == "uniform":
      return np.random.uniform(sem_mean - 5*sem_std, 
                               sem_mean + 5*sem_std, 1)[0]

  def make_me_a_binary(self, body_a, body_b, semi = None, semi_mean = None,
                       semi_std = None, ecc = None, incl = None, incl_mean = None,
                       incl_std = None):
    """Function to create a binary from two stars.
       
       Inputs:
       body_a/b:   Star 1 or star 2
       semi:       Binary semi-major axis distribution (default to a = 100 AU)
       semi_mean:  Mean value of binary semi-major axis distribution
       semi_std:   Spread in distribution of binary semi-major axis
       ecc:        Eccentricity distribution of system (default to e = 2/3)
       incl:       Inclination distribution of system (default to i = 0 degrees)
       incl_mean:  Mean value of inclination of bin. systems
       incl_std:   Spread in inclination value for bin. systems
    """
    
    true_anomaly = 0 | units.deg
    long_of_asc_node = 0 | units.deg
    arg_of_peri = 0 | units.deg

    if not (ecc):
      ecc = 2/3
    else:
      ecc = self.ecc_distr(ecc)

    if not (incl):
      incl = 0 | units.deg
    else:
      incl = self.incl_distr(incl, incl_mean, incl_std) * (1 | units.rad)

    if not (semi):
      semi = 100 | units.AU
    else:
      semi = self.sem_distr(semi, semi_mean, semi_std) * (1 | units.AU)
    
    binary = new_binary_from_orbital_elements(body_a.mass, body_b.mass,
                                     abs(semi), ecc, true_anomaly, incl,
                                     long_of_asc_node, arg_of_peri,
                                     constants.G)
    binary[0].radius = body_a.radius
    binary[1].radius = body_b.radius
    
    return binary