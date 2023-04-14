from amuse.lab import *
from amuse.units import units
from parti_initialiser import *
from evol_func import *
import numpy as np

def calc_momentum(indivp):
    return (indivp.mass * indivp.velocity).sum()

def find_nearest(array, value):
    """
    Function to find the nearest value in an array for a particular element.
    Inputs:
    array:  The array for which has all the elements
    value:  The value to compare the elements with
    """

    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

def indiv_PE_all(indivp, set):
    """
    Finding a particles' individual PE based on its closest binary
    Input:
    indivp:  The individual particle computing BE for
    set:     The complete particle set
    """

    array = []
    for comp_ in set:
        if indivp != comp_:
            distance = (indivp.position-comp_.position).length()
            temp_PE = (-constants.G*indivp.mass*comp_.mass)/abs(distance)
            array.append(temp_PE)

    return array

def merge_IMBH(parti, parti_in_enc, tcoll, int_string, code):
    """
    Function which merges two particles if the collision stopping condition has been met
    
    Inputs:
    parti:          The complete particle set being simulated
    parti_in_enc:   The particles in the collision
    tcoll:          The time-stamp for which the particles collide at
    int_string:     String telling whether simulating Hermite or GRX (for the PN cross-terms)
    code:           The integrator used
    """

    com_pos = parti_in_enc.center_of_mass()
    com_vel = parti_in_enc.center_of_mass_velocity()

    new_particle  = Particles(1)
    if calc_momentum(parti_in_enc[0]) > calc_momentum(parti_in_enc[1]):
        new_particle.key_tracker = parti_in_enc[0].key
    else: 
        new_particle.key_tracker = parti_in_enc[1].key

    new_particle.mass = parti_in_enc.total_mass()
    new_particle.collision_time = tcoll
    new_particle.position = com_pos
    new_particle.velocity = com_vel
    new_particle.radius = (2*constants.G*new_particle.mass)/(constants.c**2)
    new_particle.collision_radius = 3 * new_particle.radius
    new_particle.coll_events = 1

    if int_string != 'Hermite':
        if new_particle.mass > 1e6 | units.MSun:
            code.particles.remove_particles(parti_in_enc)
            code.large_particles.add_particles(new_particle)
        else:
            code.particles.remove_particles(parti_in_enc)
            code.small_particles.add_particles(new_particle)
    
    parti.add_particles(new_particle)
    parti.remove_particles(parti_in_enc)

    return new_particle

def nearest_neighbour(indivp, pset):
    """
    Function to find the nearest particle to some individual.
    
    Inputs:
    indivp: The individual particle
    pset:   The complete particle set
    """

    min_dist = [ ]
    for parti_ in pset:
        if indivp != parti_:
            rel_pos = indivp.position - parti_.position
            min_dist.append(rel_pos.length().value_in(units.parsec))
            
    temp = np.sort(min_dist)
    index = np.where(min_dist == temp[0])[0]
    index2 = np.where(min_dist == temp[1])[0]

    return min(min_dist), pset[index], pset[index2]

def SMBH_filter(pset):
    return pset[pset.mass < 5e4 | units.MSun]