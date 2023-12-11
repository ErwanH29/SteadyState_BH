from amuse.lab import *
from amuse.units import units
from src.parti_initialiser import *
import numpy as np

def calc_momentum(indivp):
    """
    Calculate momentum of colliding particles
    """
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

def indiv_PE(indivp, pset):
    """
    Finding a particles' individual PE based on its closest binary

    Input:
    indivp:  The individual particle computing BE for
    pset:     The complete particle set
    """

    PE_all = pset.potential_energy()
    PE_min = pset[pset.key != indivp.key].potential_energy()
    indiv_PE = PE_all - PE_min

    return indiv_PE

def merge_IMBH(parti, parti_in_enc, tcoll, int_string, code):
    """
    Merges two particles if the collision stopping condition has been met
    
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

    if int_string == "GRX":
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

    rel_pos = [(indivp.position - parti_.position).length().value_in(units.pc) for parti_ in pset]
    min_dist = np.sort(rel_pos)[1]
    index = np.where(rel_pos == min_dist)[0]
    index2 = np.where(rel_pos == np.sort(rel_pos)[2])[0]

    return min_dist, pset[index], pset[index2]