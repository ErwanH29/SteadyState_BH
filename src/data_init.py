import os
import pandas as pd

from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.lab import units, Particles, constants

from src.evol_func import indiv_PE, nearest_neighbour
from src.parti_initialiser import MW_SMBH

class data_initialiser(object):
    def chaotic_sim_tracker(self, pset, Nmerge, merger_mass, 
                            time, ejected_key, stab_time, 
                            added_mass, ejected_mass, comp_time, 
                            ejected, path, pert, fname):

        """
        Track global simulation statistics
        
        Inputs:
        pset:           The particle set
        Nmerge:         Number of mergers
        merger_mass:    Final remnant mass
        time:           The simulation end time
        ejected_key:    The key of the ejected particle
        stab_time:      Final simulation time
        added_mass:     The mass of the most recently added particle [Not used here]
        ejected_mass:   The mass of the ejected particle
        comp_time:      Total CPU time to simulate
        ejected:        Boolean identifying if ejection occured (1 = T || 0 = F)
        path:           File output directory
        pert:           The PN term simulated
        fname:          File output name
        """

        SMBH_code = MW_SMBH()
        stab_tracker = pd.DataFrame()
        df_stabtime = pd.Series({'Added Particle Mass': added_mass.in_(units.MSun),
                                 'Computation Time': str(comp_time),
                                 'Cumulative Merger Mass': merger_mass.in_(units.MSun),
                                 'Ejected Mass': ejected_mass.in_(units.MSun),
                                 'Ejected Particle': ejected_key, 
                                 'Ejection': ejected, 'Final Particles': (len(pset)-1),
                                 'Initial Distance': SMBH_code.distance.in_(units.parsec),
                                 'Initial Particle Mass': pset[2:].mass.in_(units.MSun),
                                 'Initial Particles': (len(pset)-1), 
                                 'Number of Mergers': Nmerge, 
                                 'PN Term': str(pert),
                                 'Simulated Till': time.in_(units.yr),
                                 'Stability Time': stab_time
                                 })
        stab_tracker = stab_tracker.append(df_stabtime, ignore_index = True)
        stab_tracker.to_pickle(os.path.join(path+str('no_addition/chaotic_simulation'), fname))

    def energy_tracker(self, E0, Ek, Ep, time, app_time):
        """
        Data set to track the energy evolution of the system
        
        Inputs:
        E0:       The total initial energy of the particles
        Ek/Ep:    The total kinetic/pot. energy of the particles
        time:     The initial time
        app_time: The time a new particle appears [Not used here]
        """

        energy_tracker = pd.DataFrame()
        df_energy_tracker = pd.Series({'Appearance': app_time,
                                       'Collision Mass': 0 | units.MSun,
                                       'Collision Time': 0 | units.s, 
                                       'E Total': E0, 
                                       'Kinetic E': Ek.in_(units.J),
                                       'Pot. E': Ep.in_(units.J), 
                                       'Time': time.in_(units.kyr), 
                                       'dE': 0, 'dEs': 0,
                                       'Pot. E': Ep.in_(units.J)
                                        })
        energy_tracker = energy_tracker.append(df_energy_tracker, ignore_index=True)

        return energy_tracker

    def IMBH_tracker(self, pset, time, init_pop):
        """
        Data set which holds information on each individual particles
        
        Inputs:
        pset:        The particle set
        time:        The initial time of the simulation
        init_pop:    The initial population of the simulation
        """

        IMBH_array = pd.DataFrame()
        df_IMBH    = pd.DataFrame()
        for i in range(init_pop):
            semimajor = []
            eccentric = []
            inclinate = []
            arg_peri  = []
            asc_node  = []
            true_anom = []
            neigh_key = []

            if i == 0 :
                data_arr = [pset[i].key_tracker, pset[i].mass, pset[i].position, 
                            pset[i].velocity, 0 | units.J, 0 | units.J, [0,0,0], 
                            [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]
                df_IMBH_vals = pd.Series({'{}'.format(time): data_arr})
                df_IMBH = df_IMBH.append(df_IMBH_vals, ignore_index=True)

            if i != 0:
                neighbour_dist, nearest_parti, second_nearest = nearest_neighbour(pset[i], pset)
                for part_ in [pset[0], nearest_parti, second_nearest]:
                    bin_sys = Particles()
                    bin_sys.add_particle(pset[i])
                    bin_sys.add_particle(part_)
                    kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
                    semimajor.append(kepler_elements[2].value_in(units.parsec))
                    eccentric.append(kepler_elements[3])
                    inclinate.append(kepler_elements[4])
                    arg_peri.append(kepler_elements[5])
                    asc_node.append(kepler_elements[6])
                    true_anom.append(kepler_elements[7])
                    neigh_key.append(part_.key_tracker)

                parti_KE = 0.5*pset[i].mass*pset[i].velocity.length()**2
                parti_PE = indiv_PE(pset[i], pset)
                
                data_arr = [pset[i].key_tracker, pset[i].mass, 
                            pset[i].position, pset[i].velocity, 
                            parti_KE, parti_PE, neigh_key, 
                            semimajor * 1 | units.parsec, 
                            eccentric, inclinate, arg_peri, 
                            asc_node, true_anom, neighbour_dist]
                df_IMBH_vals = pd.Series({'{}'.format(time): data_arr})
                df_IMBH = df_IMBH.append(df_IMBH_vals, ignore_index=True)
        IMBH_array = IMBH_array.append(df_IMBH, ignore_index=True)
       
        return IMBH_array

    def LG_tracker(self, time, gravity):
        """
        Track the Lagrangian radii.
        
        Inputs:
        time:       Simulation timestep
        gravity:    Gravitational integrator used
        """

        LG_array = pd.DataFrame()
        df_LG_tracker = pd.Series({'Time': time.in_(units.kyr),
                                   'LG25': LagrangianRadii(gravity.particles[1:])[5].in_(units.parsec),
                                   'LG50': LagrangianRadii(gravity.particles[1:])[6].in_(units.parsec),
                                   'LG75': LagrangianRadii(gravity.particles[1:])[7].in_(units.parsec)})
        LG_array = LG_array.append(df_LG_tracker , ignore_index=True)

        return LG_array