from evol_func import *
from file_logistics import file_counter
from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.ext.orbital_elements import orbital_elements_from_binary
import pandas as pd
import os

class data_initialiser(object):
    def chaotic_sim_tracker(self, pset, init_pop, Nmerge, cum_mergermass, time, ejected_key, 
                            stab_time, added_mass, ejected_mass, comp_time, ejected, int_string, pert):

        """
        Data set which tracks statistics on the simulations resulting in ejections
        
        Inputs:
        pset:           The particle set
        init_pop:       The initial population of the simulation
        Nmerge:         The number of mergers encountered since last particle added
        cum_mergermass: The cumulative mass merged
        time:           The simulation end time
        ejected_key:    The key of the ejected particle
        stab_time:      The length the system of N particle was stable for
        added_mass:     The mass of the most recently added particle
        ejected_mass:   The mass of the ejected particle
        comp_time:      Total time simulation lasted
        ejected:        0 for not ejected (sim. ended with merger) or 1 for ejection away from SMBH
        int_string:     String describing integrator used
        pert:           The PN term simulated
        """

        SMBH_code = MW_SMBH()
        count = file_counter(int_string)
        path = '/home/erwanh/Desktop/SteadyStateBH/Data_Process/data/GRX/'
        stab_tracker = pd.DataFrame()
        df_stabtime = pd.Series({'Added Particle Mass': added_mass.in_(units.MSun),
                                 'Computation Time': str(comp_time),
                                 'Cumulative Merger Mass': cum_mergermass.in_(units.MSun),
                                 'Ejected Mass': ejected_mass.in_(units.MSun),
                                 'Ejected Particle': ejected_key, 
                                 'Ejection': ejected, 'Final Particles': (len(pset)-1),
                                 'Initial Distance': SMBH_code.distance.in_(units.parsec),
                                 'Initial Particle Mass': init_pop[2:].mass.in_(units.MSun),
                                 'Initial Particles': (len(pset)-1), 
                                 'Number of Mergers': Nmerge, 
                                 'PN Term': str(pert),
                                 'Simulated Till': time.in_(units.yr),
                                 'Stability Time': stab_time})
        stab_tracker = stab_tracker.append(df_stabtime, ignore_index = True)
        stab_tracker.to_pickle(os.path.join(path+str('/no_addition/chaotic_simulation'), 'IMBH_'+str(int_string)+'_'+str(pert)+'_'+str(len(pset)-1)
                                            +'_sim'+str(count)+'_init_dist'+str('{:.3f}'.format(SMBH_code.distance.value_in(units.parsec)))
                                            +'_equal_mass_'+str('{:.3f}'.format(pset[2].mass.value_in(units.MSun)))+'.pkl'))

    def energy_tracker(self, E0, Ek, Ep, time, app_time):
        """
        Data set to track the energy evolution of the system
        
        Inputs:
        E0:       The total initial energy of the particles
        Ek/Ep:    The total kinetic/pot. energy of the particles
        time:     The initial time
        app_time: The time a new particle appears
        """

        energy_tracker = pd.DataFrame()
        df_energy_tracker = pd.Series({'Appearance': app_time,
                                       'Collision Mass': 0 | units.MSun,
                                       'Collision Time': 0 | units.s, 
                                       'Et': E0, 
                                       'Kinetic E': Ek.in_(units.J),
                                       'Pot. E': Ep.in_(units.J), 
                                       'Time': time.in_(units.kyr), 
                                       'dE': 0,  
                                       'dEs': 0,
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
                df_IMBH_vals = pd.Series({'{}'.format(time): [pset[i].key_tracker, pset[i].mass, pset[i].position, 
                                                              pset[i].velocity, 0 | units.J, 0 | units.J, [0,0,0], 
                                                              [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]})
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
                parti_PE = np.sum(indiv_PE_all(pset[i], pset))
                df_IMBH_vals = pd.Series({'{}'.format(time): [pset[i].key_tracker, pset[i].mass, pset[i].position, pset[i].velocity, 
                                                              parti_KE, parti_PE, neigh_key, semimajor * 1 | units.parsec, 
                                                              eccentric, inclinate, arg_peri, asc_node, true_anom, neighbour_dist]})
                df_IMBH = df_IMBH.append(df_IMBH_vals, ignore_index=True)
        IMBH_array = IMBH_array.append(df_IMBH, ignore_index=True)
       
        return IMBH_array