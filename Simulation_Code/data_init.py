from evol_func import *
from file_logistics import file_counter
from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.ext.orbital_elements import orbital_elements_from_binary
import pandas as pd
import os

class data_initialiser(object):
    def chaotic_sim_tracker(self, pset, init_pop, Nmerge, cum_mergermass, time, ejected_key, 
                            stab_time, added_mass, ejected_mass, comp_time, ejected, int_string, 
                            pert, init_IMBH, kit):

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
        init_IMBH:      The number of IMBH initially
        kit:            The simulation iteration number of the process
        """

        SMBH_code = MW_SMBH()
        count = file_counter(int_string)
        path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7'
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
        stab_tracker.to_pickle(os.path.join(path+str('/no_addition/chaotic_simulation'), 'IMBH_'+str(int_string)+'_'+str(pert)+'_'+str(init_IMBH)
                                            +'_sim'+str(count)+'_init_dist'+str('{:.3f}'.format(SMBH_code.distance.value_in(units.parsec)))
                                            +'_equal_mass_'+str('{:.3f}'.format(pset[2].mass.value_in(units.MSun)))+str(kit)+'.pkl'))

    def coll_tracker(self, int_string, init_IMBH, count, init_dist, pset, coll_time, enc_particles, ejected_key, merger_mass, pert):
        """
        In case merging event occurs, saves the data:
        
        Inputs:
        int_string:    Hermite or Hermite GRX to separate the data
        count:         The simulation #
        init_dist:     Initial distance of particles to SMBH
        pset:          The complete particle set
        coll_time:     The time the collision occured at
        enc_particles: The two merging particles
        ejected_key:   The final particle key
        merger_mass:   The merger mass
        pert:          The PN term simulated
        """

        path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7/'
        file_names = 'IMBH_'+str(int_string)+'_'+str(pert)+'_'+str(init_IMBH)+'_sim'+str(count) \
                     +'_init_dist'+str('{:.3f}'.format(init_dist.value_in(units.parsec)))+'_equal_mass_' \
                     +str('{:.3f}'.format(pset[2].mass.value_in(units.MSun)))+'.pkl'

        coll_tracker = pd.DataFrame()
        df_coll_tracker = pd.Series({'Collision Time': coll_time.in_(units.kyr),
                                     'Collided Particles': [enc_particles[0].key, enc_particles[1].key],
                                     'Initial Mass': [enc_particles[0].mass, enc_particles[1].mass] | units.MSun,
                                     'Emergent Particle': ejected_key, 'Final Mass': merger_mass | units.MSun})
        coll_tracker = coll_tracker.append(df_coll_tracker, ignore_index = True)
        coll_tracker.to_pickle(os.path.join(path+str('collision_events'), file_names))

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
                                       'Et': E0, 'Kinetic E': Ek.in_(units.J),
                                       'Pot. E': Ep.in_(units.J), 
                                       'Time': time.in_(units.kyr), 
                                       'dE': 0, 'dEs': 0, 'Pot. E': Ep.in_(units.J)
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
                df_IMBH_vals = pd.Series({#'key_tracker': pset[i].key_tracker, 
                                          '{}'.format(time): [pset[i].key_tracker, pset[i].mass, pset[i].position, 
                                                              pset[i].velocity, 0 | units.J, 0 | units.J, [0,0,0], 
                                                              [0,0,0], [0,0,0], [0,0,0],
                                                              [0,0,0], [0,0,0], [0,0,0]]})
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

    def LG_tracker(self, time, gravity):
        """
        Data set which tracks the Lagrangian radius and tidal radius of the cluster.
        
        Inputs:
        time:       The initial time of the simulation
        gravity:    The integrator used for the simulation
        """

        LG_array = pd.DataFrame()
        df_LG_tracker = pd.Series({'Time': time.in_(units.kyr),
                                   'LG25': LagrangianRadii(gravity.particles[1:])[5].in_(units.parsec),
                                   'LG50': LagrangianRadii(gravity.particles[1:])[6].in_(units.parsec),
                                   'LG75': LagrangianRadii(gravity.particles[1:])[7].in_(units.parsec)})
        LG_array = LG_array.append(df_LG_tracker , ignore_index=True)

        return LG_array

    def stable_sim_tracker(self, pset, Ninj, Nmerge, merger_mass, time, int_string, deltaE, pert):

        """
        Function which tracks information on the simulations which ended in stable state.
        This occurs if time = tend, new particle is injected or a merging event occurs.
        
        Inputs:
        pset:        The particle set
        Ninj:        Binary stating if simulation ended due to injection (1 = injection, 0 = none)
        Nmerge:      Binary stating if simulation ended due to merging event (1 = merge, 0 = none)
        merger_mass: The merger event
        time:        Duration of the stable system
        int_string:  String describing integrator used
        deltaE:      Change in energy
        """

        count = file_counter(int_string)
        SMBH_code = MW_SMBH()
        path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7'

        stable_sim_tracker = pd.DataFrame()
        df_stablesim = pd.Series({'Initial Particles': (len(pset)-1), 'Injected Event': Ninj,
                                  'Merged Event': Nmerge, 'Merger Mass': merger_mass,
                                  'Simulated Till/Stab Time': time.in_(units.yr),
                                  'Initial Distance': SMBH_code.distance.in_(units.parsec),
                                  'Initial Particle Mass': pset[2:].mass.in_(units.MSun),
                                  'Change in Energy': deltaE})
        stable_sim_tracker = stable_sim_tracker.append(df_stablesim, ignore_index = True)
        stable_sim_tracker.to_pickle(os.path.join(path+str('/stable_simulation'), 'IMBH_'+str(int_string)+'_' \
                                     +str(pert)+'_'+str(len(pset)-1)+'_sim'+str(count)+'_init_dist'+str('{:.3f}'.format(SMBH_code.distance.value_in(units.parsec))) \
                                     +'_equal_mass_'+str('{:.3f}'.format(pset[2].mass.value_in(units.MSun)))+'.pkl'))
