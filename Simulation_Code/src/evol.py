from itertools import combinations
import numpy as np
import os
import pandas as pd
import time as cpu_time

from amuse.datamodel import Particles
from amuse.units import units, constants
from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.community.hermite.interface import Hermite
from amuse.community.hermite_grx.interface import HermiteGRX

from src.data_init import DataInitialise
from src.evol_func import *
from src.file_logistics import file_counter

class EvolveSystem(object):
    def __init__(self, parti, tend, eta, idist, conv, int_str, 
                 GRX_set, no_worker):
        """Setting up the simulation code
    
           Inputs:
           parti:      The particle set needed to simulate
           tend:       The end time of the simulation
           eta:        The step size
           idist:      The initial distance between IMBH and SMBH
           conv:       Variable used to convert between nbody units and SI
           int_str:    String to dictate whether using Hermite or Hermite GRX
           GRX_set:    SMBH particle class (only if using GRX)
           no_worker:  Number of workers used
        """
        #NOTE: ENDS ON DATA SET ARE DUE TO CHANGE OF SIMULATION AIM

        self.parti = parti
        self.tend = tend
        self.eta = eta
        self.idist = idist
        self.conv = conv
        self.int_str = int_str
        self.GRX_set = GRX_set
        self.init_time = cpu_time.time()
        self.no_workers = no_worker

        self.code = None
        self.pert = None

        self.time = 0 | units.yr
        self.iter = 0
        self.Nenc = 0
        self.eject  = 0
        self.ejected_mass = 0 | units.MSun

        self.N_parti = len(self.parti)
        self.init_IMBH = self.N_parti-1
        self.extra_note = ''

    def initialise_gravity(self):
        """Function setting up the gravitational code"""

        if self.int_str == 'Hermite':
            self.code = Hermite(self.conv, number_of_workers = self.no_workers)
            self.pert = 'Newtonian'
            self.code.particles.add_particles(self.parti)
        
        else:
            particles = Particles()
            print(self.GRX_set)
            particles.add_particle(self.GRX_set)
            particles.add_particle(self.parti[1:])
            self.parti = particles

            self.code = HermiteGRX(self.conv, number_of_workers = self.no_workers)
            perturbations = ["1PN_Pairwise", "1PN_EIH", "2.5PN_EIH"]
            self.pert = perturbations[2]
            self.code.parameters.perturbation = self.pert
            self.code.parameters.integrator = 'RegularizedHermite'
            self.code.small_particles.add_particles(self.parti[1:])
            self.code.large_particles.add_particles(self.GRX_set)
            self.code.parameters.light_speed = constants.c
            print('Simulating GRX with: ', self.pert)   

        self.code.parameters.dt_param = 1e-3
        self.stopping_condition = self.code.stopping_conditions.collision_detection
        self.stopping_condition.enable()

        self.channel_IMBH = {"from_gravity": 
                             self.code.particles.new_channel_to(self.parti,
                             attributes=["x", "y", "z", "vx", "vy", "vz", "mass"],
                             target_names=["x", "y", "z", "vx", "vy", "vz", "mass"]),
                             "to_gravity": 
                             self.parti.new_channel_to(self.code.particles,
                             attributes=["mass", "collision_radius"],
                             target_names=["mass", "radius"])} 

        if self.pert == 'Newtonian':    
            self.parti_KE = self.code.particles.kinetic_energy()
            self.parti_BE = self.code.particles.potential_energy()
            self.E0 = self.parti_KE + self.parti_BE 
        else: 
            self.parti_KE = self.code.particles.kinetic_energy()
            self.parti_BE = self.code.particles.potential_energy()
            self.E0 = self.code.get_total_energy_with(self.pert)[0]

    def check_ejection(self):
        """Function checking for particle ejection from cluster"""
        for particle in self.parti[1:]:
            rel_vel = particle.velocity - self.parti[0].velocity
            dist_core = particle.position - self.parti[0].position

            dist_vect = np.sqrt(np.dot(dist_core, dist_core))
            vel_vect  = np.sqrt(np.dot(rel_vel, rel_vel))
            curr_traj = (np.dot(dist_core, rel_vel))/(dist_vect * vel_vect) #Movement towards SMBH

            parti_KE = 0.5*particle.mass*(rel_vel.length())**2
            parti_BE = np.sum(indiv_PE_all(particle, self.parti))

            if parti_KE > abs(parti_BE) and \
                dist_core.length() > 2 | units.pc and \
                    curr_traj > 0:
                self.eject = 1
                self.ejected_key_track = particle.key_tracker
                self.ejected_mass = particle.mass
                self.extra_note = 'Stopped due to particle ejection'
                print("........Ejection Detected........")
                print('Simulation will now stop')
                break

    def check_merger(self):
        """Function checking for merger events"""
        if self.stopping_condition.is_set():
            print("........Encounter Detected........")
            print('Collision at step: ', iter)
            print('Simulation will now stop')
            for ci in range(len(self.stopping_condition.particles(0))):
                self.Nenc += 1

                if self.pert == 'Newtonian':
                    enc_particles_set = Particles(particles=[self.stopping_condition.particles(0)[ci],
                                                                self.stopping_condition.particles(1)[ci]])
                    enc_particles = enc_particles_set.get_intersecting_subset_in(self.parti)
                    merged_parti = merge_IMBH(self.parti, enc_particles, self.code.model_time, self.int_str, self.code)
                else:
                    particles = self.code.stopping_conditions.collision_detection.particles
                    enc_particles_set = Particles(particles=[particles(0), particles(1)])
                    merged_parti = merge_IMBH(self.parti, enc_particles_set, self.code.model_time, self.int_str, self.code)
                self.parti.synchronize_to(self.code.particles)

                self.tcoll = self.time.in_(units.s) - self.eta*self.tend
                
                self.ejected_key_track = self.parti[-1].key_tracker
                self.extra_note = 'Stopped due to merger'

    def run_code(self):
        data_trackers = DataInitialise()
        energy_tracker = data_trackers.energy_tracker(self.E0, self.parti_KE, 
                                                      self.parti_BE, self.time, 
                                                      0 | units.s)
        IMBH_tracker = data_trackers.IMBH_tracker(self.parti, self.time, self.N_parti)
        lagrangians = data_trackers.LG_tracker(self.time, self.code)

        ejected_key_track = 000   
        rhmass = []
        while self.time < self.tend:
            tcoll = 0 | units.yr
            self.iter += 1

            rhmass_val = LagrangianRadii(self.parti[1:])[6].in_(units.pc)
            rhmass.append(rhmass_val)

            self.time += self.eta*self.tend
            self.channel_IMBH["to_gravity"].copy()
                
            self.code.evolve_model(self.time)
            self.check_ejection()
            if self.eject == 0:
                self.check_merger()

            self.channel_IMBH["from_gravity"].copy()

            if self.int_str == 'GRX':
                if self.Nenc == 0 :
                    self.parti[0].position = self.code.particles[0].position
                    self.parti[0].velocity = self.code.particles[0].velocity
                else:
                    self.parti[-1].position = self.code.particles[0].position
                    self.parti[-1].velocity = self.code.particles[0].velocity

            rows = (len(self.parti)+self.Nenc)
            df_IMBH = pd.DataFrame()

            for x in combinations(range(len(self.parti)),2):
                idx1 = x[0]
                idx2 = x[1]
                if IMBH_tracker.iloc[idx1][0][0] == self.parti[idx2].key_tracker:
                    neigh_dist, nearest_parti, second_nearest = nearest_neighbour(self.parti[idx2], self.parti)

                    semimajor = [ ]
                    eccentric = [ ]
                    inclinate = [ ]
                    arg_peri  = [ ]
                    asc_node  = [ ]
                    true_anom = [ ]
                    neigh_key = [ ]

                    if idx1 == 0 and self.Nenc == 0:
                        semimajor = [0, 0, 0]
                        eccentric = [0, 0, 0]
                        inclinate = [0, 0, 0]
                        arg_peri  = [0, 0, 0]
                        asc_node  = [0, 0, 0]
                        true_anom = [0, 0, 0]
                        neigh_key = [0, 0, 0]

                    else:
                        if self.Nenc == 0:
                            SMBH_parti = self.parti[0]
                        else:
                            SMBH_parti = self.parti[-1]
                        for part_ in [SMBH_parti, nearest_parti, second_nearest]:
                            bin_sys = Particles()  
                            bin_sys.add_particle(self.parti[idx2])
                            bin_sys.add_particle(part_)
                            kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
                            semimajor.append(kepler_elements[2].value_in(units.pc))
                            eccentric.append(kepler_elements[3])
                            inclinate.append(kepler_elements[4])
                            arg_peri.append(kepler_elements[5])
                            asc_node.append(kepler_elements[6])
                            true_anom.append(kepler_elements[7])
                            if part_ == SMBH_parti:
                                if self.Nenc == 0:
                                    neigh_key.append(self.parti[0].key_tracker)
                                else:
                                    neigh_key.append(self.parti[-1].key_tracker)
                            else:
                                neigh_key.append(part_.key_tracker)

                    parti_KE = 0.5*self.parti[idx2].mass*((self.parti[idx2].velocity-self.parti[0].velocity).length())**2
                    parti_PE = np.sum(indiv_PE_all(self.parti[idx2], self.parti))

                    df_IMBH_vals = pd.Series({'{}'.format(self.time): [self.parti[idx2].key_tracker, self.parti[idx2].mass, 
                                                                       self.parti[idx2].position, self.parti[idx2].velocity, 
                                                                       parti_KE, parti_PE, neigh_key, semimajor * 1 | units.pc, 
                                                                       eccentric, inclinate, arg_peri, asc_node, true_anom, neigh_dist]})
                    break

                else:
                    df_IMBH_vals = pd.Series({'{}'.format(self.time): [np.NaN, np.NaN | units.MSun,
                                                                      [np.NaN | units.pc, np.NaN | units.pc, np.NaN | units.pc],
                                                                      [np.NaN | units.kms, np.NaN | units.kms, np.NaN | units.kms],
                                                                       np.NaN | units.J, np.NaN | units.J, np.NaN | units.m, np.NaN,
                                                                       np.NaN, np.NaN, np.NaN , np.NaN, np.NaN]})
                df_IMBH = df_IMBH.append(df_IMBH_vals, ignore_index=True)
            IMBH_tracker = IMBH_tracker.append(df_IMBH, ignore_index=True)
            IMBH_tracker['{}'.format(self.time)] = IMBH_tracker['{}'.format(self.time)].shift(-rows)
            IMBH_tracker = IMBH_tracker.dropna(axis='rows')
            
            parti_KE = self.code.particles.kinetic_energy()
            parti_BE = self.code.particles.potential_energy()
            if self.pert == 'Newtonian':
                Etp = parti_KE + parti_BE 
                Et = Etp
            else: 
                Et = self.code.get_total_energy_with(self.pert)[0]
            de = abs(Et-self.E0)/abs(self.E0)
            
            if 20 < self.iter:
                dEs = abs(Et-energy_tracker.iloc[19][3])/abs(energy_tracker.iloc[19][3])
                df_energy_tracker = pd.Series({'Appearance': None, 'Collision Mass': None, 'Collision Time': tcoll, 
                                               'Et': Et, 'Kinetic E': parti_KE.in_(units.J), 'Pot.E': parti_BE.in_(units.J),
                                               'Time': self.time.in_(units.kyr), 'dE': de, 'dEs': dEs, 'Pot.E': parti_BE.in_(units.J)})
            else:
                df_energy_tracker = pd.Series({'Appearance': None, 'Collision Mass': None, 'Collision Time': tcoll, 
                                               'Et': Et, 'Kinetic E': parti_KE.in_(units.J), 'Pot.E': parti_BE.in_(units.J),
                                               'Time': self.time.in_(units.kyr), 'dE': de, 'dEs': 0, 'Pot.E': parti_BE.in_(units.J) })
            energy_tracker = energy_tracker.append(df_energy_tracker, ignore_index=True)

            df_lagrangians = pd.Series({'Time': self.time.in_(units.kyr),
                                        'LG25': LagrangianRadii(self.code.particles[1:])[5].in_(units.pc),
                                        'LG50': LagrangianRadii(self.code.particles[1:])[6].in_(units.pc),
                                        'LG75': LagrangianRadii(self.code.particles[1:])[7].in_(units.pc)})
   
            lagrangians = lagrangians.append(df_lagrangians, ignore_index = True)
            time1 = self.time

            comp_end = cpu_time.time()
            comp_time = comp_end-self.init_time
            if self.eject > 0 or self.Nenc > 0 or comp_time >= 80:#>= 590400:
                self.time = self.tend 
                
        time1 += self.past_time
        self.code.stop()
        
        if iter > 1:
            print('Saving Data')
            chaos_stab_timescale = time1
            count = file_counter(self.int_str)

            if (comp_time >= 590400):
                if time1 == self.tend:
                    ejected_key_track = self.parti[1].key_tracker
            
                path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7/'
                file_names = 'IMBH_'+str(self.int_str)+'_'+str(self.pert)+'_'+str(self.init_IMBH)+'_sim'+str(count)+ \
                            '_idist'+str('{:.3f}'.format(self.idist.value_in(units.pc)))+'_equal_mass_' \
                            +str('{:.3f}'.format(self.parti[2].mass.value_in(units.MSun)))+str(self.kit)+'.pkl'

                IMBH_tracker.to_pickle(os.path.join(path+str('particle_trajectory'), file_names))
                energy_tracker.to_pickle(os.path.join(path+str('energy'), file_names))
                lagrangians.to_pickle(os.path.join(path+str('lagrangians'), file_names))
                data_trackers.chaotic_sim_tracker(self.parti, self.init_IMBH, self.Nenc, None, time1, 
                                                  ejected_key_track, chaos_stab_timescale, None, 
                                                  self.ejected_mass, comp_time, self.eject, self.int_str, self.pert, self.kit)
                                                
                if self.Nenc > 0:
                    data_trackers.coll_tracker(self.int_str, self.init_IMBH, count, self.idist, self.parti, tcoll, 
                                               None, ejected_key_track, None, self.pert)

                lines = ['Simulation: ', "Total CPU Time: "+str(comp_time)+' seconds', 
                        'Timestep: '+str(self.eta),
                        'Simulated until: '+str(time1.value_in(units.yr))+str(' years'), 
                        'Cluster Distance: '+str(self.idist.value_in(units.pc))+' pcs', 
                        'Masses of IMBH: '+str(self.parti.mass.value_in(units.MSun))+' MSun',
                        "No. of initial IMBH: "+str(self.init_IMBH), 
                        'Number of new particles: '+str(len(self.parti)-1-self.init_IMBH),
                        'Total Number of (Final) IMBH: '+str(len(self.parti)-2),
                        'Number of mergers: '+str(self.Nenc), 'End Time: '+str(self.tend.value_in(units.yr))+' years', 
                        'Integrator: Hermite (NO PN)',
                        'Extra Notes: ', self.extra_note]

                with open(os.path.join(path+str('simulation_stats'), 'simulation_'+str(count)+'.txt'), 'w') as f:
                    for line in lines:
                        f.write(line)
                        f.write('\n')
                print('End time: ', time1)
            
            else:
                path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7_paused/'
                file_names = 'IMBH_'+str(self.int_str)+'_'+str(self.pert)+'_'+str(self.init_IMBH)+'_sim'+str(count)+ \
                            '_idist'+str('{:.3f}'.format(self.idist.value_in(units.pc)))+'_equal_mass_' \
                            +str('{:.3f}'.format(self.parti[2].mass.value_in(units.MSun)))+str(self.kit)+'.pkl'
                IMBH_tracker.to_pickle(os.path.join(path+str('particle_trajectory'), file_names))

        else:
            print('...No stability timescale - simulation ended too quick...')