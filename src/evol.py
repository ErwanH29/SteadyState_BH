import numpy as np
import os
import pandas as pd
import time as cpu_time

from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.lab import Particles
from amuse.units import constants, units

from src.data_init import data_initialiser
from src.evol_func import indiv_PE, merge_IMBH, nearest_neighbour
from src.file_logistics import file_counter


def evolve_system(parti, tend, eta, rvir, converter, 
                  int_string, GRX_set, Nsim, init_time, 
                  init_age):
    """
    Evolve the cluster
    
    Inputs:
    parti:       The particle set needed to simulate
    tend:        The end time of the simulation
    eta:         The step size
    rvir:        Initial cluster virial radius
    converter:   Variable used to convert between nbody units and SI
    int_string:  String to identify Hermite or GRX
    GRX_set:     GRX set SMBH particle
    Nsim:        Simulation #
    init_time:   Initial CPU time
    init_age:    Initial cluster age
    """


    if int_string == 'Hermite':
        from amuse.community.hermite.interface import Hermite
        code = Hermite(converter, number_of_workers = 1)
        pert = 'Newtonian'
        code.particles.add_particles(parti)
    
    else:
        from amuse.community.hermite_grx.interface import HermiteGRX
        particles = Particles()
        particles.add_particle(GRX_set)
        particles.add_particle(parti[1:])
        parti = particles

        code = HermiteGRX(converter, number_of_workers = 1)
        perturbations = ["1PN_Pairwise", "1PN_EIH", "2.5PN_EIH"]
        pert = perturbations[2]
        code.parameters.perturbation = pert
        code.parameters.integrator = 'RegularizedHermite'
        code.small_particles.add_particles(parti[parti.mass < 0.5*parti.mass.max()])
        code.large_particles.add_particles(GRX_set)
        code.parameters.light_speed = constants.c
        print('Simulating GRX with: ', pert)   

    code.parameters.dt_param = 1e-3
    stopping_condition = code.stopping_conditions.collision_detection
    stopping_condition.enable()

    channel_IMBH = {"from_gravity": 
                    code.particles.new_channel_to(parti,
                    attributes=["x", "y", "z", "vx", "vy", "vz", "mass"],
                    target_names=["x", "y", "z", "vx", "vy", "vz", "mass"]),
                    "to_gravity": 
                    parti.new_channel_to(code.particles,
                    attributes=["mass", "collision_radius"],
                    target_names=["mass", "radius"])} 

    time = 0 | units.yr
    iter = 0
    Nenc = 0
    remn_mass = 0 | units.MSun

    N_parti = len(parti)
    init_IMBH = N_parti-1
    extra_note = ''
    
    parti_KE = code.particles.kinetic_energy()
    parti_BE = code.particles.potential_energy()
    if pert == 'Newtonian':    
        E0 = parti_KE + parti_BE 
    else: 
        E0 = code.get_total_energy_with(pert)[0]

    data_trackers = data_initialiser()
    energy_tracker = data_trackers.energy_tracker(E0, parti_KE, parti_BE, time, None)
    IMBH_tracker = data_trackers.IMBH_tracker(parti, time, N_parti)
    lagrangians = data_trackers.LG_tracker(time, code)

    ejected_key_track = 000   
    while time < tend:
        eject  = 0
        iter += 1
        app_time = 0 | units.s
        merger_mass = 0 | units.MSun
        ejected_mass = 0 | units.MSun
        tcoll = 0 | units.yr

        time += eta*tend
        channel_IMBH["to_gravity"].copy()
        code.evolve_model(time)
        channel_IMBH["from_gravity"].copy()

        pset = parti.copy_to_memory()
        SMBH = pset[pset.mass == max(pset.mass)]
        for particle in pset[pset.key != SMBH.key]:
            rel_vel = (particle.velocity - SMBH.velocity)[0]
            dist_core = (particle.position - SMBH.position)[0]

            dist_vect = np.sqrt(np.dot(dist_core, dist_core))
            vel_vect = np.sqrt(np.dot(rel_vel, rel_vel))
            curr_traj = (np.dot(dist_core, rel_vel))/(dist_vect * vel_vect) #Movement towards SMBH
            parti_KE = 0.5*particle.mass*(rel_vel.length())**2
            parti_BE = indiv_PE(particle, parti)
            
            if parti_KE > abs(parti_BE) \
                and dist_core.length() > 2 | units.pc \
                    and curr_traj > 0:
                eject = 1
                ejected_key_track = particle.key_tracker
                ejected_mass = particle.mass
                extra_note = 'Stopped due to particle ejection'
                print("........Ejection Detected........")
                print('Simulation will now stop')
                break

            elif dist_core.length() > 6 | units.pc:
                eject = 1
                ejected_key_track = particle.key_tracker
                ejected_mass = particle.mass
                extra_note = 'Stopped due to particle ejection'
                print("........Drifter Detected........")
                print('Simulation will now stop')
                break

        if eject == 0:
            if stopping_condition.is_set():
                print("........Encounter Detected........")
                print('Collision at step: ', iter)
                print('Simulation will now stop')
                for ci in range(len(stopping_condition.particles(0))):
                    Nenc += 1
                    if pert == 'Newtonian':
                        enc_particles_set = Particles(particles=[stopping_condition.particles(0)[ci],
                                                                 stopping_condition.particles(1)[ci]])
                        enc_particles = enc_particles_set.get_intersecting_subset_in(parti)
                        merged_parti = merge_IMBH(parti, enc_particles, 
                                                  code.model_time, 
                                                  int_string, code)
                    else:
                        particles = code.stopping_conditions.collision_detection.particles
                        enc_particles_set = Particles(particles=[particles(0), particles(1)])
                        merged_parti = merge_IMBH(parti, enc_particles_set, 
                                                  code.model_time, int_string, 
                                                  code)
                    parti.synchronize_to(code.particles)

                    merger_mass = merged_parti.mass.sum()
                    remn_mass += merger_mass

                    tcoll = time.in_(units.s) - eta*tend
                    
                    ejected_key_track = parti[-1].key_tracker
                    extra_note = 'Stopped due to merger'

        channel_IMBH["from_gravity"].copy()

        if int_string == 'GRX':
            if Nenc == 0 :
                parti[0].position = code.particles[0].position
                parti[0].velocity = code.particles[0].velocity
            else:
                parti[-1].position = code.particles[0].position
                parti[-1].velocity = code.particles[0].velocity

        rows = (len(parti)+Nenc)
        df_IMBH = pd.DataFrame()
        for i in range(len(parti)):
            for j in range(len(parti)):
                if IMBH_tracker.iloc[i][0][0] == parti[j].key_tracker:
                    ndist, nparti, nnparti = nearest_neighbour(parti[j], parti)

                    semimajor = [ ]
                    eccentric = [ ]
                    inclinate = [ ]
                    arg_peri  = [ ]
                    asc_node  = [ ]
                    true_anom = [ ]
                    neigh_key = [ ]

                    if i == 0 and Nenc == 0:
                        semimajor = [0, 0, 0]
                        eccentric = [0, 0, 0]
                        inclinate = [0, 0, 0]
                        arg_peri  = [0, 0, 0]
                        asc_node  = [0, 0, 0]
                        true_anom = [0, 0, 0]
                        neigh_key = [0, 0, 0]

                    else:
                        if Nenc == 0:
                            SMBH_parti = parti[0]
                        else:
                            SMBH_parti = parti[-1]
                        for part_ in [SMBH_parti, nparti, nnparti]:
                            bin_sys = Particles()  
                            bin_sys.add_particle(parti[j])
                            bin_sys.add_particle(part_)
                            kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
                            semimajor.append(kepler_elements[2].value_in(units.pc))
                            eccentric.append(kepler_elements[3])
                            inclinate.append(kepler_elements[4])
                            arg_peri.append(kepler_elements[5])
                            asc_node.append(kepler_elements[6])
                            true_anom.append(kepler_elements[7])
                            if part_ == SMBH_parti:
                                if Nenc == 0:
                                    neigh_key.append(parti[0].key_tracker)
                                else:
                                    neigh_key.append(parti[-1].key_tracker)
                            else:
                                neigh_key.append(part_.key_tracker)

                    parti_KE = 0.5*parti[j].mass*((parti[j].velocity-parti[0].velocity).length())**2
                    parti_PE = indiv_PE(parti[j], parti)

                    df_IMBH_vals = pd.Series({'{}'.format(time): [parti[j].key_tracker, parti[j].mass, parti[j].position, parti[j].velocity, 
                                                                  parti_KE, parti_PE, neigh_key, semimajor * 1 | units.pc, 
                                                                  eccentric, inclinate, arg_peri, asc_node, true_anom, ndist]})
                    break

                else:
                    df_IMBH_vals = pd.Series({'{}'.format(time): [np.NaN, np.NaN | units.MSun,
                                                                 [np.NaN | units.pc, np.NaN | units.pc, np.NaN | units.pc],
                                                                 [np.NaN | units.kms, np.NaN | units.kms, np.NaN | units.kms],
                                                                  np.NaN | units.J, np.NaN | units.J, np.NaN | units.m, np.NaN,
                                                                  np.NaN, np.NaN, np.NaN , np.NaN, np.NaN]})
            df_IMBH = df_IMBH.append(df_IMBH_vals, ignore_index=True)
        IMBH_tracker = IMBH_tracker.append(df_IMBH, ignore_index=True)
        IMBH_tracker['{}'.format(time)] = IMBH_tracker['{}'.format(time)].shift(-rows)
        IMBH_tracker = IMBH_tracker.dropna(axis='rows')
        
        parti_KE = code.particles.kinetic_energy()
        parti_BE = code.particles.potential_energy()
        if pert.lower() == 'newtonian':
            Et = parti_KE + parti_BE 
        else: 
            Et = code.get_total_energy_with(pert)[0]
        de = abs(Et-E0)/abs(E0)
        
        if 20 < iter:
            dEs = abs(Et-energy_tracker.iloc[19][3])/abs(energy_tracker.iloc[19][3])
            df_energy_tracker = pd.Series({'Appearance': app_time, 
                                           'Collision Mass': merger_mass, 
                                           'Collision Time': tcoll, 
                                           'E Total': Et, 
                                           'Kinetic E': parti_KE.in_(units.J), 
                                           'Pot.E': parti_BE.in_(units.J),
                                           'Time': time.in_(units.kyr), 
                                           'dE': de, 'dEs': dEs, 
                                           'Pot.E': parti_BE.in_(units.J)})
        else:
            df_energy_tracker = pd.Series({'Appearance': app_time, 
                                           'Collision Mass': merger_mass, 
                                           'Collision Time': tcoll, 
                                           'E Total': Et, 
                                           'Kinetic E': parti_KE.in_(units.J), 
                                           'Pot.E': parti_BE.in_(units.J),
                                           'Time': time.in_(units.kyr), 
                                           'dE': de, 'dEs': 0, 
                                           'Pot.E': parti_BE.in_(units.J) })
        energy_tracker = energy_tracker.append(df_energy_tracker, ignore_index=True)

        df_lagrangians = pd.Series({'Time': time.in_(units.kyr),
                                    'LG25': LagrangianRadii(code.particles[1:])[5].in_(units.pc),
                                    'LG50': LagrangianRadii(code.particles[1:])[6].in_(units.pc),
                                    'LG75': LagrangianRadii(code.particles[1:])[7].in_(units.pc)})

        lagrangians = lagrangians.append(df_lagrangians, ignore_index=True)
        time1 = time

        comp_end = cpu_time.time()
        comp_time = comp_end-init_time
        if eject > 0 or Nenc > 0 or comp_time >= 590400: #590400s = ~7 day sim time
            time = tend 
    time1 += init_age
    code.stop()
    
    if iter > 1:
        print('Saving Data')
        no_plot = False
        chaos_stab_timescale = time1

        if max(parti.mass) > (10**7 | units.MSun):
            path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7/'
        elif max(parti.mass) < (10**7 | units.MSun) \
            and max(parti.mass) > (10**6 | units.MSun):
            path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e6/'
        else:
            path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e5/'
        count = file_counter(path)

        if (comp_time >= 590400):
            if time1 == tend:
                ejected_key_track = parti[1].key_tracker

                
            fname = 'IMBH_'+str(int_string)+'_'+str(pert)+'_'+str(init_IMBH)+'_sim'+str(count)+ \
                    '_rvir'+str('{:.3f}'.format(rvir.value_in(units.pc)))+'_equal_mass_' \
                    +str('{:.3f}'.format(parti[2].mass.value_in(units.MSun)))+str(Nsim)+'.pkl'

            IMBH_tracker.to_pickle(os.path.join(path+str('particle_trajectory'), fname))
            energy_tracker.to_pickle(os.path.join(path+str('energy'), fname))
            lagrangians.to_pickle(os.path.join(path+str('lagrangians'), fname))
            data_trackers.chaotic_sim_tracker(parti, Nenc, remn_mass, time1, 
                                              ejected_key_track, chaos_stab_timescale, 
                                              None, ejected_mass, comp_time, 
                                              eject, int_string, pert, path, fname)
                                            
            if Nenc > 0:
                data_trackers.coll_tracker(int_string, init_IMBH, count, rvir, parti, tcoll, 
                                        enc_particles_set, ejected_key_track, merger_mass, pert)

            lines = ['Simulation: ', "Total CPU Time: "+str(comp_time)+' seconds', 
                    'Timestep: '+str(eta),
                    'Simulated until: '+str(time1.value_in(units.yr))+str(' years'), 
                    'Cluster Distance: '+str(rvir.value_in(units.pc))+' pcs', 
                    'Masses of IMBH: '+str(parti.mass.value_in(units.MSun))+' MSun',
                    "No. of initial IMBH: "+str(init_IMBH), 
                    'Number of new particles: '+str(len(parti)-1-init_IMBH),
                    'Total Number of (Final) IMBH: '+str(len(parti)-2),
                    'Number of mergers: '+str(Nenc), 'End Time: '+str(tend.value_in(units.yr))+' years', 
                    'Integrator: Hermite (NO PN)',
                    'Extra Notes: ', extra_note]

            with open(os.path.join(path+str('simulation_stats'), 'simulation_'+str(count)+'.txt'), 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')
            print('End time: ', time1)
        
        else:
            path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7_paused/'
            fname = 'IMBH_'+str(int_string)+'_'+str(pert)+'_'+str(init_IMBH)+'_sim'+str(count)+ \
                         '_rvir'+str('{:.3f}'.format(rvir.value_in(units.pc)))+'_equal_mass_' \
                         +str('{:.3f}'.format(parti[2].mass.value_in(units.MSun)))+str(Nsim)+'.pkl'
            IMBH_tracker.to_pickle(os.path.join(path+str('particle_trajectory'), fname))

    else:
        no_plot = True
        print('...No stability timescale - simulation ended too quick...')
        

    return no_plot
