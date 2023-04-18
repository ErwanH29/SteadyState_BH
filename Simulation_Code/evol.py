from amuse.community.hermite_grx.interface import *
from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.lab import *
from amuse.units import units
from data_init import *
from evol_func import *
from parti_initialiser import *


import numpy as np
import pandas as pd
import time as cpu_time

# FOR ALICE SIMULATIONS:
#   - REMOVE ALL PRINT STATEMENTS + EXCESS FILE (ONLY KEEP CHAOTIC + STABLE + SIMULATION FILES)
#   - CHANGE number_of_workers
#   - ENSURE CORRECT FILE SAVING FORMAT (NAME)
#   - WHEN USING GRX ENSURE THAT THE INT_STRING IS CHANGED + UNCOMMENT LINES 47 - 54

def evolve_system(parti, tend, eta, init_dist, converter, int_string, GRX_set, kit):
    """
    Bulk of the simulation. Uses Hermite integrator to evolve the system while bridging it.
    Keeps track of the particles' position and stores it in a pickle file.
    
    Inputs:
    parti:       The particle set needed to simulate
    tend:        The end time of the simulation
    eta:         The step size
    init_dist:   The initial distance between IMBH and SMBH
    converter:   Variable used to convert between nbody units and SI
    int_string:  String to dictate whether using Hermite or Hermite GRX
    kit:            The simulation iteration number of the process.
    """
    
    set_printing_strategy("custom", preferred_units = [units.MSun, units.pc, units.kyr, units.AU/units.yr, units.J],
                                                       precision = 20, prefix = "", separator = " ", suffix = "")

    comp_start = cpu_time.time()
    initial_set = parti.copy()

    if int_string == 'Hermite':
        code = Hermite(converter, number_of_workers = 6)
        pert = 'Newtonian'
        code.particles.add_particles(parti)
    
    else:
        particles = Particles()
        particles.add_particle(GRX_set)
        particles.add_particle(parti[1:])
        parti = particles

        code = HermiteGRX(converter, number_of_workers = 9)
        perturbations = ["1PN_Pairwise", "1PN_EIH", "2.5PN_EIH"]
        pert = perturbations[2]
        code.parameters.perturbation = pert
        code.parameters.integrator = 'RegularizedHermite'
        code.small_particles.add_particles(parti[1:])
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
    cum_merger_mass = 0 | units.MSun
    stab_timescale = time

    N_parti = len(parti)
    init_IMBH = N_parti-1
    extra_note = ''
    
    if pert == 'Newtonian':    
        parti_KE = code.particles.kinetic_energy()
        parti_BE = code.particles.potential_energy()
        E0 = parti_KE + parti_BE 
    else: 
        parti_KE = code.particles.kinetic_energy()
        parti_BE = code.particles.potential_energy()
        E0 = code.get_total_energy_with(pert)[0]

    data_trackers = data_initialiser()
    energy_tracker = data_trackers.energy_tracker(E0, parti_KE, parti_BE, time, 0 | units.s)
    IMBH_tracker = data_trackers.IMBH_tracker(parti, time, N_parti)
    ejected_key_track = 000   
    rhmass = []
    lagrangians = data_trackers.LG_tracker(time, code)

    while time < tend:
        iter += 1
        time += eta*tend

        eject  = 0
        app_time = 0 | units.s
        merger_mass = 0 | units.MSun
        added_mass = 0 | units.MSun
        ejected_mass = 0 | units.MSun
        tcoll = 0 | units.yr

        rhmass_val = LagrangianRadii(parti[1:])[6].in_(units.pc)
        rhmass.append(rhmass_val)

        channel_IMBH["to_gravity"].copy()
            
        code.evolve_model(time)
        for particle in parti[1:]:
            rel_vel = particle.velocity - parti[0].velocity
            dist_core = particle.position - parti[0].position

            dist_vect = np.sqrt(np.dot(dist_core, dist_core))
            vel_vect  = np.sqrt(np.dot(rel_vel, rel_vel))
            curr_traj = (np.dot(dist_core, rel_vel))/(dist_vect * vel_vect) #Movement towards SMBH

            parti_KE = 0.5*particle.mass*(rel_vel.length())**2
            parti_BE = np.sum(indiv_PE_all(particle, parti))

            bin_sys = Particles()
            bin_sys.add_particle(particle)
            bin_sys.add_particle(parti[0])
            kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
            ecc = kepler_elements[3]

            if parti_KE > abs(parti_BE) and dist_core.length() > 2 | units.pc and curr_traj > 0 and ecc > 1:
                eject = 1
                ejected_key_track = particle.key_tracker
                ejected_mass = particle.mass
                extra_note = 'Stopped due to particle ejection'
                print("........Ejection Detected........")
                print('Simulation will now stop')
                break

        injbin = 0
        if eject == 0:
            if stopping_condition.is_set():
                print("........Encounter Detected........")
                print('Collision at step: ', iter)
                print('Simulation will now stop')
                merge = 1
                for ci in range(len(stopping_condition.particles(0))):
                    Nenc += 1
                    if pert == 'Newtonian':
                        energy_before = code.potential_energy + code.kinetic_energy
                        enc_particles_set = Particles(particles=[stopping_condition.particles(0)[ci],
                                                                 stopping_condition.particles(1)[ci]])
                        enc_particles = enc_particles_set.get_intersecting_subset_in(parti)
                        merged_parti = merge_IMBH(parti, enc_particles, code.model_time, int_string, code)
                        merger_mass = merged_parti.mass.sum()
                        cum_merger_mass += merger_mass
                        energy_after = code.potential_energy + code.kinetic_energy
                    else:
                        energy_before = code.get_total_energy_with(pert)[0]
                        particles = code.stopping_conditions.collision_detection.particles
                        enc_particles_set = Particles(particles=[particles(0), particles(1)])
                        merged_parti = merge_IMBH(parti, enc_particles_set, code.model_time, int_string, code)
                        merger_mass = merged_parti.mass.sum()
                        cum_merger_mass += merger_mass
                        energy_after= code.get_total_energy_with(pert)[0]
                    parti.synchronize_to(code.particles)

                    tcoll = time.in_(units.s) - eta*tend
                    deltaE = abs(energy_after-energy_before)/abs(energy_before)
                    data_trackers.stable_sim_tracker(parti, injbin, merge, merger_mass, stab_timescale, int_string, deltaE, pert) 
                    
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
                    neighbour_dist, nearest_parti, second_nearest = nearest_neighbour(parti[j], parti)

                    semimajor = []
                    eccentric = []
                    inclinate = []
                    arg_peri  = []
                    asc_node  = []
                    true_anom = []
                    neigh_key = []

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
                        for part_ in [SMBH_parti, nearest_parti, second_nearest]:
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
                    parti_PE = np.sum(indiv_PE_all(parti[j], parti))

                    df_IMBH_vals = pd.Series({'{}'.format(time): [parti[j].key_tracker, parti[j].mass, parti[j].position, parti[j].velocity, 
                                                                  parti_KE, parti_PE, neigh_key, semimajor * 1 | units.pc, 
                                                                  eccentric, inclinate, arg_peri, asc_node, true_anom, neighbour_dist]})
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
        if pert == 'Newtonian':
            Etp = parti_KE + parti_BE 
            Et = Etp
        else: 
            Et = code.get_total_energy_with(pert)[0]
        de = abs(Et-E0)/abs(E0)
        
        df_energy_tracker = pd.Series({'Appearance': app_time, 'Collision Mass': merger_mass, 'Collision Time': tcoll, 
                                       'Et': Et, 'Kinetic E': parti_KE.in_(units.J), 'Pot.E': parti_BE.in_(units.J),
                                       'Time': time.in_(units.kyr), 'dE': de, 'dEs': 0, 'Pot.E': parti_BE.in_(units.J) })
        energy_tracker = energy_tracker.append(df_energy_tracker, ignore_index=True)

        df_lagrangians = pd.Series({'Time': time.in_(units.kyr),
                                   'LG25': LagrangianRadii(code.particles[1:])[5].in_(units.pc),
                                   'LG50': LagrangianRadii(code.particles[1:])[6].in_(units.pc),
                                   'LG75': LagrangianRadii(code.particles[1:])[7].in_(units.pc)})
        lagrangians = lagrangians.append(df_lagrangians, ignore_index = True)

        time1 = time
        if eject > 0 or Nenc > 0:
            time = tend 

    code.stop()
    comp_end = cpu_time.time()
    comp_time = comp_end-comp_start 
    
    if iter > 10:
        print('Saving Data')
        no_plot = False
        chaos_stab_timescale = time1
        count = file_counter(int_string)

        if time1 == tend:
            ejected_key_track = parti[1].key_tracker
       
        path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e7/'
        file_names = 'IMBH_'+str(int_string)+'_'+str(pert)+'_'+str(init_IMBH)+'_sim'+str(count)+ \
                     '_init_dist'+str('{:.3f}'.format(init_dist.value_in(units.pc)))+'_equal_mass_' \
                     +str('{:.3f}'.format(parti[2].mass.value_in(units.MSun)))+str(kit)+'.pkl'

        IMBH_tracker.to_pickle(os.path.join(path+str('particle_trajectory'), file_names))
        energy_tracker.to_pickle(os.path.join(path+str('energy'), file_names))
        lagrangians.to_pickle(os.path.join(path+str('lagrangians'), file_names))
        data_trackers.chaotic_sim_tracker(parti, initial_set, Nenc, cum_merger_mass, time1, ejected_key_track, chaos_stab_timescale, 
                                          added_mass, ejected_mass, comp_time, eject, int_string, pert, init_IMBH, kit)
                                          
        if Nenc > 0:
            data_trackers.coll_tracker(int_string, init_IMBH, count, init_dist, parti, tcoll, 
                                       enc_particles_set, ejected_key_track, merger_mass, pert)

        lines = ['Simulation: ', "Total CPU Time: "+str(comp_end-comp_start)+' seconds', 
                 'Timestep: '+str(eta),
                 'Simulated until: '+str(time1.value_in(units.yr))+str(' years'), 
                 'Cluster Distance: '+str(init_dist.value_in(units.pc))+' pcs', 
                 'Masses of IMBH: '+str(parti.mass.value_in(units.MSun))+' MSun',
                 "No. of initial IMBH: "+str(init_IMBH), 
                 'Number of new particles: '+str(len(parti)-1-init_IMBH),
                 'Total Number of (Final) IMBH: '+str(len(parti)-2),
                 'Number of mergers: '+str(Nenc), 'End Time: '+str(tend.value_in(units.yr))+' years', 
                 'Integrator: '+str(int_string),
                 'Extra Notes: ', extra_note]

        with open(os.path.join(path+str('simulation_stats'), 'simulation_'+str(count)+'.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        print('End time: ', time1)

    else:
        no_plot = True
        print('...No stability timescale - simulation ended too quick...')

    return no_plot
