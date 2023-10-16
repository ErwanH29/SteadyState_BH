from amuse.lab import *
import numpy as np
import glob
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import natsort
import os
import pickle as pkl

class plotter_setup(object):
    def font_size(self):
        axlabel_size = 16
        tick_size = 14

        return axlabel_size, tick_size

    def tickers(self, ax, plot_type):
        """
        Function to setup axis

        Inputs:
        ax:         Plotting axis
        plot_type:  String denoting whether 2D histogram or normal plot
        """
        
        axlabel_size, self.tick_size = self.font_size()

        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        if plot_type == 'plot':
            ax.tick_params(axis="y", which = 'both', direction="in", labelsize = self.tick_size)
            ax.tick_params(axis="x", which = 'both', direction="in", labelsize = self.tick_size)
            return ax
            
        ax.tick_params(axis="y", labelsize = self.tick_size)
        ax.tick_params(axis="x", labelsize = self.tick_size)
        return ax

    def tickers_pop(self, ax, pop, int_str):
        """
        Function to setup axis for population plots

        Inputs:
        ax:       Plotting axis
        pop:      Data array cluster population
        int_str:  String characterising integrator
        """

        axlabel_size, self.tick_size = self.font_size()

        ax.set_xlabel(r'$N_{\rm{IMBH}}$', fontsize = axlabel_size)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.tick_params(axis="y", which = 'both', direction="in", 
                       labelsize = self.tick_size)
        ax.tick_params(axis="x", which = 'both', direction="in", 
                       labelsize = self.tick_size)    
        
        if int_str == 'Hermite':
            xints = [i for i in range(1+int(max(pop))) if i%10 == 0 and i > 5]
            ax.set_xticks(xints)
            ax.set_xlim(5, 105)
            return ax

        xints = [i for i in range(1+int(max(pop))) if i%5 == 0 and i > 5 and i < 45]
        ax.set_xlim(5, 45)
        ax.set_xticks(xints)
        return ax

def bulk_stat_extractor(file_string):
    """
    Function which extracts all files in a given dir.
    
    Inputs:
    file_string: The directory wished to extract files from
    """

    filename = glob.glob(file_string)
    filename = natsort.natsorted(filename)
    data = [ ]

    for file_ in range(len(filename)):
        with open(filename[file_], 'rb') as input_file:
            print('Reading file: ', input_file)
            data.append(pkl.load(input_file))

    return data

def ejected_extract_final(pset, ejected, file, crop):
    """
    Extracts the final info on the ejected particle into an array
    
    Inputs:
    set:        The complete particle set plotting
    ejected:    The ejected particle
    file:       File name
    """
    ejec_idx = None
    for parti_ in range(len(pset)):
        if pset.iloc[parti_][0][0] == ejected.iloc[0][4]: 
            ejec_idx = parti_
    
    dir = os.path.join('figures/sphere_of_influence.txt')
    fixed_crop = 1e6
    dist = -5
    tol = 10**-6
    with open(dir) as f:
        line = f.readlines()
        for iter in range(len(line)):
            if iter%3 == 0 and line[iter][90+crop:-3] == file[59+crop:]:
                if line[iter][54:65] == 'rc_0.25_4e5':
                    sim_time = line[iter+1][49:57]
                    fixed_crop = int(round(float(''.join(chr_ for chr_ in sim_time if chr_.isdigit())) * 10**-3))

                elif line[iter][54:65] == 'rc_0.25_4e6':
                    sim_time = line[iter+1][49:57]
                    fixed_crop = int(round(float(''.join(chr_ for chr_ in sim_time if chr_.isdigit())) * 10**-3))

                distance = line[iter+2][48:65]
                distance = ''.join(chr_ for chr_ in distance if chr_.isdigit())
                digits = len(distance) - 1
                distance = float(distance)/10**digits
                print('Target distance: ', distance)
                
                ejec_parti = False
                j = 0
                while not (ejec_parti):
                    j += 1
                    for parti_ in range(np.shape(pset)[0]):
                        if parti_ != 0:
                            sim_snap = pset.iloc[parti_][j]
                            SMBH_coords = pset.iloc[0][j]

                            line_x = (sim_snap[2][0] - SMBH_coords[2][0])
                            line_y = (sim_snap[2][1] - SMBH_coords[2][1])
                            line_z = (sim_snap[2][2] - SMBH_coords[2][2])
                            dist = np.sqrt(line_x**2+line_y**2+line_z**2).value_in(units.pc)

                            if abs(distance - dist) <= tol:
                                print('Detected new ejectee.')
                                ejec_parti = True
                                ejec_idx = parti_

        if ejec_idx != None:
            ejec_data = pset.iloc[ejec_idx]
            ejec_data = ejec_data.replace(np.NaN, "[Np.NaN, [np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]")
            ejec_vel  = []

            time_step = min(len(ejec_data), fixed_crop)
            ejec_data = ejec_data.iloc[:time_step]
            if dist < 0:
                tot_steps = min(time_step, 15)
            else:
                tot_steps = min(time_step, 45)
            for steps_ in range(tot_steps):
                vel_ = ejec_data.iloc[(-steps_)][3].value_in(units.kms)
                vx = vel_[0] - pset.iloc[0][(-steps_)][3][0].value_in(units.kms)
                vy = vel_[1] - pset.iloc[0][(-steps_)][3][1].value_in(units.kms)
                vz = vel_[2] - pset.iloc[0][(-steps_)][3][2].value_in(units.kms)
                ejec_vel.append(np.sqrt(vx**2+vy**2+vz**2))
                
            idx = np.where(ejec_vel == np.nanmax(ejec_vel))[0]
            ejec_vel = np.asarray(ejec_vel)
            esc_vel = ejec_vel[idx]
            print('vesc: ', esc_vel)  
            return esc_vel
        else:
            print('Ejected DNE')
            return ejec_idx

def no_file_tracker(pop_arr, pop_data, no_files, no_samples):
    """
    Function to ensure same # of files are used during data extraction
    comparing algorithms.

    Input:
    pop_arr:     Array consisting of previous data files' population
    pop_data:    Population of data file
    no_files:    Maximum number of files allowed to analyse
    no_samples:  Number of files already extracted data from
    """

    if pop_arr == pop_data:
        no_samples += 1
        pop_arr = pop_data
        if no_samples > no_files:
            process = False
        else:
            process = True
        return no_samples, process, pop_arr
        
    pop_arr = pop_data
    no_samples = 1
    process = True
    return no_samples, process, pop_arr

def folder_loop(iterf):
    """
    Function to organise arrays.
    """

    if iterf == 0:
        drange = 2
        integrator = ['Hermite', 'GRX']
        return integrator, drange
        
    drange = 1
    integrator = ['GRX']
    return integrator, drange

def folder_data_loop(iterf, int):
    """
    Function to get right iteration for plotters.
    """

    if iterf == 0:
        sim_ = int
        return sim_

    sim_ = int + (1 + iterf)
    return sim_

def moving_average(array, smoothing):
        """
        Function to remove the large fluctuations in various properties by taking the average
        
        Inputs:
        array:     Variable experiencing smoothing
        smoothing: Number of elements to average over
        """

        value = np.cumsum(array, dtype=float)
        value[smoothing:] = value[smoothing:] - value[:-smoothing]

        return value[smoothing-1:]/smoothing

def ndata_chaos(iterf, dataG, chaosG, fold_):
    """
    Function to organise data files for a given directory before extracting
    them to make new, compressed files.
    """

    tcropH = 63
    if iterf == 0:
        drange = 2
        Hermite_data = glob.glob(os.path.join('/media/erwanh/Elements/'+fold_+'/Hermite/particle_trajectory/*'))
        chaoticH = ['/media/erwanh/Elements/'+fold_+'/data/Hermite/chaotic_simulation/'+str(i[tcropH:]) for i in Hermite_data]
        filename = [natsort.natsorted(Hermite_data), natsort.natsorted(dataG)]
        filenameC = [natsort.natsorted(chaoticH), natsort.natsorted(chaosG)]
        integrator = ['Hermite', 'GRX']
        return filename, filenameC, integrator, drange

    drange = 1
    filename = [natsort.natsorted(dataG)] 
    filenameC = [natsort.natsorted(chaosG)]
    integrator = ['GRX']
    return filename, filenameC, integrator, drange


def simulation_stats_checker(folder, int_string, file_crop, crop):
    """
    Function to check the final outcomes of all the simulations

    Inputs:
    folder:     Directory containing data
    int_string: Integrator
    file_crop:  Number of data sets
    crop:       Text crop for sphere_of_influence.txt
    """

    filename = glob.glob('/media/erwanh/Elements/'+folder+'/data/'+int_string+'/simulation_stats/*')
    filename = natsort.natsorted(filename)
    pfile = glob.glob(os.path.join('/media/erwanh/Elements/'+folder+'/data/'+int_string+'/chaotic_simulation/*'))
    pfile = natsort.natsorted(pfile)

    pops = np.arange(10, 105, 5).tolist()
    pop_samp = [0, 0, 0, 0, 0, 0, 0]

    SMBH_merger = 0 
    IMBH_merger = 0
    ejection = 0
    tot_sims = 0
    complete = 0

    for file_ in range(len(filename)):
        with open(filename[file_]) as f:
            line = f.readlines()
            substring= 'No. of initial IMBH'
            for i, item in enumerate(line):
                if substring in item:
                    line_pop = line[i][21:24]   
            pop = float(line_pop)
            
            idx = pops.index(pop)
            if pop <= 40 and pop_samp[idx] < file_crop:
                pop_samp[idx] += 1
                tot_sims += 1
                line1 = line[-9][:-7]
                line2 = line[3][17:]
                data = line1.split()
                mass = max([float(i) for i in data])
                sim_time = int(''.join(chr_ for chr_ in line2 if chr_.isdigit()))

                if mass == 4001000 or mass == 40001000 or mass == 401000:
                    SMBH_merger += 1
                elif data == 2000:
                    IMBH_merger += 1
                elif sim_time == 1e9:
                    complete += 1
                else:
                    ejection += 1

    dir = os.path.join('figures/sphere_of_influence.txt')
    with open(dir) as f:
        line = f.readlines()
        for file_ in pfile:
            for iter in range(len(line)):
                if iter%3 == 0 and line[iter][90+crop:-3] == file_[63+crop:]:
                    with open(file_, 'rb') as input_file:
                        ctracker = pkl.load(input_file)
                        if 5*round(0.2*ctracker.iloc[0][6]) <= 40:
                            if ctracker.iloc[0][2] > 0 | units.MSun:
                                SMBH_merger -= 1
                                ejection += 1
                            elif ctracker.iloc[0][-4] > 0 :
                                None
                            elif ctracker.iloc[0][-2] == 100 | units.Myr :
                                complete -= 1
                                ejection += 1         

    with open('figures/'+int_string+'_'+folder+'_summary.txt', 'w') as file:
        file.write('Simulation outcomes for '+str(int_string))
        file.write('\nTotal simulations:   '+str(tot_sims))
        file.write('\nSMBH merging events: '+str(SMBH_merger))
        file.write('\nIMBH merging events: '+str(IMBH_merger))
        file.write('\nEjection events:     '+str(ejection))
        file.write('\nCompleted sims:      '+str(complete))

def sphere_of_influence():
    folders = ['rc_0.25_4e6', 'rc_0.25_4e5', 'rc_0.25_4e7']
    #folders = ['rc_0.25_4e5']
    iterf = 0
    for fold_ in folders:
        if iterf == 0:
            drange = 2
            integrator = ['Hermite', 'GRX']
        else:
            drange = 1
            integrator = ['GRX']
        
        for int_ in range(drange):
            data = natsort.natsorted(glob.glob('/media/erwanh/Elements/'+fold_+'/'+integrator[int_]+'/particle_trajectory/*'))
            for file_ in range(len(data)):
                with open(data[file_], 'rb') as input_file:
                    file_size = os.path.getsize(data[file_])
                    if file_size < 2.9e9:
                        print('Reading File ', file_, ' : ', input_file)
                        pset = pkl.load(input_file)
                        distance = [ ]
                        time_snap = [ ]

                        for parti_, j in itertools.product(range(np.shape(pset)[0]), range(np.shape(pset)[1]-1)):
                            if parti_ != 0:
                                particle = pset.iloc[parti_]
                                SMBH_data = pset.iloc[0]

                                sim_snap = particle.iloc[j]
                                SMBH_coords = SMBH_data.iloc[j]

                                line_x = (sim_snap[2][0] - SMBH_coords[2][0])
                                line_y = (sim_snap[2][1] - SMBH_coords[2][1])
                                line_z = (sim_snap[2][2] - SMBH_coords[2][2])
                                dist = np.sqrt(line_x**2+line_y**2+line_z**2).value_in(units.pc)

                                if dist >= 6.00:
                                    distance.append(dist)
                                    time_snap.append(j*1000)

                        time_snap = np.asarray([i for i in time_snap])
                        distance = np.asarray([i for i in distance])
                        if len(time_snap) > 0:
                            with open('figures/sphere_of_influence.txt', 'a') as file:
                                index = np.where(time_snap == np.min(time_snap))
                                file.write('File '+str(input_file)+'\n')
                                file.write('Particle reaches beyond sphere of influence at: '+str(time_snap[index])+' (End time: ) '+str(np.shape(pset)[1]-1)+'\n')
                                file.write('Particle distance:                              '+str(distance[index])+'\n')
        iterf += 1

def stats_chaos_extractor(dir):
    steadytime_data = bulk_stat_extractor(dir)
    no_Data = len(steadytime_data)

    fin_parti_data = np.empty(no_Data)
    stab_time_data = np.empty(no_Data)

    for i in range(no_Data):
        sim_data = steadytime_data[i]
        if isinstance(sim_data.iloc[0][9], float):
            fin_parti_data[i] = sim_data.iloc[0][6] + sim_data.iloc[0][10]
            stab_time_data[i] = sim_data.iloc[0][13].value_in(units.Myr)
        
    return fin_parti_data, stab_time_data

"""print('...Gathering simulation outcomes...')
folders = ['rc_0.25_4e6', 'rc_0.25_4e6', 'rc_0.25_4e5', 'rc_0.25_4e7']
integr = ['Hermite', 'GRX', 'GRX', 'GRX', 'GRX']
data_files = [40, 60, 30, 30]

for fold_, integ_, nofiles_ in zip(folders, integr, data_files):
    if fold_ == 'rc_0.25_4e6' and integ_ == 'Hermite':
        crop_L = 94
        file_crop = 4
    else:
        crop_L = 90
        file_crop = 0
    simulation_stats_checker(fold_, integ_, nofiles_, file_crop)"""