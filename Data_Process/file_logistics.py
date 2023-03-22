from amuse.lab import *
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import natsort
import os
import pickle as pkl

class plotter_setup(object):
    def init(self):
        self.axlabel_size = 13
        self.tilabel_size = 16
        self.tick_size = 11

    def tickers(self, ax, plot_type):
        """
        Function to setup axis
        """
        
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        if plot_type == 'plot':
            ax.tick_params(axis="y", which = 'both', direction="in", fontsize = self.tick_size)
            ax.tick_params(axis="x", which = 'both', direction="in", fontsize = self.tick_size)

        return ax

    def tickers_pop(self, ax, pop, int_str):
        """
        Function to setup axis for population plots
        """

        ax.set_xlabel(r'$N_{\rm{IMBH}}$', fontsize = 13)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.tick_params(axis="y", which = 'both', direction="in", fontsize = self.tick_size)
        ax.tick_params(axis="x", which = 'both', direction="in", fontsize = self.tick_size)    
        
        if int_str == 'Hermite':
            xints = [i for i in range(1+int(max(pop))) if i % 10 == 0 and i > 5]
            ax.set_xlim(5, 105)
        else:
            xints = [i for i in range(1+int(max(pop))) if i % 5 == 0 and i > 5 and i < 45]
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

def ejected_extract_final(set, ejected):
    """
    Extracts the final info on the ejected particle into an array
    
    Inputs:
    set:        The complete particle set plotting
    ejected:    The ejected particle
    """

    for parti_ in range(len(set)):
        if set.iloc[parti_][0][0] == ejected.iloc[0][4]: 
            ejec_data = set.iloc[parti_]   #Make data set of only ejected particle
            ejec_data = ejec_data.replace(np.NaN, "[Np.NaN, [np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]")
            ejec_vel = []
            tot_steps = min(round(len(ejec_data)**0.5), 10)
            
            for steps_ in range(tot_steps):
                vel_ = ejec_data.iloc[(-steps_)][3].value_in(units.kms)
                vx = vel_[0] - set.iloc[0][(-steps_)][3][0].value_in(units.kms)
                vy = vel_[1] - set.iloc[0][(-steps_)][3][1].value_in(units.kms)
                vz = vel_[2] - set.iloc[0][(-steps_)][3][2].value_in(units.kms)
                ejec_vel.append(np.sqrt(vx**2+vy**2+vz**2))
                
            idx = np.where(ejec_vel == np.nanmax(ejec_vel))[0]
            idx -= tot_steps
            ejec_vel = np.asarray(ejec_vel)
            esc_vel = ejec_vel[idx]
                        
            return esc_vel

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
    else:
        pop_arr = pop_data
        no_samples = 1
        process = True
    
    return no_samples, process

def folder_loop(iterf):
    """
    Function to organise arrays.
    """

    if iterf == 0:
        drange = 2
        integrator = ['Hermite', 'GRX']
    else:
        drange = 1
        integrator = ['GRX']

    return integrator, drange

def folder_data_loop(iterf, int):
    """
    Function to get right iteration for plotters.
    """

    if iterf == 0:
        sim_ = int
    else:
        sim_ = int + (1 + iterf)

    return sim_

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
    else:
        drange = 1
        filename = [natsort.natsorted(dataG)] 
        filenameC = [natsort.natsorted(chaosG)]
        integrator = ['GRX']

    return filename, filenameC, integrator, drange

def simulation_stats_checker(dist_dir, int_string):
    """
    Function to check the final outcomes of all the simulations
    """

    filename = glob.glob('/media/erwanh/Elements/'+dist_dir+'/data/'+int_string+'/simulation_stats/*')
    filename = natsort.natsorted(filename)
    SMBH_merger = 0 
    IMBH_merger = 0
    ejection = 0
    tot_sims = 0
    complete = 0

    for file_ in range(len(filename)):
        tot_sims += 1
        with open(filename[file_]) as f:
            line = f.readlines()
            line1 = line[-9][:-7]
            line2 = line[3][17:]
            data = line1.split()
            data2 = line2.split()
            data = [float(i) for i in data]
            if 4001000 in data:
                SMBH_merger += 1
            elif 2000 in data:
                IMBH_merger += 1
            elif '100000000.0' in data2:
                complete += 1
            else:
                ejection += 1

    with open('figures/'+int_string+'_'+dist_dir+'_summary.txt', 'w') as file:
        file.write('Simulation outcomes for '+str(int_string))
        file.write('\nTotal simulations:   '+str(tot_sims))
        file.write('\nSMBH merging events: '+str(SMBH_merger))
        file.write('\nIMBH merging events: '+str(IMBH_merger))
        file.write('\nEjection events:     '+str(ejection))
        file.write('\nCompleted sims:      '+str(complete))

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

print('...Gathering simulation outcomes...')
folders = ['rc_0.25_4e6', 'rc_0.25_4e6', 'rc_0.25_4e7', 'rc_0.50_4e6', 'rc_0.50_4e7']
integr = ['Hermite', 'GRX', 'GRX', 'GRX', 'GRX']
for fold_, integ_ in zip(folders, integr):
    simulation_stats_checker(fold_, integ_)