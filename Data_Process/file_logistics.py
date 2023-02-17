from amuse.lab import *
import numpy as np
import glob
import matplotlib.ticker as mtick
import natsort
import os
import pickle as pkl

class plotter_setup(object):
    def tickers(self, ax, plot_type):
        """
        Function to setup axis
        """

        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        if plot_type == 'plot':
            ax.tick_params(axis="y", which = 'both', direction="in")
            ax.tick_params(axis="x", which = 'both', direction="in")

        return ax

    def tickers_pop(self, ax, pop, int_str):
        """
        Function to setup axis for population plots
        """

        ax.set_xlabel(r'IMBH Population [$N$]')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.tick_params(axis="y", which = 'both', direction="in")
        ax.tick_params(axis="x", which = 'both', direction="in")    
        
        if int_str == 'Hermite':
            xints = [i for i in range(1+int(max(pop))) if i % 10 == 0 and i > 5]
            ax.set_xlim(5, 105)
        else:
            xints = [i for i in range(1+int(max(pop))) if i % 5 == 0 and i > 5 and i < 45]
            ax.set_xlim(5, 45)
 
        ax.set_xticks(xints)

        return ax
def bulk_stat_extractor(file_string, rewrite):
    """
    Function which extracts all files in a given dir.
    
    Inputs:
    file_string: The directory wished to extract files from
    rewrite:     (Y|N) string to dictate whether to compress the file based on needed data
    """

    filename = glob.glob(file_string)
    filename = natsort.natsorted(filename)
    data = [ ]

    if rewrite == 'N':
        for file_ in range(len(filename)):
            with open(filename[file_], 'rb') as input_file:
                print('Reading file: ', input_file)
                data.append(pkl.load(input_file))
                
    else:
        for file_ in range(len(filename)):
            with open(filename[file_], 'rb') as input_file:
                print('Reading file: ', input_file)
                rewrite_file = pkl.load(input_file)
            if np.shape(rewrite_file)[1] > 50 and np.shape(rewrite_file)[1] < 50:
                data_pts = round((np.shape(rewrite_file)[1])/25)
            elif np.shape(rewrite_file)[1] > 500 and np.shape(rewrite_file)[1] < 5000:
                data_pts = round((np.shape(rewrite_file)[1])/250)
            elif np.shape(rewrite_file)[1] > 5000:
                data_pts = round((np.shape(rewrite_file)[1])/2500)
            else:
                data_pts = np.shape(rewrite_file)[1]
            rewrite_file = rewrite_file.drop(rewrite_file.iloc[:, data_pts:-1*data_pts], axis = 1) 
            data.append(rewrite_file)

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
            if 2000 in data:
                IMBH_merger += 1
            if 4001000 not in data and 2000 not in data and '100000000.0' not in data2:
                ejection += 1
            if '100000000.0' in data2:
                complete += 1

    with open('figures/'+int_string+'_'+dist_dir+'_summary.txt', 'w') as file:
        file.write('\nSimulation outcomes for '+str(int_string))
        file.write('\nTotal simulations:   '+str(tot_sims))
        file.write('\nSMBH merging events: '+str(SMBH_merger))
        file.write('\nIMBH merging events: '+str(IMBH_merger))
        file.write('\nEjection events:     '+str(ejection))
        file.write('\nCompleted sims:      '+str(complete))
        file.write('\n========================================')

def stats_chaos_extractor(dir):
    steadytime_data = bulk_stat_extractor(dir, 'N')
    no_Data = len(steadytime_data)

    fin_parti_data = np.empty(no_Data)
    stab_time_data = np.empty(no_Data)

    for i in range(no_Data):
        sim_data = steadytime_data[i]
        if isinstance(sim_data.iloc[0][9], float):
            fin_parti_data[i] = sim_data.iloc[0][6] + sim_data.iloc[0][10]
            stab_time_data[i] = sim_data.iloc[0][13].value_in(units.Myr)
        
    return fin_parti_data, stab_time_data

simulation_stats_checker('rc_0.25', 'GRX')
simulation_stats_checker('rc_0.25', 'Hermite')
simulation_stats_checker('rc_0.5', 'GRX')
simulation_stats_checker('rc_0.5', 'Hermite')