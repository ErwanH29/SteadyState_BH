
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings

from Data_Process.file_logistics import plotter_setup, stats_chaos_extractor

warnings.filterwarnings("ignore", category=OptimizeWarning) 

class stability_plotters(object):
    """
    Class to extract data and setup plots for simulations where IMBH particles are added over time
    """

    def index_extractor(self, fparti):
        """
        Function to find # of simulations for given configuration
        
        Inputs:
        fparti:  The data of the final configuration
        """
        
        pop, psamp = np.unique(fparti, return_counts=True)

        return pop, psamp
    
    def overall_steady_plotter(self):
        """
        Plot stability time
        """
            
        def log_fit(xval, slope, beta, log_c):
            return slope*(xval*np.log(log_c*xval))**beta

        def log_fit_classic(xval, slope, beta):
            return slope*(xval*np.log(0.11*xval))**beta

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plot_ini = plotter_setup()
        axlabel_size, tick_size = plot_ini.font_size()
        
        self.folders = ['rc_0.25_4e6', 'rc_0.25_4e5', 'rc_0.25_4e7']
        colours = ['red', 'blue', 'deepskyblue', 'blueviolet', 'dodgerblue', 'skyblue']
        dirH = '/data/Hermite/chaotic_simulation/*'
        dirG = '/data/GRX/chaotic_simulation/*'
        labelsD = [r'$M_{\rm{SMBH}} = 4\times10^{6}M_{\odot}$', 
                   r'$M_{\rm{SMBH}} = 4\times10^{5}M_{\odot}$',
                   r'$M_{\rm{SMBH}} = 4\times10^{7}M_{\odot}$']
        integ_label = ['Hermite', 'GRX']

        pop = [[ ], [ ], [ ], [ ]]
        psamp = [[ ], [ ], [ ], [ ]]
        fparti = [[ ], [ ], [ ], [ ]]
        stab_time = [[ ], [ ], [ ], [ ]]

        N_parti_avg = [[ ], [ ], [ ], [ ]]
        N_parti_med = [[ ], [ ], [ ], [ ]]
        N_parti_std = [[ ], [ ], [ ], [ ]]
        avg_deviate = [[ ], [ ], [ ], [ ]]
        full_simul = [[ ], [ ], [ ], [ ]]
        std_max = [[ ], [ ], [ ], [ ]]
        std_min = [[ ], [ ], [ ], [ ]]

        N_parti_med_Nsims = [[ ], [ ], [ ], [ ], [ ]]
        N_parti_std_Nsims = [[ ], [ ], [ ], [ ], [ ]]
        stdmax_Nsims = [[ ], [ ], [ ], [ ], [ ]]
        stdmin_Nsims = [[ ], [ ], [ ], [ ], [ ]]

        temp_data = [ ]

        iterf = 0
        for fold_ in self.folders:
            if iterf==0:
                drange = 2
                ffactor = 0
                direc = [dirH, dirG]
            else:
                drange = 1
                ffactor = 1
                direc = [dirG]

            for int_ in range(drange):
                fparti_data_val, stab_time_data_val = stats_chaos_extractor('/media/erwanh/Elements/'+fold_+direc[int_])
                fparti[int_+iterf+ffactor] = fparti_data_val
                stab_time[int_+iterf+ffactor] = stab_time_data_val
            iterf += 1

        for data_ in range(4):
            pop[data_], psamp[data_] = self.index_extractor(fparti[data_])
        
        cor_times = [[ ], [ ], [ ], [ ]]
        sim_times = [[ ], [ ], [ ], [ ]]
        pops_temp = [[ ], [ ], [ ], [ ]]

        dir = os.path.join('Data_Process/figures/sphere_of_influence.txt')
        with open(dir) as f:
            line = f.readlines()
            for iter in range(len(line)):
                if iter%3==0:
                    if line[iter][54:65]=='rc_0.25_4e6':
                        if line[iter][66:73]=='Hermite':
                            index = 0
                            popt = line[iter][117:121]
                            population = float(''.join(chr_ for chr_ in popt if chr_.isdigit()))
                            pops_temp[index].append(population)

                        else:
                            index = 1
                            population = float(''.join(chr_ for chr_ in popt if chr_.isdigit()))
                            pops_temp[index].append(population)

                    elif line[iter][54:65]=='rc_0.25_4e5':
                        index = 2
                        popt = line[iter][109:111]
                        population = float(''.join(chr_ for chr_ in popt if chr_.isdigit()))
                        pops_temp[index].append(population)
                    else:
                        index = 3
                        popt = line[iter][109:111]
                        population = float(''.join(chr_ for chr_ in popt if chr_.isdigit()))
                        pops_temp[index].append(population)
                
                if iter % 3==1:
                    real_time = line[iter][49:57]
                    cor_times[index].append(float(''.join(chr_ for chr_ in real_time if chr_.isdigit()))*1e-6)

                    sim_time = line[iter][70:78]
                    sim_times[index].append(float(''.join(chr_ for chr_ in sim_time if chr_.isdigit()))*1e-3)

        tol = 1e-10
        for int_ in range(np.shape(stab_time)[0]):
            for wrong_ in range(len(sim_times[int_])):
                print('Population: ', pops_temp[int_][wrong_])
                calib = True

                index = np.where(abs(stab_time[int_] - float('%.5g' % sim_times[int_][wrong_]))<=tol)[0]
                index_pop = np.where(abs(fparti[int_] - pops_temp[int_][wrong_])<=tol)[0]

                for idx_ in index:
                    if idx_ in index_pop and (calib):
                        print("Old time:   ", stab_time[int_][idx_])
                        stab_time[int_][idx_] = cor_times[int_][wrong_]
                        print("New time:   ", stab_time[int_][idx_])
                        calib = False
                        
        data_size = [10, 30, 40, 50]
        xshift = [-0.8, -0.4, 0, 0.4, 0.8]

        for int_ in range(4):
            if int_==0:
                no_data = 40
            elif int_==1:
                no_data = 60
            else:
                no_data = 30

            for pop_, samp_ in zip(pop[int_], psamp[int_]):
                N_parti = np.argwhere(fparti[int_]==pop_)
                time_arr = stab_time[int_][N_parti]
                if int_==1:
                    for rng_ in range(len(data_size)):
                        stab_times_temp = [ ]
                        rno = random.sample(range(0, 60), data_size[rng_])
                        stab_times_temp.append(stab_time[int_][N_parti][rno])
                            
                        stability_rng = np.median(stab_times_temp)
                        N_parti_med_Nsims[rng_].append(stability_rng)
                        N_parti_std_Nsims[rng_].append(np.std(stab_times_temp))
                        q1, q3 = np.percentile(stab_times_temp, [25, 75])
                        stdmax_Nsims[rng_].append(q3)
                        stdmin_Nsims[rng_].append(q1)

                stability = np.median(time_arr[:no_data])
                stability_avg = np.mean(time_arr[:no_data])
                if int_==0 and pop_==60 or pop_==70:
                    temp_data.append(stab_time[int_][N_parti])

                N_parti_avg[int_].append(stability_avg)
                N_parti_med[int_].append(stability)
                N_parti_std[int_].append(np.std(time_arr))
                q1, q3 = np.percentile(time_arr, [25, 75])
                std_max[int_].append(q3)
                std_min[int_].append(q1)

                idx = np.where(time_arr==100)[0]
                ratio = len(idx)/len(time_arr)
                full_simul[int_].append(ratio)
                avg_deviate[int_].append(abs(np.mean(time_arr-np.std(time_arr))))

            N_parti_avg[int_] = np.asarray(N_parti_avg[int_])
            N_parti_med[int_] = np.asarray(N_parti_med[int_])
            N_parti_std[int_] = np.asarray(N_parti_std[int_])
            std_max[int_] = np.asarray(std_max[int_])
            std_min[int_] = np.asarray(std_min[int_])
            avg_deviate[int_] = np.asarray(avg_deviate[int_])
            pop[int_] = np.array([float(i) for i in pop[int_]])
        
        N_parti_med_Nsims[-1] = [i for i in N_parti_med[1]]
        stdmax_Nsims[-1] = [i for i in std_max[1]]
        stdmin_Nsims[-1] = [i for i in std_min[1]]

        ##### GRX vs. Hermite #####
        curve = [[ ], [ ]]
        slope = [[ ], [ ]]
        beta = [[ ], [ ]]
        log_c = [[ ], [ ]]
        slope_err = [[ ], [ ]]
        beta_err = [[ ], [ ]]
        log_c_err = [[ ], [ ]]
        p0 = [np.asarray([40, -1, 0.11], dtype=float), np.asarray([40, -1, 0.11], dtype=float)]
        xtemp = [np.linspace(10, 100, 1000), np.linspace(10, 100, 1000)]

        fig, ax1 = plt.subplots()
        ax1.set_ylabel(r'$\log_{10} t_{\rm{loss}}$ [Myr]', fontsize=axlabel_size) 
        ax1.set_xlim(5,105)
        ax1.set_ylim(-0.7,2.4)
        for int_ in range(2):
            for j, xpos in enumerate(pop[int_]):
                N_parti_med[int_] = np.array([float(i) for i in N_parti_med[int_]])
                if j==0:
                    ax1.scatter(pop[int_], np.log10(N_parti_med[int_]), 
                                color=colours[int_], edgecolor='black', 
                                zorder=2, label=integ_label[int_])
                else:
                    ax1.scatter(pop[int_], np.log10(N_parti_med[int_]), 
                                color=colours[int_], edgecolor='black', 
                                zorder=2)
            ax1.scatter(pop[int_], np.log10(std_min[int_]), 
                        color=colours[int_], marker='_')
            ax1.scatter(pop[int_], np.log10(std_max[int_]), 
                        color=colours[int_], marker='_')
            for val_ in range(len(std_max[int_])):
                if std_max[int_][val_]>99:
                    ax1.arrow(pop[int_][val_], np.log10(std_max[int_][val_]),
                              dx=0, dy=0.2, head_width=1.2, head_length=0.05,
                              color=colours[int_])
            ax1.plot([pop[int_], pop[int_]], 
                     [np.log10(std_min[int_]), np.log10(std_max[int_])], 
                     color=colours[int_], zorder=1)

            pops = np.asarray(pop[int_], dtype=float)
            if int_==0:
                med_time = N_parti_med[int_][1:]
                sigma = std_max[int_][1:]-N_parti_med[int_][1:]
                pops = pops[1:]
            else:
                med_time = N_parti_med[int_]
                sigma = std_max[int_]-N_parti_med[int_]
            med_time = np.asarray(med_time, dtype=float)
            params, cv = scipy.optimize.curve_fit(log_fit, pops, med_time, maxfev=10000)
            slope[int_], beta[int_], log_c[int_] = params
            slope_err[int_], beta_err[int_], log_c_err[int_] = np.sqrt(np.diag(cv))
            curve[int_] = [(log_fit(i, slope[int_], beta[int_], log_c[int_])) for i in xtemp[int_]]
        
        params, cv = scipy.optimize.curve_fit(log_fit_classic, pop[0][1:], 
                                              N_parti_med[0][1:], 
                                              p0=[400, -1], 
                                              sigma=N_parti_std[0][1:],
                                              maxfev=70000)
        slope[0], beta[0] = params
        slope_err[0], beta_err[0] = np.sqrt(np.diag(cv))
        curve[0] = [(log_fit_classic(i, slope[0], beta[0])) for i in xtemp[0]]
        
        for k_ in range(2):
            print("Vals slope: ", slope[k_], "Beta: ", beta[k_], "Coulomb: ", log_c[k_])
            print("Err slope: ", slope_err[k_], "Beta: ", beta_err[k_], "Coulomb: ", log_c_err[k_])
        ax1.plot(xtemp[0], np.log10(curve[0]), zorder=1, color='red', ls='-.')
        ax1.plot(xtemp[1], np.log10(curve[1]), zorder=1, color='blue', ls='-.')
        ax1.legend(prop={'size': axlabel_size})
        plot_ini.tickers_pop(ax1, pop[0], 'Hermite')
        plt.savefig('Data_Process/figures/steady_time/stab_time_mean_HermGRX.pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        #STOP

        ##### Only GRX #####
        fig, ax1 = plt.subplots()
        ax1.set_ylabel(r'$\log_{10} t_{\rm{loss}}$ [Myr]', fontsize=axlabel_size)
        ax1.set_xlim(5,105)
        ax1.set_ylim(-1.8,2.4)
        for int_ in range(2):
            int_ += 2
            for j, xpos in enumerate(pop[int_]):
                pops = [i+1.1*xshift[int_-1] for i in pop[int_]]
                N_parti_med[int_] = np.array([float(i) for i in N_parti_med[int_]])
                if j==0:
                    ax1.scatter(pops, np.log10(N_parti_med[int_]), 
                                color=colours[int_], edgecolor='black', 
                                zorder=3, label=labelsD[int_-1])
                else:
                    ax1.scatter(pops, np.log10(N_parti_med[int_]), 
                                color=colours[int_], edgecolor='black', 
                                zorder=3)
            ax1.scatter(pops, np.log10(std_min[int_]), 
                        color=colours[int_], marker='_', 
                        zorder=3)
            ax1.scatter(pops, np.log10(std_max[int_]), 
                        color=colours[int_], marker='_', 
                        zorder=3)
            for val_ in range(len(std_max[int_])):
                if std_max[int_][val_]>99:
                    ax1.arrow(pops[val_], np.log10(std_max[int_][val_]),
                              dx=0, dy=0.2, head_width=0.5, head_length=0.05,
                              color=colours[int_])
                    
            ax1.plot([pops, pops], [np.log10(std_min[int_]), np.log10(std_max[int_])], 
                     color=colours[int_], zorder=2)

        xtemp = np.linspace(10, 40)
        GRX_fit = [(log_fit(i, slope[1], beta[0], log_c[1])) for i in xtemp]
        ax1.plot(xtemp, np.log10(GRX_fit), zorder=1, color='blue', ls='-.', label=labelsD[0])
        ax1.legend(prop={'size': axlabel_size})
        plot_ini.tickers_pop(ax1, pop[1], 'GRX')
        plt.savefig('Data_Process/figures/steady_time/stab_time_mean_GRX.pdf', dpi = 300, bbox_inches='tight')
        plt.clf()

        ##### GRX vs. Nsims #####
        labels = [r'$N_{\rm{sims}} = 10$', 
                  r'$N_{\rm{sims}} = 30$', 
                  r'$N_{\rm{sims}} = 40$', 
                  r'$N_{\rm{sims}} = 50$', 
                  r'$N_{\rm{sims}} = 60$']
        
        fig, ax1 = plt.subplots()
        ax1.set_ylabel(r'$\log_{10} t_{\rm{loss}}$ [Myr]', fontsize = axlabel_size) 
        ax1.set_xlim(5,45)
        ctemp = ['deepskyblue', 'blueviolet', 'dodgerblue', 'blue', 'purple']
        for rng_ in range(len(data_size)+1):
            for j, xpos in enumerate(pop[1]):
                pops = [i+xshift[rng_] for i in pop[1]]
                N_parti_med_Nsims[rng_] = np.array([float(i) for i in N_parti_med_Nsims[rng_]])
                stdmin_Nsims[rng_] = np.array([float(i) for i in stdmin_Nsims[rng_]])
                stdmax_Nsims[rng_] = np.array([float(i) for i in stdmax_Nsims[rng_]])
                if j==0:
                    ax1.scatter(pops, np.log10(N_parti_med_Nsims[rng_]), 
                                color = ctemp[rng_], edgecolor = 'black', zorder = 2, label = labels[rng_])
                else:
                    ax1.scatter(pops, np.log10(N_parti_med_Nsims[rng_]), 
                                color = ctemp[rng_], edgecolor = 'black', zorder = 2)
            ax1.scatter(pops, np.log10(stdmin_Nsims[rng_]), color = ctemp[rng_], marker = '_')
            ax1.scatter(pops, np.log10(stdmax_Nsims[rng_]), color = ctemp[rng_], marker = '_')
            ax1.plot([pops, pops], [np.log10(stdmin_Nsims[rng_]), np.log10(stdmax_Nsims[rng_])], 
                      color = ctemp[rng_], zorder = 1)
        ax1.legend(prop={'size': axlabel_size})
        plot_ini.tickers_pop(ax1, pop[1], 'GRX')
        plt.savefig('Data_Process/figures/steady_time/stab_time_mean_GRX_Nsims.pdf', dpi = 300, bbox_inches='tight')
        plt.clf()

        frange = 0
        for data_ in range(4):
            if data_==0:
                integ = 'Hermite'
            else:
                integ = 'GRX'
            if data_ > 1:
                frange += 1
            with open('Data_Process/figures/steady_time/Sim_summary_'+self.folders[frange]+'_'+str(integ)+'.txt', 'w') as file:
                file.write('For '+str(integ)+', # of full simulations per population:       '+str(pop[data_].flatten()))
                file.write('\n                                                     '+str(full_simul[data_]))
                file.write('\nNumber of samples:                                   '+str(psamp[data_].flatten()))
                if data_<2:
                    file.write('\nThe slope of the curve goes as:                      '+str(slope[data_]))
                    file.write('\nThe err. of slope goes as:                           '+str(slope_err[data_]))
                    file.write('\nThe power-law of the lnN goes as:                    '+str(beta[data_]))
                    file.write('\nThe err. of the power-law goes as:                   '+str(beta_err[data_]))
                    file.write('\nThe logarithmic factor goes as:                      '+str(log_c[data_]))
                    file.write('\nThe err. of the logarithmic factor goes as:          '+str(log_c_err[data_]))
                file.write('\nThe final raw data:                                  '+str(pop[data_][pop[data_] > 5].flatten()))
                file.write('\nSimulated time [Myr]                                 '+str(N_parti_med[data_][pop[data_] > 5].flatten()))
                file.write('\nAvg. simulated time [Myr]                            '+str(N_parti_avg[data_][pop[data_] > 5].flatten())+'\n\n')