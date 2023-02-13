from amuse.lab import *
from file_logistics import *
from scipy.interpolate import make_interp_spline
from scipy.optimize import OptimizeWarning
from scipy.stats import iqr
from spatial_plotters import *

import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

warnings.filterwarnings("ignore", category=OptimizeWarning) 

class stability_plotters(object):
    """
    Class to extract data and setup plots for simulations where IMBH particles are added over time
    """
        
    def extract(self, dirC):
        """
        Function to extract data from simulations who end in ejection.
        Data is from the no_addition/chaotic_simulation directory.
        """

        self.fparti_data, self.stab_time_data, self.init_dist_data, self.imass_data = stats_chaos_extractor(dirC)  

        return self.fparti_data, self.stab_time_data, self.init_dist_data, self.imass_data

    def index_extractor(self, init_mass, final_parti_data, stab_time_data, indices, vals):
        """
        Function to extract indices of depending on the wanted plot (function of mass/distance)
        
        Inputs:
        init_mass:         The set of initial masses
        final_parti_data:  The data of the final configuration
        stab_time_data:    The stability time
        indices:           The indices corresponding to the fashion you wish to separate the plots
        vals:              The variable which will compose different coloured lines in the plot
        """
        
        mass_array = init_mass[indices]
        final_part = final_parti_data[indices]
        stab_time = stab_time_data[indices]

        filtered_m = np.where((mass_array == vals).all(1))[0]
        filt_finparti = final_part[filtered_m]
        filt_stabtime = stab_time[filtered_m]
        pop, psamp = np.unique(filt_finparti, return_counts = True)

        return pop, psamp, filt_finparti, filt_stabtime
    
    def overall_steady_plotter(self):
        """
        Function to plot stability time for constant distances
        """
            
        def log_fit(xval, slope, beta, log_c):
            return slope *(xval**0.5*np.log(log_c*xval))**beta
            
        plot_ini = plotter_setup()
        dirH = 'data/Hermite/chaotic_simulation/*'
        dirG = 'data/GRX/chaotic_simulation/*'

        folders = ['rc_0.25', 'rc_0.5']
        direc = [dirH, dirG]

        colors = ['red', 'blue', 'lightcoral', 'cornflowerblue']
        integrator = ['Hermite', 'GRX']
        distance = [r'$\langle r_h \rangle \approx 0.4$ pc', r'$\langle r_h\rangle = 0.5$ pc']

        fparti_data = [[ ], [ ], [ ], [ ]]
        stab_time_data = [[ ], [ ], [ ], [ ]]
        init_dist_data = [[ ], [ ], [ ], [ ]]
        imass_data = [[ ], [ ], [ ], [ ]]
        idist_idx_chaos = [[ ], [ ], [ ], [ ]]

        pop = [[ ], [ ], [ ], [ ]]
        psamp = [[ ], [ ], [ ], [ ]]
        fparti = [[ ], [ ], [ ], [ ]]
        stab_time = [[ ], [ ], [ ], [ ]]

        N_parti_avg = [[ ], [ ], [ ], [ ]]
        N_parti_std = [[ ], [ ], [ ], [ ]]
        avg_deviate = [[ ], [ ], [ ], [ ]]
        full_simul = [[ ], [ ], [ ], [ ]]
        std_max = [[ ], [ ], [ ], [ ]]
        std_min = [[ ], [ ], [ ], [ ]]

        temp_data = [ ]

        iter = 0
        for fold_, integ_ in itertools.product(folders, direc):
            fparti_data_val, stab_time_data_val, init_dist_data_val, imass_data_val = self.extract('/media/erwanh/Elements/'+fold_+'/'+integ_)
            fparti_data[iter] = fparti_data_val
            stab_time_data[iter] = stab_time_data_val
            init_dist_data[iter] = init_dist_data_val
            imass_data[iter] = imass_data_val
            idist_idx_chaos[iter] = np.where((np.asarray(init_dist_data[iter]) == init_dist_data[iter]))
            iter += 1

        for data_ in range(4):
            pop[data_], psamp[data_], fparti[data_], stab_time[data_] = self.index_extractor(imass_data[data_], fparti_data[data_], 
                                                                                             stab_time_data[data_], idist_idx_chaos[data_], 
                                                                                             imass_data[data_])
                                                                                             
        for int_ in range(4):
            for pop_, samp_ in zip(pop[int_], psamp[int_]):
                N_parti = np.argwhere(fparti[int_] == pop_)
                time_arr = stab_time[int_][N_parti]
                stability = np.mean(time_arr)

                if int_ == 0 and pop_ == 60 or pop_ == 70:
                    temp_data.append(stab_time[int_][N_parti])

                N_parti_avg[int_].append(stability)
                N_parti_std[int_].append(np.std(time_arr))
                q1, q3 = np.percentile(time_arr, [25, 75])
                std_max[int_].append(q3)
                std_min[int_].append(q1)

                idx = np.where(time_arr == 100)[0]
                ratio = len(idx)/len(time_arr)
                full_simul[int_].append(ratio)
                avg_deviate[int_].append(abs(np.mean(time_arr-np.std(time_arr))))

            N_parti_avg[int_] = np.asarray(N_parti_avg[int_])
            N_parti_std[int_] = np.asarray(N_parti_std[int_])
            std_max[int_] = np.asarray(std_max[int_])
            std_min[int_] = np.asarray(std_min[int_])
            avg_deviate[int_] = np.asarray(avg_deviate[int_])
            pop[int_] = np.array([float(i) for i in pop[int_]])
            
        hist_tails = np.concatenate((temp_data[0], temp_data[1]))

        fig, ax = plt.subplots()
        ax.set_title(r'$\langle t_{\rm{surv}} \rangle$ for $N = 60, N = 70$')
        ax.set_xlabel(r'Time [Myr]')
        ax.set_ylabel(r'Counts')
        ax.text(5.1, 17, r'$\langle t_{\rm{surv}}\rangle$', rotation = 270)
        ax.axvline(4.87, color = 'black', linestyle = ':')
        n1, bins, patches = ax.hist(hist_tails, 30, histtype='step', color = 'black')
        n1, bins, patches = ax.hist(hist_tails, 30, alpha = 0.3, color = 'black')
        plot_ini.tickers(ax, 'plot')
        plt.savefig('figures/steady_time/stab_time_hist_6070.pdf', dpi = 300, bbox_inches='tight')
        plt.clf()

        ##### FIGURE WITH ALL DATA IN ONE PLOT
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel(r'$\log_{10} t_{\rm{surv}}$ [Myr]') 
        ax1.set_xlim(5,105)
        ax1.set_ylim(-1.5, 2.3)
        ax1.axhline(np.log10(259))
        ax1.axhline(np.log10(882))
        for int_ in range(4):
            colour_idx = round(int_/3)
            for j, xpos in enumerate(pop[int_][pop[int_] % 10 == 0]):
                N_parti_avg[int_] = np.array([float(i) for i in N_parti_avg[int_]])
                if j == 0 and int_ <= 1:
                    ax1.scatter(pop[int_][pop[int_] % 10 == 0], np.log10(N_parti_avg[int_][pop[int_] % 10 == 0]), 
                                color = colors[int_], edgecolor = 'black', zorder = 2, label = integrator[int_%2])
                if j == 0 and int_ == 0 or j == 0 and int_ == 2:
                    ax1.scatter(pop[int_][pop[int_] % 10 == 0], np.log10(N_parti_avg[int_][pop[int_] % 10 == 0]), 
                                color = colors[int_], edgecolor = 'black', zorder = 2, label = distance[colour_idx])
                else:
                    ax1.scatter(pop[int_][pop[int_] % 10 == 0], np.log10(N_parti_avg[int_][pop[int_] % 10 == 0]), 
                                color = colors[int_], edgecolor = 'black', zorder = 2)
            ax1.scatter(pop[int_][pop[int_] % 10 == 0], np.log10(std_min[int_][pop[int_] % 10 == 0]), color = colors[int_], marker = '_')
            ax1.scatter(pop[int_][pop[int_] % 10 == 0], np.log10(std_max[int_][pop[int_] % 10 == 0]), color = colors[int_], marker = '_')
            ax1.plot([pop[int_][pop[int_] % 10 == 0], pop[int_][pop[int_] % 10 == 0]], [np.log10(std_min[int_][pop[int_] % 10 == 0]), 
                      np.log10(std_max[int_][pop[int_] % 10 == 0])], color = colors[int_], zorder = 1)
        
        p0 = (100, -5, 20)
        xtemp = np.linspace(10, 100, 1000)
        slope = [[ ], [ ]]
        beta = [[ ], [ ]]
        log_c = [[ ], [ ]]
        iter = 0
        for data_ in range(2):
            data_ = round(data_/3)
            params, cv = scipy.optimize.curve_fit(log_fit, pop[data_][pop[data_] % 10 == 0], (N_parti_avg[data_][pop[data_] % 10 == 0]), p0, maxfev = 10000, method = 'trf')
            slope[iter], beta[iter], log_c[iter] = params
            curve = [(log_fit(i, slope[iter], beta[iter], log_c[iter])) for i in xtemp]
            ax1.plot(xtemp, np.log10(curve), zorder = 1, color = 'black', ls = '-.')
            iter += 1
        ax1.legend()
        plot_ini.tickers_pop(ax1, pop[0], 'Hermite')
        plt.savefig('figures/steady_time/stab_time_mean.pdf', dpi = 300, bbox_inches='tight')
        plt.clf()


        ##### FIGURE SEPERATED BY DISTANCES
        for dist_ in range(2):
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(111)
            ax1.set_ylabel(r'$\log_{10} t_{\rm{surv}}$ [Myr]') 
            ax1.set_xlim(5,105)
            ax1.set_ylim(-1.5, 2.3)
            ax1.axhline(np.log10(259))
            ax1.axhline(np.log10(882))

            for int_ in range(2):
                colour_idx = round(int_/3)
                data_ = dist_*2+int_
                for j, xpos in enumerate(pop[data_][pop[data_] % 10 == 0]):
                    N_parti_avg[data_] = np.array([float(i) for i in N_parti_avg[data_]])
                    if j == 0 and int_ <= 1:
                        ax1.scatter(pop[data_][pop[data_] % 10 == 0], np.log10(N_parti_avg[data_][pop[data_] % 10 == 0]), 
                                    color = colors[int_], edgecolor = 'black', zorder = 2, label = integrator[int_%2])
                    else:
                        ax1.scatter(pop[data_][pop[data_] % 10 == 0], np.log10(N_parti_avg[data_][pop[data_] % 10 == 0]), 
                                    color = colors[int_], edgecolor = 'black', zorder = 2)
                ax1.scatter(pop[data_][pop[data_] % 10 == 0], np.log10(std_min[data_][pop[data_] % 10 == 0]), color = colors[int_], marker = '_')
                ax1.scatter(pop[data_][pop[data_] % 10 == 0], np.log10(std_max[data_][pop[data_] % 10 == 0]), color = colors[int_], marker = '_')
                ax1.plot([pop[data_][pop[data_] % 10 == 0], pop[data_][pop[data_] % 10 == 0]], [np.log10(std_min[data_][pop[data_] % 10 == 0]), 
                          np.log10(std_max[data_][pop[data_] % 10 == 0])], color = colors[int_], zorder = 1)
        
            params, cv = scipy.optimize.curve_fit(log_fit, pop[data_][pop[data_] % 10 == 0], (N_parti_avg[data_][pop[data_] % 10 == 0]), p0, maxfev = 10000, method = 'trf')
            slope[dist_], beta[dist_], log_c[dist_] = params
            curve = [(log_fit(i, slope[dist_], beta[dist_], log_c[dist_])) for i in xtemp]
            ax1.plot(xtemp, np.log10(curve), zorder = 1, color = 'black', ls = '-.')
            ax1.legend()
            plot_ini.tickers_pop(ax1, pop[0], 'Hermite')
            plt.savefig('figures/steady_time/stab_time_mean_'+folders[dist_]+'.pdf', dpi = 300, bbox_inches='tight')
            plt.clf()
    
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel(r'$\log_{10} t_{\rm{surv}}$ [Myr]') 
        ax1.set_xlim(8,55)
        ax1.set_ylim(-1.3, 2.3)
        xints = [i for i in range(8, 1+int(max(pop[1]))) if i % 5 == 0]
        ax1.set_xlabel(r'IMBH Population [$N$]')
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax1.tick_params(axis="y", which = 'both', direction="in")
        ax1.tick_params(axis="x", which = 'both', direction="in")     
        ax1.set_xticks(xints)
        
        N_parti_avg[int_] = np.array([float(i) for i in N_parti_avg[int_]])
        xtemp = np.linspace(10, 40, 1000)
        curve = [(log_fit(i, slope[0], beta[0], log_c[0])) for i in xtemp]
        
        colour_GRX = ['blue', 'cornflowerblue']
        for data_ in range(4):
            if data_%2 != 0:
                for j, xpos in enumerate(pop[data_][pop[data_] > 5]):
                    if j == 0:
                        ax1.scatter(pop[data_][(pop[data_] <= 40) & (pop[data_] > 5)], np.log10(N_parti_avg[data_][(pop[data_] <= 40) & (pop[data_] > 5)]), 
                                    color = colour_GRX[int(0.5*(data_-1))], edgecolor = 'black', zorder = 2, label = distance[int(0.5*(data_-1))])
                    else:
                        ax1.scatter(pop[data_][(pop[data_] <= 40) & (pop[data_] > 5)], np.log10(N_parti_avg[data_][(pop[data_] <= 40) & (pop[data_] > 5)]), 
                                    color = colour_GRX[int(0.5*(data_-1))], edgecolor = 'black', zorder = 2)
                ax1.scatter(pop[data_][(pop[data_] <= 40) & (pop[data_] > 5)], np.log10(std_min[data_][(pop[data_] <= 40) & (pop[data_] > 5)]), 
                            color = colour_GRX[int(0.5*(data_-1))], marker = '_')
                ax1.scatter(pop[data_][(pop[data_] <= 40) & (pop[data_] > 5)], np.log10(std_max[data_][(pop[data_] <= 40) & (pop[data_] > 5)]), 
                            color = colour_GRX[int(0.5*(data_-1))], marker = '_')
                ax1.plot([pop[data_][(pop[data_] <= 40) & (pop[data_] > 5)], pop[data_][(pop[data_] <= 40) & (pop[data_] > 5)]], 
                         [np.log10(std_min[data_][(pop[data_] <= 40) & (pop[data_] > 5)]), np.log10(std_max[data_][(pop[data_] <= 40) & (pop[data_] > 5)])], 
                         color = colour_GRX[int(0.5*(data_-1))], zorder = 1)
                ax1.plot(xtemp, np.log10(curve), zorder = 1, color = 'black', ls = '-.')
        ax1.legend()
        plot_ini.tickers_pop(ax1, pop[0], 'GRX')
        plt.savefig('figures/steady_time/stab_time_mean_GRX.pdf', dpi = 300, bbox_inches='tight')
        
        iter = 0
        for data_ in range(2):
            fig = plt.figure(figsize=(15, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax = [ax1, ax2]
            for int_ in range(2):
                x_arr = np.linspace(10, max(pop[iter]), 100)
                smooth_curve = make_interp_spline(pop[iter], avg_deviate[iter])
                ax[int_].set_ylabel(r'$\langle (t_{\rm{surv}} - \sigma_{\rm{surv}}) \rangle$')
                ax[int_].set_title(integrator[int_])
                ax[int_].plot(x_arr, smooth_curve(x_arr), color = colors[int_], zorder = 1)
                ax[int_].scatter(pop[iter][pop[iter] > 5], avg_deviate[iter][pop[iter] > 5], color = colors[int_], edgecolors='black', zorder = 2)   
                ax[int_].set_ylim(0,1.05*max(avg_deviate[iter][pop[iter] > 5]))
                iter += 1
            plot_ini.tickers_pop(ax1, pop[0], 'Hermite')
            plot_ini.tickers_pop(ax2, pop[1], 'GRX')
            ax1.set_xlim(6,105)
            ax2.set_xlim(6,45)
            plt.savefig('figures/steady_time/stab_time_residuals'+folders[data_]+'.pdf', dpi = 300, bbox_inches='tight')
            plt.clf()

        for data_ in range(2):
            with open('figures/steady_time/Sim_summary_'+folders[data_]+'.txt', 'w') as file:
                for int_ in range(2):
                    int_ = int_ + data_
                    file.write('\n\nFor '+str(integrator[int_-data_])+', # of full simulations per population:  '+str(pop[int_].flatten()))
                    file.write('\n                                              '+str(full_simul[int_]))
                    file.write('\nThe slope of the curve goes as:               '+str(slope[data_]))
                    file.write('\nThe power-law of the lnN goes as:             '+str(beta[data_]))
                    file.write('\nThe logarithmic factor goes as:               '+str(log_c[data_]))
                    file.write('\nThe final raw data:                           '+str(pop[int_][pop[int_] > 5].flatten()))
                    file.write('\nSimulated time [Myr]                          '+str(N_parti_avg[int_][pop[int_] > 5].flatten()))