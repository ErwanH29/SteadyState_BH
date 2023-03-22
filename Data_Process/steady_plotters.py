from amuse.lab import *
from file_logistics import *
from scipy.interpolate import make_interp_spline
from scipy.optimize import OptimizeWarning
from scipy.stats import iqr
from spatial_plotters import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

warnings.filterwarnings("ignore", category=OptimizeWarning) 

class stability_plotters(object):
    """
    Class to extract data and setup plots for simulations where IMBH particles are added over time
    """

    def index_extractor(self, final_parti_data):
        """
        Function to find # of simulations for given configuration
        
        Inputs:
        final_parti_data:  The data of the final configuration
        """
        
        pop, psamp = np.unique(final_parti_data, return_counts = True)

        return pop, psamp
    
    def overall_steady_plotter(self):
        """
        Function to plot stability time for constant distances
        """
            
        def log_fit(xval, slope, beta, log_c):
            return slope*(xval**0.5*np.log(log_c*xval))**beta
            
        plot_ini = plotter_setup()
        
        folders = ['rc_0.25_4e6', 'rc_0.25_4e7', 'rc_0.50_4e6', 'rc_0.50_4e7']
        colors = ['red', 'blue', 'deepskyblue', 'skyblue', 'slateblue', 'turquoise']
        dirH = '/data/Hermite/chaotic_simulation/*'
        dirG = '/data/GRX/chaotic_simulation/*'
        labelsD = [r'$r_c = 0.25$ pc, $M_{\rm{SMBH}} = 4\times10^{6}M_{\odot}$', 
                   r'$r_c = 0.25$ pc, $M_{\rm{SMBH}} = 4\times10^{7}M_{\odot}$',
                   r'$r_c = 0.50$ pc, $M_{\rm{SMBH}} = 4\times10^{6}M_{\odot}$',
                   r'$r_c = 0.50$ pc, $M_{\rm{SMBH}} = 4\times10^{7}M_{\odot}$']
        integ_label = ['Hermite', 'GRX']

        pop = [[ ], [ ], [ ], [ ], [ ]]
        psamp = [[ ], [ ], [ ], [ ], [ ]]
        fparti = [[ ], [ ], [ ], [ ], [ ]]
        stab_time = [[ ], [ ], [ ], [ ], [ ]]

        N_parti_avg = [[ ], [ ], [ ], [ ], [ ]]
        N_parti_std = [[ ], [ ], [ ], [ ], [ ]]
        avg_deviate = [[ ], [ ], [ ], [ ], [ ]]
        full_simul = [[ ], [ ], [ ], [ ], [ ]]
        std_max = [[ ], [ ], [ ], [ ], [ ]]
        std_min = [[ ], [ ], [ ], [ ], [ ]]

        N_parti_avg_Nsims = [[ ], [ ], [ ], [ ]]
        N_parti_std_Nsims = [[ ], [ ], [ ], [ ]]
        stdmax_Nsims = [[ ], [ ], [ ], [ ]]
        stdmin_Nsims = [[ ], [ ], [ ], [ ]]

        temp_data = [ ]

        iterf = 0
        for fold_ in folders:
            if iterf == 0:
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

        for data_ in range(5):
            pop[data_], psamp[data_] = self.index_extractor(fparti[data_])
        
        rng = np.random.default_rng()
        data_size = [30, 40, 50, 55]
        xshift = [-0.75, -0.25, 0.25, 0.75]

        for int_ in range(5):
            for pop_, samp_ in zip(pop[int_], psamp[int_]):
                N_parti = np.argwhere(fparti[int_] == pop_)
                time_arr = stab_time[int_][N_parti]
                
                if int_ == 1:
                    for rng_ in range(len(data_size)):
                        stab_times_temp = [ ]
                        for iter_ in range(100):
                            rno = rng.integers(low = 0, high = data_size[rng_], size = data_size[rng_])
                            stab_times_temp.append(stab_time[int_][N_parti][rno])
                            
                        stability_rng = np.median(stab_times_temp)
                        N_parti_avg_Nsims[rng_].append(stability_rng)
                        N_parti_std_Nsims[rng_].append(np.std(stab_times_temp))
                        q1, q3 = np.percentile(stab_times_temp, [25, 75])
                        stdmax_Nsims[rng_].append(q3)
                        stdmin_Nsims[rng_].append(q1)
                stability = np.median(time_arr[:40])

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
        ax.set_title(r'$\langle t_{\rm{dis}} \rangle$ for $N = 60, N = 70$', fontsize = plot_ini.tilabel_size)
        ax.set_xlabel(r'Time [Myr]', fontsize = plot_ini.axlabel_size)
        ax.set_ylabel(r'Counts', fontsize = plot_ini.axlabel_size)
        ax.text((N_parti_avg[0][5]+N_parti_avg[0][6])/2 + 0.3, 17, r'$\langle t_{\rm{dis}}\rangle$', rotation = 270)
        ax.axvline((N_parti_avg[0][5]+N_parti_avg[0][6])/2, color = 'black', linestyle = ':')
        n1, bins, patches = ax.hist(hist_tails, 30, histtype='step', color = 'black')
        n1, bins, patches = ax.hist(hist_tails, 30, alpha = 0.3, color = 'black')
        plot_ini.tickers(ax, 'plot')
        plt.savefig('figures/steady_time/stab_time_hist_6070.pdf', dpi = 300, bbox_inches='tight')
        plt.clf()

        ##### GRX vs. Hermite #####
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel(r'$\log_{10} \langle t_{\rm{dis}}\rangle$ [Myr]', fontsize = plot_ini.axlabel_size) 
        ax1.set_xlim(5,105)
        for int_ in range(2):
            for j, xpos in enumerate(pop[int_][pop[int_] % 10 == 0]):
                N_parti_avg[int_] = np.array([float(i) for i in N_parti_avg[int_]])
                if j == 0:
                    ax1.scatter(pop[int_][pop[int_] % 10 == 0], np.log10(N_parti_avg[int_][pop[int_] % 10 == 0]), 
                                color = colors[int_], edgecolor = 'black', zorder = 2, label = integ_label[int_])
                else:
                    ax1.scatter(pop[int_][pop[int_] % 10 == 0], np.log10(N_parti_avg[int_][pop[int_] % 10 == 0]), 
                                color = colors[int_], edgecolor = 'black', zorder = 2)
            ax1.scatter(pop[int_][pop[int_] % 10 == 0], np.log10(std_min[int_][pop[int_] % 10 == 0]), color = colors[int_], marker = '_')
            ax1.scatter(pop[int_][pop[int_] % 10 == 0], np.log10(std_max[int_][pop[int_] % 10 == 0]), color = colors[int_], marker = '_')
            ax1.plot([pop[int_][pop[int_] % 10 == 0], pop[int_][pop[int_] % 10 == 0]], [np.log10(std_min[int_][pop[int_] % 10 == 0]), 
                      np.log10(std_max[int_][pop[int_] % 10 == 0])], color = colors[int_], zorder = 1)
        
        p0 = (100, -5, 20)
        xtemp = np.linspace(10, 100, 1000)
        slope = [[ ], [ ], [ ], [ ]]
        beta = [[ ], [ ], [ ], [ ]]
        log_c = [[ ], [ ], [ ], [ ]]
        curve = [[ ], [ ], [ ], [ ]]
        params, cv = scipy.optimize.curve_fit(log_fit, pop[1], (N_parti_avg[1]), p0, maxfev = 10000, method = 'trf')
        slope[0], beta[0], log_c[0] = params
        curve[0] = [(log_fit(i, slope[0], beta[0], log_c[0])) for i in xtemp]
        ax1.plot(xtemp, np.log10(curve[0]), zorder = 1, color = 'black', ls = '-.')
        ax1.legend()
        plot_ini.tickers_pop(ax1, pop[0], 'Hermite')
        plt.savefig('figures/steady_time/stab_time_mean_HermGRX.pdf', dpi = 300, bbox_inches='tight')
        plt.clf()

        ##### Only GRX #####
        ##### Make sure to fix y lims to be the same as Hermite vs. GRX plot
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel(r'$\log_{10} \langle t_{\rm{dis}}\rangle$ [Myr]', fontsize = plot_ini.axlabel_size)
        ax1.set_xlim(5,105)
        for int_ in range(3):
            int_ += 2
            for j, xpos in enumerate(pop[int_]):
                pops = [i+1.1*xshift[int_-2] for i in pop[int_]]
                N_parti_avg[int_] = np.array([float(i) for i in N_parti_avg[int_]])
                if j == 0:
                    ax1.scatter(pops, np.log10(N_parti_avg[int_]), color = colors[int_], 
                                edgecolor = 'black', zorder = 3, label = labelsD[int_-1])
                else:
                    ax1.scatter(pops, np.log10(N_parti_avg[int_]), color = colors[int_], 
                                edgecolor = 'black', zorder = 3)
            ax1.scatter(pops, np.log10(std_min[int_]), color = colors[int_], marker = '_', zorder = 3)
            ax1.scatter(pops, np.log10(std_max[int_]), color = colors[int_], marker = '_', zorder = 3)
            ax1.plot([pops, pops], [np.log10(std_min[int_]), np.log10(std_max[int_])], color = colors[int_], zorder = 2)
            params, cv = scipy.optimize.curve_fit(log_fit, pop[int_], (N_parti_avg[int_]), p0, maxfev = 10000, method = 'trf')
            slope[int_-1], beta[int_-1], log_c[int_-1] = params

        xtemp = np.linspace(10, 40)
        curve[0] = [(log_fit(i, slope[0], beta[0], log_c[0])) for i in xtemp]
        ax1.plot(xtemp, np.log10(curve[0]), zorder = 1, color = 'black', ls = '-.', label = labelsD[0])
        ax1.legend()
        plot_ini.tickers_pop(ax1, pop[1], 'GRX')
        plt.savefig('figures/steady_time/stab_time_mean_GRX.pdf', dpi = 300, bbox_inches='tight')
        plt.clf()

        ##### GRX vs. Nsims #####
        labels = [r'$N_{\rm{sims}} = 30$', r'$N_{\rm{sims}} = 40$', r'$N_{\rm{sims}} = 50$', r'$N_{\rm{sims}} = 60$']
        labels_diff = [r'$N_{{\rm{sims}}, i} = 40, N_{{\rm{sims}}, j} = 20$', 
                       r'$N_{{\rm{sims}}, i} = 40, N_{{\rm{sims}}, j} = 30$',
                       r'$N_{{\rm{sims}}, i} = 40, N_{{\rm{sims}}, j} = 40$',
                       r'$N_{{\rm{sims}}, i} = 40, N_{{\rm{sims}}, j} = 50$']
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.set_title(r'$N_{\rm{sims}}$ vs. $\langle t_{\rm{dis}}\rangle$', fontsize = plot_ini.tilabel_size)
        ax1.set_ylabel(r'$\log_{10} \langle t_{\rm{dis}} \rangle$ [Myr]', fontsize = plot_ini.axlabel_size) 
        ax1.set_xlim(5,45)
        for rng_ in range(len(data_size)):
            for j, xpos in enumerate(pop[1]):
                pops = [i+xshift[rng_] for i in pop[1]]
                N_parti_avg_Nsims[rng_] = np.array([float(i) for i in N_parti_avg_Nsims[rng_]])
                stdmin_Nsims[rng_] = np.array([float(i) for i in stdmin_Nsims[rng_]])
                stdmax_Nsims[rng_] = np.array([float(i) for i in stdmax_Nsims[rng_]])
                if j == 0:
                    ax1.scatter(pops, np.log10(N_parti_avg_Nsims[rng_]), 
                                color = colors[rng_+1], edgecolor = 'black', zorder = 2, label = labels[rng_])
                else:
                    ax1.scatter(pops, np.log10(N_parti_avg_Nsims[rng_]), 
                                color = colors[rng_+1], edgecolor = 'black', zorder = 2)
            ax1.scatter(pops, np.log10(stdmin_Nsims[rng_]), color = colors[rng_+1], marker = '_')
            ax1.scatter(pops, np.log10(stdmax_Nsims[rng_]), color = colors[rng_+1], marker = '_')
            ax1.plot([pops, pops], [np.log10(stdmin_Nsims[rng_]), np.log10(stdmax_Nsims[rng_])], 
                      color = colors[rng_+1], zorder = 1)
        ax1.legend()
        plot_ini.tickers_pop(ax1, pop[1], 'GRX')
        plt.savefig('figures/steady_time/stab_time_mean_GRX_Nsims.pdf', dpi = 300, bbox_inches='tight')
        plt.clf()
    
        ###### N vs. Residuals #####
        iterf = 0
        for data_ in range(5):
            fig, ax = plt.subplots()
            if data_ == 0:
                title_string = 'Hermite'
                ax.set_title(title_string+str('\n')+labelsD[0], fontsize = plot_ini.tilabel_size)
                plot_ini.tickers_pop(ax, pop[0], 'Hermite')
                ax.set_xlim(6, 105)
            else:
                title_string = 'GRX'
                ax.set_title(title_string+str('\n')+labelsD[iterf+1], plot_ini.tilabel_size)
                plot_ini.tickers_pop(ax, pop[0], 'GRX')
                ax.set_xlim(6, 45)
                if data_ > 1:
                    iterf += 1
            print(folders[iterf])
            print(pop[data_], avg_deviate[data_])
            ax.set_ylabel(r'$\langle (t_{\rm{dis}} - \sigma_{\rm{dis}}) \rangle$ [Myr]', fontsize = plot_ini.axlabel_size)
            x_arr = np.linspace(10, max(pop[data_]), 100)
            smooth_curve = make_interp_spline(pop[data_], avg_deviate[data_])
            ax.plot(x_arr, smooth_curve(x_arr), color = colors[data_], zorder = 1)
            ax.scatter(pop[data_][pop[data_] > 5], avg_deviate[data_][pop[data_] > 5], color = colors[data_], edgecolors='black', zorder = 2)
            plt.savefig('figures/steady_time/stab_time_residuals_'+folders[iterf]+'_'+title_string+'.pdf', dpi = 300, bbox_inches='tight')
            plt.clf()

        for data_ in range(4):
            with open('figures/steady_time/Sim_summary_'+folders[data_]+'.txt', 'w') as file:
                integrator, drange = folder_loop(data_)

                for int_ in range(drange):
                    file.write('For '+str(integrator[int_])+', # of full simulations per population:       '+str(pop[data_+int_].flatten()))
                    file.write('\n                                                     '+str(full_simul[data_+int_]))
                    file.write('\nNumber of samples:                                   '+str(psamp[data_+int_].flatten()))
                    file.write('\nThe slope of the curve goes as:                      '+str(slope[data_+int_]))
                    file.write('\nThe power-law of the lnN goes as:                    '+str(beta[data_+int_]))
                    file.write('\nThe logarithmic factor goes as:                      '+str(log_c[data_+int_]))
                    file.write('\nThe final raw data:                                  '+str(pop[data_+int_][pop[data_+int_] > 5].flatten()))
                    file.write('\nSimulated time [Myr]                                 '+str(N_parti_avg[data_+int_][pop[data_+int_] > 5].flatten())+'\n\n')