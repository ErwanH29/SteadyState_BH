from amuse.lab import *
from file_logistics import *
from matplotlib.pyplot import *
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import statsmodels.api as sm
import warnings

np.seterr(divide='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def colour_picker():
    """
    Colour chooser for the configuration plotter
    """

    colors = ['red', 'blue', 'orange', 'purple', 'salmon', 'slateblue', 
              'gold', 'darkviolet', 'cornflowerblue',  'cyan',
              'lightskyblue', 'magenta',  'dodgerblue']

    return colors

def global_properties():
    """
    Function which plots various Kepler elements of ALL particles simulated
    """
    
    plot_ini = plotter_setup()
    pop_lower = int(input('What should be the lower limit of the population sampled? '))
    pop_upper = int(input('What should be the upper limit of the population sampled? '))
    integrator = ['Hermite', 'GRX']
    folders = ['rc_0.25_4e6', 'rc_0.25_4e7', 'rc_0.50_4e6', 'rc_0.50_4e7']

    iterf = 0
    for fold_ in folders:
        if iterf == 0:
            integrator = ['Hermite', ' GRX']
        else:
            integrator = ['GRX']

        print('Data for: ', fold_)
        SMBH_ecc = [[ ], [ ]]
        SMBH_sem = [[ ], [ ]]
        SMBH_ecca = [[ ], [ ]]
        SMBH_sema = [[ ], [ ]]
        SMBH_dist = [[ ], [ ]]
        IMBH_ecc = [[ ], [ ]]
        IMBH_sem = [[ ], [ ]]
        IMBH_ecca = [[ ], [ ]]
        IMBH_sema = [[ ], [ ]]
        IMBH_dist = [[ ], [ ]]

        dir = os.path.join('figures/steady_time/Sim_summary_'+fold_+'.txt')
        with open(dir) as f:
            line = f.readlines()
            popG = line[12][54:-2] 
            avgG = line[17][56:-2]
            popG_data = popG.split()
            avgG_data = avgG.split()
            popG = np.asarray([float(i) for i in popG_data])
            avgG = np.asarray([float(i) for i in avgG_data])

        iter = 0
        for int_ in integrator:   
            data = natsort.natsorted(glob.glob('/media/erwanh/Elements/'+fold_+'/'+(int_)+'/particle_trajectory/*'))
            if int_ != 'GRX':
                val = 63 - iterf
                energy = ['/media/erwanh/Elements/'+fold_+'/data/Hermite/energy/'+str(i[val:]) for i in data]
                chaotic = ['/media/erwanh/Elements/'+fold_+'/data/Hermite/chaotic_simulation/'+str(i[val:]) for i in data]
            else:
                val = 59 - iterf
                energy = ['/media/erwanh/Elements/'+fold_+'/data/GRX/energy/'+str(i[val:]) for i in data]
                chaotic = ['/media/erwanh/Elements/'+fold_+'/data/GRX/chaotic_simulation/'+str(i[val:]) for i in data]

            total_data = 0
            ecc_data = 0
            for file_ in range(len(data)):
                with open(chaotic[file_], 'rb') as input_file:
                    chaotic_tracker = pkl.load(input_file)
                    if chaotic_tracker.iloc[0][6] <= pop_upper and chaotic_tracker.iloc[0][6] >= pop_lower:
                        with open(data[file_], 'rb') as input_file:
                            file_size = os.path.getsize(data[file_])
                            if file_size < 2.8e9:
                                print('Reading File :', input_file)
                                ptracker = pkl.load(input_file)

                                with open(energy[file_], 'rb') as input_file:
                                    etracker = pkl.load(input_file)

                                if 10*round(0.1*chaotic_tracker.iloc[0][6]) in popG:
                                    idx = np.where(popG[popG > 5] == 10*round(0.1*chaotic_tracker.iloc[0][6]))
                                    col_len = int(min(np.round((avgG[idx])*10**3), np.shape(ptracker)[1])-1)
                                else:
                                    col_len = np.shape(ptracker)[1]

                                for parti_ in range(np.shape(ptracker)[0]):
                                    if parti_ != 0:
                                        particle = ptracker.iloc[parti_]
                                        SMBH_data = ptracker.iloc[0]

                                        time = [ ]
                                        NN = [ ]
                                        ecc_SMBH = [ ]
                                        sem_SMBH = [ ]
                                        ecc_SMBHa = [ ]
                                        sem_SMBHa = [ ]
                                        ecc_IMBH = [ ]
                                        sem_IMBH = [ ]
                                        ecc_IMBHa = [ ]
                                        sem_IMBHa = [ ]
                                        dist_SMBH = [ ]
                                        dist_IMBH = [ ]

                                        for j in range(col_len-1):
                                            total_data += 1
                                            sim_snap = particle.iloc[j]
                                            ene_snap = etracker.iloc[j]
                                            SMBH_coords = SMBH_data.iloc[j]
                                            time.append(ene_snap[6].value_in(units.Myr))

                                            if sim_snap[8][2] < 1:
                                                ecc_data += 1
                                            if sim_snap[8][1] < 1:
                                                ecc_data += 1

                                            NN.append(np.log10(sim_snap[-1]))
                                            sem_SMBH.append(np.log10(sim_snap[7][0].value_in(units.pc)))
                                            ecc_SMBH.append(np.log10(1-sim_snap[8][0]))
                                            ecc_SMBHa.append(np.log10(sim_snap[8][0]))
                                            sem_SMBHa.append(np.log10(abs(sim_snap[7][0]).value_in(units.pc)))

                                            sem_IMBH.append(np.log10(sim_snap[7][1].value_in(units.pc)))
                                            sem_IMBH.append(np.log10(sim_snap[7][2].value_in(units.pc)))
                                            ecc_IMBH.append(np.log10(1-sim_snap[8][1]))
                                            ecc_IMBH.append(np.log10(1-sim_snap[8][2]))
                                            ecc_IMBHa.append(np.log10(sim_snap[8][1]))
                                            sem_IMBHa.append(np.log10(abs(sim_snap[7][1]).value_in(units.pc)))
                                            ecc_IMBHa.append(np.log10(sim_snap[8][2]))
                                            sem_IMBHa.append(np.log10(abs(sim_snap[7][2]).value_in(units.pc)))
                                            
                                            line_x = (sim_snap[2][0] - SMBH_coords[2][0])
                                            line_y = (sim_snap[2][1] - SMBH_coords[2][1])
                                            line_z = (sim_snap[2][2] - SMBH_coords[2][2])
                                            dist_SMBH.append(np.log10(np.sqrt(line_x**2+line_y**2+line_z**2).value_in(units.pc)))
                                            dist_IMBH.append(np.log10(sim_snap[-1]))
                                            
                                        SMBH_ecc[iter].append(ecc_SMBH)
                                        SMBH_sem[iter].append(sem_SMBH)
                                        SMBH_ecca[iter].append(ecc_SMBHa)
                                        SMBH_sema[iter].append(sem_SMBHa)
                                        IMBH_ecc[iter].append(ecc_IMBH)
                                        IMBH_sem[iter].append(sem_IMBH)
                                        IMBH_ecca[iter].append(ecc_IMBHa)
                                        IMBH_sema[iter].append(sem_IMBHa)
                                        SMBH_dist[iter].append(dist_SMBH)
                                        IMBH_dist[iter].append(dist_IMBH)

            if fold_ == 'rc_0.25' and iter == 0:
                with open('figures/system_evolution/output/ecc_events.txt', 'w') as file:
                    file.write('For '+str(int_)+' ecc < 1: '+str(ecc_data)+' / '+str(total_data)+' or '+str(100*ecc_data/total_data)+'%')
            else:
                with open('figures/system_evolution/output/ecc_events.txt', 'a') as file:
                    file.write('\nFor '+str(int_)+' ecc < 1: '+str(ecc_data)+' / '+str(total_data)+' or '+str(100*ecc_data/total_data)+'%')
            iter += 1
                
        c_hist = ['red', 'blue']

        eccSMBH_flat = [[ ], [ ]]
        semSMBH_flat = [[ ], [ ]]
        eccSMBHa_flat = [[ ], [ ]]
        semSMBHa_flat = [[ ], [ ]]
        eccIMBH_flat = [[ ], [ ]]
        semIMBH_flat = [[ ], [ ]]
        eccIMBHa_flat = [[ ], [ ]]
        semIMBHa_flat = [[ ], [ ]]
        distSMBH_flat = [[ ], [ ]]
        distIMBH_flat = [[ ], [ ]]
        for j in range(2):
            for sublist in SMBH_ecc[j]:
                for item in sublist:
                    eccSMBH_flat[j].append(item)
            for sublist in SMBH_sem[j]:
                for item in sublist:
                    semSMBH_flat[j].append(item)
            for sublist in SMBH_ecca[j]:
                for item in sublist:
                    eccSMBHa_flat[j].append(item)
            for sublist in SMBH_sema[j]:
                for item in sublist:
                    semSMBHa_flat[j].append(item)

            for sublist in IMBH_ecc[j]:
                for item in sublist:
                    eccIMBH_flat[j].append(item)
            for sublist in IMBH_sem[j]:
                for item in sublist:
                    semIMBH_flat[j].append(item)
            for sublist in IMBH_ecca[j]:
                for item in sublist:
                    eccIMBHa_flat[j].append(item)
            for sublist in IMBH_sema[j]:
                for item in sublist:
                    semIMBHa_flat[j].append(item)

            for sublist in SMBH_dist[j]:
                for item in sublist:
                    distSMBH_flat[j].append(item)
            for sublist in IMBH_dist[j]:
                for item in sublist:
                    distIMBH_flat[j].append(item)

        ks_eSMBH = stats.ks_2samp(eccSMBH_flat[0], eccSMBH_flat[1])
        ks_sSMBH = stats.ks_2samp(semSMBH_flat[0], semSMBH_flat[1])
        ks_eIMBH = stats.ks_2samp(eccIMBH_flat[0], eccIMBH_flat[1])
        ks_sIMBH = stats.ks_2samp(semIMBH_flat[0], semIMBH_flat[1])
        ks_dSMBH = stats.ks_2samp(distSMBH_flat[0], distSMBH_flat[1])
        ks_dIMBH = stats.ks_2samp(distIMBH_flat[0], distIMBH_flat[1])
        
        with open('figures/system_evolution/output/KStests_eject_'+fold_+'.txt', 'w') as file:
            file.write('For all simulations. If the p-value is less than 0.05 we reject the')
            file.write('\nhypothesis that samples are taken from the same distribution')
            file.write('\n\nSMBH eccentricity 2 sample KS test:         pvalue = '+str(ks_eSMBH[1]))
            file.write('\nSMBH semi-major axis 2 sample KS test:        pvalue = '+str(ks_sSMBH[1]))
            file.write('\n\nIMBH eccentricity 2 sample KS test:         pvalue = '+str(ks_eIMBH[1]))
            file.write('\nIMBH semi-major axis 2 sample KS test:        pvalue = '+str(ks_sIMBH[1]))
            file.write('\nSMBH distance 2 sample KS test:               pvalue = '+str(ks_dSMBH[1]))
            file.write('\nIMBH distance 2 sample KS test:               pvalue = '+str(ks_dIMBH[1]))

        ##### All eccentricity vs. semimajor axis #####
        fig, ax = plt.subplots()
        n, xbins, ybins, image = hist2d(semIMBHa_flat[1][::-1], eccIMBHa_flat[1][::-1], bins = 40, range=([-6.8, 2], [-1.8, 6]))
        plt.clf()
        
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$\log_{10}a$ [pc]')
        ax.set_ylabel(r'$\log_{10}e$')
        bin2d_sim, xed, yed, image = ax.hist2d(semIMBHa_flat[1], eccIMBHa_flat[1], bins = 125, range=([-6.8, 2], [-1.8, 6]), cmap = 'viridis')
        bin2d_sim /= np.max(bin2d_sim)
        extent = [-7, 2, -2, 6]
        contours = ax.imshow((bin2d_sim), extent = extent, aspect='auto', origin = 'upper')
        ax.scatter(semSMBHa_flat[1][0], eccSMBHa_flat[1][0], color = 'blueviolet', label = 'SMBH-IMBH')
        ax.scatter(semSMBHa_flat[1], eccSMBHa_flat[1], color = 'blueviolet', s = 0.3)
        ax.contour(n.T, extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=1.25, cmap='binary', levels = 3, label = 'IMBH-IMBH')
        ax.axhline(0, linestyle = ':', color = 'white', )
        ax.text(-6, 0.2, r'$e > 1$', color = 'white', va = 'center')
        ax.text(-6, -0.22, r'$e < 1$', color = 'white', va = 'center')
        plot_ini.tickers(ax, 'histogram')
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax.legend() 
        plt.savefig('figures/system_evolution/GRX_all_ecc_sem_scatter_hist_contours_'+fold_+'.png', dpi=300, bbox_inches='tight')
        plt.clf()

        ##### CDF Plots #####
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 4,  width_ratios=(2, 2, 2, 2), height_ratios=(2, 3), left=0.1, right=0.9, bottom=0.1, 
                              top=0.9, wspace=0.25, hspace=0.1)
        axL = fig.add_subplot(gs[1, 0:2])
        axL1 = fig.add_subplot(gs[0, 0:2], sharex=axL)
        axR = fig.add_subplot(gs[1, 2:])
        axR1 = fig.add_subplot(gs[0, 2:], sharex=axR)

        axL.set_xlabel(r'$\log_{10}(1-e)_{\rm{SMBH}}$')
        axR.set_xlabel(r'$\log_{10}a_{\rm{SMBH}}$ [pc]')
        axL.set_ylabel(r'$\log_{10}$(CDF)')
        axL1.set_ylabel(r'$\rho/\rho_{\rm{max}}$')
        for int_ in range(2):
            ecc_sort = np.sort(eccSMBH_flat[int_])
            ecc_index = np.asarray([i for i in enumerate(ecc_sort)])
            axL.plot(ecc_sort, np.log10(ecc_index[:,0]/ecc_index[-1,0]), color = c_hist[int_])

            kde_ecc = sm.nonparametric.KDEUnivariate(ecc_sort)
            kde_ecc.fit()
            kde_ecc.density /= max(kde_ecc.density)
            axL1.plot(kde_ecc.support, kde_ecc.density, color = c_hist[int_], label = integrator[int_])
            axL1.fill_between(kde_ecc.support, kde_ecc.density, alpha = 0.35, color = c_hist[int_])

            sem_sort = np.sort(semSMBH_flat[int_])
            sem_index = np.asarray([i for i in enumerate(sem_sort)])
            axR.plot(sem_sort, np.log10(sem_index[:,0]/sem_index[-1,0]), color = c_hist[int_])

            kde_ecc = sm.nonparametric.KDEUnivariate(sem_sort)
            kde_ecc.fit()
            kde_ecc.density /= max(kde_ecc.density)
            axR1.plot(kde_ecc.support, kde_ecc.density, color = c_hist[int_])
            axR1.fill_between(kde_ecc.support, kde_ecc.density, alpha = 0.35, color = c_hist[int_])
        
        for ax_ in [axL, axL1, axR, axR1]:
            plot_ini.tickers(ax_, 'plot')
        for ax_ in [axL1, axR1]:
            ax_.set_ylim(0, 1.1)
        axL1.legend(loc='upper left')
        plt.savefig('figures/system_evolution/ecc_cdf_histogram_'+fold_+'.png', dpi=300, bbox_inches='tight')
        plt.clf()

        iterf += 1

def spatial_plotter(int_string):
    """
    Function to plot the evolution of the system

    output: The spatial evolution of the system
    """

    plot_ini = plotter_setup()

    ptracker_files = natsort.natsorted(glob.glob('/media/erwanh/Elements/'+(int_string)+'/particle_trajectory/*'))
    etracker_files = natsort.natsorted(glob.glob('data/'+str(int_string)+'/energy/*'))
    ctracker_files = natsort.natsorted(glob.glob('data/'+str(int_string)+'/no_addition/chaotic_simulation/*'))
    iter_file = -1
    for file_ in range(len(ptracker_files)):
        iter_file += 1
        with open(ctracker_files[file_], 'rb') as input_file:
            ctracker = pkl.load(input_file)
            if ctracker.iloc[0][5] > 0 and ctracker.iloc[0][6] >= 10:
                with open(ptracker_files[file_], 'rb') as input_file:
                    file_size = os.path.getsize(ptracker_files[file_])
                    if file_size < 2e9 and file_ > 120:
                        print('Reading File ', file_, ' : ', input_file)
                        ptracker = pkl.load(input_file)
                        with open(etracker_files[file_], 'rb') as input_file:
                            etracker = pkl.load(input_file)

                        col_len_raw = np.shape(ptracker)[1]-1
                        col_len = round(col_len_raw**0.8)
                        parti_size = 20+len(ptracker)**-0.5

                        line_x = np.empty((len(ptracker), col_len))
                        line_y = np.empty((len(ptracker), col_len))
                        line_z = np.empty((len(ptracker), col_len))

                        for i in range(len(ptracker)):
                            tptracker = ptracker.iloc[i]
                            for j in range(col_len):
                                coords = tptracker.iloc[j][2]
                                if len(coords) == 1:
                                    pass
                                else:
                                    line_x[i][j] = coords[0].value_in(units.pc)
                                    line_y[i][j] = coords[1].value_in(units.pc)
                                    line_z[i][j] = coords[2].value_in(units.pc)

                        time = np.empty((col_len_raw - 1))
                        dE_array = np.empty((col_len_raw - 1))

                        for i in range(col_len_raw-1):
                            if i == 0:
                                pass
                            else:
                                vals = etracker.iloc[i]
                                time[i-1] = vals[6].value_in(units.Myr)
                                dE_array[i-1] = vals[7]

                        c = colour_picker()
                        fig = plt.figure(figsize=(12.5, 15))
                        ax1 = fig.add_subplot(321)
                        ax2 = fig.add_subplot(322)
                        ax3 = fig.add_subplot(323)
                        ax4 = fig.add_subplot(324)
                        
                        ax1.set_title('Overall System')
                        ax2.set_title('Energy Error vs. Time')
                        ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
                        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
                        for ax_ in [ax1, ax2, ax3, ax4]:
                            plot_ini.tickers(ax_, 'plot') 

                        xaxis_lim = 1.05*np.nanmax(abs(line_x-line_x[0]))
                        yaxis_lim = 1.05*np.nanmax(abs(line_y-line_y[0]))
                        zaxis_lim = 1.05*np.nanmax(abs(line_z-line_z[0]))

                        ax1.set_xlim(-abs(xaxis_lim), abs(xaxis_lim))
                        ax1.set_ylim(-abs(yaxis_lim), yaxis_lim)
                        ax3.set_xlim(-abs(xaxis_lim), abs(xaxis_lim))
                        ax3.set_ylim(-abs(zaxis_lim), zaxis_lim)
                        ax4.set_xlim(-abs(yaxis_lim), abs(yaxis_lim))
                        ax4.set_ylim(-abs(zaxis_lim), zaxis_lim)
                        ax2.set_yscale('log')

                        ax1.set_xlabel(r'$x$ [pc]')
                        ax1.set_ylabel(r'$y$ [pc]')
                        ax2.set_xlabel(r'Time [Myr]')
                        ax2.set_ylabel(r'$\frac{|E(t)-E_0|}{|E_0|}$')
                        ax3.set_xlabel(r'$x$ [pc]')
                        ax3.set_ylabel(r'$z$ [pc]')
                        ax4.set_xlabel(r'$y$ [pc]')
                        ax4.set_ylabel(r'$z$ [pc]')
                        iter = -1
                        
                        for i in range(len(ptracker)):
                            iter += 1
                            if iter > len(c):
                                iter = 0

                            if i == 0:
                                adapt_c = 'black'
                                ax1.scatter((line_x[i]-line_x[0]), (line_y[i]-line_y[0]), 
                                            c = adapt_c, zorder = 1, s = 250)
                                ax3.scatter((line_x[i]-line_x[0]), (line_z[i]-line_z[0]), 
                                            c = adapt_c, zorder = 1, s = 250)
                                ax4.scatter((line_z[i]-line_z[0]), (line_y[i]-line_y[0]), 
                                            c = adapt_c, zorder = 1, s = 250)
                                
                            else:
                                ax1.scatter(line_x[i][-1]-line_x[0][-1], line_y[i][-1]-line_y[0][-1], 
                                            c = c[iter-2], edgecolors = 'black', s = parti_size, zorder = 3)
                                ax1.scatter(line_x[i]-line_x[0], line_y[i]-line_y[0], 
                                            c = c[iter-2], s = 1, zorder = 1) 

                                ax3.scatter(line_x[i][-1]-line_x[0][-1], line_z[i][-1]-line_z[0][-1], 
                                            c = c[iter-2], edgecolors = 'black', s = parti_size, zorder = 3)
                                ax3.scatter(line_x[i]-line_x[0], line_z[i]-line_z[0], 
                                            c = c[iter-2], s = 1, zorder = 1) 

                                ax4.scatter(line_y[i][-1]-line_y[0][-1], line_z[i][-1]-line_z[0][-1], 
                                            c = c[iter-2], edgecolors = 'black', s = parti_size, zorder = 3)
                                ax4.scatter(line_y[i]-line_y[0], line_z[i]-line_z[0], 
                                            c = c[iter-2], s = 1, zorder = 1) 
                                
                        ax2.plot(time[:-5], dE_array[:-5], color = 'black')
                        plt.savefig('figures/system_evolution/Overall_System/simulation_evolution_'+str(iter_file)+'.pdf', dpi=300, bbox_inches='tight')
                        plt.clf()
                        plt.close()     

                        fig = plt.figure(figsize=(8, 8))
                        ax3D = fig.add_subplot(121, projection="3d")
                        iter = -1
                        for i in range(len(ptracker)):
                            iter += 1
                            if iter > len(c):
                                iter = 0
                            if i == 0:
                                pass
                            else:
                                ax3D.scatter(line_x[i]-line_x[0], 
                                            line_y[i]-line_y[0], 
                                            line_z[i]-line_z[0], 
                                            c = c[iter-2], s = 1, zorder = 1)
                        ax3D.scatter(0, 0, 0, color = 'black', s = 150, zorder = 2)
                        ax3D.xaxis.set_major_formatter(mtick.FormatStrFormatter('%0.2f'))
                        ax3D.yaxis.set_major_formatter(mtick.FormatStrFormatter('%0.2f'))
                        ax3D.zaxis.set_major_formatter(mtick.FormatStrFormatter('%0.2f'))
                        ax3D.set_xlabel(r'$x$ [pc]')
                        ax3D.set_ylabel(r'$y$ [pc]')
                        ax3D.set_zlabel(r'$z$ [pc]')
                        ax3D.view_init(30, 160)
                        plt.savefig('figures/system_evolution/Overall_System/simulation_evolution_3D_'+str(iter_file)+'.pdf', dpi=300, bbox_inches='tight')

    return