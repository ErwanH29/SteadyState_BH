from amuse.lab import *
from file_logistics import *
from matplotlib.pyplot import *
from scipy import stats
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
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

def ecc_semi_histogram(integrator):
    """
    Function which plots the eccentricity and semi-major axis of the particles
    """
    
    plot_ini = plotter_setup()
    axlabel_size, tick_size = plot_ini.font_size()

    data = natsort.natsorted(glob.glob('/media/erwanh/Elements/rc_0.25_4e6/'+integrator+'/particle_trajectory/*'))

    SMBH_ecca = [ ]
    SMBH_sema = [ ]
    IMBH_ecca = [ ]
    IMBH_sema = [ ]
    
    total_data = 0
    ecc_data = 0
    for file_ in range(len(data)):
        with open(data[file_], 'rb') as input_file:
            file_size = os.path.getsize(data[file_])
            if file_size < 2.8e9:
                print('Reading File :', input_file)
                ptracker = pkl.load(input_file)
                col_len = np.shape(ptracker)[1]

                if np.shape(ptracker)[0] <= 40:
                    for parti_ in range(np.shape(ptracker)[0]):
                        if parti_ != 0:
                            particle = ptracker.iloc[parti_]

                            for j in range(col_len-1):
                                total_data += 1
                                sim_snap = particle.iloc[j]

                                if sim_snap[8][2] < 1:
                                    ecc_data += 1
                                if sim_snap[8][1] < 1:
                                    ecc_data += 1

                                SMBH_ecca.append(np.log10(sim_snap[8][0]))
                                SMBH_sema.append(np.log10(abs(sim_snap[7][0]).value_in(units.pc)))

                                if sim_snap[8][0] == sim_snap[8][1] or sim_snap[7][0] == sim_snap[7][1]:
                                    pass
                                elif sim_snap[8][0] == sim_snap[8][2] or sim_snap[7][0] == sim_snap[7][2]:
                                    pass
                                else: 
                                    IMBH_ecca.append(np.log10(sim_snap[8][1]))
                                    IMBH_sema.append(np.log10(abs(sim_snap[7][1]).value_in(units.pc)))
                                    IMBH_ecca.append(np.log10(sim_snap[8][2]))
                                    IMBH_sema.append(np.log10(abs(sim_snap[7][2]).value_in(units.pc)))
                
    with open('figures/system_evolution/output/ecc_events_ALL.txt', 'w') as file:
        file.write('For '+integrator+' ecc < 1: '+str(ecc_data)+' / '+str(total_data)+' or '+str(100*ecc_data/total_data)+'%')

    data_set = pd.DataFrame()
    for i in range(len(IMBH_sema)):
        arr = {'sem': IMBH_sema[i], 'ecc': IMBH_ecca[i]}
        raw_data = pd.Series(data=arr, index=['sem', 'ecc'])
        data_set = data_set.append(raw_data, ignore_index = True)

    ##### All eccentricity vs. semimajor axis #####
    n, xbins, ybins, image = hist2d(IMBH_sema[::-1], IMBH_ecca[::-1], bins = 50, range=([-7.88, 2.5], [-4.3, 8]))
    plt.clf()
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\log_{10}a$ [pc]')
    ax.set_ylabel(r'$\log_{10}e$')
    bin2d_sim, xed, yed, image = ax.hist2d(IMBH_sema, IMBH_ecca, bins = 300, range=([-7.88, 2.5], [-4.3, 8]), cmap = 'viridis')
    bin2d_sim /= np.max(bin2d_sim)
    extent = [-7, 2, -2, 6]
    contours = ax.imshow(np.log10(bin2d_sim), extent = extent, aspect='auto', origin = 'upper')
    ax.axhline(0, linestyle = ':', color = 'white', zorder = 1)
    ax.scatter(-0.5, -1.0, color = 'blueviolet', label = 'SMBH-IMBH', zorder = 3)
    ax.scatter(SMBH_sema, SMBH_ecca, color = 'blueviolet', s = 0.3, zorder = 4)
    ax.contour(n.T, extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
               linewidths=1.25, cmap='binary', levels = 6, label = 'IMBH-IMBH', zorder = 2)
    ax.text(-6.6, 0.45, r'$e > 1$', color = 'white', va = 'center', fontsize = axlabel_size)
    ax.text(-6.6, -0.55, r'$e < 1$', color = 'white', va = 'center', fontsize = axlabel_size)
    plot_ini.tickers(ax, 'histogram')
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.legend(prop={'size': axlabel_size})
    plt.savefig('figures/system_evolution/'+integrator+'_all_ecc_sem_scatter_hist_contours_rc_0.25_4e6.png', dpi=300, bbox_inches='tight')
    plt.clf()

def energy_scatter():
    """
    Function which plots the final energy error w.r.t simulation time
    """

    plot_ini = plotter_setup()
    axlabel_size, tick_size = plot_ini.font_size()

    integrator = ['Hermite', 'GRX']
    colors = ['red', 'blue']
    time_arr = [[ ], [ ]]
    err_ener = [[ ], [ ]]

    iterd = 0
    for int_ in integrator:
        print('Processing energy data for ', int_)
        etracker_files = natsort.natsorted(glob.glob('/media/erwanh/Elements/rc_0.25_4e6/data/'+str(int_)+'/energy/*'))

        for file_ in range(len(etracker_files)):
            with open(etracker_files[file_], 'rb') as input_file:
                etracker = pkl.load(input_file)
                time_arr[iterd].append(etracker.iloc[-1][-4].value_in(units.Myr))
                err_ener[iterd].append(etracker.iloc[-1][-2])

        fig, ax = plt.subplots()
        ax.set_ylabel(r'$\log_{10}\Delta E$', fontsize = axlabel_size)
        ax.set_xlabel(r'$\log_{10}t_{\mathrm{end}}$ [Myr]', fontsize = axlabel_size)
        plot_ini.tickers(ax, 'plot') 
        bin2d_sim, xed, yed, image = ax.hist2d(np.log10(time_arr[iterd]), np.log10(err_ener[iterd]), bins = 50, 
                                                range=([-2.2, 2.2], [-14.2, 0]), cmap = 'viridis')
        bin2d_sim /= np.max(bin2d_sim)
        extent = [-7, 2, -2, 6]
        contours = ax.imshow(np.log10(bin2d_sim), extent = extent, aspect='auto', origin = 'upper')
        plot_ini.tickers(ax, 'histogram')
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        plt.savefig('figures/system_evolution/energy_error_'+int_+'_HISTOGRAM.pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        iterd += 1

    fig, ax = plt.subplots()
    ax.set_ylabel(r'$\log_{10}\Delta E$', fontsize = axlabel_size)
    ax.set_xlabel(r'$\log_{10}t_{\mathrm{end}}$ [Myr]', fontsize = axlabel_size)
    plot_ini.tickers(ax, 'plot')
    for int_ in range(len(integrator)): 
        ax.scatter(np.log10(time_arr[int_]), np.log10(err_ener[int_]), color = colors[int_], s = 3, label = integrator[int_])
    ax.legend()
    plt.savefig('figures/system_evolution/energy_error.pdf', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()  


    return

def global_properties():
    """
    Function which plots various Kepler elements of ALL particles simulated
    """
    
    plot_ini = plotter_setup()
    
    pop_lower = 5#int(input('What should be the lower limit of the population sampled? '))
    pop_upper = 40#int(input('What should be the upper limit of the population sampled? '))
    integrator = ['Hermite', 'GRX']
    folders = ['rc_0.25_4e6', 'rc_0.25_4e7', 'rc_0.50_4e6', 'rc_0.50_4e7']

    iterf = 0
    for fold_ in folders:
        if iterf == 0:
            integrator = ['Hermite', 'GRX']
        else:
            integrator = ['GRX']

        print('Data for: ', fold_)
        SMBH_ecc = [[ ], [ ]]
        SMBH_sem = [[ ], [ ]]
        SMBH_dist = [[ ], [ ]]
        IMBH_ecc = [[ ], [ ]]
        IMBH_sem = [[ ], [ ]]
        IMBH_dist = [[ ], [ ]]

        dir = os.path.join('figures/steady_time/Sim_summary_'+fold_+'_GRX.txt')
        with open(dir) as f:
            line = f.readlines()
            popG = line[0][54:-2] 
            avgG = line[7][56:-2]
            popG_data = popG.split()
            avgG_data = avgG.split()
            popG = np.asarray([float(i) for i in popG_data])
            avgG = np.asarray([float(i) for i in avgG_data])
            
        iter = 0
        no_files = 40
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
            pop_checker = [0]
            no_samples = 0
            for file_ in range(len(data)):
                with open(chaotic[file_], 'rb') as input_file:
                    chaotic_tracker = pkl.load(input_file)
                    pop = 5*round(0.2*chaotic_tracker.iloc[0][6])
                    if chaotic_tracker.iloc[0][6] <= pop_upper and chaotic_tracker.iloc[0][6] >= pop_lower:
                        with open(data[file_], 'rb') as input_file:
                            file_size = os.path.getsize(data[file_])
                            if file_size < 2.9e9:
                                print('Reading File', file_, ': ', input_file)
                                no_samples, process = no_file_tracker(pop_checker[0], pop, no_files, no_samples)

                                if (process):
                                    ptracker = pkl.load(input_file)

                                    with open(energy[file_], 'rb') as input_file:
                                        etracker = pkl.load(input_file)

                                    if pop in popG:
                                        idx = np.where(popG[popG > 5] == pop)
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
                                            ecc_IMBH = [ ]
                                            sem_IMBH = [ ]
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

                                                if sim_snap[8][0] == sim_snap[8][1] or sim_snap[7][0] == sim_snap[7][1]:
                                                    pass
                                                else: 
                                                    sem_IMBH.append(np.log10(sim_snap[7][1].value_in(units.pc)))
                                                    sem_IMBH.append(np.log10(sim_snap[7][2].value_in(units.pc)))
                                                    ecc_IMBH.append(np.log10(1-sim_snap[8][1]))
                                                    ecc_IMBH.append(np.log10(1-sim_snap[8][2]))
                                                
                                                line_x = (sim_snap[2][0] - SMBH_coords[2][0])
                                                line_y = (sim_snap[2][1] - SMBH_coords[2][1])
                                                line_z = (sim_snap[2][2] - SMBH_coords[2][2])
                                                dist_SMBH.append(np.log10(np.sqrt(line_x**2+line_y**2+line_z**2).value_in(units.pc)))
                                                dist_IMBH.append(np.log10(sim_snap[-1]))
                                                
                                            SMBH_ecc[iter].append(ecc_SMBH)
                                            SMBH_sem[iter].append(sem_SMBH)
                                            IMBH_ecc[iter].append(ecc_IMBH)
                                            IMBH_sem[iter].append(sem_IMBH)
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
        eccIMBH_flat = [[ ], [ ]]
        semIMBH_flat = [[ ], [ ]]
        distSMBH_flat = [[ ], [ ]]
        distIMBH_flat = [[ ], [ ]]
        for j in range(2):
            for sublist in SMBH_ecc[j]:
                for item in sublist:
                    eccSMBH_flat[j].append(item)
            for sublist in SMBH_sem[j]:
                for item in sublist:
                    semSMBH_flat[j].append(item)

            for sublist in IMBH_ecc[j]:
                for item in sublist:
                    eccIMBH_flat[j].append(item)
            for sublist in IMBH_sem[j]:
                for item in sublist:
                    semIMBH_flat[j].append(item)

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
    """
    
    def plotter_code(merger_idx, chaos_data, file_, integrator):
        """
        Plotter function
        
        Inputs:
        merger_idx: Idx to denote what outcome probed
        chaos_data: Chaos data files
        file_:      The file # extracted
        integrator: The integrator used 
        """

        plot_ini = plotter_setup()
        axlabel_size, tick_size = plot_ini.font_size()

        if merger_idx == -4:
            outcome = 'merger'
            bool = chaos_data.iloc[0][merger_idx]
            idx = 0
        else:
            outcome = 'ejection'
            bool = chaos_data.iloc[0][merger_idx].number

        if bool > 0:
            with open(ptracker_files[file_], 'rb') as input_file:
                file_size = os.path.getsize(ptracker_files[file_])
                if file_size < 2e9:
                    print('Reading File ', file_, ' : ', input_file)
                    ptracker = pkl.load(input_file)

                    col_len = 150
                    parti_size = 20+len(ptracker)**-0.5

                    line_x = np.empty((len(ptracker), col_len))
                    line_y = np.empty((len(ptracker), col_len))
                    line_z = np.empty((len(ptracker), col_len))

                    for i in range(len(ptracker)):
                        tptracker = ptracker.iloc[i]
                        for j in range(col_len):
                            data_it = col_len - j
                            coords = tptracker.iloc[-data_it][2]
                            line_x[i][j] = coords[0].value_in(units.pc)
                            line_y[i][j] = coords[1].value_in(units.pc)
                            line_z[i][j] = coords[2].value_in(units.pc)

                        if outcome == 'merger':       
                            if i != 0 and math.isnan(line_x[i][-1]) or tptracker.iloc[-1][1].value_in(units.kg) > 10**36:
                                idx = i

                    c = colour_picker()
                    fig, ax = plt.subplots()
                    
                    plot_ini.tickers(ax, 'plot') 

                    xaxis_lim = 1.05*np.nanmax(abs(line_x-line_x[0]))
                    yaxis_lim = 1.05*np.nanmax(abs(line_y-line_y[0]))

                    ax.set_xlim(-abs(xaxis_lim), abs(xaxis_lim))
                    ax.set_ylim(-abs(yaxis_lim), yaxis_lim)

                    ax.set_xlabel(r'$x$ [pc]', fontsize = axlabel_size)
                    ax.set_ylabel(r'$y$ [pc]', fontsize = axlabel_size)
                    
                    iter = 0
                    for i in range(len(ptracker)):
                        if iter > len(c):
                            iter = 0

                        if i == 0:
                            adapt_c = 'black'
                            ax.scatter((line_x[i]-line_x[0]), (line_y[i]-line_y[0]), 
                                        c = adapt_c, zorder = 1, s = 250)
                            
                        else:
                            ax.scatter(line_x[i][-1]-line_x[0][-1], line_y[i][-1]-line_y[0][-1], 
                                        c = c[iter-2], edgecolors = 'black', s = parti_size, zorder = 3)
                            ax.scatter(line_x[i]-line_x[0], line_y[i]-line_y[0], 
                                        c = c[iter-2], s = 1, zorder = 1) 
                            if outcome == 'merger':
                                if i == idx:
                                    ax.scatter(line_x[i][-2]-line_x[0][-2], line_y[i][-2]-line_y[0][-2], 
                                            c = c[iter-2], edgecolors = 'black', s = 4*parti_size, zorder = 3)
                                else:
                                    ax.scatter(line_x[i][-2]-line_x[0][-2], line_y[i][-2]-line_y[0][-2], 
                                            c = c[iter-2], edgecolors = 'black', s = 0.8*parti_size, zorder = 3)
                        iter += 1
                    print('figures/system_evolution/Overall_System/simulation_evolution_pop_'+str(len(ptracker))+'_'+str(file_)+'_'+outcome+'.pdf')
                    print(ptracker_files[file_])
                    plt.savefig('figures/system_evolution/Overall_System/simulation_evolution_pop_'+str(integrator)+str(len(ptracker))+'_'+str(file_)+'_'+outcome+'.pdf', dpi=300, bbox_inches='tight')
                    plt.clf()
                    plt.close()


    ptracker_files = natsort.natsorted(glob.glob('/media/erwanh/Elements/rc_0.25_4e6/'+(int_string)+'/particle_trajectory/*'))
    ctracker_files = natsort.natsorted(glob.glob('/media/erwanh/Elements/rc_0.25_4e6/data/'+str(int_string)+'/chaotic_simulation/*'))

    print('Spatial evolution plotter')
    bools = [True, False]
    for bool_ in bools:
        merger_bool = bool_
        if (merger_bool):
            outcome_idx = -4
        else:
            outcome_idx = 3

        iter_file = 0
        for file_ in range(len(ptracker_files)):
            with open(ctracker_files[file_], 'rb') as input_file:
                ctracker = pkl.load(input_file)
                plotter_code(outcome_idx, ctracker, file_, int_string)
            iter_file += 1
            
    return