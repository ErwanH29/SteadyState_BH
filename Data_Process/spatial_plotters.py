import glob
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import natsort
import numpy as np
import os
import pickle as pkl
import scipy.optimize
from scipy import stats
import statsmodels.api as sm
import warnings

from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.lab import Particles
from amuse.units import units
from matplotlib.pyplot import hist2d

from file_logistics import moving_average, no_file_tracker, plotter_setup

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

def orbital_plotter_setup(flat_arr, iloop, ax_bot, 
                          ax_top, pcolours, plabels, 
                          bool):
    """
    Function to plot the CDF and KDE of orbital properties.

    Inputs:
    flat_arr:  The flattened data set
    iloop:     The loop iteration
    ax_bot:    The CDF plot
    ax_top:    The KDE plot
    pcolours:  The colour scheme
    plabels:   The labels for the plot
    bool:      Boolean to limit labelling
    """

    plot_ini = plotter_setup()
    axlabel_size, tick_size = plot_ini.font_size()
    
    plot_ini = plotter_setup()
    data_sort = np.sort(flat_arr[iloop])
    data_idx = np.asarray([i for i in enumerate(data_sort)])
    ax_bot.plot(data_sort, np.log10(data_idx[:,0]/data_idx[-1,0]), color=pcolours[iloop])

    kde = sm.nonparametric.KDEUnivariate(data_sort)
    kde.fit()
    kde.density /= max(kde.density)
    if (bool):
        ax_top.plot(kde.support, kde.density, color=pcolours[iloop], label=plabels[iloop])
        ax_top.legend(loc='upper left')
        ax_top.legend(prop={'size': axlabel_size-5})
    else:
        ax_top.plot(kde.support, kde.density, color=pcolours[iloop])
    ax_top.fill_between(kde.support, kde.density, alpha=0.35, color=pcolours[iloop])

    for ax_ in [ax_bot, ax_top]:
        plot_ini.tickers(ax_, 'plot')
        ax_.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax_top.set_ylim(1e-2, 1.1)

    return ax_bot, ax_top

def ecc_semi_histogram(integrator):
    """
    Plot {a, e} CDF of IMBH
    """
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plot_ini = plotter_setup()
    axlabel_size, tick_size = plot_ini.font_size()

    data = natsort.natsorted(glob.glob('/media/erwanh/Elements1/rc_0.25_4e6/'+integrator+'/particle_trajectory/*'))

    SMBH_ecca = [ ]
    SMBH_sema = [ ]
    IMBH_ecca = [ ]
    IMBH_sema = [ ]

    pop_checker = 0
    no_files = 60
    no_samples = 0
    
    total_data = 0
    ecc_data = 0
    for file_ in range(len(data)):
        with open(data[file_], 'rb') as input_file:
            file_size = os.path.getsize(data[file_])
            if file_size < 2.8e9:
                print('Reading File :', input_file)
                ptracker = pkl.load(input_file)
                dt_len = np.shape(ptracker)[1]
                pop = np.shape(ptracker)[0]
                if np.shape(ptracker)[0] <= 40:
                    no_samples, process, pop_checker = no_file_tracker(
                                                           pop_checker, 
                                                           5*round(0.2*pop), 
                                                           no_files, 
                                                           no_samples)
                    if (process):
                        for parti_, j in itertools.product(range(pop), range(dt_len-1)):
                            if parti_ != 0:
                                particle = ptracker.iloc[parti_]
                                total_data += 1
                                sim_snap = particle.iloc[j]

                                if sim_snap[8][2] < 1:
                                    ecc_data += 1
                                if sim_snap[8][1] < 1:
                                    ecc_data += 1
                                
                                SMBH_ecca.append(np.log10(sim_snap[8][0]))
                                SMBH_sema.append(np.log10(abs(sim_snap[7][0]).value_in(units.pc)))

                                if sim_snap[8][0]==sim_snap[8][1] or sim_snap[7][0]==sim_snap[7][1]:
                                    pass
                                elif sim_snap[8][0]==sim_snap[8][2] or sim_snap[7][0]==sim_snap[7][2]:
                                    pass
                                else: 
                                    IMBH_ecca.append(np.log10(sim_snap[8][1]))
                                    IMBH_sema.append(np.log10(abs(sim_snap[7][1]).value_in(units.pc)))
                                    IMBH_ecca.append(np.log10(sim_snap[8][2]))
                                    IMBH_sema.append(np.log10(abs(sim_snap[7][2]).value_in(units.pc)))

    ##### All eccentricity vs. semimajor axis #####
    n, xbins, ybins, image = hist2d(IMBH_sema[::-1], IMBH_ecca[::-1], bins=300, range=([-7.88, 2.5], [-4.3, 8]))
    plt.clf()
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\log_{10}a$ [pc]', fontsize=axlabel_size)
    ax.set_ylabel(r'$\log_{10}e$', fontsize=axlabel_size)
    bin2d_sim, xed, yed, image = ax.hist2d(IMBH_sema, IMBH_ecca, bins=700, 
                                           range=([-7.88, 2.5], [-4.3, 8]), 
                                           cmap='Greys')
    bin2d_sim /= np.max(bin2d_sim)
    extent = [-7, 2, -2, 6]
    contours = ax.imshow(np.log10(bin2d_sim), extent=extent, 
                         aspect='auto', cmap="Blues", 
                         origin='upper')
    ax.axhline(0, linestyle=':', color='white', zorder=1)
    ax.scatter(-0.5, -1.0, color='blueviolet', label='SMBH-IMBH', zorder=3)
    ax.scatter(SMBH_sema, SMBH_ecca, color='black', s=0.3, zorder=4)
    cset = ax.contour(n.T, extent=[xbins.min(),xbins.max(), ybins.min(),ybins.max()],
                      linewidths=1.25, alpha=0)
    inside_data = np.full_like(IMBH_sema, False, dtype=bool)
    p = cset.collections[0].get_paths()
    for level in p:
        inside_data |= level.contains_points(list(zip(*(IMBH_sema, IMBH_ecca))))
        ax.scatter(np.asarray(IMBH_sema)[~inside_data], np.asarray(IMBH_ecca)[~inside_data], 
                   color='blue', s=0.5, label='IMBH-IMBH')
    ax.text(-6.6, 0.45, r'$e > 1$', color='white', va='center', fontsize=axlabel_size)
    ax.text(-6.6, -0.55, r'$e < 1$', color='white', va='center', fontsize=axlabel_size)
    plot_ini.tickers(ax, 'histogram')
    ax.tick_params(axis="y", which='both', labelsize=tick_size)
    ax.tick_params(axis="x", which='both', labelsize=tick_size)
    ax.legend(prop={'size': axlabel_size})
    plt.savefig('figures/system_evolution/'+integrator+'_all_ecc_sem_hist.png', dpi=300, bbox_inches='tight')
    plt.clf()

def energy_scatter():
    """
    Function which plots the final energy error w.r.t simulation time
    """

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plot_ini = plotter_setup()
    axlabel_size, tick_size = plot_ini.font_size()

    integrator = ['Hermite', 'GRX']
    colors = ['red', 'blue']
    time_arr = [[ ], [ ]]
    err_ener = [[ ], [ ]]

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2,  width_ratios=[5, 2], height_ratios=[1],
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax)
    ax1.tick_params(axis="y", labelleft=False)
    ax.set_ylabel(r'$\log_{10}\Delta E$', fontsize=axlabel_size)
    ax.set_xlabel(r'$\log_{10}t_{\mathrm{end}}$ [Myr]', fontsize=axlabel_size)
    ax1.set_xlabel(r'$\rho/\rho_{\rm{max}}$', fontsize=axlabel_size)
    for ax_ in [ax, ax1]:
        plot_ini.tickers(ax_, 'plot')
    for int_ in range(len(integrator)): 
        ax.scatter(np.log10(time_arr[int_]), np.log10(err_ener[int_]), color=colors[int_], s=3)
        kde_ene = sm.nonparametric.KDEUnivariate(np.log10(err_ener[int_]))
        kde_ene.fit()
        kde_ene.density /= max(kde_ene.density)
        ax1.plot(kde_ene.density, kde_ene.support, color=colors[int_], label=integrator[int_])
        ax1.fill_between(kde_ene.density, kde_ene.support, alpha=0.35, color=colors[int_])
    ax1.legend(prop={'size': axlabel_size})
    ax.set_ylim(-15, 0)
    plt.savefig('figures/system_evolution/energy_error.pdf', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()  

    return

def global_properties():
    """
    Function which plots various Kepler elements of ALL particles simulated
    """
    
    print('...Plotters Hermite vs. GRX CDF KDE...')

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plot_ini = plotter_setup()
    axlabel_size, tick_size = plot_ini.font_size()
    
    pop_lower = 5
    pop_upper = 40
    integrator = ['Hermite', 'GRX']
    
    SMBH_ecc = [[ ], [ ]]
    SMBH_sem = [[ ], [ ]]

    no_files = 60
    dir = os.path.join('figures/steady_time/Sim_summary_rc_0.25_4e6_GRX.txt')
    with open(dir) as f:
        line = f.readlines()
        popG = line[0][54:-2] 
        avgG = line[10][55:-2]
        popG_data = popG.split()
        avgG_data = avgG.split()
        popG = np.asarray([float(i) for i in popG_data])
        avgG = np.asarray([float(i) for i in avgG_data])
    
    iter = 0
    for int_ in integrator:   
        data = natsort.natsorted(glob.glob('/media/erwanh/Elements/rc_0.25_4e6/'+(int_)+'/particle_trajectory/*'))
        if int_ != 'GRX':
            val = 63
            chaotic = ['/media/erwanh/Elements/rc_0.25_4e6/data/Hermite/chaotic_simulation/'+str(i[val:]) for i in data]
        else:
            val = 59
            chaotic = ['/media/erwanh/Elements/rc_0.25_4e6/data/GRX/chaotic_simulation/'+str(i[val:]) for i in data]

        total_data = 0
        ecc_data = 0
        pop_checker = 0
        no_samples = 0
        for file_ in range(len(data)):
            with open(chaotic[file_], 'rb') as input_file:
                chaotic_tracker = pkl.load(input_file)
                pop = 5*round(0.2*chaotic_tracker.iloc[0][6])
                if pop <= pop_upper and pop >= pop_lower:
                    with open(data[file_], 'rb') as input_file:
                        file_size = os.path.getsize(data[file_])
                        if file_size<2.9e9:
                            print('Reading File', file_, ': ', input_file)
                            no_samples, process, pop_checker = no_file_tracker(pop_checker, pop, no_files, no_samples)
                            
                            if (process):
                                ptracker = pkl.load(input_file)
                                idx = np.where(popG[popG > 5]==pop)
                                dt_len = int(min(np.round((avgG[idx])*1e3), np.shape(ptracker)[1])-1)

                                for parti_ in range(np.shape(ptracker)[0]):
                                    if parti_ != 0:
                                        total_data += 1
                                        particle = ptracker.iloc[parti_]
                                        sim_snap = particle.iloc[dt_len-2]
                                            
                                        SMBH_sem[iter].append(np.log10(abs(sim_snap[7][0].value_in(units.pc))))
                                        SMBH_ecc[iter].append(np.log10(1-sim_snap[8][0]))

        with open('figures/system_evolution/output/ecc_events_rc_0.25_'+str(int_)+'.txt', 'w') as file:
            file.write('For '+str(int_)+' ecc < 1: '+str(ecc_data)+' / '+str(total_data)+' or '+str(100*ecc_data/total_data)+'%')

        iter += 1
                
    ks_eSMBH = stats.ks_2samp(SMBH_ecc[0], SMBH_ecc[1])
    ks_sSMBH = stats.ks_2samp(SMBH_sem[0], SMBH_sem[1])
    with open('figures/system_evolution/output/KStests_eject_rc_0.25_4e6.txt', 'w') as file:
        file.write('For all simulations. If the p-value is less than 0.05 we reject the')
        file.write('\nhypothesis that samples are taken from the same distribution')
        file.write('\n\nSMBH eccentricity 2 sample KS test:         pvalue = '+str(ks_eSMBH[1]))
        file.write('\nSMBH semi-major axis 2 sample KS test:        pvalue = '+str(ks_sSMBH[1]))

    ##### CDF Plots #####
    c_hist = ['red', 'blue']
        
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 4,  width_ratios=(2,2,2,2), height_ratios=(2,3), 
                          left=0.1, right=0.9, bottom=0.1, 
                          top=0.9, wspace=0.35, hspace=0.15)
    axL = fig.add_subplot(gs[1, 0:2])
    axL1 = fig.add_subplot(gs[0, 0:2], sharex=axL)
    axR = fig.add_subplot(gs[1, 2:])
    axR1 = fig.add_subplot(gs[0, 2:], sharex=axR)
    axL.set_xlabel(r'$\log_{10}(1-e)_{\rm{IMBH}}$', fontsize=axlabel_size)
    axR.set_xlabel(r'$\log_{10}a_{\rm{IMBH}}$ [pc]', fontsize=axlabel_size)
    axL.set_ylabel(r'$\log_{10}$(CDF)', fontsize=axlabel_size)
    axL1.set_ylabel(r'$\rho/\rho_{\rm{max}}$', fontsize=axlabel_size)
    for int_ in range(2):
        axL, axL1 = orbital_plotter_setup(SMBH_ecc, int_, axL, axL1, c_hist, integrator, True)
        axR, axR1 = orbital_plotter_setup(SMBH_sem, int_, axR, axR1, c_hist, integrator, False)
    plt.savefig('figures/system_evolution/ecc_SMBH_cdf_histogram_rc_0.25_4e6.pdf', dpi=300, bbox_inches='tight')
    plt.clf()

def global_properties_GRX_pops():
    """
    Function which plots various Kepler elements of ALL particles simulated
    """

    print('...Plotters Population CDF KDE...')
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plot_ini = plotter_setup()
    axlabel_size, tick_size = plot_ini.font_size()
    
    pop_tracker = 10
    labels = [r'Hermite$_{\mathrm{4e6}}$',
              r'GRX$_{\mathrm{4e6}}$',
              r'GRX$_{\mathrm{4e5}}$',
              r'GRX$_{\mathrm{4e7}}$']
    folders = ['rc_0.25_4e6', 'rc_0.25_4e5', 'rc_0.25_4e7']
    time_share = [True]

    dir = os.path.join('figures/steady_time/Sim_summary_rc_0.25_4e5_GRX.txt')
    with open(dir) as f:
        line = f.readlines()
        popG = line[0][54:-2] 
        avgG = line[10][55:-2]
        popG_data = popG.split()
        avgG_data = avgG.split()
        popG = np.asarray([float(i) for i in popG_data])
        avgG = np.asarray([float(i) for i in avgG_data])
            
        no_files = 60
        val = 59
        iterf = 0 
        
        for share_ in time_share:
            iterd = 0
            SMBH_ecc = [[ ], [ ], [ ], [ ]]
            SMBH_sem = [[ ], [ ], [ ], [ ]]
            SMBH_dis = [[ ], [ ], [ ], [ ]]
            
            for fold_ in folders:
                print('Data for: ', fold_, iterd)
                if iterf==0:
                    drange = 2
                    integrator = ['Hermite', 'GRX']
                else:
                    drange = 1
                    integrator = ['GRX']

                for int_ in range(drange):
                    if integrator[int_]=='Hermite':
                        val = 63
                    else:
                        val = 59
                    data = natsort.natsorted(glob.glob('/media/erwanh/Elements/'+fold_+'/'+integrator[int_]+'/particle_trajectory/*'))
                    chaotic = ['/media/erwanh/Elements/'+fold_+'/data/'+integrator[int_]+'/chaotic_simulation/'+str(i[val:]) for i in data]
                    total_data = 0
                    ecc_data = 0
                    pop_checker = 0
                    no_samples = 0
                    for file_ in range(len(data)):
                        with open(chaotic[file_], 'rb') as input_file:
                            chaotic_tracker = pkl.load(input_file)
                            pop = 5*round(0.2*chaotic_tracker.iloc[0][6])
                            if pop==pop_tracker:
                                with open(data[file_], 'rb') as input_file:
                                    file_size = os.path.getsize(data[file_])
                                    if file_size < 2.9e9:
                                        print('Reading File', file_, ': ', input_file)
                                        no_samples, process, pop_checker = no_file_tracker(pop_checker, pop, no_files, no_samples)

                                        if (process):
                                            ptracker = pkl.load(input_file)
                                            idx = np.where(popG[popG > 5]==pop)
                                            if (share_):
                                                dt_len = int(min(np.round((avgG[idx])*1e3), np.shape(ptracker)[1])-1)
                                                time_string = 'shared_cropped_time'
                                            else:
                                                dt_len = np.shape(ptracker)[1]-1
                                                time_string = 'total_simulation_time'

                                            for parti_, j in itertools.product(range(np.shape(ptracker)[0]), range(dt_len - 1)):
                                                if parti_ != 0:
                                                    particle = ptracker.iloc[parti_]
                                                    SMBH_data = ptracker.iloc[0]

                                                    for j in range(dt_len-1):
                                                        total_data += 1
                                                        sim_snap = particle.iloc[j]
                                                        SMBH_coords = SMBH_data.iloc[j]

                                                        if sim_snap[8][2] < 1:
                                                            ecc_data += 1
                                                        if sim_snap[8][1] < 1:
                                                            ecc_data += 1
                                                    
                                                        if j==dt_len - 2:
                                                            SMBH_sem[iterd].append(np.log10(abs(sim_snap[7][0].value_in(units.pc))))
                                                            SMBH_ecc[iterd].append(np.log10(1-sim_snap[8][0]))
                                                            
                                                            line_x = (sim_snap[2][0] - SMBH_coords[2][0])
                                                            line_y = (sim_snap[2][1] - SMBH_coords[2][1])
                                                            line_z = (sim_snap[2][2] - SMBH_coords[2][2])
                                                            dist = np.sqrt(line_x**2+line_y**2+line_z**2)
                                                            SMBH_dis[iterd].append(np.log10(dist.value_in(units.pc)))

                    iterd += 1                      
                iterf += 1
                    
            c_hist = ['red', 'blue', 'deepskyblue', 'royalblue']

            ##### CDF Plots #####
            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(2, 4,  width_ratios=(2, 2, 2, 2), height_ratios=(2, 3), left=0.1, right=0.9, bottom=0.1, 
                                top=0.9, wspace=0.35, hspace=0.15)
            axL = fig.add_subplot(gs[1, 0:2])
            axL1 = fig.add_subplot(gs[0, 0:2], sharex=axL)
            axR = fig.add_subplot(gs[1, 2:])
            axR1 = fig.add_subplot(gs[0, 2:], sharex=axR)
            axL.set_xlabel(r'$\log_{10}(1-e)_{\rm{SMBH}}$', fontsize = axlabel_size)
            axR.set_xlabel(r'$\log_{10}a_{\rm{SMBH}}$ [pc]', fontsize = axlabel_size)
            axL.set_ylabel(r'$\log_{10}$(CDF)', fontsize = axlabel_size)
            axL1.set_ylabel(r'$\rho/\rho_{\rm{max}}$', fontsize = axlabel_size)
            for int_ in range(4):
                axL, axL1 = orbital_plotter_setup(SMBH_ecc, int_, axL, axL1, c_hist, labels, True)
                axR, axR1 = orbital_plotter_setup(SMBH_sem, int_, axR, axR1, c_hist, labels, False)
            plt.savefig('figures/system_evolution/ecc_SMBH_cdf_histogram_GRX_'+time_string+'.pdf', dpi=300, bbox_inches='tight')
            plt.clf()

def lagrangian_tracker():
    def plotter_func(folder_str, pset_bool):
        """
        Function to plot average Lagrangian radii of the simulations whose population = pop_filt.
        
        Inputs:
        folder_str: Folder containing the data
        pset_bool:  Boolean 
                    (True  - calculate Lagrangians from particle set)
                    (False - extract Lagrangians from corresponding file)
        """

        dir = os.path.join('figures/steady_time/Sim_summary_'+str(folder_str)+'_GRX.txt')
        with open(dir) as f:
            line = f.readlines()
            popG = line[0][54:-2] 
            avgG = line[10][55:-2]
            popG_data = popG.split()
            avgG_data = avgG.split()
            popG = np.asarray([float(i) for i in popG_data])
            avgG = np.asarray([float(i) for i in avgG_data])

        data = natsort.natsorted(glob.glob('/media/erwanh/Elements/'+str(folder_str)+'/GRX/particle_trajectory/*'))
        chaotic = ['/media/erwanh/Elements/'+str(folder_str)+'/data/GRX/chaotic_simulation/'+str(i[59:]) for i in data]

        no_files = 60
        pop_checker = 0
        no_samples = 0
        pop_filt = 10
        fact = 1e4
        idx = np.where(popG[popG > 5]==pop_filt)

        no_runs  = np.zeros([1, 10**5])
        L25t_arr = np.zeros([1, 10**5])
        L50t_arr = np.zeros([1, 10**5])
        L75t_arr = np.zeros([1, 10**5])
        for file_ in range(len(chaotic)):
            with open(chaotic[file_], 'rb') as input_file:
                chaotic_tracker = pkl.load(input_file)
                pop = 5*round(0.2*chaotic_tracker.iloc[0][6])
                if pop==pop_filt: 
                    if (pset_bool):
                        with open(data[file_], 'rb') as input_file:
                            file_size = os.path.getsize(data[file_])
                            if file_size < 2.9e9:
                                print('Reading File', file_, ': ', input_file)
                                no_samples, process, pop_checker = no_file_tracker(pop_checker, pop, no_files, no_samples)
                                if (process):   
                                    ptracker = pkl.load(input_file)
                                    dt_len = int(min(np.round((avgG[idx])*1e3*fact), np.shape(ptracker)[1])-1)
                                    pset = Particles(pop-1, mass = 1e3 | units.MSun)
                                    for j in range(dt_len-1):
                                        simsnap = ptracker.iloc[:,j]
                                        for parti_ in range(pop):
                                            if parti_ != 0:
                                                psnap = simsnap.iloc[parti_]
                                                pset[parti_-1].x = psnap[2][0]
                                                pset[parti_-1].y = psnap[2][1]
                                                pset[parti_-1].z = psnap[2][2]
                                                pset[parti_-1].vx = psnap[3][0]
                                                pset[parti_-1].vy = psnap[3][1]
                                                pset[parti_-1].vz = psnap[3][2]
                                                    
                                        no_runs[0][j]  += 1
                                        L25t_arr[0][j] += LagrangianRadii(pset)[5].value_in(units.pc)
                                        L50t_arr[0][j] += LagrangianRadii(pset)[6].value_in(units.pc)
                                        L75t_arr[0][j] += LagrangianRadii(pset)[7].value_in(units.pc)
                    else:
                        lagdata = ['/media/erwanh/Elements/'+str(folder_str)+'/data/GRX/lagrangians/'+str(i[59:]) for i in data]
                        with open(lagdata[file_], 'rb') as input_file:
                            no_samples, process, pop_checker = no_file_tracker(pop_checker, pop, no_files, no_samples)
                            if (process):  
                                print('Reading File', file_, ': ', input_file)
                                ltracker = pkl.load(input_file)
                                dt_len = int(min(np.round((avgG[idx])*1e3), np.shape(ltracker)[0])-1)
                                for j in range(dt_len):
                                    no_runs[0][j]  += 1
                                    L25t_arr[0][j] += ltracker.iloc[j][0].value_in(units.pc)
                                    L50t_arr[0][j] += ltracker.iloc[j][1].value_in(units.pc)
                                    L75t_arr[0][j] += ltracker.iloc[j][2].value_in(units.pc)

        tim_arr = [ ] 
        L25_arr = [ ]
        L50_arr = [ ]
        L75_arr = [ ]

        time_iter = 0
        for i, j, k, l in zip (L25t_arr[0], L50t_arr[0], L75t_arr[0], no_runs[0]):
            time_iter += 1e-3
            if l > 0:
                L25_arr.append(i/l)
                L50_arr.append(j/l)
                L75_arr.append(k/l)
                tim_arr.append(time_iter)
                
        return tim_arr, L25_arr, L50_arr, L75_arr

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plot_ini = plotter_setup()
    axlabel_size, tick_size = plot_ini.font_size()

    folders = ['rc_0.25_4e6', 'rc_0.25_4e7']
    colours = ['red', 'blue', 'blueviolet']
    labelDat = [r'$M_{\mathrm{SMBH}} = 4\times10^{6}\ M_\odot$', 
                r'$M_{\mathrm{SMBH}} = 4\times10^{7}\ M_\odot$']
    pset = [True, False]

    fig, ax = plt.subplots()
    ax.set_ylabel(r'$r_{x}$ [pc]', fontsize=axlabel_size) 
    ax.set_xlabel(r'$t$ [Myr]', fontsize=axlabel_size) 
    iterf = 0
    smooth = 5000
    for fold_, bool_ in zip(folders, pset):
        tim_arr, L25_arr, L50_arr, L75_arr = plotter_func(fold_, bool_)
        tim_arr = moving_average(tim_arr, smooth)
        L25_arr = moving_average(L25_arr, smooth)
        L50_arr = moving_average(L50_arr, smooth)
        L75_arr = moving_average(L75_arr, smooth)
        
        ax.plot(tim_arr, L25_arr, color=colours[iterf+1], label=labelDat[iterf])
        ax.plot(tim_arr, L50_arr, color=colours[iterf+1], linestyle='-.')
        ax.plot(tim_arr, L75_arr, color=colours[iterf+1], linestyle = ':')
        
        iterf += 1
    ax.legend(prop={'size': axlabel_size})
    plot_ini.tickers(ax, 'plot')
    plt.savefig('figures/system_evolution/GRX_Lagrangians_Evolution.pdf', dpi=300, bbox_inches='tight')

def velocity_plotter():
    """
    Function to plot the velocity distribution after one relaxation time
    """
    
    def maxwell_boltzmann(vel, slope, sigma):
        return slope*(vel**2/sigma**3)*np.exp(-(vel)**2/(2*sigma**2))

    labels = ["Hermite", "GRX"]
    rad_vel = [[ ], [ ]]
    iter = 0
    for int_ in labels:
        ptracker = natsort.natsorted(glob.glob('/media/erwanh/Elements/rc_0.25_4e6/'+(int_)+'/particle_trajectory_c/*'))
        for f_ in ptracker:
            with open(f_, 'rb') as input_file:
                file_size = os.path.getsize(f_)
                if file_size < 2.9e9:
                    print('Reading:', f_)
                    ptracker = pkl.load(input_file)
                    simsnap = ptracker.iloc[:,1200]
                    for parti_ in range(np.shape(ptracker)[0]):
                        psnap = simsnap.iloc[parti_]
                        if parti_==0:
                            smbh_vx = psnap[3][0]
                            smbh_vy = psnap[3][1]
                            smbh_vz = psnap[3][2]
                        if parti_ != 0:
                            dvx = (psnap[3][0]-smbh_vx).value_in(units.kms)
                            dvy = (psnap[3][1]-smbh_vy).value_in(units.kms)
                            dvz = (psnap[3][2]-smbh_vz).value_in(units.kms)
                            dv = np.sqrt((dvx**2+dvy**2+dvz**2))+100
                            rad_vel[iter].append(dv)
        iter+=1
    
    c = colour_picker()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plot_ini = plotter_setup()
    axlabel_size, tick_size = plot_ini.font_size()

    fig, ax = plt.subplots()
    plot_ini.tickers(ax, 'plot') 
    ax.set_ylim(0,1.04)
    ax.set_xlim(0,1150)
    ax.set_xlabel(r"$|{v}_{\mathrm{SMBH}}|$ [km s$^{-1}$]", fontsize=tick_size)
    ax.set_ylabel(r"$\rho/\rho_{\mathrm{max}}$", fontsize=tick_size)
    for int_ in range(2):
        kdef_SMBH = sm.nonparametric.KDEUnivariate((rad_vel[int_]))
        kdef_SMBH.fit(kernel="gau")
        kdef_SMBH.density = (kdef_SMBH.density/max(kdef_SMBH.density))
        ax.plot(kdef_SMBH.support, kdef_SMBH.density, 
                color=c[int_], label=labels[int_])
        ax.fill_between(kdef_SMBH.support, kdef_SMBH.density, 
                        alpha=0.35, color=c[int_])
        xtemp = np.linspace(min(kdef_SMBH.support), max(kdef_SMBH.support), 1100)
        params, cv = scipy.optimize.curve_fit(maxwell_boltzmann, 
                                              kdef_SMBH.support, 
                                              kdef_SMBH.density,
                                              maxfev=70000,
                                              p0=[200,150])
        slope, beta = params
        print("Slope: ", slope, "Sigma: ", beta)
        slope_err, beta_err = np.sqrt(np.diag(cv))
        print("Slope Err: ", slope_err, "Sigma ERr: ", beta_err)
        curve = np.asarray([(maxwell_boltzmann(i, slope, beta-5)) for i in xtemp])
        curve *= max(kdef_SMBH.density)/max(curve)
        ax.plot(xtemp, curve, color=c[int_], ls=":", lw=3)
    ax.legend(prop={'size': axlabel_size})
    ax.tick_params(axis="y", which='both', labelsize=tick_size)
    ax.tick_params(axis="x", which='both', labelsize=tick_size)
    plt.savefig("figures/vel_disp.pdf", dpi=300, bbox_inches='tight')

def spatial_plotter():
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

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plot_ini = plotter_setup()
        axlabel_size, tick_size = plot_ini.font_size()

        if merger_idx==-4:
            outcome = 'merger'
            bool = chaos_data.iloc[0][merger_idx]
            idx = 0
        else:
            outcome = 'ejection'
            bool = chaos_data.iloc[0][merger_idx].number
        
        if bool > 0:
            with open(ptracker, 'rb') as input_file:
                file_size = os.path.getsize(ptracker)
                if file_size < 2e9:
                    print('Reading File ', file_, ' : ', input_file)
                    ptracker = pkl.load(input_file)

                    if np.shape(ptracker)[0] > 15:
                        dt_len = min(150, round(np.shape(ptracker)[1]))
                    else:
                        dt_len = 150

                    line_x = np.empty((len(ptracker), dt_len))
                    line_y = np.empty((len(ptracker), dt_len))
                    line_z = np.empty((len(ptracker), dt_len))
                    for i in range(len(ptracker)):
                        tptracker = ptracker.iloc[i]
                        for j in range(dt_len):
                            data_it = dt_len - j
                            coords = tptracker.iloc[-data_it][2]
                            line_x[i][j] = coords[0].value_in(units.pc)
                            line_y[i][j] = coords[1].value_in(units.pc)
                            line_z[i][j] = coords[2].value_in(units.pc)

                        if outcome=='merger':       
                            if i != 0 and math.isnan(line_x[i][-1]) or tptracker.iloc[-1][1].value_in(units.kg) > 1e36:
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

                        for j in range(len(line_z[i])):
                            if i != 0 and line_z[i][j] - line_z[0][j] >= 0:
                                plot_bool = True
                            elif i != 0 and np.sqrt((line_x[i][j] - line_x[0][j])**2 + (line_z[i][j] - line_z[0][j])**2 + (line_y[i][j] - line_y[0][j])**2) >= 0.125:
                                plot_bool = True
                            elif i==0:
                                plot_bool = True
                            else:
                                plot_bool = False
                            if (plot_bool):
                                if i==0:
                                    adapt_c = 'black'
                                    ax.scatter((line_x[i][j]-line_x[0][j]), (line_y[i][j]-line_y[0][j]), 
                                                c = adapt_c, zorder = 1, s = 400)
                                    
                                else:
                                    ax.scatter(line_x[i][j]-line_x[0][j], line_y[i][j]-line_y[0][j], 
                                                c = c[iter-2], s = 1, zorder = 1) 
                                    if outcome=='merger':
                                        if i==idx:
                                            ax.scatter(line_x[i][-2]-line_x[0][-2], line_y[i][-2]-line_y[0][-2], 
                                                    c = c[iter-2], edgecolors = 'black', s = 100, zorder = 3)
                                        else:
                                            ax.scatter(line_x[i][-2]-line_x[0][-2], line_y[i][-2]-line_y[0][-2], 
                                                    c = c[iter-2], edgecolors = 'black', s = 30, zorder = 3)
                                    else:
                                        ax.scatter(line_x[i][-1]-line_x[0][-1], line_y[i][-1]-line_y[0][-1], 
                                                    c = c[iter-2], edgecolors = 'black', s = 30, zorder = 3)
                        iter += 1
                    plot_ini.tickers(ax, 'plot')
                    fname = 'figures/system_evolution/Overall_System/simulation_evolution_pop_'+str(integrator)+str(len(ptracker))+'_'+str(file_)+'_'+outcome+'1.pdf'
                    plt.savefig(fname, dpi=300, bbox_inches='tight')
                    plt.clf()
                    plt.close()

    print('Spatial evolution plotter')
    bools = [True, False]
    integ = ['Hermite', 'GRX']

    for int_ in integ:
        ptracker_files = natsort.natsorted(glob.glob('/media/erwanh/Elements/rc_0.25_4e5/'+(int_)+'/particle_trajectory/*'))
        ctracker_files = natsort.natsorted(glob.glob('/media/erwanh/Elements/rc_0.25_4e5/data/'+str(int_)+'/chaotic_simulation/*'))
        for bool_ in bools:
            merger_bool = bool_
            if (merger_bool):
                outcome_idx = -4
            else:
                outcome_idx = 3

            for file_ in range(len(ptracker_files)):
                with open(ctracker_files[file_], 'rb') as input_file:
                    ctracker = pkl.load(input_file)
                    plotter_code(outcome_idx, ctracker, file_, int_)
            
    return