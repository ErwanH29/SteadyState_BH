from file_logistics import *
from amuse.ext.galactic_potentials import MWpotentialBovy2015

import fnmatch
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import warnings

class ejection_stats(object):
    """
    Class which extracts data from simulations ending with ejections
    """

    def __init__(self):
        self.folders = ['rc_0.25_4e6', 'rc_0.25_4e7', 'rc_0.50_4e6', 'rc_0.50_4e7']
        self.colours = ['red', 'blue', 'deepskyblue', 'skyblue', 'slateblue']
        warnings.filterwarnings("ignore", category=RuntimeWarning) 
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

    def new_data_extractor(self):
        """
        Function compressing newly simulated data to more manipulatable files
        """

        iterf = 0
        for fold_ in self.folders:
            GRX_data = glob.glob(os.path.join('/media/erwanh/Elements/'+fold_+'/GRX/particle_trajectory/*'))
            chaotic_G = ['/media/erwanh/Elements/'+fold_+'/data/GRX/chaotic_simulation/'+str(i[59:]) for i in GRX_data]
            if iterf == 0:
                drange = 2
                Hermite_data = glob.glob(os.path.join('/media/erwanh/Elements/'+fold_+'/Hermite/particle_trajectory/*'))
                chaotic_H = ['/media/erwanh/Elements/'+fold_+'/data/Hermite/chaotic_simulation/'+str(i[63:]) for i in Hermite_data]
                filest = [natsort.natsorted(Hermite_data), natsort.natsorted(GRX_data)] 
                filesc = [natsort.natsorted(chaotic_H), natsort.natsorted(chaotic_G)]
                integrator = ['Hermite', 'GRX']
            else:
                drange = 1
                filest = [natsort.natsorted(GRX_data)] 
                filesc = [natsort.natsorted(chaotic_G)]
                integrator = ['GRX']

            for int_ in range(drange):
                for file_ in range(len(filest[int_])):
                    with open(filesc[int_][file_], 'rb') as input_file:
                        ctracker = pkl.load(input_file)
                        if ctracker.iloc[0][5] > 0 and ctracker.iloc[0][6] > 5:
                            with open(filest[int_][file_], 'rb') as input_file:
                                print('Ejection detected. Reading File :', input_file)
                                path = '/media/erwanh/Elements/'+fold_+'/data/ejection_stats/'
                                count = len(fnmatch.filter(os.listdir(path), '*.*'))
                                ptracker = pkl.load(input_file)
                                vesc = ejected_extract_final(ptracker, ctracker)

                                stab_tracker = pd.DataFrame()
                                df_stabtime = pd.Series({'Integrator': integrator[int_],
                                                        'Population': np.shape(ptracker)[0],
                                                        'Simulation Time': np.shape(ptracker)[1] * 10**-3,
                                                        'vesc': vesc})
                                stab_tracker = stab_tracker.append(df_stabtime, ignore_index = True)
                                stab_tracker.to_pickle(os.path.join(path, 'IMBH_'+str(integrator[int_])+'_ejec_data_indiv_parti_'+str(count)+'.pkl'))
            iterf += 1

    def combine_data(self, folder):
        """
        Function to extract data
        """
        
        if folder == self.folders[0]:
            self.tot_pop = [[ ], [ ]]
            self.sim_time = [[ ], [ ]]
            self.vesc = [[ ], [ ]]
            drange = 2
        else:
            self.tot_pop = [[ ]]
            self.sim_time = [[ ]]
            self.vesc = [[ ]]
            drange = 1

        ejec_data = natsort.natsorted(glob.glob('/media/erwanh/Elements/'+folder+'/data/ejection_stats/*'))
        for file_ in range(len(ejec_data)):
            with open(ejec_data[file_], 'rb') as input_file:
                data_file = pkl.load(input_file)
                if data_file.iloc[0][0] == 'Hermite' or folder != self.folders[0]:
                    int_idx = 0
                else:
                    int_idx = 1
                self.tot_pop[int_idx].append(data_file.iloc[0][1])
                self.sim_time[int_idx].append(data_file.iloc[0][2])
                self.vesc[int_idx].append(data_file.iloc[0][3])

        for int_ in range(drange):
            self.vesc[int_] = np.asarray(self.vesc[int_], dtype = 'float')

    def vejec_plotters(self):
        """
        Function to plot the ejection velocity
        """

        plot_ini = plotter_setup()
        MW_code = MWpotentialBovy2015()
        vesc_MW = (np.sqrt(2)*MW_code.circular_velocity(0.4 | units.parsec) + np.sqrt(2*constants.G*(4e6 | units.MSun)/(0.1 | units.parsec))).value_in(units.kms)
        configD = [r'$r_c = 0.25$ pc, $M_{\rm{SMBH}} = 4\times10^{6}M_{\odot}$', 
                   r'$r_c = 0.25$ pc, $M_{\rm{SMBH}} = 4\times10^{7}M_{\odot}$',
                   r'$r_c = 0.50$ pc, $M_{\rm{SMBH}} = 4\times10^{6}M_{\odot}$',
                   r'$r_c = 0.50$ pc, $M_{\rm{SMBH}} = 4\times10^{7}M_{\odot}$']

        iterf = 0
        for fold_ in self.folders:
            print('Plotting data for: ', fold_)
            self.combine_data(fold_)
            vels = []
            avg_surv = [ ]
            if iterf == 0:
                drange = 2
                integrator = ['Hermite', 'GRX']
                cfactor = 0
            else:
                drange = 1
                integrator = ['GRX']
                cfactor = 1

            for int_ in range(drange):
                tot_pop = np.asarray([5*round(i/5) for i in self.tot_pop[int_]]) 
                in_pop = np.unique(tot_pop)
                for pop_ in in_pop:
                    idx = np.where((tot_pop == pop_))[0]
                    avg_surv.append(np.nanmean(np.asarray(self.sim_time[int_])[idx]))

            norm_min = np.log10(min(avg_surv))
            norm_max = np.log10(max(avg_surv))
            normalise = plt.Normalize(norm_min, norm_max)

            with open('figures/ejection_stats/output/ejec_stats_'+fold_+'.txt', 'w') as file:
                for int_ in range(drange):
                    sim_time = np.asarray(self.sim_time[int_])
                    vesc = np.asarray(self.vesc[int_])

                    tot_pop = np.asarray([5*round(i/5) for i in self.tot_pop[int_]]) 
                    in_pop = np.unique(tot_pop)
                    avg_vesc = np.empty(len(in_pop))
                    avg_surv = np.empty(len(in_pop))

                    pops = [ ]
                    samples = [ ]
                    minvel = [ ]
                    maxvel = [ ]
                    iter = 0
                    for pop_ in in_pop:
                        pops.append(pop_)
                        idx = np.where(tot_pop == pop_)[0]
                        samples.append(len(vesc[idx]))
                        avg_vesc[iter] = np.nanmean(vesc[idx])
                        avg_surv[iter] = np.nanmean(sim_time[idx])
                        minvel.append(np.nanmin(vesc[idx]))
                        maxvel.append(np.nanmax(vesc[idx]))
                        iter += 1
                    vels.append(max(vesc))
                    pops = np.asarray(pops)
                    avg_vesc = np.asarray(avg_vesc)

                    file.write('\nData for '+str(integrator[int_]))
                    file.write('\nPopulations counts                           ' + str(in_pop) + ' : ' + str(samples))
                    file.write('\nPopulations average escape velocity          ' + str(in_pop) + ' : ' + str(avg_vesc) + ' kms')
                    file.write('\nPopulations min. escape velocity             ' + str(in_pop) + ' : ' + str(minvel) + ' kms')
                    file.write('\nPopulations max. escape velocity             ' + str(in_pop) + ' : ' + str(maxvel) + ' kms')
                    file.write('\nPopulations average escape time              ' + str(in_pop) + ' : ' + str(avg_surv) + ' Myr')
                    file.write('\n========================================================================')

                    fig = plt.figure(figsize=(5, 8))
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)
                    ax1.set_title(integrator[int_]+'\n'+configD[iterf])
                    ax1.set_xlabel(r'$v_{ejec}$ [km s$^{-1}$]')
                    ax2.set_ylabel(r'$\rho/\rho_{\rm{max}}$')
                    ax1.set_ylabel(r'$\langle v_{\rm{ejec}} \rangle$ [km s$^{-1}$]')
                    ax2.axvline(vesc_MW, linestyle = ':', color = 'black')
                    ax2.text(655, 0.2, r'$v_{\rm{esc, MW}}$', rotation = 270)

                    colour_axes = ax1.scatter(pops, avg_vesc, edgecolors='black', c = np.log10(avg_surv), norm = normalise, zorder = 3)
                    cbar = plt.colorbar(colour_axes, ax=ax1, label = r'$\log_{10} \langle t_{\rm{ejec}}\rangle$ [Myr]')
                    n1, bins, patches = ax2.hist(vesc, 20)
                    ax2.clear()
                    n, bins, patches = ax2.hist(vesc, 20, histtype = 'step', color=self.colours[int_+iterf+cfactor], weights=[1/n1.max()]*len(vesc))
                    n, bins, patches = ax2.hist(vesc, 20, color=self.colours[int_+iterf+cfactor], alpha = 0.4, weights=[1/n1.max()]*len(vesc))
                    
                    plot_ini.tickers_pop(ax1, self.tot_pop[int_], integrator[int_])
                    plot_ini.tickers(ax2, 'plot')
                    plt.savefig('figures/ejection_stats/vejection_'+fold_+'_'+str(integrator[int_])+'.pdf', dpi = 300, bbox_inches='tight')
            iterf += 1

class event_tracker(object):
    """
    Class plotting the fraction of simulation ending with mergers per population
    """

    def __init__(self):

        plot_ini = plotter_setup()
        colours = ['red', 'blue', 'deepskyblue', 'skyblue', 'slateblue']
        folders = ['rc_0.25_4e6', 'rc_0.25_4e7', 'rc_0.50_4e6', 'rc_0.50_4e7']
        labelsI = ['Hermite', 'GRX']
        labelsD = [r'$r_c = 0.25$ pc, $M_{\rm{SMBH}} = 4\times10^{6}M_{\odot}$', 
                   r'$r_c = 0.25$ pc, $M_{\rm{SMBH}} = 4\times10^{7}M_{\odot}$',
                   r'$r_c = 0.50$ pc, $M_{\rm{SMBH}} = 4\times10^{6}M_{\odot}$',
                   r'$r_c = 0.50$ pc, $M_{\rm{SMBH}} = 4\times10^{7}M_{\odot}$']

        chaos_data = glob.glob('/media/erwanh/Elements/rc_0.25_4e6/data/Hermite/chaotic_simulation/*')
        chaos_data_GRX = glob.glob('/media/erwanh/Elements/rc_0.25_4e6/data/GRX/chaotic_simulation/*')
        chaos_data = [natsort.natsorted(chaos_data), natsort.natsorted(chaos_data_GRX)]

        init_pop = [[ ], [ ]]
        merger = [[ ], [ ]]
        in_pop = [[ ], [ ]]
        frac_merge = [[ ], [ ]]

        fig, ax = plt.subplots()
        ax.set_title(r'$r_c = 0.25$ pc, $M_{\rm{SMBH}} = 4\times10^{6} M_{\odot}$')
        ax.set_ylabel(r'$N_{\rm{merge}}/N_{\rm{sim}}$')
        ax.set_ylim(0, 1.05)
        for int_ in range(2):
            for file_ in range(len(chaos_data[int_])):
                with open(chaos_data[int_][file_], 'rb') as input_file:
                    data = pkl.load(input_file)
                    init_pop[int_].append(len(data.iloc[0][8])+1)
                    merger[int_].append(data.iloc[0][10])

            in_pop[int_] = np.unique(init_pop[int_])
            for pop_ in in_pop[int_][(in_pop[int_] > 5)]:
                indices = np.where((init_pop[int_] == pop_))[0]
                temp_frac = [merger[int_][i] for i in indices]
                frac_merge[int_].append(np.mean(temp_frac))
            ax.scatter(in_pop[int_][(in_pop[int_] > 5)], frac_merge[int_], color = colours[int_], edgecolors = 'black')
        plot_ini.tickers_pop(ax, in_pop[0], labelsI[0])
        plt.savefig('figures/ejection_stats/SMBH_merge_fraction_HermGRX.pdf', dpi=300, bbox_inches='tight')

        fig, ax = plt.subplots()
        ax.set_ylabel(r'$N_{\rm{merge}}/N_{\rm{sim}}$')
        ax.set_ylim(0,1.05)
        init_pop = [[ ], [ ], [ ], [ ]]
        merger = [[ ], [ ], [ ], [ ]]
        in_pop = [[ ], [ ], [ ], [ ]]
        frac_merge = [[ ], [ ], [ ], [ ]]

        iterf = 0
        for fold_ in folders:
            chaos_data_GRX = glob.glob('/media/erwanh/Elements/'+fold_+'/data/GRX/chaotic_simulation/*')
            chaos_data = [natsort.natsorted(chaos_data_GRX)]

            for file_ in range(len(chaos_data[0])):
                with open(chaos_data[0][file_], 'rb') as input_file:
                    data = pkl.load(input_file)
                    init_pop[iterf].append(len(data.iloc[0][8])+1)
                    merger[iterf].append(data.iloc[0][10])

            in_pop[iterf] = np.unique(init_pop[iterf])
            for pop_ in in_pop[iterf][(in_pop[iterf] > 5)]:
                indices = np.where((init_pop[iterf] == pop_))[0]
                temp_frac = [merger[iterf][i] for i in indices]
                frac_merge[iterf].append(np.mean(temp_frac))
            ax.scatter(in_pop[iterf][(in_pop[iterf] > 5)], frac_merge[iterf], color = colours[iterf+1], label = labelsD[iterf], edgecolors = 'black')
            iterf += 1
        ax.legend()
        plot_ini.tickers_pop(ax, in_pop[1], labelsI[1])
        plt.savefig('figures/ejection_stats/SMBH_merge_fraction_GRX_All.pdf', dpi=300, bbox_inches='tight')