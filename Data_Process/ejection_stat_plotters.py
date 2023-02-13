from file_logistics import *
from amuse.ext.galactic_potentials import MWpotentialBovy2015

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import warnings

class ejection_stats(object):
    """
    Class which extracts EJECTED data files
    """
    def __init__(self):
        self.integrator = ['Hermite', 'GRX']
        warnings.filterwarnings("ignore", category=RuntimeWarning) 
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

    def new_data_extractor(self):
        """
        Function to extract newly simulated data into reduced manipulatable files
        """

        Hermite_data = glob.glob(os.path.join('/media/erwanh/Elements/Hermite/particle_trajectory_ejec_new/*'))
        chaotic_H = ['data/Hermite/no_addition/chaotic_simulation/'+str(i[60:]) for i in Hermite_data]
        GRX_data = glob.glob(os.path.join('/media/erwanh/Elements/GRX/particle_trajectory_ejec_new/*'))
        chaotic_G = ['data/GRX/no_addition/chaotic_simulation/'+str(i[56:]) for i in GRX_data]

        filest = [natsort.natsorted(Hermite_data), natsort.natsorted(GRX_data)] 
        filesc = [natsort.natsorted(chaotic_H), natsort.natsorted(chaotic_G)] 
        for int_ in range(2):
            for file_ in range(len(filest[int_])):
                with open(filesc[int_][file_], 'rb') as input_file:
                    print('Reading file: ', input_file)
                    ctracker = pkl.load(input_file)
                    if ctracker.iloc[0][5] > 0 and ctracker.iloc[0][6] > 5:
                        print(ctracker)
                        with open(filest[int_][file_], 'rb') as input_file:
                            print('Reading File :', input_file)
                            count = len(fnmatch.filter(os.listdir('data/ejection_stats/'), '*.*'))
                            ptracker = pkl.load(input_file)
                            print(ptracker)
                            print(ejected_extract_final(ptracker, ctracker))

                            ex, ey, ez, vesc, ejec_KE, ejec_PE, Nclose, ebool = ejected_extract_final(ptracker, ctracker)

                            path = '/home/erwanh/Desktop/SteadyStateBH/Data_Process/data/ejection_stats/'
                            stab_tracker = pd.DataFrame()
                            df_stabtime = pd.Series({'self.integrator': self.integrator[int_],
                                                     'Population': np.shape(ptracker)[0],
                                                     'Simulation Time': np.shape(ptracker)[1] * 10**-3,
                                                     'xPos': ex, 'yPos': ey, 'zPos': ez,
                                                     'vesc': vesc, 'KE': ejec_KE, 'PE': ejec_PE,
                                                     'Nclose': Nclose, 'Ejected Bool': ebool})
                            stab_tracker = stab_tracker.append(df_stabtime, ignore_index = True)
                            stab_tracker.to_pickle(os.path.join(path, 'IMBH_'+str(self.integrator[int_])+'_ejec_data_indiv_parti_'+str(count)+'.pkl'))

    def combine_data(self):
        """
        Function to extract meaningful results from compressed files
        """
        
        self.tot_pop = [[ ], [ ]]
        self.sim_time = [[ ], [ ]]
        self.ex = [[ ], [ ]]
        self.ey = [[ ], [ ]]
        self.ez = [[ ], [ ]]
        self.vesc = [[ ], [ ]]
        self.eKE = [[ ], [ ]]
        self.ePE = [[ ], [ ]]
        self.Nclose = [[ ], [ ]]
        self.ejecBool = [[ ], [ ]]

        ejec_data = natsort.natsorted(glob.glob('data/ejection_stats/*'))
        for file_ in range(len(ejec_data)):
            with open(ejec_data[file_], 'rb') as input_file:
                data_file = pkl.load(input_file)
                if data_file.iloc[0][0] == 'Hermite':
                    int_idx = 0
                else:
                    int_idx = 1

                self.tot_pop[int_idx].append(data_file.iloc[0][1])
                self.sim_time[int_idx].append(data_file.iloc[0][2])
                self.ex[int_idx].append(data_file.iloc[0][3])
                self.ey[int_idx].append(data_file.iloc[0][4])
                self.ez[int_idx].append(data_file.iloc[0][5])
                self.vesc[int_idx].append(data_file.iloc[0][6])
                self.eKE[int_idx].append(data_file.iloc[0][7])
                self.ePE[int_idx].append(data_file.iloc[0][8])
                self.Nclose[int_idx].append(data_file.iloc[0][9])
                self.ejecBool[int_idx].append(data_file.iloc[0][10])

        for int_ in range(2):
            self.ex[int_] = np.asarray(self.ex[int_], dtype = 'float')
            self.ey[int_] = np.asarray(self.ey[int_], dtype = 'float')
            self.ez[int_] = np.asarray(self.ez[int_], dtype = 'float')
            self.vesc[int_] = np.asarray(self.vesc[int_], dtype = 'float')
            self.eKE[int_] = np.asarray(self.eKE[int_], dtype = 'float')
            self.ePE[int_] = np.asarray(self.ePE[int_], dtype = 'float')

    def energy_plotters(self):
        """
        Function to plot KE vs. PE of the ejected particle
        """

        plot_ini = plotter_setup()
        MW_code = MWpotentialBovy2015()

        ejec_KEt = np.asarray(self.eKE)

        ymin = [min(self.ePE[0]/max(ejec_KEt[0])), min(self.ePE[1]/max(ejec_KEt[1]))]
        ymax = [max(self.ePE[0]/max(ejec_KEt[0])), max(self.ePE[1]/max(ejec_KEt[1]))]
        for int_ in range(1):
            ejec_KE = np.asarray(ejec_KEt[int_][np.isfinite(ejec_KEt[int_])])
            ejec_PE = np.asarray(self.ePE[int_][np.isfinite(ejec_KEt[int_])])
            ex = self.ex[int_][np.isfinite(self.ex[int_])]
            ey = self.ey[int_][np.isfinite(self.ey[int_])] 
            ez = self.ez[int_][np.isfinite(self.ez[int_])]

            ejected_KEb = np.asarray(ejec_KE[(ejec_KE < 1e46) & (abs(ejec_PE) < 1e46)])
            ejected_PEb = np.asarray(ejec_PE[(ejec_KE < 1e46) & (abs(ejec_PE) < 1e46)])
            final_posb = np.asarray([(i**2+j**2+z**2)**0.5 for i, j, z in zip(ex, ey, ez)])[(ejec_KE < 1e46) & (abs(ejec_PE) < 1e46)]
            final_posb[np.isnan(final_posb)] = 0
            ejected_KEb /= max(ejected_KEb)
            ejected_PEb /= max(ejected_KEb)

            ejected_KE = np.asarray([i/np.nanmax(ejec_KE) for i in ejec_KE])
            ejected_PE = np.asarray([i/np.nanmax(ejec_KE) for i in ejec_PE])
            final_pos = np.asarray([(i**2+j**2+z**2)**0.5 for i, j, z in zip(ex, ey, ez)])
            final_pos[np.isnan(final_pos)] = 0

            plt.figure(figsize=(13, 4))
            gs = gridspec.GridSpec(1, 2)
            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[0,1])
            for ax_ in [ax1, ax2]:
                ax_.set_ylabel(r'$E_P/E_{{K, {\rm{max}}}}$')
                ax_.set_xlabel(r'$E_K/E_{{K, {\rm{max}}}}$')
                ax_.set_xlim(0, 1.1)
                ax_.set_ylim(1.1*ymin[int_], 0.9*ymax[int_])
                plot_ini.tickers(ax_, 'plot')
                
            ax1.set_title(str(self.integrator[int_]))
            colour_axes = ax1.scatter(ejected_KE, ejected_PE, edgecolor = 'black', c=final_pos, s = 20, zorder = 2)
            cbar = plt.colorbar(colour_axes, ax=ax1, label = r'$r_{\rm{SMBH}}$ [pc]')

            bin2d_sim, xedge, yedge, img = ax2.hist2d(ejected_KE, ejected_PE, bins=(200, 200), range=([0,1.1],[1.1*ymin[int_], 0.9*ymax[int_]]))
            bin2d_sim /= np.max(bin2d_sim)
            extent = [0, 1.1, 1.1*ymin[int_], 0.9*ymax[int_]]
            contours = ax2.imshow(np.log10(bin2d_sim), extent = extent, aspect='auto', origin = 'upper')
            plt.savefig('figures/ejection_stats/KEPE_diagram_'+str(self.integrator[int_])+'_.pdf', dpi=500, bbox_inches='tight')

    def vejec_plotters(self):
        """
        Function to plot the ejection velocity
        """

        plot_ini = plotter_setup()
        MW_code = MWpotentialBovy2015()
        vesc_MW = (np.sqrt(2)*MW_code.circular_velocity(0.4 | units.parsec) + np.sqrt(2*constants.G*(4e6 | units.MSun)/(0.1 | units.parsec))).value_in(units.kms)

        ymin = []
        ymax = []
        vels = []
        
        avg_surv = [ ]
        avg_pos = [[ ], [ ]]
        for int_ in range(2):
            if int_ == 0:
                tot_pop = np.asarray([10*round(0.1*i) for i in self.tot_pop[int_]])
            else:
                tot_pop = np.asarray([5*round(i/5) for i in self.tot_pop[int_]])
                
            in_pop = np.unique(tot_pop)
            for pop_ in in_pop:
                idx = np.where((tot_pop == pop_))[0]
                avg_surv.append(np.nanmean(np.asarray(self.sim_time[int_])[idx]))
                posx = np.asarray(self.ex[int_])[idx]**2
                posy = np.asarray(self.ey[int_])[idx]**2
                posz = np.asarray(self.ez[int_])[idx]**2
                avg_pos[int_].append(np.nanmean(np.sqrt(posx+posy+posz)))

        norm_min = np.log10(min(avg_surv))#, min(avg_surv[1])))
        norm_max = np.log10(max(avg_surv))
        normalise = plt.Normalize(norm_min, norm_max)

        ax_iter = -1
        ax_title = ['Hermite', 'GRX']
        colours = ['red', 'blue']

        with open('figures/ejection_stats/output/ejec_stats.txt', 'w') as file:
            for int_ in range(2):
                ax_iter += 1
                if int_ == 0:
                    tot_pop = np.asarray([10*round(0.1*i) for i in self.tot_pop[int_]])
                else:
                    tot_pop = np.asarray([5*round(i/5) for i in self.tot_pop[int_]])

                sim_time = np.asarray(self.sim_time[int_])
                vesc = np.asarray(self.vesc[int_])

                in_pop = np.unique(tot_pop)
                avg_vesc = np.empty(len(in_pop))
                avg_surv = np.empty(len(in_pop))

                pops = [ ]
                samples = [ ]
                minvel = [ ]
                maxvel = [ ]
                MWvels = [ ]
                iter = -1

                for pop_ in in_pop:
                    iter += 1
                    pops.append(pop_)
                    idx = np.where((tot_pop == pop_))[0]
                    samples.append(len(vesc[idx]))
                    avg_vesc[iter] = np.nanmean(vesc[idx])
                    avg_surv[iter] = np.nanmean(sim_time[idx])
                    minvel.append(np.nanmin(vesc[idx]))
                    maxvel.append(np.nanmax(vesc[idx]))
                    MWvels.append(vesc[idx][vesc[idx] > 660])

                    ymin.append(min(avg_vesc))
                    ymax.append(max(avg_vesc))
                vels.append(max(vesc))
                pops = np.asarray(pops)
                avg_vesc = np.asarray(avg_vesc)

                file.write('\nData for '+str(self.integrator[int_]))
                file.write('\nPopulations counts                           ' + str(in_pop) + ' : ' + str(samples))
                file.write('\nPopulations average escape velocity          ' + str(in_pop) + ' : ' + str(avg_vesc) + ' kms')
                file.write('\nPopulations average escape location distance ' + str(in_pop) + ' : ' + str(avg_pos[int_]) + ' pc')
                file.write('\nPopulations min. escape velocity             ' + str(in_pop) + ' : ' + str(minvel) + ' kms')
                file.write('\nPopulations max. escape velocity             ' + str(in_pop) + ' : ' + str(maxvel) + ' kms')
                file.write('\nPopulations average escape time              ' + str(in_pop) + ' : ' + str(avg_surv) + ' Myr')
                file.write('\n========================================================================')

                fig = plt.figure(figsize=(5, 8))
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                
                if int_ == 0:
                    colour_axes = ax1.scatter(pops, avg_vesc, edgecolors='black', c = np.log10(avg_surv), norm = normalise, zorder = 3)
                else:
                    colour_axes = ax1.scatter(pops[pops <= 40], avg_vesc[pops <= 40], edgecolors='black', c = np.log10(avg_surv[pops <= 40]), norm = normalise, zorder = 3)
                

                n1, bins, patches = ax2.hist(vesc, 20)
                ax2.clear()
                n, bins, patches = ax2.hist(vesc, 20, histtype = 'step', color=colours[int_], weights=[1/n1.max()]*len(vesc))
                n, bins, patches = ax2.hist(vesc, 20, color=colours[int_], alpha = 0.4, weights=[1/n1.max()]*len(vesc))

                ax1.set_title(ax_title[int_])
                ax1.set_xlabel(r'$v_{ejec}$ [km s$^{-1}$]')
                ax2.set_ylabel(r'$\rho/\rho_{\rm{max}}$')
                ax1.set_ylabel(r'$\langle v_{\rm{ejec}} \rangle$ [km s$^{-1}$]')
                ax2.axvline(vesc_MW, linestyle = ':', color = 'black')
                ax2.text(655, 0.2, r'$v_{\rm{esc, MW}}$', rotation = 270)
                
                plot_ini.tickers_pop(ax1, self.tot_pop[int_], ax_title[int_])
                plot_ini.tickers(ax2, 'plot')

                cbar = plt.colorbar(colour_axes, ax=ax1, label = r'$\log_{10} \langle t_{\rm{ejec}}\rangle$ [Myr]')
                plt.savefig('figures/ejection_stats/vejection_'+str(ax_title[int_])+'.pdf', dpi = 300, bbox_inches='tight')

class event_tracker(object):
    """
    Class to take stats of ejection vs. mergers
    """
    def __init__(self):

        plot_ini = plotter_setup()
        chaos_data = glob.glob('data/Hermite/no_addition/chaotic_simulation/*')
        chaos_data_GRX = glob.glob('data/GRX/no_addition/chaotic_simulation/*')
        chaos_data = [natsort.natsorted(chaos_data), natsort.natsorted(chaos_data_GRX)]

        init_pop = [[ ], [ ]]
        merger = [[ ], [ ]]
        in_pop = [[ ], [ ]]
        frac_merge = [[ ], [ ]]
        colours = ['red', 'blue']
        labels = ['Hermite', 'GRX']

        fig = plt.figure(figsize=(5, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax = [ax1, ax2]
        for ax_ in ax:
            ax_.set_ylabel(r'$N_{\rm{merge}}/N_{\rm{sim}}$')
            ax_.set_ylim(0,1.05)

        for int_ in range(2):
            for file_ in range(len(chaos_data[int_])):
                with open(chaos_data[int_][file_], 'rb') as input_file:
                    data = pkl.load(input_file)
                    init_pop[int_].append(len(data.iloc[0][8])+1)
                    merger[int_].append(data.iloc[0][10])

            in_pop[int_] = np.unique(init_pop[int_])
            iter = -1
            for pop_ in in_pop[int_][(in_pop[int_] > 5)]:
                iter +=1
                indices = np.where((init_pop[int_] == pop_))[0]
                temp_frac = [merger[int_][i] for i in indices]
                frac_merge[int_].append(np.mean(temp_frac))
            if int_ == 0:
                ax[int_].scatter(in_pop[int_][(in_pop[int_] > 5)], frac_merge[int_], color = colours[int_], edgecolors = 'black')
            else:
                ax[int_].scatter(in_pop[int_][(in_pop[int_] > 5) & (in_pop[int_] < 45)], frac_merge[int_][:-1], color = colours[int_], edgecolors = 'black')
        plot_ini.tickers_pop(ax1, in_pop[0], labels[0])
        plot_ini.tickers_pop(ax2, in_pop[1], labels[1])

        plt.savefig('figures/ejection_stats/SMBH_merge_fraction.pdf', dpi=300, bbox_inches='tight')


cst = event_tracker()