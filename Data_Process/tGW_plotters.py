from amuse.lab import *
from file_logistics import *
from scipy.special import jv 

import fnmatch
import LISA_Curves.LISA as li
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import statsmodels.api as sm
import warnings

class gw_calcs(object):
    """
    Class which forms plots of the sustainable hierarchical and binaries 
    along with final time-step binary/hierarchical data.
    Sustainable condition: If same partner for iter > 5, or roughly 5000 years
    """

    def __init__(self):
        """
        Extracts the required data
        """

        np.seterr(divide='ignore')
        warnings.filterwarnings("ignore", category=RuntimeWarning) 
        warnings.filterwarnings("ignore", category=UserWarning) 
        self.H0 = 67.4 | units.kms / units.Mpc #Taken from arXiv:1807.06209 in km/s/Mpc
        self.tH = self.H0**-1 | units.Myr
        self.integrator = ['Hermite', 'GRX']
        self.folders = ['rc_0.25_4e6', 'rc_0.25_4e7', 'rc_0.5_4e6', 'rc_0.5_4e7']
        self.colors = ['red', 'blue', 'lightcoral', 'cornflowerblue']
        self.distance = [r'$\langle r_c = 0.25 \rangle$', r'$\langle r_c = 0.50 \rangle$']
        
    def new_data_extractor(self):
        """
        Script to extract data from recently simulated runs
        """

        print('!!!!!! WARNING THIS WILL TAKE A WHILE !!!!!!!')

        iterf = 0
        for fold_ in self.folders:
            tcropH = 59 - iterf
            tcropG = 55 - iterf

            GRX_data = glob.glob(os.path.join('/media/erwanh/Elements/'+fold_+'/GRX/particle_trajectory/*'))
            chaoticG = ['/media/erwanh/Elements/'+fold_+'/data/GRX/chaotic_simulation/'+str(i[tcropG:]) for i in GRX_data]

            if iterf == 0:
                drange = 2
                Hermite_data = glob.glob(os.path.join('/media/erwanh/Elements/'+fold_+'/Hermite/particle_trajectory/*'))
                chaoticH = ['/media/erwanh/Elements/'+fold_+'/data/Hermite/chaotic_simulation/'+str(i[tcropH:]) for i in Hermite_data]
                filename = [natsort.natsorted(Hermite_data), natsort.natsorted(GRX_data)] 
                filenameC = [natsort.natsorted(chaoticH), natsort.natsorted(chaoticG)]
                integrator = ['Hermite', 'GRX']
            else:
                drange = 1
                filename = [natsort.natsorted(GRX_data)] 
                filenameC = [natsort.natsorted(chaoticG)]
                integrator = ['GRX']

            for int_ in range(drange):
                for file_ in range(len(filename[int_])):
                    with open(filenameC[int_][file_], 'rb') as input_file:
                        chaotic_tracker = pkl.load(input_file)
                        if chaotic_tracker.iloc[0][6] <= 40 and chaotic_tracker.iloc[0][6] > 5:
                            with open(filename[int_][file_], 'rb') as input_file:
                                print('Reading file', file_, ': ', filename[int_][file_])
                                data = pkl.load(input_file)
                                sim_time = np.shape(data)[1]-1
                                SMBH_sys_mass = data.iloc[0][0][1]

                                for parti_ in range(np.shape(data)[0]):
                                    count = len(fnmatch.filter(os.listdir('/media/erwanh/Elements/'+fold_+'/data/tGW/'), '*.*'))
                                    mass1 = data.iloc[parti_][0][1]
                                    tot_mass = SMBH_sys_mass + mass1
                                    if not (isinstance(data.iloc[parti_][-1][0], np.uint64)) or data.iloc[parti_][-1][1] == tot_mass:
                                        merge_Bool = 1
                                    else: 
                                        merge_Bool = 0

                                    mass_IMBH = []
                                    semi_SMBH_GW_indiv = []
                                    semi_NN_GW_indiv = []
                                    semi_t_GW_indiv = []
                                    ecc_SMBH_GW_indiv = []
                                    ecc_NN_GW_indiv = []
                                    ecc_t_GW_indiv = []
                                    nharm_SMBH_GW_indiv = []
                                    nharm_NN_GW_indiv = []
                                    nharm_t_GW_indiv = []

                                    Nclose_indiv = 0
                                    Nfbnn_indiv = 0
                                    Nfbt_indiv = 0

                                    freq_SMBH_GW_indiv = [-5]
                                    freq_NN_GW_indiv = [-5]
                                    freq_t_GW_indiv = [-5]
                                    strain_SMBH_GW_indiv = [-5]
                                    strain_NN_GW_indiv = [-5]
                                    strain_t_GW_indiv = [-5]
                                    time_SMBH_GW_indiv = [-5]
                                    time_NN_GW_indiv = [-5]
                                    time_t_GW_indiv = [-5]
                                    SMBH_NN_event = [-5]
                                    SMBH_t_event = [-5]

                                    if parti_ != 0:
                                        for col_ in range(np.shape(data)[1]-1):
                                            sem_SMBH = abs(data.iloc[parti_][col_][7][0])
                                            ecc_SMBH = data.iloc[parti_][col_][8][0]

                                            strain_SMBH = self.gw_strain(sem_SMBH, ecc_SMBH, mass1, SMBH_sys_mass)
                                            freq_SMBH = self.gw_freq(sem_SMBH, ecc_SMBH, mass1, SMBH_sys_mass)
                                            nharm_SMBH = self.gw_harmonic_mode(ecc_SMBH)
                                            if freq_SMBH > 10**-12:
                                                strain_SMBH_GW_indiv.append(strain_SMBH)
                                                freq_SMBH_GW_indiv.append(freq_SMBH)
                                                nharm_SMBH_GW_indiv.append(nharm_SMBH)
                                                time_SMBH_GW_indiv.append(10**-3 * col_)
                                                semi_SMBH_GW_indiv.append(sem_SMBH)
                                                ecc_SMBH_GW_indiv.append(ecc_SMBH)
                                            
                                            semi_major_nn = abs(data.iloc[parti_][col_][7][1])
                                            semi_major_t = abs(data.iloc[parti_][col_][7][2])
                                            ecc_nn = (data.iloc[parti_][col_][8][1])
                                            ecc_t = (data.iloc[parti_][col_][8][2])
                                            for part_ in range(np.shape(data)[0]):
                                                if data.iloc[part_][0][0] == data.iloc[parti_][col_][6][1]:
                                                    mass2 = data.iloc[part_][0][1]

                                            strain_nn = self.gw_strain(semi_major_nn, ecc_nn, mass1, mass2)
                                            freq_nn = self.gw_freq(semi_major_nn, ecc_nn, mass1, mass2)
                                            nharm_nn = self.gw_harmonic_mode(ecc_nn)
                                            if freq_nn > 10**-12:
                                                strain_NN_GW_indiv.append(strain_nn)
                                                freq_NN_GW_indiv.append(freq_nn)
                                                nharm_NN_GW_indiv.append(nharm_nn)
                                                time_NN_GW_indiv.append(10**-3 * col_)
                                                semi_NN_GW_indiv.append(semi_major_nn)
                                                ecc_NN_GW_indiv.append(ecc_nn)

                                                linex = data.iloc[parti_][col_][2][0] - data.iloc[0][col_][2][0]
                                                liney = data.iloc[parti_][col_][2][1] - data.iloc[0][col_][2][1]
                                                linez = data.iloc[parti_][col_][2][2] - data.iloc[0][col_][2][2]
                                                dist_SMBH = (linex**2+liney**2+linez**2).sqrt()
                                                dist_NN = data.iloc[parti_][col_][-1]

                                                if dist_SMBH.value_in(units.pc) == dist_NN or mass2 > 10**5 | units.MSun:
                                                    Nfbnn_indiv += 1
                                                    SMBH_NN_event.append(1)
                                                else:
                                                    Nfbnn_indiv += 0.5
                                                    SMBH_NN_event.append(-5)
                                                    
                                            tSMBH = False
                                            if sem_SMBH == semi_major_t or ecc_SMBH == ecc_t:
                                                mass2 = SMBH_sys_mass
                                                tSMBH = True

                                            strain_t = self.gw_strain(semi_major_t, ecc_t, mass1, mass2)
                                            freq_t = self.gw_freq(semi_major_t, ecc_t, mass1, mass2)
                                            nharm_t = self.gw_harmonic_mode(ecc_t)
                                            if freq_t > 10**-12:
                                                strain_t_GW_indiv.append(strain_t)
                                                freq_t_GW_indiv.append(freq_t)
                                                nharm_t_GW_indiv.append(nharm_t)
                                                time_t_GW_indiv.append(10**-3 * col_) 
                                                semi_t_GW_indiv.append(semi_major_t)
                                                ecc_t_GW_indiv.append(ecc_t)    

                                                if (tSMBH):
                                                    Nfbt_indiv += 1
                                                    SMBH_t_event.append(1)
                                                else:
                                                    Nfbt_indiv += 0.5
                                                    SMBH_t_event.append(-5)

                                            if data.iloc[parti_][col_][-1] < 5e-3:
                                                Nclose_indiv += 1

                                        mass_IMBH = np.asarray(mass_IMBH)
                                        Ntot_indiv = Nfbnn_indiv + Nfbt_indiv

                                        path = '/media/erwanh/Elements/'+fold_+'/data/tGW/'
                                        stab_tracker = pd.DataFrame()
                                        df_stabtime = pd.Series({'Integrator': integrator[int_],
                                                                'Simulation Time': 10**3*sim_time,
                                                                'Population': 10*round(0.1*np.shape(data)[0]),
                                                                'mass SMBH': SMBH_sys_mass,
                                                                'mass IMBH1': mass1,
                                                                'mass IMBH2': mass2,
                                                                'No. Binary Events': Nfbnn_indiv,
                                                                'No. Tertiary Events': Nfbt_indiv,
                                                                'No. Total Events': Ntot_indiv,
                                                                'No. Close Encounter': Nclose_indiv,
                                                                'SMBH Binary Frequencies': freq_SMBH_GW_indiv,
                                                                'SMBH Binary Strain': strain_SMBH_GW_indiv,
                                                                'SMBH Binary Time': time_SMBH_GW_indiv,
                                                                'SMBH Binary Semi major': semi_SMBH_GW_indiv,
                                                                'SMBH Binary Eccentricity': ecc_SMBH_GW_indiv,
                                                                'FlyBy Binary Frequencies': freq_NN_GW_indiv,
                                                                'Flyby Binary Strain': strain_NN_GW_indiv,
                                                                'Flyby Binary Time': time_NN_GW_indiv,
                                                                'Flyby Binary Semi major': semi_NN_GW_indiv,
                                                                'Flyby Binary Eccentricity': ecc_NN_GW_indiv,
                                                                'Flyby SMBH Event': SMBH_NN_event,                 #Boolean (1 = SMBH events, -5 = IMBH-IMBH)
                                                                'Flyby Tertiary Frequencies': freq_t_GW_indiv,
                                                                'Flyby Tertiary Strain': strain_t_GW_indiv,
                                                                'Flyby Tertiary Time': time_t_GW_indiv,
                                                                'Flyby Tertiary Semi major': semi_t_GW_indiv,
                                                                'Flyby Tertiary Eccentricity': ecc_t_GW_indiv,
                                                                'Tertiary SMBH Event': SMBH_t_event,
                                                                'Merger Boolean': merge_Bool})
                                        stab_tracker = stab_tracker.append(df_stabtime, ignore_index = True)
                                        stab_tracker.to_pickle(os.path.join(path, 'IMBH_'+str(integrator[int_])+'_tGW_data_indiv_parti_'+str(count)+'_'+str(parti_)+'_local1.pkl'))

            iterf += 1

    def combine_data(self, filter, integ, folder):
        """
        Function which extracts ALL data.

        Inputs:
        filter:       Delineate results based on population or integrator used.
        integ:        The integrator the data is for
        folder:       The directory which the data resides in (based on cluster radius)
        """

        self.sim_time = [ ]
        self.pop = [ ]
        self.close_enc = [ ]
        self.mass_SMBH = [ ]
        self.mass_parti = [ ]
        self.mass_IMBH = [ ]

        self.freq_flyby_SMBH = [ ]
        self.strain_flyby_SMBH = [ ]
        self.time_flyby_SMBH = [ ]
        self.semi_flyby_SMBH = [ ]
        self.ecc_flyby_SMBH = [ ]

        self.freq_flyby_nn = [ ]
        self.strain_flyby_nn = [ ]
        self.time_flyby_nn = [ ]
        self.semi_flyby_nn = [ ]
        self.ecc_flyby_nn = [ ]

        self.freq_flyby_t = [ ]
        self.strain_flyby_t = [ ]
        self.time_flyby_t = [ ]
        self.semi_flyby_t = [ ]
        self.ecc_flyby_t = [ ]

        self.tot_sim_time = [ ]
        self.tot_events = [ ]
        self.fb_nn_events = [ ]
        self.fb_nn_SMBH = [ ]
        self.fb_t_events = [ ]
        self.fb_t_SMBH = [ ]

        if integ == 'Hermite':
            int_idx = 0
        else:
            int_idx = 1

        print('Extracting data for ', self.integrator[int_idx], ' and distance ', folder)
        tGW_data = natsort.natsorted(glob.glob('/media/erwanh/Elements/'+folder+'/data/tGW/*'))
        if filter == 'pop_filt':
            for file_ in range(len(tGW_data)):
                with open(tGW_data[file_], 'rb') as input_file:
                    data_file = pkl.load(input_file)
                    if data_file.iloc[0][0] == self.integrator[int_idx]:
                        print('TRue')
                        self.sim_time.append(data_file.iloc[0][1])
                        self.pop.append(int(data_file.iloc[0][2]))
                        self.mass_SMBH.append(data_file.iloc[0][3])
                        self.mass_parti.append(data_file.iloc[0][4])
                        self.mass_IMBH.append(data_file.iloc[0][5])
                        self.fb_nn_events.append(data_file.iloc[0][6])
                        self.fb_t_events.append(data_file.iloc[0][7])
                        self.tot_events.append(data_file.iloc[0][8])
                        self.close_enc.append(data_file.iloc[0][9])

                        self.freq_flyby_SMBH.append(data_file.iloc[0][10])
                        self.strain_flyby_SMBH.append(data_file.iloc[0][11])
                        self.time_flyby_SMBH.append(data_file.iloc[0][12])
                        self.semi_flyby_SMBH.append(data_file.iloc[0][13])
                        self.ecc_flyby_SMBH.append(data_file.iloc[0][14])

                        self.freq_flyby_nn.append(data_file.iloc[0][15])
                        self.strain_flyby_nn.append(data_file.iloc[0][16])
                        self.time_flyby_nn.append(data_file.iloc[0][17])
                        self.semi_flyby_nn.append(data_file.iloc[0][18])
                        self.ecc_flyby_nn.append(data_file.iloc[0][19])
                        self.fb_nn_SMBH.append(data_file.iloc[0][20])

                        self.freq_flyby_t.append(data_file.iloc[0][21])
                        self.strain_flyby_t.append(data_file.iloc[0][22])
                        self.time_flyby_t.append(data_file.iloc[0][23])
                        self.semi_flyby_t.append(data_file.iloc[0][24])
                        self.ecc_flyby_t.append(data_file.iloc[0][25])
                        self.fb_t_SMBH.append(data_file.iloc[0][26])

        if filter == 'integrator':
            for file_ in range(len(tGW_data)):
                with open(tGW_data[file_], 'rb') as input_file:
                    data_file = pkl.load(input_file)
                    if data_file.iloc[0][0] == integ:
                        self.sim_time.append(data_file.iloc[0][1])
                        self.pop.append(int(data_file.iloc[0][2]))
                        self.mass_SMBH.append(data_file.iloc[0][3])
                        self.mass_parti.append(data_file.iloc[0][4])
                        self.mass_IMBH.append(data_file.iloc[0][5])
                        self.fb_nn_events.append(data_file.iloc[0][6])
                        self.fb_t_events.append(data_file.iloc[0][7])
                        self.tot_events.append(data_file.iloc[0][8])
                        self.close_enc.append(data_file.iloc[0][9])

                        self.freq_flyby_SMBH.append(data_file.iloc[0][10])
                        self.strain_flyby_SMBH.append(data_file.iloc[0][11])
                        self.time_flyby_SMBH.append(data_file.iloc[0][12])
                        self.semi_flyby_SMBH.append(data_file.iloc[0][13])
                        self.ecc_flyby_SMBH.append(data_file.iloc[0][14])

                        self.freq_flyby_nn.append(data_file.iloc[0][15])
                        self.strain_flyby_nn.append(data_file.iloc[0][16])
                        self.time_flyby_nn.append(data_file.iloc[0][17])
                        self.semi_flyby_nn.append(data_file.iloc[0][18])
                        self.ecc_flyby_nn.append(data_file.iloc[0][19])
                        self.fb_nn_SMBH.append(data_file.iloc[0][20])

                        self.freq_flyby_t.append(data_file.iloc[0][21])
                        self.strain_flyby_t.append(data_file.iloc[0][22])
                        self.time_flyby_t.append(data_file.iloc[0][23])
                        self.semi_flyby_t.append(data_file.iloc[0][24])
                        self.ecc_flyby_t.append(data_file.iloc[0][25])
                        self.fb_t_SMBH.append(data_file.iloc[0][26])

        self.sim_time = np.asarray(self.sim_time, dtype = 'object')
        self.pop = np.asarray(self.pop, dtype = 'object')
        self.mass_SMBH = np.asarray(self.mass_SMBH, dtype = 'object')
        self.mass_parti = np.asarray(self.mass_parti, dtype = 'object')
        self.mass_IMBH = np.asarray(self.mass_IMBH, dtype = 'object')
        self.fb_nn_events = np.asarray(self.fb_nn_events, dtype = 'object')
        self.fb_t_events = np.asarray(self.fb_t_events, dtype = 'object')
        self.tot_events = np.asarray(self.tot_events, dtype = 'object')
        self.close_enc = np.asarray(self.close_enc, dtype = 'object')
        self.freq_flyby_SMBH = np.asarray(self.freq_flyby_SMBH, dtype = 'object')
        self.strain_flyby_SMBH = np.asarray(self.strain_flyby_SMBH, dtype = 'object')
        self.time_flyby_SMBH = np.asarray(self.time_flyby_SMBH, dtype = 'object')
        self.semi_flyby_SMBH = np.asarray(self.semi_flyby_SMBH, dtype = 'object')
        self.ecc_flyby_SMBH = np.asarray(self.ecc_flyby_SMBH, dtype = 'object')
        self.freq_flyby_nn = np.asarray(self.freq_flyby_nn, dtype = 'object')
        self.strain_flyby_nn = np.asarray(self.strain_flyby_nn, dtype = 'object')
        self.time_flyby_nn = np.asarray(self.time_flyby_nn, dtype = 'object')
        self.semi_flyby_nn = np.asarray(self.semi_flyby_nn, dtype = 'object')
        self.ecc_flyby_nn = np.asarray(self.ecc_flyby_nn, dtype = 'object')
        self.fb_nn_SMBH = np.asarray(self.fb_nn_SMBH, dtype = 'object')
        self.freq_flyby_t = np.asarray(self.freq_flyby_t, dtype = 'object')
        self.strain_flyby_t = np.asarray(self.strain_flyby_t, dtype = 'object')
        self.time_flyby_t = np.asarray(self.time_flyby_t, dtype = 'object')
        self.semi_flyby_t = np.asarray(self.semi_flyby_t, dtype = 'object')
        self.ecc_flyby_t = np.asarray(self.ecc_flyby_t, dtype = 'object')
        self.fb_t_SMBH = np.asarray(self.fb_t_SMBH, dtype = 'object')

    def coll_radius(self, mass_arr):
        return 3 * (2*constants.G*mass_arr)/(constants.c**2)

    def forecast_interferometer(self, ax, m1, m2):
        """
        Function to plot the LISA and muAres frequency range in Ge a vs. (1-e) plots
        """
        
        ecc_range = np.linspace(0.0001, (1-10**-8), 50)

        self.Ares_semimaj_max = self.gw_cfreq_semi(ecc_range[1:], 1 | units.Hz, m1, m2)
        self.Ares_semimaj_min = self.gw_cfreq_semi(ecc_range[1:], 1e-7 | units.Hz, m1, m2)
        self.Ares_semimaj = self.gw_cfreq_semi(ecc_range[1:], 1e-3 | units.Hz, m1, m2)

        self.LISA_semimaj_max = self.gw_cfreq_semi(ecc_range[1:], 1 | units.Hz, m1, m2)
        self.LISA_semimaj_min = self.gw_cfreq_semi(ecc_range[1:], 1e-5 | units.Hz, m1, m2)
        self.LISA_semimaj = self.gw_cfreq_semi(ecc_range[1:], 1e-2 | units.Hz, m1, m2)

        ecc_range = [np.log(1-i) for i in ecc_range[1:]]
        self.text_angle = np.degrees(np.arctan((ecc_range[30]-ecc_range[20])/(self.Ares_semimaj[30]-self.Ares_semimaj[20])))

        ax.plot(self.Ares_semimaj_min, ecc_range, linestyle = ':', color = 'white', zorder = 2)
        ax.plot(self.Ares_semimaj, ecc_range, linestyle = '-.', color = 'white', zorder = 3)
        ax.plot(self.Ares_semimaj_max, ecc_range, linestyle = ':', color = 'white', zorder = 4)
        ax.fill_between(np.append(self.Ares_semimaj_min, self.Ares_semimaj_max[::-1]), 
                        np.append(ecc_range[:], ecc_range[::-1]), alpha = 0.6, color = 'black', zorder = 1)
        ax.plot(self.LISA_semimaj_min, ecc_range, linestyle = ':', color = 'white', zorder = 2)
        ax.plot(self.LISA_semimaj, ecc_range, linestyle = '-.', color = 'white', zorder = 3)
        ax.plot(self.LISA_semimaj_max, ecc_range, linestyle = ':', color = 'white', zorder = 4)
        ax.fill_between(np.append(self.LISA_semimaj_min, self.LISA_semimaj_max[::-1]), 
                        np.append(ecc_range[:], ecc_range[::-1]), alpha = 0.6, color = 'black', zorder = 1)

        return ax

    def gfunc(self, ecc):
        nharm = self.gw_harmonic_mode(ecc)
        return nharm**4/32 * ((jv(nharm-2, nharm*ecc)-2*ecc*jv(nharm-1, nharm*ecc) + 2/nharm * jv(nharm, nharm*ecc) + 2*ecc*jv(nharm+1, nharm*ecc) - jv(nharm+2, nharm*ecc))**2 + (1-ecc**2)*(jv(nharm-2, nharm*ecc) - 2*jv(nharm, nharm*ecc) + jv(nharm+2, nharm*ecc))**2 + 4/(3*nharm**2)*(jv(nharm, nharm*ecc)**2))
    
    def gw_cfreq_semi(self, ecc_arr, freq_val, m1, m2):
        """
        Function to get constant frequency curves based on eqn. 43 of Samsing et al. 2014.
        Frequency values are based on Samsing et al. 2014 and correspond to LIGO (200 Hz) with range 10 < f < 10 000 [https://link.aps.org/doi/10.1103/PhysRevD.93.112004]
        LISA (1e-2 Hz) peak sensitivity with range 1e-4 < f < 1 [https://lisa.nasa.gov/]
        
        Inputs:
        ecc_arr:  The eccentricity array
        freq_val: The constant frequency wishing to plot
        m1/m2:    The binary mass
        """
        
        term1 = np.sqrt(constants.G*(m1+m2))/np.pi
        semi_maj = [np.log10(((term1 * (1+i)**1.1954/(1-i**2)**1.5 * freq_val**-1)**(2/3)).value_in(units.pc)) for i in ecc_arr]
        return semi_maj

    def gw_freq(self, semi, ecc, m1, m2):
        """
        Frequency equation is based on Samsing et al. 2014 eqn (43). 
        
        Inputs:
        semi:   The semi-major axes of the system
        ecc:    The eccentricity of the binary system
        m1/m2:  The individual mass components of the binary system
        """

        nharm = self.gw_harmonic_mode(ecc)
        freq =  (2*np.pi)**-1*np.sqrt(constants.G*(m1+m2)/abs(semi)**3) * nharm
        return freq.value_in(units.Hz)

    def gw_harmonic_mode(self, ecc):
        """
        Finding the peak harmonic of gravitational frequency for a given eccentric orbit.
        Equation 36 of Wen (2003)
        """

        nharm = 2*(1+ecc)**1.1954/(1-ecc**2)**1.5
        return nharm

    def gw_dfreq(self, ecc, m1, m2, semi, chirp_mass, ecc_func):
        """
        Function to take into account the limited LISA observation time, T ~ 5yrs
        Based on equation (6) of Kremer et al. 2019.
        The redshift is calculated from Ned Wright's calculator with data from arXiv:1807.06209
        (omegaM = 0.315, omegaL = 0.685, H0 = 67.4 km/s/Mpc)
        """

        redshift = 0.1972
        nharm = self.gw_harmonic_mode(ecc)
        forb = np.sqrt(constants.G*(m1+m2))/(2*np.pi) * abs(semi)**-1.5 * (redshift+1)**-1

        dfreq = (96*nharm)/(10*np.pi)*(constants.G*chirp_mass)**(5/3)/(constants.c**5) * (2*np.pi * forb)**(11/3) * abs(ecc_func)
        return dfreq

    def gw_strain(self, semi, ecc, m1, m2):
        """
        Use of eqn (7) Kremer et al. 2018.
        Use of eqn (20) of Peters and Matthews (1963).
        At 1Gpc, z ~ 0.0228 from cosmology calc.
        
        Inputs:
        semi:   The semi-major axes of the system
        ecc:    The eccentricity of the binary system
        m1/m2:  The individual mass components of the binary system
        """

        dist = 1 | units.Gpc  # Taken from [https://imagine.gsfc.nasa.gov/features/cosmic/milkyway_info.html]
        redshift = 0.0228
        ecc = abs(ecc)

        chirp_mass = (m1*m2)**0.6/(m1+m2)**0.2 * (1+redshift)**-1
        cfactor = 2/(3*np.pi**(4/3)) * (constants.G**(5/3))/(constants.c**3) * (dist*(1+redshift))**-2
        ecc_func = (1+(73/24)*ecc**2+(37/96)*ecc**4)*(1-ecc**2)**-3.5

        nharm = self.gw_harmonic_mode(ecc)
        freq = self.gw_freq(semi, ecc, m1, m2) * (1 | units.Hz)
        dfreq = self.gw_dfreq(ecc, m1, m2, semi, chirp_mass, ecc_func)
        factor = dfreq * (5 * (1 | units.yr))/freq

        strain = min(1, factor) * cfactor * chirp_mass**(5/3) * freq**(-1/3) * (2/nharm)**(2/3) * (self.gfunc(ecc)/ecc_func)
        return (strain.value_in(units.s**-1.6653345369377348e-16))**0.5

    def gw_timescale(self, semi, ecc, m1, m2):
        """
        Function to calculate the GW timescale based on Peters (1964).
        
        Inputs:
        semi:    The semi-major axis of the binary
        ecc:     The eccentricity of the binary
        m1/m2:   The binary component masses
        outputs: The gravitational wave timescale
        """

        red_mass = (m1*m2)/(m1+m2)
        tot_mass = m1 + m2
        tgw = (5/256) * (constants.c)**5/(constants.G**3)*(semi**4*(1-ecc**2)**3.5)/(red_mass*tot_mass**2)
        return tgw

    def scatter_hist(self, x1, y1, x2, y2, ax, ax_histf, ax_histh, label1, label2, data_exist, data_filt):
        """
        Function to plot the frequency/strain histogram along its scatter plot.
        Use of: https://arxiv.org/pdf/2007.04241.pdf and https://arxiv.org/pdf/1408.0740.pdf 
        
        Inputs:
        x:           The strain arrays
        y:           The frequency arrays
        ax:          axis where the scatter plot is located
        ax_histf:    axis where the strain histogram is placed
        ax_histh:    axis where the frequency histogram is placed
        label:       Labels for the legend
        data_exist:  Whether x2 and y2 contain data
        data_filt:   To crop data files too large to estimate KDE
        """

        ######## PLOT {ALL | ONLY SMBH | ONLY NN | ONLY TERT}


        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)

        # the scatter plot:
        ax.scatter(np.log10(x1), np.log10(y1), color = 'blueviolet', s = 0.75)
        ax.scatter(np.log10(x2), np.log10(y2), color = 'orange', s = 0.75)
        
        ax_histf.tick_params(axis="x", labelbottom=False)
        ax_histh.tick_params(axis="y", labelleft=False)

        if (data_filt):
            no_data = round(len(x1)**0.9)
            no_data2 = round(len(x2)**0.5)
        else:
            no_data = len(x1)
            no_data2 = len(x2)

        kdef_SMBH = sm.nonparametric.KDEUnivariate(np.log10(x1[:no_data]))
        kdef_SMBH.fit()
        kdef_SMBH.density = (kdef_SMBH.density/max(kdef_SMBH.density))
        ax_histf.plot(kdef_SMBH.support, kdef_SMBH.density, color = 'blueviolet', label = label1)
        ax_histf.fill_between(kdef_SMBH.support, kdef_SMBH.density, alpha = 0.35, color = 'blueviolet')

        kdeh_SMBH = sm.nonparametric.KDEUnivariate(np.log10(y1[:no_data]))
        kdeh_SMBH.fit()
        kdeh_SMBH.density = (kdeh_SMBH.density / max(kdeh_SMBH.density))
        ax_histh.plot(kdeh_SMBH.density, kdeh_SMBH.support, color = 'blueviolet')
        ax_histh.fill_between(kdeh_SMBH.density, kdeh_SMBH.support, alpha = 0.35, color = 'blueviolet')

        if (data_exist):
            kdef_IMBH = sm.nonparametric.KDEUnivariate(np.log10(x2[:no_data2]))
            kdef_IMBH.fit()
            kdef_IMBH.density = (kdef_IMBH.density / max(kdef_IMBH.density))
            ax_histf.plot(kdef_IMBH.support, kdef_IMBH.density, color = 'orange', label = label2)
            ax_histf.fill_between(kdef_IMBH.support, kdef_IMBH.density, alpha = 0.35, color = 'orange')
            kdeh_IMBH = sm.nonparametric.KDEUnivariate(np.log10(y2[:no_data2]))
            kdeh_IMBH.fit()
            kdeh_IMBH.density = (kdeh_IMBH.density / max(kdeh_IMBH.density))
            ax_histh.plot(kdeh_IMBH.density, kdeh_IMBH.support, color = 'orange')
            ax_histh.fill_between(kdeh_IMBH.density, kdeh_IMBH.support, alpha = 0.35, color = 'orange')
           
        ax_histf.set_ylim(0, 1.05)
        ax_histf.set_ylabel(r'$\rho/\rho_{\rm{max}}$')
        ax_histf.legend()

        ax_histh.set_xlim(0, 1.05) 
        ax_histh.set_xlabel(r'$\rho/\rho_{\rm{max}}$')

        # LISA
        lisa = li.LISA() 
        x_temp = np.linspace(10**-5, 1, 1000)
        Sn = lisa.Sn(x_temp)

        # SKA
        SKA = np.load(os.path.join(os.path.dirname(__file__), 'SGWBProbe/files/hc_SKA.npz'))
        SKA_freq = SKA['x']
        SKA_hc = SKA['y']
        SKA_strain = SKA_hc**2/SKA_freq

        # muAres 
        Ares = np.load(os.path.join(os.path.dirname(__file__), 'SGWBProbe/files/S_h_muAres_nofgs.npz'))
        Ares_freq = Ares['x']
        Ares_strain = Ares['y']

        ax.plot(np.log10(x_temp), np.log10(np.sqrt(x_temp*Sn)), color = 'slateblue')
        ax.plot(np.log10(Ares_freq), np.log10(np.sqrt(Ares_freq*Ares_strain)), linewidth='1.5', color='red')
        ax.plot(np.log10(SKA_freq), np.log10(np.sqrt(SKA_freq*SKA_strain)), linewidth='1.5', color='orangered')
        ax.text(-9.25, -15.8, 'SKA', fontsize ='small', rotation = 322, color = 'orangered')
        ax.text(-4.28, -18.2, 'LISA', fontsize ='small', rotation = 309, color = 'slateblue')
        ax.text(-6.13, -19, r'$\mu$Ares', fontsize ='small', rotation = 312, color = 'red')
        
    def orbital_hist_plotter(self):
        """
        Function which plots all transient events into a histogram.
        Separates events depending on IMBH-IMBH or SMBH-IMBH.
        """

        plot_init = plotter_setup()

        xmin = -7
        xmax = 0
        ymin = -7
        iterf = 0

        print('Plotting Orbital Parameter Diagram')
        for fold_ in self.folders:
            IMBH_sem = [[ ], [ ]]
            IMBH_ecc = [[ ], [ ]]

            SMBH_sem = [[ ], [ ]]
            SMBH_ecc = [[ ], [ ]]

            for int_ in range(2):
                self.combine_data('pop_filt', self.integrator[int_], fold_)
                print(self.mass_parti)

                for parti_ in range(len(self.semi_flyby_nn)): #Looping through every individual particle
                    if self.pop[parti_] <= 40:
                        for event_ in range(len(self.semi_flyby_nn[parti_])): #Looping through every detected event
                            semi_fb_nn = self.semi_flyby_nn[parti_][event_]
                            if semi_fb_nn < 1 | units.parsec:
                                if np.asarray(self.fb_nn_SMBH[parti_][event_]) < 0 and self.mass_IMBH[parti_] < 10**5 | units.MSun:
                                    ecc_fb_nn = self.ecc_flyby_nn[parti_][event_]
                                    IMBH_sem[int_].append(semi_fb_nn.value_in(units.pc))
                                    IMBH_ecc[int_].append(np.log10(1-ecc_fb_nn))

                                else:
                                    ecc_fb_nn = self.ecc_flyby_nn[parti_][event_]
                                    SMBH_sem[int_].append(semi_fb_nn.value_in(units.pc))
                                    SMBH_ecc[int_].append(np.log10(1-ecc_fb_nn))

                for parti_ in range(len(self.semi_flyby_t)):
                    if self.pop[parti_] <= 40:
                        for event_ in range(len(self.semi_flyby_t[parti_])):
                            semi_fb_t = self.semi_flyby_t[parti_][event_]
                            if semi_fb_t < 1 | units.parsec:
                                if np.asarray(self.fb_t_SMBH[parti_][event_]) < 0 and self.mass_IMBH[parti_] < 10**5 | units.MSun:
                                    ecc_fb_t = self.ecc_flyby_t[parti_][event_]
                                    IMBH_sem[int_].append(semi_fb_t.value_in(units.pc))
                                    IMBH_ecc[int_].append(np.log10(1-ecc_fb_t))

                                else:
                                    ecc_fb_t = self.ecc_flyby_t[parti_][event_]
                                    SMBH_sem[int_].append(semi_fb_t.value_in(units.pc))
                                    SMBH_ecc[int_].append(np.log10(1-ecc_fb_t))

                for parti_ in range(len(self.semi_flyby_SMBH)):
                    if self.pop[parti_] <= 40:
                        for event_ in range(len(self.semi_flyby_SMBH[parti_])):
                            semi_fb_SMBH = self.semi_flyby_SMBH[parti_][event_]
                            if semi_fb_SMBH < 1 | units.parsec:
                                ecc_fb_SMBH = self.ecc_flyby_SMBH[parti_][event_]
                                SMBH_sem[int_].append(semi_fb_SMBH.value_in(units.pc))
                                SMBH_ecc[int_].append(np.log10(1-ecc_fb_SMBH))

                IMBH_ecc[int_] = np.asarray(IMBH_ecc[int_])
                IMBH_sem[int_] = np.asarray(IMBH_sem[int_]) 
                SMBH_ecc[int_] = np.asarray(SMBH_ecc[int_])
                SMBH_sem[int_] = np.asarray(SMBH_sem[int_])
            
            ############### PLOTTING OF a vs. (1-e) FOR BIN ##############
            x_arr = np.linspace(-10, 0, 2000)
            print(np.shape(self.mass_parti))
            rmass_IMBH = (self.mass_parti[0]**2)*(2*self.mass_parti[0])
            rmass_SMBH = (self.mass_parti[0]*self.mass_SMBH[0])*(self.mass_parti[0]+self.mass_SMBH[0])
            const_tgw = [np.log10(1-np.sqrt(1-((256*self.tH*(constants.G**3)/(5*constants.c**5)*rmass_IMBH*(10**(i) * (1 | units.pc)) **-4))**(1/3.5))) for i in x_arr]
            const_tgw2 = [np.log10(1-np.sqrt(1-((256*self.tH*(constants.G**3)/(5*constants.c**5)*rmass_SMBH*(10**(i) * (1 | units.pc)) **-4))**(1/3.5))) for i in x_arr]

            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(2, 2,  width_ratios=(4, 2), height_ratios=(2, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.06, hspace=0.06)
            ax = fig.add_subplot(gs[1, 0])
            ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
            ax2 = fig.add_subplot(gs[1, 1], sharey=ax)

            self.forecast_interferometer(ax, self.mass_parti[0], self.mass_SMBH[0])
            ax.text(-5, -3, r'$\mu$Ares ($f_{\rm{peak}} = 10^{-3}$ Hz)', verticalalignment = 'center', fontsize ='small', rotation=self.text_angle+7, color = 'white')
            ax.text(-5.7, -3, r'LISA ($f_{\rm{peak}} = 10^{-2}$ Hz)', verticalalignment = 'center', fontsize ='small', rotation=self.text_angle+7, color = 'white')
                    
            ax.scatter(np.log10(SMBH_sem[0]), SMBH_ecc[0], color = self.colors[2*iterf], s = 0.75, zorder = 5)
            ax.scatter(np.log10(SMBH_sem[1]), SMBH_ecc[1], color = self.colors[2*iterf+1], s = 0.75, zorder = 5)
            
            ax1.tick_params(axis="x", labelbottom=False)
            ax2.tick_params(axis="y", labelleft=False)

            no_data0S = round(len(SMBH_ecc[0])**0.9)
            no_data1S = round(len(SMBH_ecc[1])**0.9)
            no_data0I = round(len(IMBH_ecc[0])**0.9)
            no_data1I = round(len(IMBH_ecc[1])**0.9)

            kdeh_IMBH = sm.nonparametric.KDEUnivariate(SMBH_ecc[0][:no_data0S])
            kdeh_IMBH.fit()
            kdeh_IMBH.density = (kdeh_IMBH.density / max(kdeh_IMBH.density))
            ax2.plot(kdeh_IMBH.density, (kdeh_IMBH.support), color = self.colors[2*iterf])
            ax2.fill_between(kdeh_IMBH.density, (kdeh_IMBH.support), alpha = 0.35, color = self.colors[2*iterf])
            
            kdef_IMBH = sm.nonparametric.KDEUnivariate(np.log10(SMBH_sem[0][:no_data0S]))
            kdef_IMBH.fit()
            kdef_IMBH.density = (kdef_IMBH.density / max(kdef_IMBH.density))
            ax1.plot(kdef_IMBH.support, (kdef_IMBH.density), color = self.colors[2*iterf], label = 'Hermite')
            ax1.fill_between(kdef_IMBH.support, (kdef_IMBH.density), alpha = 0.35, color = self.colors[2*iterf])

            kdef_SMBH = sm.nonparametric.KDEUnivariate(np.log10(SMBH_sem[1][:no_data1S]))
            kdef_SMBH.fit()
            kdef_SMBH.density = (kdef_SMBH.density/max(kdef_SMBH.density))
            ax1.plot(kdef_SMBH.support, (kdef_SMBH.density), color = self.colors[2*iterf + 1], label = 'GRX')
            ax1.fill_between(kdef_SMBH.support, (kdef_SMBH.density), alpha = 0.35, color = self.colors[2*iterf + 1])

            kdeh_SMBH = sm.nonparametric.KDEUnivariate(SMBH_ecc[1][:no_data1S])
            kdeh_SMBH.fit()
            kdeh_SMBH.density = (kdeh_SMBH.density / max(kdeh_SMBH.density))
            ax2.plot(kdeh_SMBH.density, (kdeh_SMBH.support), color = self.colors[2*iterf + 1])
            ax2.fill_between(kdeh_SMBH.density, (kdeh_SMBH.support), alpha = 0.35, color = self.colors[2*iterf + 1])

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, 0)
            ax1.set_ylabel(r'$\rho/\rho_{\rm{max}}$')
            ax1.set_ylim(0,1.05)
            ax2.set_xlim(0,1.05)
            ax1.legend()
            ax2.set_xlabel(r'$\rho/\rho_{\rm{max}}$')

            ax.set_ylabel(r'$\log_{10}(1-e)$')
            ax.set_xlabel(r'$\log_{10} a$ [pc]')
            plot_init.tickers(ax, 'plot')
            plot_init.tickers(ax1, 'plot')
            plot_init.tickers(ax2, 'plot')

            ax.plot(x_arr, const_tgw2, color = 'black', zorder = 6)
            ax.text(-0.75, -3, r'$t_{\rm{GW}} > t_H$', verticalalignment = 'center', fontsize ='small', rotation=self.text_angle+27, color = 'white', 
                    path_effects=[pe.withStroke(linewidth=1, foreground="black")], zorder = 6)
            ax.text(-0.95, -3.5, r'$t_{\rm{GW}} < t_H$', verticalalignment = 'center', fontsize ='small', rotation=self.text_angle+27, color = 'white', 
                    path_effects=[pe.withStroke(linewidth=1, foreground="black")], zorder = 6)
            plt.savefig('figures/gravitational_waves/HistScatter_ecc_semi_SMBH_'+fold_+'.png', dpi=300, bbox_inches='tight')
            plt.clf()

            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(2, 2,  width_ratios=(4, 2), height_ratios=(2, 4),
                                left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0.05, hspace=0.05)
            ax = fig.add_subplot(gs[1, 0])
            ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
            ax2 = fig.add_subplot(gs[1, 1], sharey=ax)

            self.forecast_interferometer(ax, self.mass_parti[0], self.mass_IMBH[0][0])
            ax.text(-6.1, -3, r'$\mu$Ares ($f_{\rm{peak}} = 10^{-3}$ Hz)', verticalalignment = 'center', fontsize ='small', rotation=self.text_angle+7, color = 'white')
            ax.text(-6.8, -3, r'LISA ($f_{\rm{peak}} = 10^{-2}$ Hz)', verticalalignment = 'center', fontsize ='small', rotation=self.text_angle+7, color = 'white')

            ax.scatter(np.log10(IMBH_sem[0]), IMBH_ecc[0], color = self.colors[2*iterf], s = 0.75, zorder = 5)
            ax.scatter(np.log10(IMBH_sem[1]), IMBH_ecc[1], color = self.colors[2*iterf + 1], s = 0.75, zorder = 5)
            
            ax1.tick_params(axis="x", labelbottom=False)
            ax2.tick_params(axis="y", labelleft=False)

            kdeh_IMBH = sm.nonparametric.KDEUnivariate(IMBH_ecc[0][:no_data0I])
            kdeh_IMBH.fit()
            kdeh_IMBH.density = (kdeh_IMBH.density / max(kdeh_IMBH.density))
            ax2.plot(kdeh_IMBH.density, (kdeh_IMBH.support), color = self.colors[2*iterf])
            ax2.fill_between(kdeh_IMBH.density, (kdeh_IMBH.support), alpha = 0.35*(1+iterf), color = self.colors[2*iterf])
            
            kdef_IMBH = sm.nonparametric.KDEUnivariate(np.log10(IMBH_sem[0][:no_data0I]))
            kdef_IMBH.fit()
            kdef_IMBH.density = (kdef_IMBH.density / max(kdef_IMBH.density))
            ax1.plot(kdef_IMBH.support, (kdef_IMBH.density), color = self.colors[2*iterf], label = 'Hermite')
            ax1.fill_between(kdef_IMBH.support, (kdef_IMBH.density), alpha = 0.35*(1+iterf), color = self.colors[2*iterf])

            kdef_SMBH = sm.nonparametric.KDEUnivariate(np.log10(IMBH_sem[1][:no_data1I]))
            kdef_SMBH.fit()
            kdef_SMBH.density = (kdef_SMBH.density/max(kdef_SMBH.density))
            ax1.plot(kdef_SMBH.support, (kdef_SMBH.density), color = self.colors[2*iterf + 1], label = 'GRX')
            ax1.fill_between(kdef_SMBH.support, (kdef_SMBH.density), alpha = 0.35*(1+iterf), color = self.colors[2*iterf + 1])

            kdeh_SMBH = sm.nonparametric.KDEUnivariate(IMBH_ecc[1][:no_data1I])
            kdeh_SMBH.fit()
            kdeh_SMBH.density = (kdeh_SMBH.density / max(kdeh_SMBH.density))
            ax2.plot(kdeh_SMBH.density, (kdeh_SMBH.support), color = self.colors[2*iterf + 1])
            ax2.fill_between(kdeh_SMBH.density, (kdeh_SMBH.support), alpha = 0.35*(1+iterf), color = self.colors[2*iterf + 1])

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, 0)
            ax1.set_ylim(0,1.05)
            ax2.set_xlim(0,1.05)
            ax1.set_ylabel(r'$\rho/\rho_{\rm{max}}$')
            ax1.legend()
            ax2.set_xlabel(r'$\rho/\rho_{\rm{max}}$')

            ax.set_ylabel(r'$\log_{10}(1-e)$')
            ax.set_xlabel(r'$\log_{10} a$ [pc]')
            plot_init.tickers(ax, 'plot')
            plot_init.tickers(ax1, 'plot')
            plot_init.tickers(ax2, 'plot')
            ax.plot(x_arr, const_tgw, color = 'black', zorder = 6)
            ax.text(-2.55, -3, r'$t_{\rm{GW}} > t_H$', verticalalignment = 'center', fontsize ='small', rotation=self.text_angle+27, color = 'white', 
                    path_effects=[pe.withStroke(linewidth=1, foreground="black")], zorder = 6)
            ax.text(-3.08, -3.1, r'$t_{\rm{GW}} < t_H$', verticalalignment = 'center', fontsize ='small', rotation=self.text_angle+27, color = 'white', 
                    path_effects=[pe.withStroke(linewidth=1, foreground="black")], zorder = 6)
            plt.savefig('figures/gravitational_waves/HistScatter_ecc_semi_IMBH_'+fold_+'.png', dpi=300, bbox_inches='tight')
            plt.clf()

            iterf += 1

    def strain_freq_plotter(self):
        """
        Function which plots the amplitude histogram for all sets of particle.
        """
        
        plot_ini = plotter_setup()
        data_idx = 0                 #Hardcode (0 = Hermite) | (1 = GRX)
        iterf = 0
        print('Plotting Strain Frequency Diagram')
        for fold_ in self.folders:
            self.combine_data('integrator', self.integrator[data_idx], fold_)
            IMBH_strain = [ ]
            IMBH_freq = [ ]
            SMBH_strain = [ ]
            SMBH_freq = [ ]
            
            for parti_ in range(len(self.semi_flyby_nn)): #Looping through every individual particle
                if self.pop[parti_] <= 40:                #Hardcode value
                    for event_ in range(len(self.semi_flyby_nn[parti_])): #Looping through every detected event
                        semi_fb_nn = self.semi_flyby_nn[parti_][event_]
                        if semi_fb_nn < 1 | units.parsec and self.strain_flyby_nn[parti_][event_] > 0:
                            if np.asarray(self.fb_nn_SMBH[parti_][event_]) < 0:
                                IMBH_strain.append(self.strain_flyby_nn[parti_][event_])
                                IMBH_freq.append(self.freq_flyby_nn[parti_][event_])
                            else:
                                SMBH_strain.append(self.strain_flyby_nn[parti_][event_])
                                SMBH_freq.append(self.freq_flyby_nn[parti_][event_])

            for parti_ in range(len(self.semi_flyby_t)):
                if self.pop[parti_] <= 40:
                    for event_ in range(len(self.semi_flyby_t[parti_])):
                        semi_fb_t = self.semi_flyby_t[parti_][event_]
                        if semi_fb_t < 1 | units.parsec and self.strain_flyby_t[parti_][event_] > 0:
                            if np.asarray(self.fb_t_SMBH[parti_][event_]) < 0:
                                IMBH_strain.append(self.strain_flyby_t[parti_][event_])
                                IMBH_freq.append(self.freq_flyby_t[parti_][event_])
                            else:
                                SMBH_strain.append(self.strain_flyby_t[parti_][event_])
                                SMBH_freq.append(self.freq_flyby_t[parti_][event_])

            for parti_ in range(len(self.semi_flyby_SMBH)):
                if self.pop[parti_] <= 40:
                    for event_ in range(len(self.semi_flyby_SMBH[parti_])):
                        semi_fb_SMBH = self.semi_flyby_SMBH[parti_][event_]
                        if semi_fb_SMBH < 1 | units.parsec:
                            SMBH_strain.append(self.strain_flyby_SMBH[parti_][event_])
                            SMBH_freq.append(self.freq_flyby_SMBH[parti_][event_])

            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(2, 2,  width_ratios=(4, 2), height_ratios=(2, 4),
                                left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0.05, hspace=0.05)
            ax = fig.add_subplot(gs[1, 0])
            ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
            ax2 = fig.add_subplot(gs[1, 1], sharey=ax)
            
            self.scatter_hist(SMBH_freq, SMBH_strain,
                              IMBH_freq, IMBH_strain, 
                              ax, ax1, ax2, 'SMBH-IMBH', 'IMBH-IMBH',
                              True, True)
            ax.set_xlabel(r'$\log_{10}f$ [Hz]')
            ax.set_ylabel(r'$\log_{10}h$')
            ax1.set_title(str(self.integrator[data_idx]))
            plot_ini.tickers(ax, 'plot')
            plot_ini.tickers(ax1, 'plot')
            plot_ini.tickers(ax2, 'plot')
            ax.set_ylim(-30, -12.2)
            ax.set_xlim(-12.5, 0.1)
            plt.savefig('figures/gravitational_waves/'+str(self.integrator[data_idx])+'GW_freq_strain_maximise_diagram_N<=40'+fold_+'.png', dpi = 500, bbox_inches='tight')
            plt.clf()
