from amuse.lab import *
from file_logistics import *
from scipy.optimize import curve_fit
from tGW_plotters import *

import fnmatch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pickle as pkl
import warnings

class sustainable_sys(object):
    """
    Class which extracts ALL information of particle_trajectory files in a memory-optimised 
    manner and appends the required data values into arrays.
    """

    def __init__(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.folders = ['rc_0.25_4e6', 'rc_0.25_4e5', 'rc_0.25_4e7']
        self.colors = ['red', 'blue', 'deepskyblue', 
                       'slateblue', 'turquoise', 'skyblue']
        self.labelsD = [r'$M_{\rm{SMBH}} = 4\times10^{6}M_{\odot}$', 
                        r'$M_{\rm{SMBH}} = 4\times10^{5}M_{\odot}$',
                        r'$M_{\rm{SMBH}} = 4\times10^{7}M_{\odot}$']
        self.legend_id = [3, 4]
        self.vdisp = 150000/3

        self.fcrop = True
        if (self.fcrop):
            self.frange = 1
        else:
            self.frange = 3

    def new_data_extractor(self):
        """
        Script to extract data from recently simulated runs
        """
        
        GW_calcs = gw_calcs()
        pop_tracker = 40

        print('!!!!!! WARNING THIS WILL TAKE A WHILE !!!!!!!')
        iterf = 0
        for fold_ in self.folders[:self.frange]:
            print('Files for: ', fold_)
            tcropG = 59
            GRX_data = glob.glob('/media/erwanh/Elements/'+fold_+'/GRX/particle_trajectory/*')
            chaoticG = ['/media/erwanh/Elements/'+fold_+'/data/GRX/chaotic_simulation/'+str(i[tcropG:]) for i in GRX_data]
            filename, filenameC, integrator, drange = ndata_chaos(iterf, GRX_data, chaoticG, fold_)
            for int_ in range(1):
                for file_ in range(len(filename[int_])):
                    with open(filenameC[int_][file_], 'rb') as input_file:
                        chaotic_tracker = pkl.load(input_file)
                        proceed = True
                        if chaotic_tracker.iloc[0][6] <= pop_tracker and (proceed):
                            with open(filename[int_][file_], 'rb') as input_file:
                                file_size = os.path.getsize(filename[int_][file_])
                                if file_size<2.8e9:
                                    data = pkl.load(input_file)
                                    pop = 5*round(0.2*np.shape(data)[0])
                                    print('Reading file', file_, ':', input_file)
                                    for parti_ in range(np.shape(data)[0]):
                                        pop_bin = [ ]
                                        pop_ter = [ ]

                                        semi_NN_avg = [ ]
                                        semi_NN_min = [ ]
                                        semi_t_avg = [ ]
                                        semi_t_min = [ ]

                                        bsys_time = [ ]
                                        bform_time = [ ]
                                        bin_key = [ ]
                                        new_bin_sys = [ ]
                                        tsys_time = [ ]
                                        tform_time = [ ]
                                        ter_key = [ ]
                                        new_ter_sys = [ ]

                                        GW_freqbin = [ ]
                                        GW_strainbin = [ ]
                                        GW_timeb = [ ]
                                        GW_bmass = [ ]
                                        GW_freqter = [ ]
                                        GW_strainter = [ ]
                                        GW_timet = [ ]
                                        GW_tmass = [ ]

                                        hard_bin = [ ]
                                        hard_ter = [ ]
                                        dedt = [ ]
                                        dadt = [ ]

                                        bin_sys = False
                                        ter_sys = False
                                        semi_nn_avg_temp = []
                                        semi_t_avg_temp = []
                                        temp_dedt = [] 
                                        temp_dadt = []
                                        bform_time.append(-5)
                                        tform_time.append(-5)

                                        if parti_ != 0:
                                            for col_ in range(np.shape(data)[1]-1):
                                                nn_semi = abs(data.iloc[parti_][col_][7][1])
                                                nn_ecc = data.iloc[parti_][col_][8][1]
                                                mass1 = data.iloc[parti_][0][1]

                                                for part_ in range(np.shape(data)[0]):
                                                    if data.iloc[part_][0][0] == data.iloc[parti_][col_][6][1]:
                                                        mass2 = data.iloc[part_][0][1]
                                                        mass = max(mass1, mass2)
                                                        Ehard = (constants.G*mass)/(2*50*(self.vdisp*(1 | units.ms))**2)
                                                        Ebin = 0.1*(constants.G*mass)/(2*(self.vdisp*(1 | units.ms))**2)

                                                        hard = False
                                                        bin = False
                                                        if nn_semi<Ehard and nn_ecc<1:   #Hard binary conditions (Quinlan 1996b)
                                                            hard = True
                                                            bin = True
                                                            hard_bin.append(1)
                                                            
                                                        if not (hard) and nn_semi<Ebin and abs(nn_ecc)<1:  #Value chosen for a<1000AU
                                                            hard_bin.append(-5)
                                                            bin = True

                                                        if (bin):
                                                            if not (bin_sys):  #First formation time
                                                                formation_time = col_*1e3
                                                                bform_time[-1] = formation_time
                                                                bin_sys = True
                                                            bin_key.append(data.iloc[parti_][col_][6][1])
                                                            if data.iloc[parti_][col_][6][1] != bin_key[-1] or len(new_bin_sys)<1:
                                                                new_bin_sys.append(1)
                                                            else:
                                                                new_bin_sys.append(-5)

                                                            bsys_time.append(col_)
                                                            semi_nn_avg_temp.append(nn_semi.value_in(units.pc))
                                                            pop_bin.append(pop)

                                                            strain = GW_calcs.gw_strain(nn_semi, nn_ecc, mass1, mass2)
                                                            freq = GW_calcs.gw_freq(nn_semi, nn_ecc, mass1, mass2)
                                                            GW_time = GW_calcs.gw_timescale(nn_semi, nn_ecc, mass1, mass2)
                                                            GW_strainbin.append(strain)
                                                            GW_freqbin.append(freq)
                                                            GW_timeb.append(float(GW_time.value_in(units.Myr)))
                                                            GW_bmass.append(mass2.value_in(units.MSun))

                                                            semi_outer = abs(data.iloc[parti_][col_][7][2])
                                                            ecc_outer = data.iloc[parti_][col_][8][2]

                                                            #Calculate tertiary. The stability equality is based on Mardling and Aarseth 2001
                                                            if ecc_outer<1:
                                                                for part_ in range(np.shape(data)[0]):
                                                                    if data.iloc[part_][0][0] == data.iloc[parti_][col_][6][2]:
                                                                        mass_outer = data.iloc[part_][0][1]

                                                                semi_ratio = semi_outer/nn_semi
                                                                equality = 2.8*((1+mass_outer/(mass1+mass2))*(1+ecc_outer)/(1-ecc_outer)**0.5)**0.4
                                                                if semi_ratio>equality:
                                                                    if not (ter_sys):
                                                                        formation_time = col_*1e3
                                                                        tform_time[-1] = formation_time
                                                                        ter_sys = True
                                                                    ter_key.append([i for i in data.iloc[parti_][col_][6][2]][0])
                                                                    if data.iloc[parti_][col_][6][2] != ter_key[-1] or len(new_ter_sys)<2:
                                                                        new_ter_sys.append(1)
                                                                    else:
                                                                        new_ter_sys.append(-5)

                                                                    tsys_time.append(col_)
                                                                    semi_t_avg_temp.append(semi_outer.value_in(units.pc))
                                                                    pop_ter.append(pop)

                                                                    GW_time = GW_calcs.gw_timescale(semi_outer, ecc_outer, mass1, mass_outer)
                                                                    strain = GW_calcs.gw_strain(semi_outer, ecc_outer, mass1, mass_outer)
                                                                    freq = GW_calcs.gw_freq(semi_outer, ecc_outer, mass1, mass_outer)
                                                                    GW_strainter.append(strain)
                                                                    GW_freqter.append(freq)
                                                                    GW_timet.append(float(GW_time.value_in(units.Myr)))
                                                                    GW_tmass.append([mass2.value_in(units.MSun), 
                                                                                     mass_outer.value_in(units.MSun)])

                                                                    if (hard):
                                                                        hard_ter.append(1)
                                                                    else:
                                                                        hard_ter.append(-5)

                                            bin_sys = np.shape(np.unique(bin_key))[0]
                                            ter_sys = np.shape(np.unique(ter_key))[0]

                                            if len(semi_nn_avg_temp)>0:
                                                semi_NN_avg.append(np.mean(semi_nn_avg_temp))
                                                semi_NN_min.append(np.min(semi_nn_avg_temp))
                                            else:
                                                semi_NN_avg.append(-5)
                                                semi_NN_min.append(-5)

                                            if len(semi_t_avg_temp)>0:
                                                semi_t_avg.append(np.mean(semi_t_avg_temp))
                                                semi_t_min.append(np.min(semi_t_avg_temp))
                                            else:
                                                semi_t_avg.append(-5)
                                                semi_t_min.append(-5)
                                                        
                                            bsys_time.append((len(np.unique(bsys_time)))/(col_))
                                            tsys_time.append((len(np.unique(tsys_time)))/(col_))
                                            dedt.append(np.mean(temp_dedt))
                                            dadt.append(np.mean(temp_dadt))

                                            path = '/media/erwanh/Elements/'+fold_+'/data/bin_hier_systems/'
                                            stab_tracker = pd.DataFrame()
                                            df_stabtime = pd.Series({'File_number': file_,
                                                                     'Integrator': integrator[int_],
                                                                     'Population': pop,
                                                                     'Binary Pop.': pop_bin,
                                                                     '# Binary Sys.': bin_sys,
                                                                     'First bin. form': bform_time,
                                                                     'Binary System Delineate': new_bin_sys,
                                                                     'Tertiary Pop.': pop_ter,
                                                                     '# Tertiary Sys.': ter_sys,
                                                                     'First ter. form': tform_time,
                                                                     'Tertiary System Delineate': new_ter_sys,
                                                                     'Hard Bin. Bool': hard_bin,
                                                                     'Hard Ter. Bool': hard_ter,
                                                                     'Bin. GW freq': GW_freqbin,
                                                                     'Bin. GW strain': GW_strainbin,
                                                                     'Bin. GW time': GW_timeb,           #Merging time in Myr
                                                                     'Bin. GW mass': GW_bmass,           #Constituent mass in MSun
                                                                     'Ter. GW freq': GW_freqter,
                                                                     'Ter. GW strain': GW_strainter,
                                                                     'Ter. GW time': GW_timet,
                                                                     'Ter. GW mass': GW_tmass,
                                                                     'Average dedt': dedt,
                                                                     'Average dadt': dadt,
                                                                     'Bin. Semi-major Avg': semi_NN_avg,
                                                                     'Bin. Semi-major Min': semi_NN_min,
                                                                     'Ter. Semi-major Avg': semi_t_avg,
                                                                     'Ter. Semi-major Min': semi_t_min,
                                                                     'Total sim. length': col_ * 1000,
                                                                     })
                                            stab_tracker = stab_tracker.append(df_stabtime, ignore_index = True)
                                            stab_tracker.to_pickle(os.path.join(path, 'IMBH_'+str(integrator[int_])+'_system_data_indiv_parti_file_'+str(file_)+'_'+str(parti_)+'_local2.pkl'))
            iterf += 1

    def array_rewrite(self, arr, arr_type, filt):
        """
        Function to rewrite array to manipulatable float format

        Inputs:
        arr:      The original array with the data
        arr_type: String stating whether array is nested or not 
        filt:     Boolean to filter out unwanted values
        """

        new_arr = [ ]
        if arr_type == 'nested':
            for sublist in arr:
                for item_ in sublist:
                    if (filt):
                        if item_>0:
                            new_arr.append(item_)
                    else:
                        new_arr.append(item_)
        else:
            for item_ in arr:
                if (filt):
                    if item_>0:
                        new_arr.append(item_)
                else:
                    new_arr.append(item_)
        
        return new_arr
                
    def combine_data(self):
        """
        Function which extracts ALL data and computes
        various quantitative results.
        """

        print('Extracting data')
        GW_calcs = gw_calcs()
        integrator = ['Hermite', 'GRX']

        self.file_names = [[ ], [ ], [ ], [ ], [ ]]
        self.integrator = [[ ], [ ], [ ], [ ], [ ]]
        self.pop = [[ ], [ ], [ ], [ ], [ ]]
        self.pop_bin = [[ ], [ ], [ ], [ ], [ ]]
        self.bin_sys = [[ ], [ ], [ ], [ ], [ ]]
        self.bform_time = [[ ], [ ], [ ], [ ], [ ]]
        self.new_bin_sys = [[ ], [ ], [ ], [ ], [ ]]
        self.pop_ter = [[ ], [ ], [ ], [ ], [ ]]
        self.ter_sys = [[ ], [ ], [ ], [ ], [ ]]
        self.tform_time = [[ ], [ ], [ ], [ ], [ ]]
        self.new_ter_sys = [[ ], [ ], [ ], [ ], [ ]]
        self.hard_bin = [[ ], [ ], [ ], [ ], [ ]]
        self.hard_ter = [[ ], [ ], [ ], [ ], [ ]]
        self.GW_freqbin = [[ ], [ ], [ ], [ ], [ ]]
        self.GW_strainbin = [[ ], [ ], [ ], [ ], [ ]]
        self.GW_timeb = [[ ], [ ], [ ], [ ], [ ]]
        self.GW_bmass = [[ ], [ ], [ ], [ ], [ ]]
        self.GW_freqter = [[ ], [ ], [ ], [ ], [ ]]
        self.GW_strainter = [[ ], [ ], [ ], [ ], [ ]]
        self.GW_timet = [[ ], [ ], [ ], [ ], [ ]]
        self.GW_tmass = [[ ], [ ], [ ], [ ], [ ]]
        self.semi_NN_avg = [[ ], [ ], [ ], [ ], [ ]]
        self.semi_NN_min = [[ ], [ ], [ ], [ ], [ ]]
        self.semi_t_avg = [[ ], [ ], [ ], [ ], [ ]]
        self.semi_t_min = [[ ], [ ], [ ], [ ], [ ]]
        self.tot_sim = [[ ], [ ], [ ], [ ], [ ]]

        iterf = 0
        for fold_ in self.folders[:self.frange]:
            system_data = natsort.natsorted(glob.glob('/media/erwanh/Elements/'+fold_+'/data/bin_hier_systems/*'))
            for file_ in range(len(system_data)):
                with open(system_data[file_], 'rb') as input_file:
                    data_file = pkl.load(input_file)
                    if data_file.iloc[0][1] == 'Hermite':
                        int_ = 0
                    else:
                        int_ = iterf + 1

                    self.file_names[int_].append(data_file.iloc[0][0])
                    self.integrator[int_].append(data_file.iloc[0][1])
                    self.pop[int_].append(int(data_file.iloc[0][2]))
                    self.pop_bin[int_].append(data_file.iloc[0][3])
                    self.bin_sys[int_].append(data_file.iloc[0][4])
                    self.bform_time[int_].append(data_file.iloc[0][5])
                    self.new_bin_sys[int_].append(data_file.iloc[0][6])
                    self.pop_ter[int_].append(data_file.iloc[0][7])
                    self.ter_sys[int_].append(data_file.iloc[0][8])
                    self.tform_time[int_].append(data_file.iloc[0][9])
                    self.new_ter_sys[int_].append(data_file.iloc[0][10])
                    self.hard_bin[int_].append(data_file.iloc[0][11])
                    self.hard_ter[int_].append(data_file.iloc[0][12])
                    self.GW_freqbin[int_].append(data_file.iloc[0][13])
                    self.GW_strainbin[int_].append(data_file.iloc[0][14])
                    self.GW_timeb[int_].append(data_file.iloc[0][15])
                    self.GW_bmass[int_].append(data_file.iloc[0][16])
                    self.GW_freqter[int_].append(data_file.iloc[0][17])
                    self.GW_strainter[int_].append(data_file.iloc[0][18])
                    self.GW_timet[int_].append(data_file.iloc[0][19])
                    self.GW_tmass[int_].append(data_file.iloc[0][20])
                    self.semi_NN_avg[int_].append(data_file.iloc[0][23])
                    self.semi_NN_min[int_].append(data_file.iloc[0][24])
                    self.semi_t_avg[int_].append(data_file.iloc[0][25])
                    self.semi_t_min[int_].append(data_file.iloc[0][26])
                    self.tot_sim[int_].append(data_file.iloc[0][27])

            iterf += 1
            
        self.GWfreq_bin = [[ ], [ ], [ ], [ ], [ ]]
        self.GWstra_bin = [[ ], [ ], [ ], [ ], [ ]]
        self.GWfreq_ter = [[ ], [ ], [ ], [ ], [ ]]
        self.GWstra_ter = [[ ], [ ], [ ], [ ], [ ]]
        for int_ in range(5):
            self.GWfreq_bin[int_] = self.array_rewrite(self.GW_freqbin[int_], 'nested', False)
            self.GWstra_bin[int_] = self.array_rewrite(self.GW_strainbin[int_], 'nested', False)
            self.GWfreq_ter[int_] = self.array_rewrite(self.GW_freqter[int_], 'nested', False)
            self.GWstra_ter[int_] = self.array_rewrite(self.GW_strainter[int_], 'nested', False) 

        self.unique_pops = [[ ], [ ], [ ], [ ], [ ]]
        self.binary_systems = [[ ], [ ], [ ], [ ], [ ]]
        self.binary_total = [[ ], [ ], [ ], [ ], [ ]]
        self.binary_occupation = [[ ], [ ], [ ], [ ], [ ]]
        self.binary_init = [[ ], [ ], [ ], [ ], [ ]]
        self.binary_IMBH = [[ ], [ ], [ ], [ ], [ ]]
        self.binary_hard = [[ ], [ ], [ ], [ ], [ ]]
        self.GWb_mergers = [[ ], [ ], [ ], [ ], [ ]]

        self.tertiary_systems = [[ ], [ ], [ ], [ ], [ ]]
        self.tertiary_total = [[ ], [ ], [ ], [ ], [ ]]
        self.tertiary_occupation = [[ ], [ ], [ ], [ ], [ ]]
        self.tertiary_init = [[ ], [ ], [ ], [ ], [ ]]
        self.tertiary_IMBH = [[ ], [ ], [ ], [ ], [ ]]
        self.tertiary_hard = [[ ], [ ], [ ], [ ], [ ]]
        self.GWt_mergers = [[ ], [ ], [ ], [ ], [ ]]

        self.GWfreq_binIMBH = [[ ], [ ], [ ], [ ], [ ]]
        self.GWstra_binIMBH = [[ ], [ ], [ ], [ ], [ ]]
        self.GWfreq_terIMBH = [[ ], [ ], [ ], [ ], [ ]]
        self.GWstra_terIMBH = [[ ], [ ], [ ], [ ], [ ]]

        self.GWfreq_binHard = [[ ], [ ], [ ], [ ], [ ]]
        self.GWstra_binHard = [[ ], [ ], [ ], [ ], [ ]]
        self.GWfreq_binHardIMBH = [[ ], [ ], [ ], [ ], [ ]]
        self.GWstra_binHardIMBH = [[ ], [ ], [ ], [ ], [ ]]

        self.bocc_lower = [[ ], [ ], [ ], [ ], [ ]]
        self.bocc_highr = [[ ], [ ], [ ], [ ], [ ]]
        self.hocc_lower = [[ ], [ ], [ ], [ ], [ ]]
        self.hocc_highr = [[ ], [ ], [ ], [ ], [ ]]
        
        sims = [[40, 40, 40, 40], 
                [60, 60, 60, 60, 60, 60, 60], 
                [40, 40, 40, 40, 40, 40, 40],
                [40, 40, 40, 40, 40, 40, 40],
                [40, 40, 40, 40, 40, 40, 40]]

        iterf = 0
        for fold_ in self.folders[:self.frange]:
            integrator, drange = folder_loop(iterf)
            with open('figures/binary_hierarchical/output/system_summary_'+fold_+'.txt', 'w') as file:
                for int_ in range(drange):
                    print('Collecting data for: ', fold_, integrator[int_])
                    sim_ = folder_data_loop(iterf, int_)
                        
                    bform_med = [ ]
                    bform_higq = [ ]
                    bform_lowq = [ ]

                    tform_med = [ ]
                    tform_higq = [ ]
                    tform_lowq = [ ]

                    pop_arr = np.unique(self.pop[sim_])
                    file.write('Data for '+str(integrator[int_]+' in pc'))
                    iter = -1
                    for pop_ in pop_arr:
                        iter += 1
                        self.unique_pops[sim_].append(pop_)
                        idx = np.argwhere(self.pop[sim_] == pop_).flatten()
                        bform_time = [ ]
                        tform_time = [ ]
                        GWb_time = [ ]
                        GWt_time = [ ]
                        semi_NN_avg = [ ]
                        semi_NN_min = [ ]
                        semi_t_avg = [ ]
                        semi_t_min = [ ]
                        bin_occ = [ ]
                        ter_occ = [ ]

                        bin_data_merge = 0
                        bin_data = 0
                        IMBH_bin = 0
                        hard_bin = 0
                        ter_data_merge = 0
                        ter_data = 0
                        IMBH_ter = 0
                        hard_ter = 0 
                        for data_ in idx:
                            hard_Bool = False
                            for item_ in self.bform_time[sim_][data_]:
                                if item_>0:
                                    bform_time.append(item_)
                            for item_ in self.tform_time[sim_][data_]:
                                if item_>0:
                                    tform_time.append(item_)
                            idx_nbin = np.argwhere(np.asarray(self.new_bin_sys[sim_][data_])>0).flatten()
                            idx_tbin = np.argwhere(np.asarray(self.new_ter_sys[sim_][data_])>0).flatten()

                            GW_btime_temp = np.asarray(self.GW_timeb[sim_][data_])[idx_nbin-1]
                            GW_ttime_temp = np.asarray(self.GW_timet[sim_][data_])[idx_tbin-1]
                            bin_data += len(idx_nbin)
                            ter_data += len(idx_tbin)
                            for idx_ in idx_nbin:
                                if np.asarray(self.GW_bmass[sim_][data_])[idx_-1]<1e6:
                                    IMBH_bin += 1
                                if np.asarray(self.hard_bin[sim_][data_])[idx_-1]>0:
                                    hard_bin += 1
                                    hard_Bool = True

                            for idx_ in idx_tbin:
                                if np.shape(self.GW_tmass[sim_][data_])[0]>0:
                                    for mass_ in np.asarray(self.GW_tmass[sim_][data_])[idx_-1]:
                                        if mass_>1e6:
                                            IMBH_ter -= 1
                                if np.asarray(self.hard_ter[sim_][data_])[idx_-1].all()>0:
                                    hard_ter += 1
                            
                            if (hard_Bool):
                                if len(idx_nbin) == 1:
                                    self.GWfreq_binIMBH[sim_].append([float(i) for i in self.GW_freqbin[sim_][data_]])
                                    self.GWstra_binIMBH[sim_].append([float(i) for i in self.GW_strainbin[sim_][data_]])
                                else:
                                    prev_iter = 0
                                    for crop_ in idx_nbin:
                                        self.GWfreq_binIMBH[sim_].append([float(i) for i in self.GW_freqbin[sim_][data_][prev_iter:crop_]])
                                        self.GWstra_binIMBH[sim_].append([float(i) for i in self.GW_strainbin[sim_][data_][prev_iter:crop_]])
                                        prev_iter = crop_

                                if len(idx_tbin) == 1:
                                    self.GWfreq_terIMBH[sim_].append(self.GW_freqter[sim_][data_])
                                    self.GWstra_terIMBH[sim_].append(self.GW_strainter[sim_][data_])
                                else:
                                    prev_iter = 0
                                    for crop_ in idx_nbin:
                                        self.GWfreq_terIMBH[sim_].append(self.GW_freqter[sim_][data_][prev_iter:crop_])
                                        self.GWstra_terIMBH[sim_].append(self.GW_strainter[sim_][data_][prev_iter:crop_])
                                        prev_iter = crop_

                            if GW_btime_temp<(GW_calcs.tH).value_in(units.Myr):
                                bin_data_merge += 1
                                GWb_time.append(float(GW_btime_temp))
                            for item_ in GW_ttime_temp:
                                if item_<(GW_calcs.tH).value_in(units.Myr):
                                    ter_data_merge += 1
                                    GWt_time.append(float(GW_btime_temp))
                            for item_ in self.semi_NN_avg[sim_][data_]:
                                if item_>0:
                                    semi_NN_avg.append(item_)
                            for item_ in self.semi_NN_min[sim_][data_]:
                                if item_>0:
                                    semi_NN_min.append(item_)
                            for item_ in self.semi_t_avg[sim_][data_]:
                                if item_>0:
                                    semi_t_avg.append(item_)
                            for item_ in self.semi_t_min[sim_][data_]:
                                if item_>0:
                                    semi_t_min.append(item_)

                            bin_occ.append(1e3*(len(self.pop_bin[sim_][data_])/self.tot_sim[sim_][data_]))
                            ter_occ.append(1e3*(len(self.pop_ter[sim_][data_])/self.tot_sim[sim_][data_]))
                        bform_time = np.asarray(bform_time)
                        tform_time = np.asarray(tform_time)

                        self.binary_total[sim_].append(bin_data)
                        self.binary_IMBH[sim_].append(IMBH_bin)
                        self.binary_hard[sim_].append(hard_bin)
                        self.GWb_mergers[sim_].append(bin_data_merge)
                        self.tertiary_total[sim_].append(ter_data)
                        self.tertiary_IMBH[sim_].append((ter_data+IMBH_ter))
                        self.tertiary_hard[sim_].append(hard_ter)
                        self.GWt_mergers[sim_].append(ter_data_merge)

                        self.binary_systems[sim_].append(bin_data/sims[sim_][iter])
                        median_bocc = np.median(bin_occ)
                        q1, q3 = np.percentile(bin_occ, [25, 75])
                        self.binary_occupation[sim_].append(median_bocc)
                        self.bocc_highr[sim_].append(q3)
                        self.bocc_lower[sim_].append(q1)

                        self.tertiary_systems[sim_].append(ter_data/sims[sim_][iter])
                        median_tocc = np.median(ter_occ)
                        q1, q3 = np.percentile(ter_occ, [25, 75])
                        self.tertiary_occupation[sim_].append(median_tocc)
                        self.hocc_highr[sim_].append(q3)
                        self.hocc_lower[sim_].append(q1)

                        median_binform = np.median(bform_time)
                        q1, q3 = np.percentile(bform_time, [25, 75])
                        bform_med.append('{:.7f}'.format(median_binform))
                        bform_higq.append('{:.7f}'.format(q3 - median_binform))
                        bform_lowq.append('{:.7f}'.format(q1 - median_binform))

                        median_terform = np.median(tform_time)
                        if median_terform>0:
                            q1, q3 = np.percentile(tform_time, [25, 75])
                        else:
                            q1, q3 = 0, 0
                        tform_med.append('{:.7f}'.format(median_terform))
                        tform_higq.append('{:.7f}'.format(q3 - median_terform))
                        tform_lowq.append('{:.7f}'.format(q1 - median_terform))

                        self.binary_init[sim_].append(len(bform_time[bform_time <= 1000]))
                        self.tertiary_init[sim_].append(len(tform_time[tform_time <= 1000]))

                    nbin = [[ ], [ ], [ ], [ ], [ ], [ ], [ ]]
                    nSMBHbin = [[ ], [ ], [ ], [ ], [ ], [ ], [ ]]
                    nIMBHbin = [[ ], [ ], [ ], [ ], [ ], [ ], [ ]]
                    nhardbin = [[ ], [ ], [ ], [ ], [ ], [ ], [ ]]
                    nter = [[ ], [ ], [ ], [ ], [ ], [ ], [ ]]
                    nhardter = [[ ], [ ], [ ], [ ], [ ], [ ], [ ]]
                    
                    for file_ in np.unique(self.file_names[sim_]):
                        bin_syst = 0
                        SMBH_bin = 0
                        IMBH_bin = 0
                        hard_bin = 0
                        ter_syst = 0
                        hard_ter = 0

                        idx_file = np.argwhere(self.file_names[sim_] == file_).flatten()
                        iter = np.argwhere(5*round(0.2*self.pop[sim_][idx_file[0]]) == pop_arr)[0][0]
                        for data_ in idx_file:
                            idx_nbin = np.argwhere(np.asarray(self.new_bin_sys[sim_][data_])>0).flatten()
                            idx_tbin = np.argwhere(np.asarray(self.new_ter_sys[sim_][data_])>0).flatten()
                            bin_syst += len(idx_nbin)
                            ter_syst += len(idx_tbin)

                            for idx_ in idx_nbin:
                                if np.asarray(self.GW_bmass[sim_][data_])[idx_-1]<1e6:
                                    IMBH_bin += 1
                                else:
                                    SMBH_bin += 1
                                if np.asarray(self.hard_bin[sim_][data_])[idx_-1]>0:
                                    hard_bin += 1
                                    hard_Bool = True
                                    
                            for idx_ in idx_tbin:
                                if np.asarray(self.hard_ter[sim_][data_])[idx_-1]>0:
                                    hard_ter += 1

                        nbin[iter].append(bin_syst)
                        nSMBHbin[iter].append(SMBH_bin)
                        nIMBHbin[iter].append(IMBH_bin)
                        nhardbin[iter].append(hard_bin)
                        nter[iter].append(ter_syst)
                        nhardter[iter].append(hard_ter)

                    median_bin = [ ]
                    lower_bin = [ ]
                    upper_bin = [ ]
                    hard_bin = [ ]
                    lower_hard = [ ]
                    upper_hard = [ ]
                    median_ter = [ ]
                    lower_ter = [ ]
                    upper_ter = [ ]
                    for idx_ in range(np.shape(nbin)[0]):
                        median_bin.append(np.median(nbin[idx_]))
                        if np.median(nbin[idx_])>0:
                            q1, q3 = np.percentile(nbin[idx_], [25, 75])
                        else:
                            q1, q3 = 0, 0
                        lower_bin.append(q1)
                        upper_bin.append(q3)

                        hard_bin.append(np.median(nhardbin[idx_]))
                        if np.median(nhardbin[idx_])>0:
                            q1, q3 = np.percentile(nhardbin[idx_], [25, 75])
                        else:
                            q1, q3 = 0, 0
                        lower_hard.append(q1)
                        upper_hard.append(q3)

                        median_ter.append(np.median(nter[idx_]))
                        if np.median(nter[idx_])>0:
                            q1, q3 = np.percentile(nter[idx_], [25, 75])
                        else:
                            q1, q3 = 0, 0
                        lower_ter.append(q1)
                        upper_ter.append(q3)

                    file.write('\nBINARY DATA')
                    file.write('\nAverage binary formation time [yrs]:             '+str(pop_arr)+' : '+str(bform_med))
                    file.write('\nLower binary formation time [yrs]:               '+str(pop_arr)+' : '+str(bform_lowq))
                    file.write('\nHigher binary formation time [yrs]:              '+str(pop_arr)+' : '+str(bform_higq))
                    file.write('\n# Binary systems at initialisation:              '+str(pop_arr)+' : '+str(self.binary_init[sim_])+' / '+str(self.binary_total[sim_]))
                    file.write('\nFraction of IMBH-IMBH binaries:                  '+str(pop_arr)+' : '+str(self.binary_IMBH[sim_])+' / '+str(self.binary_total[sim_]))
                    file.write('\nFraction of hard binaries:                       '+str(pop_arr)+' : '+str(self.binary_hard[sim_])+' / '+str(self.binary_total[sim_]))
                    file.write('\nFraction of mergers within Hubble time:          '+str(pop_arr)+' : '+str(self.GWb_mergers[sim_])+' / '+str(self.binary_total[sim_]))
                    file.write('\nMedian # binaries:                               '+str(pop_arr)+' : '+str(median_bin))
                    file.write('\nLower percentile # binaries:                     '+str(pop_arr)+' : '+str(lower_bin))
                    file.write('\nUpper percentile # binaries:                     '+str(pop_arr)+' : '+str(upper_bin))
                    file.write('\nMedian # hard binaries:                          '+str(pop_arr)+' : '+str(hard_bin))
                    file.write('\nLower percentile # hard binaries:                '+str(pop_arr)+' : '+str(lower_hard))
                    file.write('\nUpper percentile # hard binaries:                '+str(pop_arr)+' : '+str(upper_hard))

                    file.write('\n\nTERTIARY DATA')
                    file.write('\n# Tertiary systems at initialisation:            '+str(pop_arr)+' : '+str(self.tertiary_init[sim_])+' / '+str(self.tertiary_total[sim_]))
                    file.write('\nFraction of IMBH-IMBH tertiaries:                '+str(pop_arr)+' : '+str(self.tertiary_IMBH[sim_])+' / '+str(self.tertiary_total[sim_]))
                    file.write('\nFraction of hard tertiaries:                     '+str(pop_arr)+' : '+str(self.tertiary_hard[sim_])+' / '+str(self.tertiary_total[sim_]))
                    file.write('\nFraction of mergers within Hubble time:          '+str(pop_arr)+' : '+str(self.GWt_mergers[sim_])+' / '+str(self.tertiary_total[sim_]))
                    file.write('\nMEdian tertiary formation time [yrs]:            '+str(pop_arr)+' : '+str(tform_med))
                    file.write('\nLower tertiary formation time [yrs]:             '+str(pop_arr)+' : '+str(tform_lowq))
                    file.write('\nHigher tertiary formation time [yrs]:            '+str(pop_arr)+' : '+str(tform_higq))
                    file.write('\nMedian # tertiary:                               '+str(pop_arr)+' : '+str(median_ter))
                    file.write('\nLower percentile # tertiary:                     '+str(pop_arr)+' : '+str(lower_ter))
                    file.write('\nUpper percentile # tertiary:                     '+str(pop_arr)+' : '+str(upper_ter))
            iterf += 1
            
    def GW_emissions(self):
        
        GW_calcs = gw_calcs()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plot_ini = plotter_setup()
        axlabel_size, ticklabel_size = plot_ini.font_size()

        plt.clf()

        ####### PLOT FOR ALL ########
        iterf = 0
        for fold_ in self.folders[:self.frange]:
            integrator, drange = folder_loop(iterf)

            for int_ in range(drange):
                sim_ = folder_data_loop(iterf, int_)
                fig = plt.figure(figsize=(8, 6))
                gs = fig.add_gridspec(2, 2,  width_ratios=(4, 2), height_ratios=(2, 4),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.06, hspace=0.06)
                ax = fig.add_subplot(gs[1, 0])
                ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
                ax2 = fig.add_subplot(gs[1, 1], sharey=ax)
                ax.set_xlabel(r'$\log_{10}f$ [Hz]', fontsize = axlabel_size)
                ax.set_ylabel(r'$\log_{10}h$', fontsize = axlabel_size)
                
                tertiary = False
                if len(self.GWfreq_ter[sim_])>0:
                    tertiary = True
                GW_calcs.scatter_hist(self.GWfreq_bin[sim_], self.GWstra_bin[sim_],
                                      self.GWfreq_ter[sim_], self.GWstra_ter[sim_],
                                      ax, ax1, ax2, 'Binary', 'Hierarchical',
                                      tertiary, False)

                for ax_ in [ax, ax1, ax2]:
                    plot_ini.tickers(ax_, 'plot')
                    ax_.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    ax_.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                ax.set_ylim(-38, -12.2)
                ax.set_xlim(-13.8, 0.2)
                plt.savefig('figures/binary_hierarchical/'+str(integrator[int_])+'_'+fold_+'GW_scatter_allbins__diagram.png', dpi = 500, bbox_inches='tight')
                plt.clf()
                
                fig = plt.figure(figsize=(8, 6))
                gs = fig.add_gridspec(2, 2,  width_ratios=(4, 2), height_ratios=(2, 4),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.06, hspace=0.06)
                ax = fig.add_subplot(gs[1, 0])
                ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
                ax2 = fig.add_subplot(gs[1, 1], sharey=ax)

                GWfreq_binIMBH = self.array_rewrite(self.GWfreq_binIMBH[sim_], 'nested', False)
                GWstra_binIMBH = self.array_rewrite(self.GWstra_binIMBH[sim_], 'nested', False)
                GWfreq_terIMBH = self.array_rewrite(self.GWfreq_terIMBH[sim_], 'nested', False)
                GWstra_terIMBH = self.array_rewrite(self.GWstra_terIMBH[sim_], 'nested', False)

                tertiary = False
                if len(GWstra_terIMBH)>0:
                    tertiary = True

                GW_calcs.scatter_hist(GWfreq_binIMBH, GWstra_binIMBH,
                                      GWfreq_terIMBH, GWstra_terIMBH,
                                      ax, ax1, ax2, 'Hard Binary', 'Hierarchical',
                                      tertiary, False)

                ax.set_xlabel(r'$\log_{10}f$ [Hz]', fontsize = axlabel_size)
                ax.set_ylabel(r'$\log_{10}h$', fontsize = axlabel_size)
                for ax_ in [ax, ax1, ax2]:
                    plot_ini.tickers(ax_, 'plot')
                    ax_.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    ax_.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                ax.set_ylim(-28.5, -12.2)
                ax.set_xlim(-13.8, 0.2)
                plt.savefig('figures/binary_hierarchical/'+str(integrator[int_])+'_'+fold_+'_GW_hardbins_diagram.png', dpi = 500, bbox_inches='tight')
                plt.clf()
            iterf += 1

    def single_streak_plotter(self):
        """
        Function to illustrate the streak-like characteristic emerging from the simulation
        """

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plot_ini = plotter_setup()
        axlabel_size, tick_size = plot_ini.font_size()
        
        integrators = ['Hermite', 'GRX']

        x_temp = np.linspace(1e-5, 1, 1000)

        lisa = li.LISA() 
        Sn = lisa.Sn(x_temp)
        SKA = np.load(os.path.join(os.path.dirname(__file__), 'SGWBProbe/files/hc_SKA.npz'))
        SKA_freq = SKA['x']
        SKA_hc = SKA['y']
        SKA_strain = SKA_hc**2/SKA_freq
        Ares = np.load(os.path.join(os.path.dirname(__file__), 'SGWBProbe/files/S_h_muAres_nofgs.npz'))
        Ares_freq = Ares['x']
        Ares_strain = Ares['y']

        for int_ in range(1):
            int_ += 1
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.log10(x_temp), np.log10(np.sqrt(x_temp*Sn)), 
                    color = 'slateblue', zorder = 1)
            ax.plot(np.log10(Ares_freq), np.log10(np.sqrt(Ares_freq*Ares_strain)), 
                    linewidth='1.5', color='red', zorder = 3)
            ax.plot(np.log10(SKA_freq), np.log10(np.sqrt(SKA_freq*SKA_strain)), 
                    linewidth='1.5', color='orangered', zorder = 4)
            ax.text(-9.7, -16.1, 'SKA', fontsize = axlabel_size, 
                    rotation=310, color = 'orangered')
            ax.text(-4.7, -18.2, 'LISA', fontsize = axlabel_size, 
                    rotation = 300, color = 'slateblue')
            ax.text(-6.5, -19, r'$\mu$Ares', fontsize = axlabel_size, 
                    rotation = 300, color = 'red')

            idx = [7, 19]  #Hardcoded values
            GWfreq_binIMBH = self.array_rewrite(self.GWfreq_binIMBH[int_][idx[int_]], 'not', False)
            GWstra_binIMBH = self.array_rewrite(self.GWstra_binIMBH[int_][idx[int_]], 'not', False)
            GWfreq_terIMBH = self.array_rewrite(self.GWfreq_terIMBH[int_][idx[int_]], 'not', False)
            GWstra_terIMBH = self.array_rewrite(self.GWstra_terIMBH[int_][idx[int_]], 'not', False)

            GWtime_binIMBH = [(1e-3*i) for i in range(len(GWfreq_binIMBH))]
            GWtime_terIMBH = [(1e-3*i) for i in range(len(GWfreq_terIMBH))]
            colour_axes = ax.scatter(np.log10(GWfreq_binIMBH), np.log10(GWstra_binIMBH), s = 7, c = GWtime_binIMBH, zorder = 5)
            if len(GWfreq_terIMBH)>0:
                ax.scatter(np.log10(GWfreq_terIMBH), np.log10(GWstra_terIMBH), c = GWtime_terIMBH)
            cbar = plt.colorbar(colour_axes, ax=ax)
            cbar.set_label(label = r'$t_{\rm{sys}}$ [Myr]', fontsize =  axlabel_size)
            cbar.ax.tick_params(labelsize = axlabel_size)
            
            ax.set_xlabel(r'$\log_{10}f$ [Hz]', fontsize = axlabel_size)
            ax.set_ylabel(r'$\log_{10}h$', fontsize = axlabel_size)
            plot_ini.tickers(ax, 'plot')
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax.set_ylim(-28.5, -12.2)
            ax.set_xlim(-13.8, 0.2)
            plt.savefig('figures/binary_hierarchical/'+str(integrators[int_])+'GW_single_streak_diagram.pdf', dpi = 500, bbox_inches='tight')
            plt.clf()
    
    def sys_occupancy_plotter(self):
        """
        Function to plot various 'sustainable system' plots
        """

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plot_ini = plotter_setup()
        axlabel_size, tick_size = plot_ini.font_size()

        integrator = ['Hermite', 'GRX']
        colors = ['red', 'blue']
        markers = ['o', 's']
        xshift = [-1, 1]

        normalise_p1 = plt.Normalize(0, (max(self.binary_systems[0])))

        xtemp = np.linspace(10, 40, 1000)

        with open('figures/binary_hierarchical/output/line_of_best_fit_m4e6.txt', 'w') as file:
            fig, ax = plt.subplots()
            for int_ in range(2):

                ini_pop = np.unique(self.pop[int_])
                if int_ == 0:
                    ini_pop = [i+xshift[int_] for i in ini_pop]
                else:
                    for i in range(len(ini_pop)):
                        if (ini_pop[i]/2)%5 == 0:
                            ini_pop[i] += +xshift[int_]

                best_fit = np.polyfit(ini_pop, np.log10(self.binary_occupation[int_]), 1)
                curve = np.poly1d(best_fit)
                file.write('Data for      '+str(integrator[int_]))
                file.write('\nFactor:       '+str(best_fit[0]))
                file.write('\ny-intercept:  '+str(best_fit[1])+'\n\n')

                ax.set_ylabel(r'$\log_{10}(t_{\rm{sys}} / t_{\rm{sim}})$', fontsize=axlabel_size)
                ax.plot(xtemp, curve(xtemp), color=colors[int_], linestyle=':', zorder=1)
                colour_axes = ax.scatter(ini_pop, np.log10(self.binary_occupation[int_]), 
                                         edgecolors='black', s=88, marker=markers[int_],
                                         c=(self.binary_systems[int_]), norm=(normalise_p1), 
                                         label=integrator[int_], zorder=2)
                ax.scatter(ini_pop, np.log10(self.bocc_highr[int_]), 
                           c='black', marker='_')
                ax.scatter(ini_pop, np.log10(self.bocc_lower[int_]), 
                           c='black', marker='_')
                ax.plot([ini_pop, ini_pop], [np.log10(self.bocc_lower[int_]), np.log10(self.binary_occupation[int_])], c='black', zorder=1)
                ax.plot([ini_pop, ini_pop], [np.log10(self.binary_occupation[int_]), np.log10(self.bocc_highr[int_])], c='black', zorder=1)
            plot_ini.tickers_pop(ax, self.pop[1], 'GRX')
            ax.legend(prop={'size': axlabel_size})
            cbar = plt.colorbar(colour_axes, ax=ax)
            cbar.set_label(label=r'$\mathrm{med}(N_{\rm{sys}})$ ', fontsize= axlabel_size)
            cbar.ax.tick_params(labelsize=axlabel_size)
            plt.savefig('figures/binary_hierarchical/sys_form_m4e6.pdf', dpi=300, bbox_inches='tight')

