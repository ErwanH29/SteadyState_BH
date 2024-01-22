import time as cpu_time
from tGW_plotters import *
from sustainable_bintert_plotters import *
from steady_plotters import *
from ejection_stat_plotters import *
from spatial_plotters import *

print('...steady_plotter...')
start_steady = cpu_time.time()
cls = stability_plotters()
cls.overall_steady_plotter()
end_steady = cpu_time.time()
print('Plotting time: ', end_steady - start_steady, ' seconds')
STOP
sphere_of_influence()
STOP

print('...tGW_plotters...')
start_tgw = cpu_time.time()
cls = gw_calcs()
#ecc_mergers()
#cls.new_data_extractor()
#cls.strain_freq_plotter()
cls.GW_event_tracker()
#cls.orbital_hist_plotter()
end_tgw = cpu_time.time()
print('Plotting time: ', end_tgw - start_tgw, ' seconds')
STOP

print('...spatial_plotters...')
start_spatial = cpu_time.time()
#spatial_plotter()
#lagrangian_tracker()
#energy_scatter()
#global_properties()
#global_vels_GRX_pops()
#global_properties_GRX_pops()
ecc_semi_histogram('GRX')
end_spatial = cpu_time.time()
print('Plotting time: ', end_spatial - start_spatial, ' seconds')
STOP


# ======================================================================= #






print('...sustainable_bintert_plotters...')
cls = sustainable_sys()
#cls.new_data_extractor()
cls.combine_data()
#cls.GW_emissions()
#cls.single_streak_plotter()
cls.sys_occupancy_plotter()



print('... ejection_stat_plotters ...')
start_ejec = cpu_time.time()
event_tracker()
cls = ejection_stats()
#cls.new_data_extractor()
cls.vejec_plotters()
end_ejec = cpu_time.time()
print('Plotting time: ', end_ejec - start_ejec, ' seconds')
STOP


