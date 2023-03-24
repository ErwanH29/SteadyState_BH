import time as cpu_time
from tGW_plotters import *
from sustainable_bintert_plotters import *
from steady_plotters import *
from ejection_stat_plotters import *
from spatial_plotters import *

print('...spatial_plotters...')
start_spatial = cpu_time.time()
#ecc_semi_histogram('GRX')
energy_scatter()
spatial_plotter('GRX')
#global_properties()
end_spatial = cpu_time.time()
print('Plotting time: ', end_spatial - start_spatial, ' seconds')
STOP

print('...tGW_plotters...')
start_tgw = cpu_time.time()
cls = gw_calcs()
cls.new_data_extractor()
#cls.orbital_hist_plotter()
cls.strain_freq_plotter()
end_tgw = cpu_time.time()
print('Plotting time: ', end_tgw - start_tgw, ' seconds')
STOP



print('...steady_plotter...')
start_steady = cpu_time.time()
cls = stability_plotters()
cls.overall_steady_plotter()
end_steady = cpu_time.time()
print('Plotting time: ', end_steady - start_steady, ' seconds')
STOP





print('... ejection_stat_plotters ...')
start_ejec = cpu_time.time()
event_tracker()
cls = ejection_stats()
#cls.new_data_extractor()
cls.vejec_plotters()
end_ejec = cpu_time.time()
print('Plotting time: ', end_ejec - start_ejec, ' seconds')

print('...sustainable_bintert_plotters...')
cls = sustainable_sys()
#cls.new_data_extractor()   
cls.combine_data()
#cls.GW_emissions()
#cls.single_streak_plotter()
cls.sys_occupancy_plotter()
cls.sys_popul_plotter()