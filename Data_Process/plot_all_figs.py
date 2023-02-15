import time as cpu_time
from tGW_plotters import *
from sustainable_bintert_plotters import *
from steady_plotters import *
from ejection_stat_plotters import *
from spatial_plotters import *

start_time = cpu_time.time()

print('...tGW_plotters...')
cst = gw_calcs()
#cst.new_data_extractor()
cst.orbital_hist_plotter()
cst.strain_freq_plotter()
STOP

STOP





print('... ejection_Stat_plotters ...')
cst = event_tracker()
cst = ejection_stats()
cst.new_data_extractor()
cst.combine_data()
cst.energy_plotters()
cst.vejec_plotters()
end_ejec = cpu_time.time()


print('...steady_plotter...')
cst = stability_plotters()
cst.overall_steady_plotter()

print('...spatial_plotters...')
global_properties()
spatial_plotter('GRX')

print('...sustainable_bintert_plotters...')
cst = sustainable_sys()
#cst.new_data_extractor()   
cst.combine_data()
cst.GW_emissions()
cst.single_streak_plotter()
#cst.sys_occupancy_plotter()
#cst.sys_popul_plotter()