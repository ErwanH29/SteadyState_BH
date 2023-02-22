from amuse.lab import *
from file_logistics import *
from spatial_plotters import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

def PS_function(zmass, phi0, mc, alpha):
    """
    The Press-Schecter stellar mass function (Press, Schechter 1974).
    Value is in (Mpc)^-3
    
    Inputs:
    zmass:  The redshift-dependent mass integrating over
    phi0:   The redshift-dependent normalisation factor
    mc:     The redshift-dependent characteristic mass. Describes the knee
    alpha:  The redshift-dependent power-law mass slope
    """

    ps_func = phi0 * (zmass/mc)**alpha*np.exp(-zmass/mc)
    return ps_func

def calc_loop(iter, const_arr, range_a, range_b, phi0, phi0_diff, mc, mc_diff, alpha, alpha_diff, phen):
    """
    Function to calculate the events per cubic Gpc per year for both a constant GC formation
    and a redshift-dependent one
    
    Inputs:
    iter:        Current iteration of the for loop
    range_a:     The number of data points in the current computing redshift interval
    range_b:     The number of data points in the previous redshift interval
    phi0:        The phi0 constant
    phi0_diff:   The change in phi0 over the redshift interval
    mc:          The mc constant
    mc_diff:     The change in mc over the redshift interval
    alpha:       The power-law relating how mass evolves over redshift
    alpha_diff:  The change in the power-law over the redshift interval
    phen:        The phenomena (ULX | SMBH-IMBH merger) wishing to compute rate of
    """
    
    eff = 1
    const_rate = const_arr[phen]
    NIMBH = 1

    phi0_val = phi0 + phi0_diff * (iter-range_b)/(range_a-range_b)
    mc_val = mc + mc_diff * (iter-range_b)/(range_a-range_b)
    alpha_val = alpha + alpha_diff * (iter-range_b)/(range_a-range_b)
    PS_val = quad(PS_function, 10**8, 10**14, args=(phi0_val, mc_val, alpha_val))[0]

    Nevents_const = (4/3)*np.pi*eff*NIMBH*const_rate*(6*10**10)**-1 * PS_val

    return Nevents_const

def cmove_dist(z):
    """
    The distance based on redshift.
    The constants are taken from Planck 2018.
    Value is in Mpc.
    Inputs:
    z:      The redshift to integrate over
    """

    H0 = 67
    c = constants.c.value_in(units.kms)
    omegaL = 0.673
    omegaM = 0.315
    return c/H0 * (np.sqrt(omegaL+omegaM*(1+z)**3))**-1

def phenomena_event():
    """
    Function which manipulates the data and plots it depending on the
    phenomena (phen) asked.
    """

    plot_init = plotter_setup()

    # Data from Furlong et al. (2015)
    merger_rate = np.linspace(0, 1, 1000) #0.9/7 -> 7 / 0.9 merge per Myr
    redshift_range = [0, 0.5, 1, 2, 3]
    phi0_range = np.asarray([8.4, 8.4, 7.4, 4.5, 2.2]) 
    phi0_range *= 10**-4
    mc_range = [10**11.14, 10**11.11, 10**11.06, 10**10.91, 10**10.78]
    alpha_range = [-1.43, -1.45, -1.48, -1.57, -1.66]


    ps_int = []
    cmove_dist_int = []
    for i in range(len(redshift_range)-1):
        ps_int.append(quad(PS_function, 10**8, 10**14, args=(phi0_range[i], mc_range[i], alpha_range[i]))[0])
        cmove_dist_int.append(quad(cmove_dist, redshift_range[i], redshift_range[i+1])[0])

    phi0_diffs = []
    mc_diffs = []
    alpha_diffs = []
    for i in range(len(phi0_range)-1):
        phi0_diffs.append(phi0_range[i+1] - phi0_range[i])
        mc_diffs.append(mc_range[i+1] - mc_range[i])
        alpha_diffs.append(alpha_range[i+1] - alpha_range[i])
    phi0_diffs = np.asarray(phi0_diffs)
    mc_diffs = np.asarray(mc_diffs)
    alpha_diffs = np.asarray(alpha_diffs)

    value_const_temp = 0
    NIMBH = 1

    cum_GCc_all = [ ]
    for rate_ in merger_rate:
        cum_GCc = [ ]
        for i in range(len(ps_int)):
            value_const_temp += (4/3)*np.pi*NIMBH*rate_*(6*10**10)**-1*(ps_int[i] * cmove_dist_int[i]**3) * 10**-6
            cum_GCc.append(value_const_temp)
        cum_GCc_all.append(cum_GCc)
    zvals = [0, 0.5, 1, 2, 3]
    cum_GCc.insert(0,0)

    xnew = np.linspace(0,3,1000)
    fc = interp1d(zvals, cum_GCc, kind='quadratic')
    fc = fc(xnew)

    fig, ax = plt.subplots()
    ax.set_ylabel(r'$\log_{10} N_{\rm{Events}}$ [yr$^{-1}$]', color = 'black')
    ax.set_xlim(0,3)
    ax.set_ylim(0.1,3.05)
    ax.set_xlabel(r'Redshift') 
    ax.tick_params(axis="y", direction="in", labelcolor='black')
    plot_init.tickers(ax, 'plot')
    ax.plot(xnew, np.log10(fc),  color = 'black')
    plt.savefig('figures/forecast/merger_rate.pdf', dpi=300, bbox_inches='tight')

    print('Total IMBH-SMBH mergers: ', max(fc))

phenomena_event()