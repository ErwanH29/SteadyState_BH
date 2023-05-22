from amuse.lab import *
from file_logistics import *
from spatial_plotters import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

warnings.filterwarnings("ignore")

def PS_function(zmass, phi0, mc, alpha):
    """
    The Press-Schecter stellar mass function (Press & Schechter 1974).
    Value is in (Mpc)^-3
    
    Inputs:
    zmass:  The redshift-dependent mass integrated over
    phi0:   The redshift-dependent normalisation factor
    mc:     The redshift-dependent characteristic mass. Describes the knee
    alpha:  The redshift-dependent power-law mass slope
    """

    ps_func = phi0 * (zmass/mc)**alpha*np.exp(-zmass/mc)
    return ps_func

def cmove_dist(z):
    """
    The distance based on redshift. The constants are taken from Planck 2018.
    Value is in Mpc.
    
    Inputs:
    z:      The redshift to integrate over
    """

    H0 = 67.4 #From arXiv:1807.06209
    c = constants.c.value_in(units.kms)
    omegaL = 0.685
    omegaM = 0.315
    return c/H0 * (np.sqrt(omegaL+omegaM*(1+z)**3))**-1

def plotter():

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plot_ini = plotter_setup()
    dummy1, dummy2, axlabel_size, tick_size = plot_ini.font_size()
        

    # Data from Furlong et al. (2015)
    merger_rate = np.linspace(1e-3, 1, 100) #0.9/7 -> 7 / 0.9 merge per Myr
    mrate_fixed = [1e-3, 1e-2, 1e-1, 1]
    zrange = [0, 0.5, 1, 2, 3]
    phi0 = np.asarray([8.4, 8.4, 7.4, 4.5, 2.2]) 
    phi0 *= 1e-4
    mcluster = [10**11.14, 10**11.11, 10**11.06, 10**10.91, 10**10.78]
    alpha = [-1.43, -1.45, -1.48, -1.57, -1.66]
    eps = 0.8

    ps_int = []
    cmove_dist_int = []
    for i in range(len(zrange)-1):
        ps_int.append(quad(PS_function, 1e8, 1e14, args=(phi0[i], mcluster[i], alpha[i]))[0])
        cmove_dist_int.append(quad(cmove_dist, zrange[i], zrange[i+1])[0])

    ytext = r'$\Gamma_{\rm{events}}$ [yr$^{-1}$]'
    xtext = r'$\Gamma_{\rm{infall}}$ [Myr$^{-1}$]'
    
    fig, ax = plt.subplots()
    ax.set_ylabel(r'$\log_{10}$'+ytext, fontsize = axlabel_size)
    ax.set_xlabel(r'$\log_{10}$'+xtext, fontsize = axlabel_size)
    ax.tick_params(axis="y", direction="in", labelcolor='black')
    ax.set_ylim(0.1,4.4)
    plot_ini.tickers(ax, 'plot')

    cum_merger = [ ]
    for rate_ in merger_rate:
        mergerval_temp = 0
        for i in range(len(ps_int)):
            mergerval_temp += (4/3)*eps*np.pi*rate_*(6e10)**-1*(ps_int[i] * cmove_dist_int[i]**3) * 1e-6
        cum_merger.append(mergerval_temp)
    ax.plot(np.log10(merger_rate), np.log10(cum_merger),  color = 'black')
    
    cum_merger_fixed = [ ]
    itert = 0
    for rate_ in mrate_fixed:
        mergerval_temp = 0
        for i in range(len(ps_int)):
            mergerval_temp += (4/3)*eps*np.pi*rate_*(6e10)**-1*(ps_int[i] * cmove_dist_int[i]**3) * 1e-6
        cum_merger_fixed.append(mergerval_temp)
        if itert <= 1:
            ax.text(np.log10(rate_)+0.05, np.log10(mergerval_temp)-0.5, 
                    xtext+' = '+r"$10^{{{0:.3g}}}$".format(np.log10(rate_))+'\n'+ytext+' = '+"{:.3f}".format(mergerval_temp)[:4], 
                    horizontalalignment='left', fontsize = 13)
        else:
            if itert == 2:
                no_dig = 3
                merger_text = mergerval_temp
            else:
                no_dig += 1
                merger_text = 10*round(0.1*mergerval_temp)
            ax.text(np.log10(rate_)-0.03, np.log10(mergerval_temp)+0.1, 
                    xtext+' = '+r"$10^{{{0:.3g}}}$".format(np.log10(rate_))+'\n'+ytext+' = '+"{:.3f}".format(merger_text)[:no_dig], 
                    horizontalalignment='right', fontsize = 13)
        itert += 1

    ax.scatter(np.log10(mrate_fixed), np.log10(cum_merger_fixed), color = 'black')
    plt.savefig('figures/forecast/merger_rate.pdf', dpi=300, bbox_inches='tight')
    
plotter()
