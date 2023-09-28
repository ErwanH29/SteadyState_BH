import glob
import fnmatch
import os

def file_counter():
    """
    Function which counts the number of files in a directory.
    """
    
    dir_path = '/home/s2009269/data1/GRX_Orbit_Data_rc0.25_m4e6/simulation_stats/'
    count = len(fnmatch.filter(os.listdir(dir_path), '*.*'))
    return count

def file_reset(dir):
    """
    Function to remove all files from a directory
    """
    
    char = input("WARNING: You are about to delete all files in"+str(dir)+". Are you sure? (Y | N)")
    if char.lower() == "y":
        filelist = glob.glob(os.path.join(dir, "*"))
        for f in filelist:
            os.remove(f)
