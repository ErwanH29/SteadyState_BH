import glob
import fnmatch
import os

def file_counter(path):
    """
    Counts the number of files in a directory.
    """
    
    dir_path = path+'simulation_stats/'
    count = len(fnmatch.filter(os.listdir(dir_path), '*.*'))
    return count

def file_reset(dir):
    """
    Remove all files from a directory
    """

    filelist = glob.glob(os.path.join(dir, "*"))
    for f in filelist:
        os.remove(f)
