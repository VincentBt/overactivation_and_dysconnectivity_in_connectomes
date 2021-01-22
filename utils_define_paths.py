import os
import multiprocessing


def check_folder_exists(path_folder):
    """
    Check that the folder at path path_folder exists and if not, create it
    """
    if path_folder[-1] == '/':
        path_folder = path_folder[:-1]
    path_parent = '/'.join(path_folder.split('/')[:-1]) + '/' #parent dir
    name_folder = path_folder.split('/')[-1]
    if name_folder not in os.listdir(path_parent):
        print("Creating a folder to save the simulation files, at path: {}".format(path_folder))
        os.mkdir(path_folder)
    

def define_paths():
    """
    Defines path_dir_save and path_data, depending on the computer used
    Defines also n_parallel_processes for compute_effects_CI_vs_BP.py
    """
    n_cpu_cores = multiprocessing.cpu_count() #number of CPU cores
    
    which_computer = os.path.abspath("").split("/")[-3]
    if which_computer == 'ENS Ulm':
        path_dir_save = '../../results_code/simulations_CI_BP/compute_effects_CI_vs_BP/'
        path_nextcloud = '/home/vincent/Nextcloud/'  #on my Asus laptop
        path_data = '../../other code (not mine)/data (not mine)/connectomics/Renaud/'
        dir_save_fig = '../../Figures/SCZResearch2020/'
        n_parallel_processes = 4
    elif (which_computer == 'Vincent_Bouttier') and (os.path.abspath("").split("/")[-4] == 'cure2_ubuntu'): #Ubuntu in Lille
        #save into the NextCloud server (in Windows) which synchronizes with my Asus laptop
        path_dir_save = '/mnt/d/ESPACE DE TRAVAIL COMMUN/Vincent/NextCloud_Vincent_Bouttier/'
        path_nextcloud = '/mnt/d/ESPACE DE TRAVAIL COMMUN/Vincent/NextCloud_Vincent_Bouttier/' #on Lille's computer
        path_data = '../../other code (not mine)/data (not mine)/connectomics/Renaud/'
        dir_save_fig = '../../Figures/SCZResearch2020/'
        n_parallel_processes = 36
    else:
#         print("Computer unknown")
        path_abs = os.path.abspath("")
        path_dir_save = path_abs + '/' + 'data_simulations' + '/'
#         check_folder_exists(path_dir_save) #create folder if it does not exist
        path_data = path_abs + '/' + 'data' + '/'
        dir_save_fig = path_abs + '/' + 'figures' + '/'
        assert n_cpu_cores > 4
        n_parallel_processes = 4 #in order not to make the computer crash
#     print("path_dir_save = {}".format(path_dir_save))
#     print("path_data = {}".format(path_data))
    return path_dir_save, path_data, n_parallel_processes, dir_save_fig
    
    
path_dir_save, path_data, n_parallel_processes, dir_save_fig = define_paths()