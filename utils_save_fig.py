import matplotlib.pyplot as plt
from utils_define_paths import * #defines dir_save_fig (global variable)
import os

def save_fig(figname, type_graph, savefigs=True):
    if savefigs:
        parent_dir = '/'.join(dir_save_fig[:-1].split('/')[:-1]) + '/'
        name_dir_save = dir_save_fig[:-1].split('/')[-1]
        if name_dir_save not in os.listdir(parent_dir):
            os.mkdir(parent_dir + name_dir_save)
            print("created folder '{}' at path {}".format(name_dir_save, parent_dir))
        for format_file in ['png', 'svg']: #, 'eps'
            if format_file not in os.listdir(dir_save_fig):
                os.mkdir(dir_save_fig + format_file)
                print("create folder '{}' at path {}".format(format_file, dir_save_fig))
            if type_graph == 'realistic_connectome_AAL':
                dir_save = dir_save_fig + format_file + '/' + type_graph + '/' #saving in a subfolder, in order not to mix things (realistic_connectome_AAL is not used in the article, contrary to modular_SW and realistic_connectome_AAL2)
            else: #default case
                dir_save = dir_save_fig + format_file + '/'
            plt.savefig(dir_save + figname + '_' + type_graph + '.' + format_file, bbox_inches='tight')
            print("saved figure '{}' at path {}".format(figname, dir_save + figname + '_' + type_graph + '.' + format_file))

                
def save_text(figname, type_graph, list_strings, savefigs=True):
    if savefigs:
        with open(dir_save_fig + figname + '_' + type_graph + '.txt', "w") as file: #write mode 
            for s in list_strings:
                file.write("{} \n".format(s))