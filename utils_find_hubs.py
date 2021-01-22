from utils_graph_measures import *
from graph_generator import *

############################################## NOTES: ##############################################################
# Hubs = high degree and high nodal centrality. Low clustering??

# Other possible metrics to compute:
# - modularity
# - clustering coefficient (check that I compute the right one)
# - Participation coefficient +++
# - Betweenness-centrality +++ (see Lord et al 2017)

# TODO = classifiy the nodes as connector hubs (high degree-centrality, high betweenness-centrality, high participation coefficient) / provincial hubs (high degree-centrality, low betweenness-centrality, low participation coefficient) / other nodes (low degree-centrality, low betweenness-centrality, low participation coefficient)
####################################################################################################################

def sort_dict_by_value(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))


####################################################################################################################
##### Criteria from Bertolero et al 2018 (A mechanistic model of connector hubs, modularity and cognition) #########             
def find_connector_hubs(G):
    """
    Criteria from [Bertolero et al 2018, Nature Hum Behav]: connector hubs = top 20% highest participation coefficient nodes
    """
    d = get_participation_coefficient_undirected(G)
    threshold = np.quantile(list(d.values()), q=0.8)
#     connector_hubs = [key for key,val in d.items() if val >= threshold]
    connector_hubs = [key for key,val in sort_dict_by_value(d).items() if val >= threshold]
    return connector_hubs

def find_local_hubs(G):
    """
    Criteria from [Bertolero et al 2018, Nature Hum Behav]: local hubs = top 20% highest within-community strength nodes
    """
    d = within_community_strength_undirected(G)
    threshold = np.quantile(list(d.values()), q=0.8)
#     local_hubs = [key for key,val in d.items() if val >= threshold]
    local_hubs = [key for key,val in sort_dict_by_value(d).items() if val >= threshold]
    return local_hubs



############################################################################################################################
##### Criteria adapted from Bertolero et al 2018 (A mechanistic model of connector hubs, modularity and cognition) #########   
##### (but imposing that a node has only one category: local hub / connector hub / other node ##############################
##### difference with above = a local hub cannot have a high participation coefficient #####################################

def type_connector_hub(G, node):
    """
    Classif connector hub vs other nodes
    """
    is_connector_hub = node in find_connector_hubs(G)
    if is_connector_hub:
        return 'connector_hub'
    else:
        return 'other_node'
    

def type_local_hub(G, node):
    """
    Classif local hub vs other nodes
    """
    is_local_hub = node in find_local_hubs(G)
    if is_local_hub:
        return 'local_hub'
    else:
        return 'other_node'
    
    
def type_hub(G, node):
    """
    Classif with local hub / connector hub / other nodes
    (where a node has only one category, and more particularly a local hub cannot be a connector hub)
    """
    is_local_hub = node in find_local_hubs(G)
    is_connector_hub = node in find_connector_hubs(G)
    if is_connector_hub:
        return 'connector_hub'
    elif is_local_hub:
        return 'local_hub'
    else:
        return 'other_node'





####################################################################################################################
##### Critera from [Functional cartography of complex metabolic networks, Guimera & Amaral, Nature 2008] ###########

def classify_nodes(G, plot_inter_vs_intra_connectivity=True):
    """
    Criteria from [Functional cartography of complex metabolic networks, Guimera & Amaral, Nature 2008]
    (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2175124/)
    """
    d_within_community_strength_undirected = within_community_strength_undirected(G)
    d_participation_coefficient = get_participation_coefficient_undirected(G)
    assert set(list(d_within_community_strength_undirected.keys())) == set(list(d_participation_coefficient.keys()))
    d_within_community_strength_undirected = {key:d_within_community_strength_undirected[key] for key in d_participation_coefficient}
    list_nodes = list(d_participation_coefficient.keys())
    inter_connectivity = np.array(list(d_participation_coefficient.values()))
    intra_connectivity = np.array(list(d_within_community_strength_undirected.values()))
    R_nodes = {}
    R_nodes[1] = np.nonzero((inter_connectivity <= 0.05) * (intra_connectivity < 2.5))[0]
    R_nodes[2] = np.nonzero((0.05 < inter_connectivity) * (inter_connectivity <= 0.625) * (intra_connectivity < 2.5))[0]
    R_nodes[3] = np.nonzero((0.625 < inter_connectivity) * (inter_connectivity <= 0.8) * (intra_connectivity < 2.5))[0]
    R_nodes[4] = np.nonzero((0.8 < inter_connectivity) * (intra_connectivity < 2.5))[0]
    R_nodes[5] = np.nonzero((inter_connectivity <= 0.3) * (intra_connectivity > 2.5))[0]
    R_nodes[6] = np.nonzero((0.3 < inter_connectivity) * (inter_connectivity <= 0.75) * (intra_connectivity > 2.5))[0]
    R_nodes[7] = np.nonzero((0.75 < inter_connectivity) * (intra_connectivity > 2.5))[0]
#     print({'R'+str(key)+'_nodes':len(val) for key,val in R_nodes.items()}) #how many nodes of each class
    R_nodes = {'R'+str(key)+'_nodes':[list_nodes[el] for el in val] for key,val in R_nodes.items()}
    mapping = {
        'R1_nodes': 'Ultra-peripheral nodes',
        'R2_nodes': 'Peripheral nodes',
        'R3_nodes': 'Non-hub connectors',
        'R4_nodes': 'Non-hub kinless nodes',
        'R5_nodes': 'Provincial hubs',
        'R6_nodes': 'Connector hubs',
        'R7_nodes': 'Kinless hubs',
    }
    
    if plot_inter_vs_intra_connectivity:
        fig, ax = plt.subplots(1, figsize=(12,6))
        plt.scatter(inter_connectivity, intra_connectivity, color='black')
        plt.xlabel('inter_connectivity (participation coefficient)')
        plt.ylabel('intra_connectivity (z-score)')
        colors_modules = {
            'R1': 'black',
            'R2': 'red',
            'R3': 'green',
            'R4': 'blue',
            'R5': 'gold',
            'R6': 'purple',
            'R7': 'grey',
        }
        plt.text(0.025,0,'Ultra-peripheral nodes', rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=14, color=colors_modules['R1'])
        plt.text(0.35,0,'Peripheral nodes', rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=14, color=colors_modules['R2'])
        plt.text(0.7,0,'Non-hub connector nodes', rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=14, color=colors_modules['R3'])
        plt.text(0.9,0,'Non-hub kinless nodes', rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=14, color=colors_modules['R4'])
        plt.text(0.15,3,'Provincial hubs', rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=14, color=colors_modules['R5'])
        plt.text(0.525,3,'Connector hubs', rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=14, color=colors_modules['R6'])
        plt.text(0.875,3,'Kinless hubs', rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=14, color=colors_modules['R7'])
        import matplotlib.patches as patches
        min_y = np.min([np.min(intra_connectivity),-2])
        max_y = np.max([np.max(intra_connectivity),4])
        # print("min_y", min_y)
        ax.add_patch(patches.Rectangle((0,min_y),0.05,2.5-min_y,linewidth=2,edgecolor=colors_modules['R1'],facecolor='none',label='R1'))
        ax.add_patch(patches.Rectangle((0.05,min_y),0.625-0.05,2.5-min_y,linewidth=2,edgecolor=colors_modules['R2'],facecolor='none',label='R2'))
        ax.add_patch(patches.Rectangle((0.625,min_y),0.8-0.625,2.5-min_y,linewidth=2,edgecolor=colors_modules['R3'],facecolor='none',label='R3'))
        ax.add_patch(patches.Rectangle((0.8,min_y),0.2,2.5-min_y,linewidth=2,edgecolor=colors_modules['R4'],facecolor='none',label='R4'))
        ax.add_patch(patches.Rectangle((0,2.5),0.3,max_y-2.5,linewidth=2,edgecolor=colors_modules['R5'],facecolor='none',label='R5'))
        ax.add_patch(patches.Rectangle((0.3,2.5),0.75-0.3,max_y-2.5,linewidth=2,edgecolor=colors_modules['R6'],facecolor='none',label='R6'))
        ax.add_patch(patches.Rectangle((0.75,2.5),0.25,max_y-2.5,linewidth=2,edgecolor=colors_modules['R7'],facecolor='none',label='R7'))
        plt.legend(bbox_to_anchor=(1.15, 0.7))
        plt.ylim(top=4)
        plt.xlim(left=0, right=1.05)
        plt.show()
    
    return {mapping[key]: val for key,val in R_nodes.items()}




####################################################################################################################
##################################  My criteria (arbitrary)  #######################################################

def find_hubs(G, quantile=0.91):
    """
    Criterion entirely based on degree
    """
    dict_degrees = dict(G.degree())
#     print(dict_degrees)
    threshold = np.quantile(list(dict_degrees.values()), q=quantile)
#     print("threshold", threshold)
    nodes_highest_degree_10_percent = [key for key,val in dict_degrees.items() if val >= threshold]
#     print(nodes_highest_degree_10_percent)
#     print()
    return nodes_highest_degree_10_percent

def find_antihubs(G, quantile=0.11):
    """
    Criterion entirely based on degree
    """
    dict_degrees = dict(G.degree())
    threshold = np.quantile(list(dict_degrees.values()), q=quantile)
    nodes_highest_degree_10_percent = [key for key,val in dict_degrees.items() if val <= threshold]
    return nodes_highest_degree_10_percent
