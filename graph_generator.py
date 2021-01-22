import bct
import numpy as np
import networkx as nx
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from utils_define_paths import *


connections_Lord = [('L', 'R'), ('L', 'O'), ('L', 'H'), ('L', 'E'), ('L', 'D'), ('O', 'H'), ('O', 'P'),
                    ('O', 'R'), ('O', 'D'), ('O', 'E'), ('O', 'Q'), ('H', 'P'), ('H', 'E'), ('H', 'D'),
                    ('H', 'Q'), ('D', 'P'), ('D', 'E'), ('D', 'S'), ('E', 'Q'), ('E', 'A'), ('Q', 'R'),
                    ('Q', 'P'), ('Q', 'T'), ('Q', 'M'), ('P', 'A'), ('P', 'R'), ('I', 'A'), ('I', 'T'),   
                    ('I', 'B'), ('I', 'F'), ('F', 'B'), ('F', 'G'), ('G', 'B'), ('G', 'T'), ('G', 'C'),
                    ('C', 'B'), ('C', 'T'), ('C', 'A'), ('C', 'M'), ('B', 'K'), ('T', 'B'), ('T', 'J'),
                    ('A', 'S'), ('A', 'K'), ('A', 'M'), ('A', 'J'), ('K', 'M'), ('K', 'J'), ('K', 'S'),
                    ('J', 'M'), ('J', 'S'), ('J', 'N'), ('N', 'S'), ('S', 'M')
                   ]

#If G is directed (type DiGraph) then G.neighbors(node1) returns the list of successors of node1, (= nodes node2 such that (node1,node2) is a directed edge of G). To get the list of all neighbors (= edges (node1,node2) or (node2,node1)), use nx.all_neighbors(G, node) (or G.to_undirected().neighbors(node))
#If G is undirected (type Graph) then G.neighbors(node1) returns the neighbors of G (= nodes node2 such that (node1,node2) is a undirected edge of G, i.e. (node1,node2) is a directed edge or (node2,node1) is a directed edge).




def weighting_graph(graph_G, type_graph_input=None, method_weighting='normal_J', sigma_weighting=1, w_uniform=None,
                    method_orientation='random'
                   ):
    """
    Orientation ('up' or 'down', for now random) and weighting of the graph
    input graph_G is unoriented, and output graph_G is also unoriented (type Graph) but has information about up/down for each unoriented edge (node1,node2) i.e. gives the direction of node1->node2
    graph_G should have the information 'up'/'down' at each edge (unless we change and take the convention that the direction is already given by (node1,node2) in the keys, which defines the 'up' edge always...)
    method_weighting should be in ['normal_J', 'normal_tanhJ', 'uniform_factors', 'bimodal_w']
    Careful: in case we binarize the graph (i.e. edge or no edge), e.g. based on some threshold of the connection, then we can potentially get some unconnected nodes. But this is not what we do here (as we give weights all different from 0)
    """
    if type_graph_input is not None:
        print("Providing type_graph as input to the function is deprecated - it is directly inferred from graph_G")
        assert type_graph_input is None #raises error
    type_graph = graph_G.type_graph #inferred directly
    f_random_sign_interaction = lambda x: np.random.choice([x, 1-x])

    if method_weighting == 'uniform_w':
        print("method_weighting = 'uniform_w' (careful not to confuse with bimodal_w")
        method_weighting = 'bimodal_w'
    
    #do some checks
    if type_graph in ['realistic_connectome_AAL', 'realistic_connectome_AAL2', 'realistic_connectome_HCP']:
        assert method_weighting in ['bimodal_w', 'use_proba_conn', 'uniform_w']
    else:
        assert method_weighting in ['normal_J', 'normal_tanhJ', 'uniform_factors', 'bimodal_w', 'uniform_w']
        
    if method_weighting == 'use_proba_conn':
        #here we do the hyp that anatomical weights from the data are proportional to w - 1/2 (i.e. the effective weights)
        #We could consider variants of this (correlation? J where 2*w-1=tanh(J)? ...etc)
        all_p = np.array([graph_G.edges[edge]['weight'] for edge in graph_G.edges])
        max_p = np.max(all_p) * 1. #1.5 #the multiplication factor is an important parameter (it determines the dynamical regime of the system, and in particular whether there is frustration or not). 2 = linear regime / 1 = oscillations --> take something in between to be at the edge
        def transf_p_to_w(p, max_p):
            '''Forces the result to be between 0.5 and 1 (as p>=0)'''
            return 0.5 + 0.5 / max_p * p
        for node1, node2, d in graph_G.edges(data=True):
            d['weight'] = f_random_sign_interaction(transf_p_to_w(d['weight'], max_p)) #modifies the weights
            d['orientation'] = get_orientation_edge(node1, node2, method_orientation=method_orientation)
    
    elif method_weighting == 'bimodal_w': #bimodal (w or 1-w)
        if w_uniform is None:
            w_uniform = 0.63 if not('realistic_' in type_graph) else 0.56 #0.65 if not('realistic_' in type_graph) else 0.57
        for node1, node2, d in graph_G.edges(data=True):
            d['weight'] = f_random_sign_interaction(w_uniform)
            d['orientation'] = get_orientation_edge(node1, node2, method_orientation=method_orientation)
    
    elif method_weighting == 'uniform_w': #uniform (w)
        #If all connections >0.5, a simple pulse pushes the system to a stable state, potentially with high beliefs (in absolute value). It seems that there are 2 stable states, with beliefs of opposite signs but the same absolute value. --> that is why we use instead a "bimodal" distribution of weights, in which w can take 2 values, which are symmetrical w.r.t. 0.5 (neutral value)
        if w_uniform is None:
            w_uniform = 0.65 if not('realistic_' in type_graph) else 0.57
        for node1, node2, d in graph_G.edges(data=True):
            d['weight'] = w_uniform
            d['orientation'] = get_orientation_edge(node1, node2, method_orientation=method_orientation)
    
    elif method_weighting == 'normal_w': #w has a normal distribution
#         if w_uniform is None:
        low_w, high_w = (0.5, 0.7) if '_SW' in type_graph else (0.5, 0.6)
        for node1, node2, d in graph_G.edges(data=True):
            from scipy.stats import truncnorm
            def get_truncated_normal(mean=0, sd=1, low=0.5, upp=1):
                return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
            def get_sample_truncated_normal(mean=0, sd=1, low=0.5, upp=1):
                return get_truncated_normal(mean, sd, low, upp).rvs()
            d['weight'] = f_random_sign_interaction(get_sample_truncated_normal(mean=0.55, sd=0.15, low=low_w, upp=high_w))
            d['orientation'] = get_orientation_edge(node1, node2, method_orientation=method_orientation)
    
    elif 'factor' not in method_weighting:
        def get_random_tanh_J(method_weighting='normal_J', sigma_weighting=1):
            if method_weighting == 'normal_J':
                #Normal distribution for J_ij (spin-glass)
                return np.tanh(np.random.normal(0, sigma_weighting))
            elif method_weighting == 'normal_tanhJ':
                #Taking directly normal distribution for tanh(J_ij)
                return np.random.normal(0, sigma_weighting)
        ###########  define the weights w_ij  #############
        assert method_weighting in ['normal_J', 'normal_tanhJ'] #how to generate J_ij (or tanh_J_ij)
        for node1, node2, d in graph_G.edges(data=True):
            d['weight'] = 1/2 + 1/2 * get_random_tanh_J(method_weighting=method_weighting, sigma_weighting=sigma_weighting) #J = 2*w - 1 thus w = (1 + J) / 2
            d['orientation'] = get_orientation_edge(node1, node2, method_orientation=method_orientation)
    else: #'factor' in method_weighting
        ###########  define the factors f_ij  #############
        assert method_weighting in ['uniform_factors']
        for node1, node2, d in graph_G.edges(data=True):
            d['factor'] = np.random.uniform(low=0, high=1, size=(2,2))
            d['orientation'] = get_orientation_edge(node1, node2, method_orientation=method_orientation)
            
    #Here we make sure that there are no unconnected nodes ("G = G_orient(graph)" does not remove unconnected nodes)
    #Detect the presence of unconnected nodes (if needed, add some code that removes them)
    d_degrees = dict(graph_G.degree(graph_G.nodes))
    list_unconnected_nodes = np.array(list(d_degrees.keys()))[np.array(list(d_degrees.values())) == 0]
    #I think list_unconnected_nodes is the same as list(nx.isolates(G))
    if len(list_unconnected_nodes) != 0:
        print("{} nodes are unconnected: {}".format(len(list_unconnected_nodes), list_unconnected_nodes))

    graph_G.type_graph = type_graph
    return graph_G


def get_orientation_edge(node1, node2, method_orientation='random'):
    assert method_orientation in ['random', 'full_hierarchical']
    if method_orientation == 'random':
        return np.random.choice(['up', 'down'])
    elif method_orientation == 'full_hierarchical':
        return 'up' if node1<node2 else 'down'
        

def get_adjacency_matrix(graph_G):
    """
    graph_G needs to be undirected, 
    graph_G needs to have weights (w) and not factors (f_ij) otherwise the adjacency matrix cannot be defined
    """
    G_weighted = G_orient_and_double(graph_G)
    adjacency_matrix = nx.to_numpy_matrix(G_weighted, nonedge=0.5)
    J = np.array(2*adjacency_matrix - 1) #connectivity matrix (taking 2*w-1 instead of w, because we do a parallel with rate networks and F(x,w)~(2w-1)*tanh(x))
    # N = 100
    # J = np.random.normal(size=(N,N), scale=1/np.sqrt(N)) #random Gaussian matrix
    assert np.sum(J != J.T) == 0 #Check that J is symmetrical
    return J


def G_from_graph(graph_G):
    print("G_from_graph is a deprecated function - use G_orient instead")
    return G_orient(graph_G)


def G_remove_orientation(graph_G_oriented):
    """
    THis function is identity if the input graph_G_oriented is already undirected
    """
    graph_G_undirected = graph_G_oriented.to_undirected()
    graph_G_undirected.type_graph = graph_G_oriented.type_graph #we need to copy the type_graph attribute
    return graph_G_undirected


def G_orient(graph_G): #previous name: G_from_graph
    """
    Creates the Networx graph (directed)
    This function is not needed to run BP/CI because the variable used is an undirected graph (with info about what is the down or up sense in order to define alpha_c and alpha_d)
    This function is needed:
    1) to plot the graph (with directionality of edges)
    2) to compute graph measures which depend on the direction of edges (it is useful in the case where alpha_c != alpha_d)
    After G is transformed into a directed graph with this function (while it was previously unoriented), one should be careful with the G.neighbors(...) function because that gives only the neighbors corresponding to an edge exiting the node!! ---> instead of using G.neighbors(node), use nx.all_neighbors(G, node) (or G.to_undirected().neighbors(node))
    This function also removes unconnected nodes from the graph (which is useful for the simulation: we totally forget the unconnected nodes so they cannot be stimulated for instance). That is useful if we use G.nodes in the code... (instead of list(activations_history_CI.keys()))
    Potentially, the weight information is not so much needed (at least to plot the graph)
    """
    graph_G_oriented = nx.DiGraph()
    for node1, node2, d in graph_G.edges(data=True): #(node1, node2), d in graph_G.items(): #In case of problem, change the code: if d doesn't exist, then use 1
        if d['orientation'] == 'down': #d[1]
            graph_G_oriented.add_edge(node1, node2, **{key:d[key] for key in d.keys() if key!='orientation'}) #before: **d, but we remove from d the info about the orientation #weight=d['weight']
        elif d['orientation'] == 'up': #d[1]
            graph_G_oriented.add_edge(node2, node1, **{key:d[key] for key in d.keys() if key!='orientation'})
        else:
            print("Every edge should be directed up or down")
            sys.exit()
    graph_G_oriented.type_graph = graph_G.type_graph
    return graph_G_oriented



def G_orient_and_double(graph_G): #previous name: "G_from_graph_with_weights"
    """
    Creates a directed graph from the undirected graph, by not only defining on directed edge (as in G_orient, which is helpful for the graph plotting) but both directed edges
    Only use: to define the adjacency matrix (function "get_adjacency_matrix")
    """
    graph_G_oriented = nx.DiGraph()
    for node1, node2, d in graph_G.edges(data=True):
        graph_G_oriented.add_edge(node1, node2, **{key:d[key] for key in d.keys() if key!='orientation'}) #before: **d, but we remove from d the info about the orientation #weight=d['weight']
        graph_G_oriented.add_edge(node2, node1, **{key:d[key] for key in d.keys() if key!='orientation'})
    graph_G_oriented.type_graph = graph_G.type_graph
    return graph_G_oriented


def G_from_connections(list_connections):
    """
    Returns a graph from a list of edges
    """
#     G = nx.DiGraph()
#     for (node1, node2) in connections:
#         G.add_edge(node1, node2)
#     return G
    graph_G = nx.Graph()
    graph_G.add_edges_from(list_connections)
    return graph_G


#THE 2 FUNCTIONS BELOW (orient_edge + create_graph) HELP DEFINING WISELY THE UP/DOWN SENSE OF THE CONNECTIONS. IT IS USEFUL ONLY FOR alpha_c != alpha_d (which is not something we consider now)

# def orient_edge(edge, method_orientation='full_hierarchical'):
#     """
#     See function get_orientation_edge
#     """
#     if method_orientation == 'full_hierarchical':
#         node1, node2 = edge
#         if node1 < node2:
#             return 'up'
#         else:
#             return 'down'
#     elif method_orientation == 'random':
#         return np.random.choice(['up', 'down'])

    
# def create_graph(which='hierarchical', method_orientation='full_hierarchical'):
#     """
#     Creates an unoriented graph, with non-redundant edges (i.e. if (node1,node2) exists, then (node2,node1) does not)
#     The 'up' and 'down' define feedforward vs feedback connections
#     """
    
#     if which == 'hierarchical':
#         # This is what I used with the bct module for 32 node graphs (for 64 you'd have to change the first parameter to 6)- 
#         graph_array = bct.makefractalCIJ(5,5,3)[0]   #hierarchial
#         for i in range(25):    #removing random edges
#             k = np.random.randint(0,32)
#             j = np.random.randint(0,32)
#             while(graph_array[k][j] != 1):
#                 k = np.random.randint(0,32)
#                 j = np.random.randint(0,32)
#             graph_array[k][j] = 0
#             graph_array[j][k] = 0
#     elif which == 'modular':
#         graph_array = bct.makeevenCIJ(32,540,4)     #modular

#     if which in ['hierarchical', 'modular']:
#         graph_edges = np.nonzero(graph_array)
#         graph_edges = np.concatenate((graph_edges[0][:,None], graph_edges[1][:,None]), axis=1)
#         graph_edges = {tuple(edge) for edge in graph_edges}
#         graph_edges_asym = {(node1, node2) for (node1,node2) in graph_edges if (node2,node1) not in graph_edges}
#         graph_edges_sym = {(node1, node2) for (node1,node2) in graph_edges if (((node2,node1) in graph_edges) and (node1<node2))}
#         graph_edges = graph_edges_asym.union(graph_edges_sym)
#         graph = {edge: (1, orient_edge(edge, method_orientation)) for edge in graph_edges}
    
#     elif which == 'watts_strogatz':
#         #create a small-world unoriented graph
#         graph = nx.watts_strogatz_graph(n=68, k=4, p=0.3) #p=1 for random / p=0 for regular / p~0.5 for SW
    
#     elif which == 'Lord':
#         graph = G_from_connections(connections_Lord)
        
#     elif which == 'bistable_perception':
#         connections_easy = [('A', 'B'), ('B', 'C')]
#         graph = G_from_connections(connections_easy)
        
#     elif which == 'simple':
#         connections_easy = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('D','B'), ('A','C')]
#         graph = G_from_connections(connections_easy)
    
#     elif which == 'complete':
#         graph = nx.complete_graph(5)
        
#     if which not in ['hierarchical', 'modular']:
#         #orient the edges randomly
#         #graph = {edge : (w, numpy.random.choice(['up', 'down'])) for edge in graph.edges()}
#         graph = {edge : (1, orient_edge(edge, method_orientation)) for edge in graph.edges()}
        
#     return graph #careful: sometimes the output is directed (e.g. because of G_from_connections) 


def read_AAL2_file(which='AAL2', remove_cerebellum_and_vermis=True, remove_ofc=True):
    """
    read AAL2.txt (contains the region number, the region name and another number (??)) of the AAL2 parcellation with 120 nodes
    From these 120 nodes we remove the nodes from 95 to 120, i.e. the cerebellum
    We also remove the nodes from OFC=orbito frontal cortex (which are not much connected to the network, not even between themselves)
    """
    assert which in ['AAL2', 'AAL']
    
    
    dir_files = path_data #defined in utils_define_paths.py (global variable)
    
    if which == 'AAL2':
        #specify the location of file AAL2.txt (brain atlas coming from the HCP842 tractography files on http://brain.labsolver.org/diffusion-mri-templates/tractography)
        file_name = 'AAL2.txt'
        filename = dir_files + file_name
        #read file
        list_i_node = []
        list_names = []
        list_i2_node = []
        with open(filename) as file:
            lines = [el for el in file.read().split('\n')]
            lines = lines[:-1] #remove the last line (blank)
            for line in lines:
                i_node, name_node, i2_node = [el for el in line.split()]
                list_i_node.append(int(i_node))
                list_names.append(name_node)
                list_i2_node.append(int(i2_node))
        if remove_cerebellum_and_vermis:
            #remove the cerebellum + vermis regions
            list_i_node, list_names = filter_cerebellum_and_vermis(list_i_node, list_names)
        if remove_ofc:
            #remove the ofc regions
            list_i_node, list_names = filter_ofc(list_i_node, list_names)
#         return list_i_node, list_names, list_i2_node
        return list_i_node, list_names

    elif which == 'AAL': #merged version of AAL2, less common and with less regions
        file_name = 'AAL-merged.txt'
        filename = dir_files + file_name
        #read file
        list_i_node = []
        list_names = []
#         list_i2_node = []
        with open(filename) as file:
            lines = [el for el in file.read().split('\n')]
            lines = lines[:-1] #remove the last line (blank)
#             print(lines)
            for line in lines:
                i_node, name_node = [el for el in line.split()]
                list_i_node.append(int(i_node))
                list_names.append(name_node)
#                 list_i2_node.append(int(i2_node))
        #remove the cerebellum + vermis regions
        if remove_cerebellum_and_vermis:
            list_i_node, list_names = filter_cerebellum_and_vermis(list_i_node, list_names)
        if remove_ofc:
            list_i_node, list_names = filter_ofc(list_i_node, list_names)
        return list_i_node, list_names

def binary_filter_cerebellum_and_vermis(list_names):
    is_cerebellum_or_vermis = np.array([('Cerebelum' in node) or ('Vermis' in node)  #AAL2
                                        or ('cerebellum' in node) or ('vermis' in node)  #AAL-merged
                                        for node in list_names])
    return is_cerebellum_or_vermis

def binary_filter_ofc(list_names):
    is_ofc = np.array([('OFC' in node) #AAL2
                       or ('ofc' in node) #AAL-merged
                       for node in list_names])
    return is_ofc


def filter_cerebellum_and_vermis(list_i_node, list_names):
    is_cerebellum_or_vermis = binary_filter_cerebellum_and_vermis(list_names)
    list_names = list(np.array(list_names)[~is_cerebellum_or_vermis])
    list_i_node = list(np.array(list_i_node)[~is_cerebellum_or_vermis])
    return list_i_node, list_names

def filter_ofc(list_i_node, list_names):
    is_ofc = binary_filter_ofc(list_names)
    list_names = list(np.array(list_names)[~is_ofc])
    list_i_node = list(np.array(list_i_node)[~is_ofc])
    return list_i_node, list_names
    

def get_modules_and_belonging(type_graph, remove_cerebellum_and_vermis=True, remove_ofc=True):
    assert type_graph in ['realistic_connectome_AAL', 'realistic_connectome_AAL2']
    
    if type_graph == 'realistic_connectome_AAL':
        use_corresp_1 = True #do not change (as this makes sense only for AAL2)
        modules_realistic_connectome = {
            'AUD': [29, 30, 69, 70, 71, 72], #AUDITORY NETWORK
            'SSM': [15, 16, 51, 52, 55, 56, 57, 58, 65, 66, 67, 68], #SOMATO-SENSORI-MOTOR NETWORK
            'VIS': [5, 6, 19, 20, 27, 28, 35, 36, 37, 38], #VISUAL NETWORK
            'DAN': [53, 54], #DORSAL ATTENTION NETWORK
            'CON': [41, 42, 43, 44, 45, 46], #CINGULO-OPERCULAR NETWORK
            'SN': [1, 2, 33, 34, 39, 40], #SALIENCE NETWORK
            'FPN': [21, 22, 23, 24], #FRONTO-PARIETAL NETWORK
            'DMN': [3, 4, 13, 14, 17, 18, 25, 26, 31, 32, 59, 60], #DEFAULT-MODE NETWORK
            'CB': [9, 10, 11, 12, 75], #Cerebellar
            'SubC': [7, 8, 49, 50, 61, 62, 73, 74], #SUBCORTICAL NETWORK
        #     'Not attributed': [9, 10, 11, 12, 47, 48, 51, 52, 63, 64, 75]
        #     'Not attributed': [47, 48, 51, 52, 63, 64]
            'Not attributed': [47, 48, 63, 64]
        } #Correspondance (done by Renaud Jardri) between AAL-merged regions and modules assigned from [A mechanistic model of connector hubs, modularity and cognition, Bertolero 2018] (see in particular Fig 4), which was previously used in [Powers 2011].
        #In the actual paper: also ventral attention, cerebellar, (memory retrieval), subdivision hand vs mouth for somatosensory
        
    elif type_graph == 'realistic_connectome_AAL2':
        use_corresp_1 = True #True = Bertolero / False = anatomical
        
        if use_corresp_1: 
            modules_realistic_connectome = {
                'AUD': [83,84,67,68,85,86,87,88,89,90,91,92,93,94], #AUDITORY NETWORK
                'SSM': [37,38,73,74,61,62,1,2,13,14,15,16], #SOMATO-SENSORI-MOTOR NETWORK
                'VIS': [47,48,49,50,59,60,51,52,53,54,55,56,57,58], #VISUAL NETWORK
                'DAN': [63,64,65,66], #DORSAL ATTENTION NETWORK
                'CON': [31,32,25,26,29,30], #CINGULO-OPERCULAR NETWORK    #Why not 27,28?
                'SN': [45,46,33,34,27,28], #SALIENCE NETWORK
                'FPN': [7,8,9,10,11,12,5,6,21,22], #FRONTO-PARIETAL NETWORK
                'DMN': [69,70,35,36,39,40,3,4,19,20,41,42,43,44,71,72], #DEFAULT-MODE NETWORK
                'CB': [99,100,101,102,103,104,105,106,107,108,109,110,111,112,95,96,97,98,113,114,115,116,117,118,119,120], #Cerebellar
                'SubC': [75,76,79,80,77,78,81,82], #SUBCORTICAL NETWORK
                'Not attributed': [17,18,23,24]
            } #Correspondance between AAL2 regions and modules assigned from [A mechanistic model of connector hubs, modularity and cognition, Bertolero 2018], based on the one above (for AAL-merged) + using the correspondance between AAL2 and AAL-merged
            #Question (to discuss with Renaud): why isn't the Amygdala in SubC? Why aren't the OFC regions all together?
        else:
            modules_realistic_connectome = {
                'Central region': ['Precentral_L', 'Postcentral_L', 'Rolandic_Oper_L'],
                'Frontal lobe – lateral surface': ['Frontal_Sup_2_L', 'Frontal_Mid_2_L', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Tri_L'],
                'Frontal lobe – Medial surface': ['Frontal_Sup_Medial_L', 'Supp_Motor_Area_L', 'Paracentral_Lobule_L'],
                'Frontal lobe – Orbital surface': ['Frontal_Med_Orb_L', 'Frontal_Inf_Orb_2_L', 'Rectus_L', 'OFCmed_L', 'OFCant_L', 'OFCpost_L', 'OFClat_L', 'Olfactory_L'],
                'Temporal lobe': ['Temporal_Sup_L', 'Heschl_L', 'Temporal_Mid_L', 'Temporal_Inf_L'],
                'Parietal lobe – lateral surface': ['Parietal_Sup_L', 'Parietal_Inf_L', 'Angular_L', 'SupraMarginal_L'],
                'Parietal lobe – medial surface': ['Precuneus_L'],
                'Occipital lobe – lateral surface': ['Occipital_Sup_L', 'Occipital_Mid_L', 'Occipital_Inf_L'],
                'Occipital lobe – medial and inferior surfaces': ['Cuneus_L', 'Calcarine_L', 'Lingual_L', 'Fusiform_L'],
                'Limbic lobe': ['Temporal_Pole_Sup_L', 'Temporal_Pole_Mid_L', 'Cingulate_Ant_L', 'Cingulate_Mid_L', 'Cingulate_Post_L', 'Hippocampus_L', 'ParaHippocampal_L', 'Insula_L'],
                'Sub cortical grey nuclei': ['Amygdala_L', 'Caudate_L', 'Putamen_L', 'Pallidum_L', 'Thalamus_L'],
                'Cerebellum': ['Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10']
            } #Anatomical modules, from the AAL2 paper: https://www.sciencedirect.com/science/article/pii/S1053811915006953 (+I added the cerebellum, which was not on the list)
            #add the _R regions
            modules_realistic_connectome = {module_name: list(itertools.chain.from_iterable([[el, el[:-2] + '_R'] 
                                                                                             for el in list_nodes_module]))
                                            for module_name,list_nodes_module in modules_realistic_connectome.items()}

    #create a dict (belonging_modules) with node_name:module_name (instead of module_name:i_node as for modules_realistic_connectome)
    which_type = type_graph.replace('realistic_connectome_', '')
    list_i_node, list_names = read_AAL2_file(which_type,
                                             remove_cerebellum_and_vermis=remove_cerebellum_and_vermis,
                                             remove_ofc=remove_ofc
                                            )
    mapping_nodes = dict(zip(list_i_node,list_names))
    #mapping_inverse = dict(zip(list(mapping_nodes.values()), list(mapping_nodes.keys())))
    belonging_modules = {}
    for module_name, list_nodes_module in modules_realistic_connectome.items():
        for node in list_nodes_module:
            if use_corresp_1: #Modules from Bertolero
                if node in mapping_nodes.keys(): #that removes all nodes which are part of the cerebellum or vermis (if remove_cerebellum_and_vermis = True) and of ofc (if remove_ofc = True)
                    belonging_modules[mapping_nodes[node]] = module_name            
            else: #Anatomical modules
                if node in mapping_nodes.values(): #that removes all nodes which are part of the cerebellum or vermis (if remove_cerebellum_and_vermis = True) and of ofc (if remove_ofc = True)
                    belonging_modules[node] = module_name
    
    #create a dict with module_name:list_nodes_names_in_module
    modules_realistic_connectome = {
        module_name: list(np.array(list(belonging_modules.keys()))[np.array(list(belonging_modules.values())) == module_name])
        for module_name in modules_realistic_connectome.keys()
    }
    modules_realistic_connectome = {key:val for key,val in modules_realistic_connectome.items() if len(val) != 0} #remove empty modules (e.g. cerebellum if remove_cerebellum_and_vermis = True, or OFC if remove_ofc = True)
        
    return modules_realistic_connectome, belonging_modules


def generate_graph(type_graph='modular_SW', 
                   remove_cerebellum_and_vermis=True, remove_ofc=True, 
                   binarize_realistic_connectome=False,
                   list_connections=None
                  ):
    '''
    Creates the graph structure, without orientation or weights (except from the case of the realistic connectome, where DTI data is used to get probabilities of connection which is included in the weights, but are later renormalized in the weigthing function of the graph)
    Generates an unoriented network, with non-redundant edges (i.e. if (node1,node2) exists, then (node2,node1) does not) and where the 'up' and 'down' define feedforward vs feedback connections.
    To check whether the graph generated is "fine" (i.e. similar to a brain network), I could compute several graph metrics as in Crossley et al (Brain 2014): see note on Mendeley (paragraph Results/Characteristics of the normal human brain (DTI) connectome "TODO: I should do that with the graphs I use or generate"
    Options remove_cerebellum_and_vermis and remove_ofc are only useful if type_graph = 'realistic_connectome_AAL' or 'realistic_connectome_AAL2'
    '''
    
    #run some checks
    assert not(binarize_realistic_connectome and (type_graph not in ['realistic_connectome_AAL', 'realistic_connectome_AAL2',
'realistic_connectome_HCP']))
    assert not((list_connections is not None) and (type_graph != 'manual'))
    assert type_graph in ['small_world', 'Lord_paper', 'connections_easy', 'complete', 'manual',
                          'modular_SW', 'modular_SW_big', 'hierarchical_SW',
                          'realistic_connectome_AAL', 'realistic_connectome_AAL2',
                          'realistic_connectome_HCP'
                         ]
    
#     #Create a small-world unoriented graph
#     graph = nx.watts_strogatz_graph(n=68, k=4, p=0.3) #p=1 for random / p=0 for regular / p~0.5 for SW
# #     graph = G_from_connections(connections_Lord)
# #     connections_easy = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('D','B'), ('A','C')]
# #     graph = G_from_connections(connections_easy)
# #     graph = nx.complete_graph(5)
#     G = G_orient(graph)
    
    # graph_array = bct.makefractalCIJ(5,2.5,3)[0]
    # j=0
    # for i in range(40):
    #     while(graph_array[k][j]!=1):
    #         k = np.random.randint(0,32)
    #         j = np.random.randint(0,32)
    #     graph_array[k][j] = 0
    #     graph_array[j][k] = 0
    
    
    if type_graph in ['small_world', 'Lord_paper', 'connections_easy', 'complete', 'manual']:
        if type_graph == 'small_world':
            #create a small-world unoriented graph
            graph = nx.watts_strogatz_graph(n=68, k=4, p=0.5)#p=03 or 0.5 for instance #p=1 for random / p=0 for regular / p~0.5 for SW
        #special graphs (determined i.e. not random)
        elif type_graph == 'manual':
            assert list_connections is not None
            graph = G_from_connections(list_connections)
        elif type_graph == 'Lord_paper':
            graph = G_from_connections(connections_Lord)
        elif type_graph == 'connections_easy':
            connections_easy = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('D','B'), ('A','C')]
            graph = G_from_connections(connections_easy)
        elif type_graph == 'complete':
            graph = nx.complete_graph(5)
        G = nx.Graph(graph) #doesn't remove the unconnected nodes
        G.type_graph = type_graph
        
        #Some nodes have no connections with others. To check the list of these nodes, use the following code
        d_degrees = dict(G.degree(G.nodes))
        list_unconnected_nodes = np.array(list(d_degrees.keys()))[np.array(list(d_degrees.values())) == 0]
        #I think list_unconnected_nodes is the same as list(nx.isolates(G))
        if len(list_unconnected_nodes) != 0:
            print("{} nodes are unconnected: {}".format(len(list_unconnected_nodes), list_unconnected_nodes))
        
        return G#, graph
    
    elif type_graph in ['modular_SW', 'modular_SW_big', 'hierarchical_SW']:
        #Using modular SW graph or hierarchial SW graph
        if type_graph == 'modular_SW':
            n_nodes = 32
            n_edges = 425
            n_nodes_remove = 20
            graph_array = bct.makeevenCIJ(n_nodes,n_edges,3)     #modular (n_nodes, n_edges, module size 2^n)  Creates fully-connected modules + some connections between modules (note that removing some random edges as below aims at making less connected nodes within the same module)
        #Using modular SW graph or hierarchial SW graph
        elif type_graph == 'modular_SW_big':
            n_nodes = 128 #must be a power of 2
            n_edges = 2425
            graph_array = bct.makeevenCIJ(n_nodes,n_edges,3)     #modular (n_nodes, n_edges, module size 2^n)  Creates fully-connected modules + some connections between modules (note that removing some random edges as below aims at making less connected nodes within the same module)
            n_nodes_remove = 100
        elif type_graph == 'hierarchical_SW': #For hierarchical networks, the connection density fall-off was chosen as 2.5
            n_nodes = 32
            graph_array = bct.makefractalCIJ(5,2.5,3)[0]   #hierarchial
            n_nodes_remove = 20
        else:
            print('pb: type_graph unknown')
            sys.exit()
        np.random.seed() #ensures that we get different graphs (otherwise the same graph is generated over and over, as all workers are given the same random state)
        #1- removing unidirectional edges
        for i in range(n_nodes):
            for j in range(n_nodes):
                if graph_array[i][j]==1 and graph_array[j][i]==0:
                    graph_array[i][j]=0
        #2- removing random edges
        for i in range(n_nodes_remove):    
            k = np.random.randint(0,n_nodes)
            j = np.random.randint(0,n_nodes)
            while graph_array[k][j] != 1:
                k = np.random.randint(0, n_nodes)
                j = np.random.randint(0, n_nodes)
            graph_array[k][j] = 0
            graph_array[j][k] = 0
        
        G = nx.Graph(graph_array)
        G.type_graph = type_graph
        
        #Detect the presence of unconnected nodes
        d_degrees = dict(G.degree(G.nodes))
        list_unconnected_nodes = np.array(list(d_degrees.keys()))[np.array(list(d_degrees.values())) == 0]
        #I think list_unconnected_nodes is the same as list(nx.isolates(G))
        if len(list_unconnected_nodes) != 0:
            print("Some nodes are unconnected: {}".format(list_unconnected_nodes))
            sys.exit()
        
        return G#, graph_array
    
    elif type_graph in ['realistic_connectome_AAL', 'realistic_connectome_AAL2']:
        
        which_type = type_graph.replace('realistic_connectome_', '') #'AAL' or 'AAL2'
        
        #specify the location of the .mat file (structural connectivity) coming from the HCP842 tractography files on http://brain.labsolver.org/diffusion-mri-templates/tractography
        dirname = path_data #defined by function define_paths
        if type_graph == 'realistic_connectome_AAL2':
            filename = 'normative_tracts_AAL2_dilatedby1voxel.trk.gz.AAL2_dilatedby1voxel.count.pass.connectivity'
        else:
            filename = 'normative_tracts_AAL-merged.trk.gz.AAL-merged.count.pass.connectivity'
            
        mat = scipy.io.loadmat(dirname + filename + '.mat') #load the data
        m = mat['connectivity'] #Note that mat has also fields 'name' and 'atlas'. m is a symmetric matrix
#         print(m.shape)
        
        def filter_m(m, remove_cerebellum_and_vermis, remove_ofc):
            """
            Removes potentially the cerebellum, vermis and OFC regions 
            See AAL2.txt and AAL-merged.txt for the name of regions
            """
            list_i_node, list_names = read_AAL2_file(which=which_type, remove_cerebellum_and_vermis=False, remove_ofc=False)
            to_remove = np.array([False]*len(list_names))
            if remove_cerebellum_and_vermis:
                is_cerebellum_or_vermis = binary_filter_cerebellum_and_vermis(list_names)
                to_remove = to_remove + is_cerebellum_or_vermis
            if remove_ofc:
                is_ofc = binary_filter_ofc(list_names)
                to_remove = to_remove + is_ofc
        #     to_remove = is_cerebellum_or_vermis + is_ofc
            m = m[~to_remove,:][:, ~to_remove]
            return m
        m = filter_m(m, remove_cerebellum_and_vermis, remove_ofc)
#         print(m.shape)

        p_conn = m / 134610 #normalization to get probabilities instead of numbers

        #histogram of the proba of connection (p_conn)
        # plt.hist(p_conn.flatten(), bins=40)
        # plt.show()

        #Plot the matrix p_conn (see https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec)
        plot_matrix = False
        if plot_matrix:
            fig = plt.figure(figsize=(14,7))
            ax = sns.heatmap(
                p_conn, 
            #     cmap=sns.diverging_palette(20, 220, n=200),
                square=True
            )
            # ax.set_xticklabels(
            # ax.get_xticklabels(),
            # rotation=45,
            # horizontalalignment='right')
            plt.show()

        plot_struct_matrix = False
        if plot_struct_matrix:
            if type_graph == 'realistic_connectome_AAL2':
                list_threshold_p = [0.003, 0.004, 0.006, 0.008, 0.01, 0.012]
            else: #AAL-merged
                list_threshold_p = [0.003, 0.004, 0.006, 0.008, 0.01, 0.012]
            for threshold_p in list_threshold_p: # threshold_p = 0.02
                n_edges = np.sum(p_conn > threshold_p)
                print(n_edges)
                size_graph = p_conn.shape[0]
                struct_conn = p_conn > threshold_p
                #Plot the matrix struct_conn (see https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec)
#                 fig = plt.figure(figsize=(14,7))
#                 ax = sns.heatmap(
#                     struct_conn, 
#                 #     cmap=sns.diverging_palette(20, 220, n=200),
#                     square=True
#                 )
                fig, ax = plt.subplots(1, figsize=(10,10))
                plt.imshow(struct_conn, cmap="Greys", interpolation="none")
                plt.title('threshold = {} gives graph density = {}'.format(threshold_p, np.round(n_edges / (size_graph * (size_graph-1)),3)))
                plt.show()
            
            
        if type_graph == 'realistic_connectome_AAL2':
            threshold_p = 0.007
        else: #AAL-merged
            threshold_p = 0.003
        n_edges = np.sum(p_conn > threshold_p)
#         print("n_edges = {}".format(n_edges))
        if binarize_realistic_connectome:
            graph_array = p_conn > threshold_p #binarizes the connections (can result into having some unconnected nodes)
        else: #default
            graph_array = p_conn #does not binarize the connections (no unconnected nodes here)
        G = nx.Graph(graph_array) #doesn't remove the unconnected nodes
        
#         G = nx.relabel_nodes(G, lambda x: x+1) #relabel the node indices beginning from 1 (instead of 0)
        list_i_node, list_names = read_AAL2_file(which=which_type,
                                                 remove_cerebellum_and_vermis=remove_cerebellum_and_vermis,
                                                 remove_ofc=remove_ofc
                                                )
        assert len(list_names) == len(list(G.nodes))
        G = nx.relabel_nodes(G, lambda x: list_names[x]) #give the right region names
        G.type_graph = type_graph
        
        #Detect the presence of unconnected nodes
        d_degrees = dict(G.degree(G.nodes))
#         print(d_degrees)
        list_unconnected_nodes = np.array(list(d_degrees.keys()))[np.array(list(d_degrees.values())) == 0]
        #I think list_unconnected_nodes is the same as list(nx.isolates(G))
        if len(list_unconnected_nodes) != 0:
#             print("{} nodes are unconnected: {}".format(len(list_unconnected_nodes), list_unconnected_nodes))
            G.remove_nodes_from(list(nx.isolates(G))) #removes unconnected nodes, in place
#             print("...removed these nodes")
            print("...removed the {} unconnected nodes ({})".format(len(list_unconnected_nodes), list_unconnected_nodes))
#             sys.exit()
        
        return G#, graph_array
    
    elif type_graph == 'realistic_connectome_HCP':
        
        #specify the location of the .mat file (structural connectivity) coming from https://doi.org/10.1101/2020.06.22.166041 (Fig 1; comes from HCP = human connectome project)
        dirname = path_data
        filename = 'averageConnectivity_Fpt'

        import mat73
        mat = mat73.loadmat(dirname + filename + '.mat') #load the data

        m = mat['rawStreamlineCounts'] #Note that mat has also field 'parcelIDs' (list of regions)
        # print(m.shape)
        p_conn = m #normalization to get probabilities instead of numbers

        #histogram of the proba of connection (p_conn)
        # plt.hist(p_conn.flatten(), bins=40)
        # plt.show()

        #Plot the matrix p_conn (see https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec)
        plot_matrix = False
        if plot_matrix:
            fig = plt.figure(figsize=(14,7))
            ax = sns.heatmap(
                p_conn, 
            #     cmap=sns.diverging_palette(20, 220, n=200),
                square=True
            )
            # ax.set_xticklabels(
            # ax.get_xticklabels(),
            # rotation=45,
            # horizontalalignment='right')
            plt.show()

        # for threshold_p in [-3.5, -3.2, -3, -2.5, -2]: # threshold_p = 0.02
        #     n_edges = np.sum(p_conn > threshold_p)
        #     print(n_edges)
        #     print("probability of edge: {}".format(n_edges / p_conn.shape[0]**2))
        #     struct_conn = p_conn > threshold_p
        #     #Plot the matrix struct_conn (see https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec)
        #     fig = plt.figure(figsize=(14,7))
        #     ax = sns.heatmap(
        #         struct_conn, 
        #     #     cmap=sns.diverging_palette(20, 220, n=200),
        #         square=True
        #     )
        # #     ax.set_xticklabels(
        # #     ax.get_xticklabels(),
        # #     rotation=45,
        # #     horizontalalignment='right')
        #     plt.title('threshold = {}'.format(threshold_p))
        #     plt.show()

        threshold_p = -3.2
        n_edges = np.sum(p_conn > threshold_p)
#         print("n_edges = {}".format(n_edges))
        print("probability of edge: {}".format(n_edges / p_conn.shape[0]**2))
        graph_array = p_conn > threshold_p
        G = nx.Graph(graph_array)

        #Some nodes have no connections with others. To check the list of these nodes, use the following code
        d_degrees = dict(G.degree(G.nodes))
#         print(d_degrees)
        list_unconnected_nodes = np.array(list(d_degrees.keys()))[np.array(list(d_degrees.values())) == 0]
        #I think list_unconnected_nodes is the same as list(nx.isolates(G))
        if len(list_unconnected_nodes) != 0:
            print("Some nodes are unconnected: {}".format(list_unconnected_nodes))


#         G = nx.relabel_nodes(G, lambda x: x+1) #relabel the node indices beginning from 1 (instead of 0)
        list_regions = [el[0] for el in mat['parcelIDs']] #list of regions
        G = nx.relabel_nodes(G, lambda x: list_regions[x])
        G.type_graph = type_graph
        return G#, graph_array
    
    #note: to get the adjacency matrix from G, just do:
#     adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)
#     #Plot adjacency matrix in toned-down black and white
#     fig = plt.figure(figsize=(5, 5))
#     plt.imshow(adjacency_matrix, cmap="Greys", interpolation="none")
#     plt.show()
    else:
        print("Unknown graph type")

    
