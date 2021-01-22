#TODO++: check whether the graph measures are applied to a graph with or without weights, and directed or undirected
# G = G_from_graph(graph) #directed graph (without the weights information)

import networkx as nx
from brainconn.modularity import community_louvain
from brainconn.centrality import participation_coef
import numpy as np
from graph_generator import *
import sys


def plot_measure(measure, graph_G, order_keys=None):
    """ 
    Plots the degree
    order_keys is a dictionnary
    """
    assert measure == 'degree'
    
    degree_nodes = {node: graph_G.degree(node) for node in graph_G.nodes}
    if order_keys is not None:
        degree_nodes_good_order = {key: degree_nodes[key] for key in order_keys.keys()} #to have the same order of keys
        degree_nodes = degree_nodes_good_order
    plt.figure(figsize=(20,10))
    plt.plot(list(degree_nodes.values()))
    # plt.plot(x_pos, list(degree_nodes.values()))
    # plt.xticks(x_pos, labels=list(degree_nodes.keys()), rotation=90, size=12)
    plt.ylabel('degree of node', size=15)
    plt.show()

def get_legend_measures(measure):
#     return 'normalized ' + measure
#     return 'normalized ' + measure.replace('_undirected','').replace('_',' ')
    return measure.replace('_undirected','').replace('_',' ')


def get_modules(G):
    """
    Do community detection in the general case, or give imposed modules in the case realistic_connectome_AAL
    Instead of community_louvain, Bertolero et al 2018 (A mechanistic model of connector hubs, modularity and cognition) uses InfoMap for community detection, which I cannot find in Networkx:  I could only find https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html, https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/community/modularity_max.html, https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/community/quality.html, and https://networkx.github.io/documentation/stable/reference/algorithms/community.html
    In [Guimera & Amaral, Functional cartography of complex metabolic networks, Nature 2008], the authors use simulated annealing for module identification (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2175124/#SD1) with the goal being to find the partition with largest modularity.
    
    Works on undirected or directed graphs
    """
    if not hasattr(G, 'type_graph'): #check that G has a type_graph and print a message if it's not the case
        print('G has no type_graph attribute')
        sys.exit()
        
    if G.type_graph in ['realistic_connectome_AAL', 'realistic_connectome_AAL2']:
        _, belonging_modules = get_modules_and_belonging(G.type_graph, remove_cerebellum_and_vermis=False, remove_ofc=False) #here we don't remove anything, even if G has some regions removed: indeed, 1- we cannot know easily the information about remove_cerebellum_and_vermis and remove_ofc based on G, and 2-belonging is filtered when used below, so it doesn't matter if it has additionnal elements which are not needed.

#         print(type(G), len(list(G.nodes)))
#         for node in G.nodes:
#             print("node: {}".format(node))
#             print("belongs to {}".format(belonging_modules[node]))
#         return np.array([belonging_modules[node] for node in G.nodes])
        return dict(zip(list(G.nodes), np.array([belonging_modules[node] for node in G.nodes])))
    else: #default case
#         return community_louvain(nx.to_numpy_array(G))[0]
        return dict(zip(list(G.nodes), community_louvain(nx.to_numpy_array(G.to_undirected()))[0]))
#         return dict(zip(list(G.nodes), community_louvain(nx.to_numpy_array(G))[0]))
#         bct.community_louvain(nx.to_numpy_array(G.to_undirected()))[0]
    
def get_participation_coefficient(G):
    """
    Works on undirected or directed graphs (because nx.to_numpy_array and get_modules apply on both undirected and directed graphs)
    In fact get_participation_coefficient and get_participation_coefficient_undirected return the same result (up to the computer precision ~ 1e-16) if G is undirected vs undirected-like (i.e. directed but with both (node1,node2) and (node2,node1), e.g. created with function G_orient_and_double
    """
    return dict(zip(G.nodes(), 
                    participation_coef(nx.to_numpy_array(G), list(get_modules(G).values()))))
#     return dict(zip(G.nodes(), 
#                     participation_coef(nx.to_numpy_array(G), get_modules(G))))
#     try:
#         return dict(zip(G.nodes(), 
#                         participation_coef(nx.to_numpy_array(G), list(get_modules(G).values())))
#     except RuntimeError:
#         return 0#'pb'


def get_participation_coefficient_undirected(G):
    """
    See above: get_participation_coefficient and get_participation_coefficient_undirected return the same result (up to the computer precision ~ 1e-16) if G is undirected vs undirected-like (i.e. directed but with both (node1,node2) and (node2,node1), e.g. created with function G_orient_and_double
    """
#     return dict(zip(G.to_undirected().nodes(),
#                     participation_coef(nx.to_numpy_array(G.to_undirected()), list(get_modules(G.to_undirected()).values()))))
#     return get_participation_coefficient(G.to_undirected())
    G_undirected = G_remove_orientation(G)
    return get_participation_coefficient(G_undirected)


def within_community_strength_undirected(G):
    G_undirected = G_remove_orientation(G)
    modules = get_modules(G_undirected)
    list_modules = np.unique(list(modules.values()))
    modules_contain = {module:[node for node in modules.keys() if modules[node] == module] for module in list_modules}
    d = {}
    for module, list_nodes_module in modules_contain.items():
#         print("###############################################")
        G_module = G_undirected.subgraph(list_nodes_module) #subgraph of G (just one module)
        k_i = dict(G_module.degree()) #degrees_inside_module
#         print(k_i)
        mean_k_si = np.mean(list(k_i.values())) #mean_degree_inside_module
        sig_k_si = np.std(list(k_i.values())) #std_degree_inside_module
        if sig_k_si == 0: #all degrees are the same
            z_i = {key:0 for key,val in k_i.items()}
        else: #default case
            z_i = {key:(val-mean_k_si)/sig_k_si for key,val in k_i.items()}
#         print(z_i)
#         print("###############################################")
        d.update(z_i)
#     return d
    assert set(list(d.keys())) == set(list(G.nodes))
    return {key: d[key] for key in list(G.nodes)}


#It seems that there are often convergence problems for nx.eigenvector_centrality (I suppose it depends on the type of graph). There are also quite often problems with get_participation_coefficient (because of community_louvain)


fun_centrality_measures = {
    "degree_centrality": nx.degree_centrality,
    "neighbors_length_2": lambda G: get_number_nodes_at_length_n(G, 2),
    "neighbors_length_3": lambda G: get_number_nodes_at_length_n(G, 3),
    "neighbors_length_4": lambda G: get_number_nodes_at_length_n(G, 4),
#     "in_degree_centrality": nx.in_degree_centrality,
#     "out_degree_centrality": nx.out_degree_centrality,
#     "eigenvector_centrality": nx.eigenvector_centrality,
#     "katz_centrality": nx.katz_centrality, #doesn't converge for realistic_connectome_AAL
#     "katz_centrality_undirected": lambda G: nx.katz_centrality(G.to_undirected()), #doesn't converge for realistic_connectome_AAL
    "closeness_centrality": nx.closeness_centrality,
    "closeness_centrality_undirected": lambda G: nx.closeness_centrality(G.to_undirected()),
    "betweenness_centrality": nx.betweenness_centrality,
    "betweenness_centrality_undirected": lambda G: nx.betweenness_centrality(G.to_undirected()),
    "load_centrality": nx.load_centrality,
    "load_centrality_undirected": lambda G: nx.load_centrality(G.to_undirected()),
    "harmonic_centrality": nx.harmonic_centrality,
    "harmonic_centrality_undirected": lambda G: nx.harmonic_centrality(G.to_undirected()),
    "local_reaching_centrality": lambda G: {node: nx.local_reaching_centrality(G, node) for node in G.nodes},
    "local_reaching_centrality_undirected": lambda G: {node: nx.local_reaching_centrality(G.to_undirected(), node) for node in G.nodes},
#     "participation_coefficient": get_participation_coefficient,
    "participation_coefficient_undirected": get_participation_coefficient_undirected,
    "within_community_strength_undirected": within_community_strength_undirected
}


fun_degree_measures = {
    "degree": lambda G: dict(G.degree()), #dict(G.degree(G.nodes))
    "in_degree": lambda G: dict(G.in_degree()), #dict(G.in_degree(G.nodes))
    "out_degree": lambda G: dict(G.out_degree()) #dict(G.out_degree(G.nodes))
}


fun_clustering_measures = {
    "clustering": nx.clustering,
    "clustering_undirected": lambda G: nx.clustering(G.to_undirected()),
    "square_clustering": nx.square_clustering,
    "square_clustering_undirected": lambda G: nx.square_clustering(G.to_undirected())
}


fun_assortativity_measures = {
    "average_neighbor_degree": nx.average_neighbor_degree
}

fun_global_measures = {
    "global_density": lambda G: dict(zip(G.nodes, [nx.density(G.to_undirected())]*len(G))),
    "global_average_shortest_path_length": lambda G: dict(zip(G.nodes, [nx.average_shortest_path_length(G.to_undirected())]*len(G))),
    "global_reaching_centrality": lambda G: dict(zip(G.nodes, [nx.global_reaching_centrality(G.to_undirected())]*len(G))),
    "global_degree_assortativity_coefficient": lambda G: dict(zip(G.nodes, [nx.degree_assortativity_coefficient(G.to_undirected())] * len(G))),
    "global_degree_pearson_correlation_coefficient": lambda G: dict(zip(G.nodes,
                                                              [nx.degree_pearson_correlation_coefficient(G.to_undirected())] * len(G))),
    "global_transitivity": lambda G: dict(zip(G.nodes, [nx.transitivity(G.to_undirected())]*len(G))),
    "global_average_clustering": lambda G: dict(zip(G.nodes, [nx.average_clustering(G.to_undirected())]*len(G)))
}
                       

all_fun_measures = {
    **fun_centrality_measures,
#     **fun_degree_measures,
    **fun_clustering_measures,
    **fun_assortativity_measures,
    **fun_global_measures
}


#graph measures
fun_measures = {
#     "closeness_centrality_undirected": lambda graph_G: nx.closeness_centrality(graph_G.to_undirected()),
#     "betweenness_centrality_undirected": lambda graph_G: nx.betweenness_centrality(graph_G.to_undirected()),
    "participation_coefficient_undirected": get_participation_coefficient_undirected,
#     "clustering_undirected": lambda graph_G: nx.clustering(graph_G.to_undirected()),
    "degree": lambda graph_G: dict(graph_G.degree(graph_G.nodes)),
    "within_community_strength_undirected": within_community_strength_undirected
}

    

def get_centrality_measures(G):
    """
    #Participation coefficient: pb with true_divide --> TODO = inspect that (the participation coefficient function is partly random and can be of bad quality, because of the module detection in community_louvain
    """
    return {key: fun(G) for key, fun in fun_centrality_measures.items()}
    
    
def get_degree_measures(G):
    return {key: fun(G) for key, fun in fun_degree_measures.items()}
    
    
def get_clustering_measures(G):
    return {key: fun(G) for key, fun in fun_clustering_measures.items()}
    
    
def get_assortativity_measures(G):
    return {key: fun(G) for key, fun in fun_assortativity_measures.items()}


def get_number_nodes_at_length_n(G, n):
    """
    Note that for n=1, this returns the number of neighbors of each node
    If n>1, this returns for each node the number of nodes at distance n of this node
    """
    d_path_lengths = dict(nx.shortest_path_length(G.to_undirected()))
    number_nodes_at_length_n = {node: list(val.values()).count(n) for node, val in d_path_lengths.items()}
    return number_nodes_at_length_n


#GRAPH MEASURES (see https://networkx.github.io/documentation/stable/reference/algorithms/index.html)

#Global measures
# print(nx.info(G))
# print(nx.density(G))
# print(nx.average_shortest_path_length(G))
# pprint(nx.global_reaching_centrality(G))
# pprint(nx.degree_assortativity_coefficient(G))
# pprint(nx.degree_pearson_correlation_coefficient(G))
# pprint(nx.transitivity(G))
# pprint(nx.average_clustering(G))

#Centrality
# pprint(nx.degree_centrality(G))
# pprint(nx.in_degree_centrality(G))
# pprint(nx.out_degree_centrality(G))
# pprint(nx.eigenvector_centrality(G)) #same as nx.eigenvector_centrality_numpy(G)
# pprint(nx.katz_centrality(G)) #same asnx.katz_centrality_numpy(G)
# pprint(nx.closeness_centrality(G))
# pprint(nx.betweenness_centrality(G))
# pprint(nx.load_centrality(G))
# pprint(nx.harmonic_centrality(G))
# pprint({node:nx.local_reaching_centrality(G,node) for node in G.nodes})

#Degree
# print(G.in_degree(G.nodes))
# print(G.out_degree(G.nodes))
# print(G.degree(G.nodes)) #same as print(nx.degree(G))
# plt.hist(dict(G.degree(G.nodes)).values())
# plt.title("degree distribution")
# plt.show()
# print(nx.degree_histogram(G)) #degree distribution

#Clustering
# pprint(nx.clustering(G))
# pprint(nx.square_clustering(G))

#Assortativity
# pprint(nx.average_neighbor_degree(G))


#Not implemented for directed graphs
# print(nx.algorithms.smallworld.sigma(G))
# print(nx.algorithms.smallworld.omega(G))
# print(nx.algorithms.efficiency_measures.local_efficiency(G))
# print(nx.algorithms.efficiency_measures.global_efficiency(G))
# print(nx.rich_club_coefficient(G, normalized=False))
# print(nx.cycle_basis(G))
# pprint(nx.current_flow_closeness_centrality(G))
# pprint(nx.current_flow_betweenness_centrality(G))
# pprint(nx.edge_current_flow_betweenness_centrality(G))
# pprint(nx.approximate_current_flow_betweenness_centrality(G))
# pprint(nx.communicability_betweenness_centrality(G))
# pprint(nx.subgraph_centrality(G))
# pprint(nx.subgraph_centrality_exp(G))
# pprint(nx.estrada_index(G))
# pprint(nx.second_order_centrality(G))
# pprint(nx.voterank(G))
# pprint(nx.information_centrality(G))
# pprint(nx.triangles(G))
# pprint(nx.generalized_degree(G)) #to use on the graph made undirected?
    