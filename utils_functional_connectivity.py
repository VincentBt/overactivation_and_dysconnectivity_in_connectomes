import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr
import sys
from utils_graph_measures import get_modules
import numpy as np
# import statsmodels
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multitest as multi
import bct

def get_fmri_activation(activations_history_CI):
    """
    Computes the average over time of the instantaneous activation (using length 10)
    This is the quantity used to compute the correlations (rather than the instantaneous activation)
    """
    list_nodes = list(activations_history_CI.keys())
    # print("list_nodes:", list_nodes)
    # print("len(list_nodes) =", len(list_nodes))
    # gsize = len(list_nodes)
    T = len(activations_history_CI[list(activations_history_CI.keys())[0]]) #len(B_history_CI[list(B_history_CI.keys())[0]])

    #Taking average of 10 elements
    average = []
    for j,node in enumerate(list_nodes):
        average.append([np.mean(activations_history_CI[node][i:i+10]) for i in range(2, T+1-10) if (i-2)%10==0])
    average = np.array(average)
#     print("average.shape", average.shape)
    df = pd.DataFrame(average)
    #df = pd.DataFrame(activations_history_BP).T  #previous code (not smoothing temporally)
    df.index = list_nodes
    return df


def get_corr(df):
    """
    Calculate the correlation between individuals. 
    We have to transpose first because the corr function calculate the pairwise correlations between columns.
    Note that the default method is the Pearson correlation
    """
    #Calculate the correlation between individuals. We have to transpose first, because the corr function calculate the pairwise correlations between columns.
    corr = df.T.corr() #same as corr = np.corrcoef(df.to_numpy().T)
    #print(np.arctanh(corr)) #Fisher transformation
    return corr

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

def get_pval(df):
    """
    Computes the p-values for each combination of nodes (node1, node2)
    Note that all p-values for (node_i, node_i) are computed = 1 (logically they should be 0, but it's fine like that here because eventually we want to keep the edges with small p-values). On the contrary, the old function calculate_pvalues computes p-values = 0 for (node_i, node_i); that's the only difference between the 2 functions
    """
    pval = df.T.corr(method=pearsonr_pval)
    return pval
   
#Correlation with p value
# def calculate_pvalues(df):
#     df = df._get_numeric_data()
#     dfcols = pd.DataFrame(columns=df.columns)
#     pvalues = dfcols.transpose().join(dfcols, how='outer')
#     for r in df.columns:
#         for c in df.columns:
#             if c == r:
#                 df_corr = df[[r]].dropna()
#             else:
#                 df_corr = df[[r,c]].dropna()
#             pvalues[r][c] = stats.pearsonr(df_corr[r], df_corr[c])[1]
#     return pvalues
def calculate_pvalues(df):
    print("Old function - got replaced by get_pval (use get_pval(df) instead of calculate_pvalues(df.T))")
    sys.exit()
#     return get_pval(df.T) #the .T is here because I used calculate_pvalues(df.T), but get_pval already has a .T inside the function...


def plot_corr(corr):
    #Plot the correlation matrix (see https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec)
    fig = plt.figure(figsize=(14,7))
    ax = sns.heatmap(
        corr, 
    #     vmin=-1, vmax=1, center=0,
    #     cmap=sns.diverging_palette(20, 220, n=200),
        square=True)
    # ax.set_xticklabels(
    #     ax.get_xticklabels(),
    #     rotation=45,
    #     horizontalalignment='right')
#     plt.show()


def correction_pvalues(pval):
    """
    FDR correction
    To check that there is no pb, do:
    # print(np.min(pv[truth == False]), np.max(pv[truth == False]))
    # print(np.min(pv[truth == True]), np.max(pv[truth == True]))
    # plt.scatter(truth, pv)
    # plt.xlabel("truth")
    # plt.ylabel("pv")
    # plt.show()
    # plt.scatter(pval, pv)
    # plt.xlabel("pval")
    # plt.ylabel("pv")
    # plt.show()
    # plt.scatter(truth, corr)
    # plt.xlabel("truth")
    # plt.ylabel("corr")
    # plt.show()
    """
#     truth, pv = multi.fdrcorrection(pval) #, method="negcorr") #wrong when pval is a multidimensional array (that's why we convert it to a unidimensional array just below)
    truth, pv = multi.fdrcorrection(pval.to_numpy().flatten()) #, method="negcorr")
    truth = truth.reshape((pval.shape[0], pval.shape[1]))
    pv = pv.reshape((pval.shape[0], pval.shape[1]))
#     print("truth")
#     print(truth.shape)

    #checks that truth == (pv < 0.05), i.e. that truth = True iff pv < 0.05
    assert np.max(pv[truth == True]) <= 0.05
    assert np.min(pv[truth == False]) >= 0.05
    assert np.sum(truth != (pv < 0.05)) == 0 #i.e. truth == (pv < 0.05)
    
    pv_df = pd.DataFrame(pv)
    pv_df.index = pval.index
    pv_df.columns = pval.columns
#     print("pv_df")
#     display(pv_df)

#     return pv
    return pv_df


def flatten_df(df):
    """
    Transform df into a links data frame (3 columns only)
    """
#     list_nodes = list(df.columns)
    links = df.stack().reset_index()
    links.columns = ['node1', 'node2', 'value']
#     display(links)
    return links


def build_functional_graph_from_corr(corr, threshold_corr=0.5):
    """
    Keep only |correlation| > threshold and remove self edges (cor(A,A)=1)
    """
    links = flatten_df(corr)
#     display(links)
#     plt.hist(links['value'])
#     plt.title("Repartition of the correlations between nodes")
#     plt.show()
    links_filtered = links.loc[((links['value'] > threshold_corr) | (links['value'] < -threshold_corr)) 
                               & (links['node1'] != links['node2'])]
    print("Number of highly correlated FC edges:", len(links_filtered) // 2)

    # Build the graph from the edges
    G_fun_conn = nx.from_pandas_edgelist(links_filtered, 'node1', 'node2') # nx.from_pandas_dataframe(links_filtered, 'node1', 'node2') #outdated
    print("Number of functional edges (unoriented): {}".format(len(G_fun_conn.edges)))
    n_potential_edges = int(len(G_fun_conn) * (len(G_fun_conn) - 1) / 2)
    print("n_potential_edges = {}".format(n_potential_edges))
    print("Density of the functional graph = {}%".format(int(100 * len(G_fun_conn.edges) / n_potential_edges)))
    
    return G_fun_conn


def build_functional_graph_from_pval(pv, threshold_pval=1e-9):
    """
    Keep only p-values < threshold and remove self edges
    """
#     print("n_edges = {}".format(np.sum(pv.to_numpy() < threshold_pval)))
    links = flatten_df(pv)
#     print("n_edges = {}".format(np.sum(links['value'].to_numpy() < threshold_pval)))
#     display(links)
#     plt.hist(links['value'])
#     plt.title("Repartition of the p-values between nodes")
#     plt.show()
    links_filtered = links.loc[(links['value'] < threshold_pval)
                               & (links['node1'] != links['node2'])]
    print("Number of highly FC edges with low p-value:", len(links_filtered) // 2)
    
    # Build the graph from the edges
    G_fun_conn = nx.from_pandas_edgelist(links_filtered, 'node1', 'node2') # nx.from_pandas_dataframe(links_filtered, 'node1', 'node2') #outdated
    print("Number of functional edges (unoriented): {}".format(len(G_fun_conn.edges)))
    n_potential_edges = int(len(G_fun_conn) * (len(G_fun_conn) - 1) / 2)
    print("n_potential_edges = {}".format(n_potential_edges))
    print("Density of the functional graph = {}%".format(int(100 * len(G_fun_conn.edges) / n_potential_edges)))

    return G_fun_conn


def number_common_edges(G1, G2):
    print("number_common_edges is deprecated - used instead get_n_common_edges")
    sys.exit()
    

def get_common_edges(G1, G2, type_edge='unoriented'):
    if type_edge == 'unoriented':
        l = []
        for i in G2.edges:
            if i in G1.edges:
                l.append(i)
        return l
    else:
        s1 = set(list(G1.edges))
        s2 = set(list(G2.edges))
        return s1.intersection(s2)
    
    
def get_n_common_edges(G1, G2, type_edge='unoriented'):
    return len(get_common_edges(G1, G2, type_edge=type_edge))


def plot_G_fun_conn(G_fun_conn, G):
    """
    Plot the graph G_fun_conn (with pos corresponding to G)
    """
    plt.figure(figsize=(20,10))
    nx.draw(G_fun_conn, pos=graphviz_layout(G.to_undirected(), prog='dot'), 
            with_labels=True, node_color='orange', node_size=1200, edge_color='black', 
            linewidths=1, font_size=15) 
#     nx.draw(G_fun_conn, 
#             with_labels=True, node_color='orange', node_size=400, edge_color='black',
#             linewidths=1, font_size=15)
    plt.show()
    
    
def compute_modularity_ratio(G_fun_conn, G, output='ratio'):
    """
    Intra/Extra modularity ratio
    TODO: make sure that the modules computed are the same for a given graph (no randomness) e.g. for BP and CI
    """
    assert output in ['ratio', 'percent']
    
    intra, extra = 0, 0
    belonging = get_modules(G) #computes the modules
    for (node1, node2) in G_fun_conn.edges:
        if belonging[node1] == belonging[node2]:
            intra += 1
        else:
            extra += 1
    if output == 'ratio' and extra == 0:
        extra = 1
        return intra / extra
    else: #percentage of intramodular edges
        return intra / (intra + extra)
