import numpy as np
# from utils_plot_dict import plot_dict
from graph_generator import *
import sys
from utils_CI_BP import create_alpha_dict, generalized_xor
from operator import xor


def get_rank(M):
    return np.linalg.matrix_rank(M)


def get_stability_matrix(graph_G, 
                         alpha=None, alpha_c=None, alpha_d=None, dict_alpha_impaired=None,
                         theta=None):
    """
    returns S = F'(0) for CI: m_{t+1} = F(m_t) where m is a vector of all the m_{ij}
    i.e. returns the Jacobian of the system = stability matrix (for the fixed point 0)
    theta us a dict containing the constant external inputs (local fields) - by default 0
    """
    #running some checks: whether alpha is given, or (alpha_c,alpha_d) is given, or dict_alpha_impaired is given
    alpha_dict = create_alpha_dict(graph_G, 
                                   alpha=alpha, alpha_c=alpha_c, alpha_d=alpha_d, dict_alpha_impaired=dict_alpha_impaired)
#     assert xor((alpha_c is not None) and (alpha_d is not None), dict_alpha_impaired is not None) #exclusive or
#     assert not((alpha is None) and ((alpha_c is None) and (alpha_d is None)))
#     assert not((alpha is not None) and ((alpha_c is not None) or (alpha_d is not None)))
#     if alpha is None: #(alpha_c, alpha_d) is given
#         assert alpha_c == alpha_d #alpha_c != alpha_d is not implemented
#         alpha = alpha_c
    
    #recognize automatically whether the graph has weights or factors
    if 'weight' in list(graph_G.edges(data=True))[0][2].keys():
        which = 'symmetrical_with_w' #with w_ij (symmetrical case)
    else:
        which = 'general_case' #with f_ij
    
    list_edges = set(graph_G.edges).union(set({(node2,node1) for (node1,node2) in graph_G.edges})) #so that there is both i->j and j->i #before: "graph.keys()"
    
    if which == 'general_case':
#         print("Recover f_ij from the graph (for now it's impossible as we only associate w_ij)")
#         sys.exit()
        S = np.zeros((len(list_edges), len(list_edges)))
        for ind_1, edge1 in enumerate(list_edges):
            i, j = edge1
            neighbors_i = list(graph_G.neighbors(i)) #because graph_G is undirected #G.to_undirected()
            for ind_2, edge2 in enumerate(list_edges):
                k, l = edge2
                if (i == l) and (k in neighbors_i):
                    if edge1 in graph_G.edges: #in graph.keys():
                        f_ij = graph_G.edges[edge1[0], edge1[1]]['factor'] #graph[edge1[0], edge1[1]][0]
                        k_ij = f_ij[1,1]/(f_ij[1,0]+f_ij[1,1]) - f_ij[0,1]/(f_ij[0,0]+f_ij[0,1]) #(d/(c+d) - b/(a+b))
                    else:
                        f_ij = graph_G.edges[edge1[1], edge1[0]]['factor'] #graph[edge1[1], edge1[0]][0]
                        k_ij = f_ij[1,1]/(f_ij[0,1]+f_ij[1,1]) - f_ij[1,0]/(f_ij[0,0]+f_ij[1,0]) #(d/(b+d) - c/(a+c)) #because f_ji = f_ij.T
                    S[ind_1, ind_2] = k_ij
                    if k == j:
                        S[ind_1, ind_2] *= (1 - alpha_dict[i, j]) #alpha
        return S
    
    elif which == 'symmetrical_with_w':
        S = np.zeros((len(list_edges), len(list_edges)))
        for ind_1, edge1 in enumerate(list_edges):
            i, j = edge1
            neighbors_i = list(graph_G.neighbors(i)) #because graph_G is undirected #G.to_undirected()
            for ind_2, edge2 in enumerate(list_edges):
                k, l = edge2
                if (i == l) and (k in neighbors_i):
                    if edge1 in graph_G.edges: #graph.keys():
                        w_ij = graph_G.edges[edge1[0], edge1[1]]['weight'] #graph[edge1[0], edge1[1]][0]
                    else:
                        w_ij = graph_G.edges[edge1[1], edge1[0]]['weight'] #graph[edge1[1], edge1[0]][0] #because we take here symmetrical weights
                    tanh_Jij = 2 * w_ij - 1
                    if theta is None:
                        S[ind_1, ind_2] = tanh_Jij #for theta=0, it gives tanh_Jij
                    else: #case where some local fields are != 0
                        S[ind_1, ind_2] = tanh_Jij * (1 - np.tanh(theta[i])**2) / (1 - tanh_Jij**2 * np.tanh(theta[i])**2)
                    if k == j:
                        S[ind_1, ind_2] *= (1 - alpha_dict[i, j])
        return S
    
    
def get_stability_matrix_CIbeliefs(graph_G, dict_alpha=None):
    """
    returns S = F'(0) for CIbeliefs: X_{t+1} = F(X_t) where X is a vector of all the B_i(t-k) (where k in [0,k_max])
    i.e. returns the Jacobian of the system = stability matrix (for the fixed point 0)
    """
    print(dict_alpha)
    #recognize automatically whether the graph has weights or factors
    if 'weight' in list(graph_G.edges(data=True))[0][2].keys():
        which = 'symmetrical_with_w' #with w_ij (symmetrical case)
    else:
        which = 'general_case' #with f_ij
    
    list_edges = set(graph_G.edges).union(set({(node2,node1) for (node1,node2) in graph_G.edges})) #so that there is both i->j and j->i #before: "graph.keys()"
    list_nodes = list(graph_G.nodes)
    k_max = len(dict_alpha[list(dict_alpha.keys())[0]])
    print("k_max = {}".format(k_max))
    
    if which == 'general_case':
        S = np.zeros((len(list_nodes)*(k_max+1), len(list_nodes)*(k_max+1)))
        for ind_1, (i,k1) in enumerate(itertools.product(list_nodes, range(k_max+1))):
            neighbors_i = list(graph_G.neighbors(i)) #because graph_G is undirected
            for ind_2, (j,k2) in enumerate(itertools.product(list_nodes, range(k_max+1))):
                if k1 == 0:
                    if j in neighbors_i:
                        if (i,j) in graph_G.edges: #graph.keys():
                            f_ij = graph_G.edges[i, j]['factor'] #graph[edge1[0], edge1[1]][0]
                            k_ij = f_ij[1,1]/(f_ij[1,0]+f_ij[1,1]) - f_ij[0,1]/(f_ij[0,0]+f_ij[0,1]) #(d/(c+d) - b/(a+b))
                        else:
                            f_ij = graph_G.edges[edge1[1], edge1[0]]['factor'] #graph[edge1[1], edge1[0]][0]
                            k_ij = f_ij[1,1]/(f_ij[0,1]+f_ij[1,1]) - f_ij[1,0]/(f_ij[0,0]+f_ij[1,0]) #(d/(b+d) - c/(a+c)) #because f_ji = f_ij.T
                        S[ind_1, ind_2] = k_ij
#                         print(i,j,k1,k2, S[ind_1, ind_2])
                    elif (i == j) and (k2>=1):
                        S[ind_1, ind_2] = - dict_alpha[j][-k2]
#                         print("alpha",i,j,k1,k2, S[ind_1, ind_2])
                elif (i == j) and (k1 - 1 == k2): #and (k1 >=1) automatically
                    S[ind_1, ind_2] = 1 #just a copy: the function is the identity
        return S
    
    elif which == 'symmetrical_with_w':
        S = np.zeros((len(list_nodes)*(k_max+1), len(list_nodes)*(k_max+1)))
        for ind_1, (i,k1) in enumerate(itertools.product(list_nodes, range(k_max+1))):
            neighbors_i = list(graph_G.neighbors(i)) #because graph_G is undirected
            for ind_2, (j,k2) in enumerate(itertools.product(list_nodes, range(k_max+1))):
                if k1 == 0:
                    if j in neighbors_i:
                        if (i,j) in graph_G.edges: #graph.keys():
                            w_ij = graph_G.edges[i, j]['weight']
                        else:
                            w_ij = graph_G.edges[j, i]['weight'] #because we take here symmetrical weights
                        tanh_Jij = 2 * w_ij - 1
                        S[ind_1, ind_2] = tanh_Jij
#                         print(i,j,k1,k2, S[ind_1, ind_2])
                    elif (i == j) and (k2>=1):
                        S[ind_1, ind_2] = - dict_alpha[j][-k2] #alpha[-1] corresponds to k=2
#                         print("alpha",i,j,k1,k2, S[ind_1, ind_2])
                elif (i == j) and (k1 - 1 == k2): #and (k1 >=1) automatically
                    S[ind_1, ind_2] = 1 #just a copy: the function is the identity
        return S

    
def plot_eigenvalues_from_graph(graph_G, 
                                alpha=None, alpha_c=None, alpha_d=None, dict_alpha_impaired=None,
                                order='1'):
    """
    Can help to determine the stability of the system (approx with rate model, at different possible orders)
    """
    #run some checks
    assert not((alpha is None) and (alpha_c is None) and (alpha_d is None) and (order != '0'))
    assert order in ['exact', '0', '1', 'inf'] #exact means stability matrix (F'(0) where M^{t+1} = F'(M^t)); '0'/'1'/'inf' mean the order of the approximation (while approximating the system with a "rate network" i.e. without M but only with B)
    if order != 0: #checking that whether alpha is given, or (alpha_c,alpha_d) is given, or dict_alpha_impaired is given
        assert generalized_xor(alpha is not None, 
                               (alpha_c is not None) and (alpha_d is not None), 
                               dict_alpha_impaired is not None)
    # N = 100
    # J = np.random.normal(size=(N,N), scale=1/np.sqrt(N)) #random Gaussian matrix
    if order == 'exact':
        S = get_stability_matrix(graph_G, alpha=alpha, 
                                 alpha_c=alpha_c, alpha_d=alpha_d, 
                                 dict_alpha_impaired=dict_alpha_impaired)
        plot_eigenvalues(S)
    else:
        print("this is an approximation (order = {})".format(order))
        J = get_adjacency_matrix(graph_G)
        if dict_alpha_impaired is not None:
            print("not implemented") #one would need to change get_corrected_matrix and define alpha_matrix (based on graph_G and dict_alpha_impaired)
            sys.exit()
        J_corrected = get_corrected_matrix(J, alpha=alpha, alpha_c=alpha_c, alpha_d=alpha_d, order=order)
        plot_eigenvalues(J_corrected)
    
    
def plot_eigenvalues(M=None, eig_values=None):
    """
    M is the connectivity matrix (or the stability matrix)
    """
    assert ((M is not None) or (eig_values is not None)) and not((M is not None) and (eig_values is not None)) #= exclusive OR
    
    #compute the eigenvalues of M
    if M is not None: #otherwise eig_values are already given
        eig_values = np.linalg.eig(M)[0]
    eig_values_real = eig_values.real
    eig_values_imag = eig_values.imag

    #plot the eigenvalues of M
    plt.figure(figsize=(6,6))
    plt.scatter(eig_values_real, eig_values_imag)
    #plot the unity circle
    def circle(angle):
        return (np.cos(angle), np.sin(angle))
    angle_all = np.linspace(0, 2*np.pi, 100)
    x_circle, y_circle = circle(angle_all)
    plt.plot(x_circle, y_circle, color='red')
    plt.show()

    
def get_corrected_matrix(J, 
                         alpha=None, alpha_c=None, alpha_d=None, alpha_matrix=None, 
                         order='1'):
    """
    order: order of the linearization of CI (0 or 1 or inf) to have something similar to a "rate network"
    For non-uniform alpha, alpha_matrix (with corresponding edges with J) needs to be provided in input (or alternatively (alpha_c; alpha_d), only in the case where alpha_c = alpha_d)
    """
    #run some checks
    assert order in ['0', '1', 'inf']
    assert not((alpha is None) and (alpha_c is None) and (alpha_d is None) and (alpha_matrix is None) and (order != '0'))
    if order != 0: #checking that whether alpha is given, or (alpha_c,alpha_d) is given
        assert generalized_xor(alpha is not None, (alpha_c is not None) and (alpha_d is not None), alpha_matrix is not None)
    if (alpha_c is not None) and (alpha_d is not None):
        print("alpha_c != alpha_d: one should provide alpha_matrix instead") #if alpha_c != alpha_d, we need to know the alpha_ij for each directed edge (i,j), thus we need to know the orientation of each edge --> give graph_G or alpha_matrix (Here we chose alpha_matrix)
        assert alpha_c == alpha_d
        alpha = alpha_c
    if order == '0': #as if alpha was = 0
        return J
    if alpha is not None:
        if order == '1': #order 1 in alpha
            return J - alpha * np.diag(np.diagonal(J.dot(J))) #d= np.sum(J * J.T, axis=1)
        elif order == 'inf':
            J_hat = J / (1 - alpha**2 * J * J.T)
            J_hat_hat = np.diag(np.sum(J * J.T / (1 - alpha**2 * J * J.T), axis=1))
            return J_hat - alpha * J_hat_hat
    else:
        print("not implemented")
#         if order == '1': #order 1 in alpha
#             return J - np.diag(np.diagonal(J.dot(J * alpha_matrix))) #TODO: check that the alpha_matrix is here indeed
#         else:
#             #TODO (but I think that there is no formula in the case where alpha_ij != alpha_ji)
        sys.exit()
        
        
def plot_hist_real_part_eigenvalues(J, list_alpha, order='1'):
    """
    input: 'order' is the order where we stop write the imbricated messages in CI
    Non-applicable to non-uniform alpha
    """
    assert order in ['1', 'inf']
    for alpha in list_alpha:
        J_corrected = get_corrected_matrix(J, alpha)
        #compute and plot the eigenvalues of J_corrected
        eig_values = np.linalg.eig(J_corrected)[0]
        eig_values_real = eig_values.real
        eig_values_imag = eig_values.imag
        sns.distplot(eig_values_real, hist = False, kde = True,
                     kde_kws = {'linewidth': 3}, #kde_kws = {'shade': True, 'linewidth': 3}, 
                     label = 'alpha = {}'.format(alpha)) #color='darkblue'
#         plt.hist(eig_values_real, label='alpha = {}'.format(alpha), alpha=0.5)
    plt.legend()
    plt.xlabel('Re(eigenvalue)')
    plt.ylabel('distribution')
    plt.show()


def upper_bound_largest_eigenvalue(M):
    """
    Gives quickly an upper bound of the largest eigenvalue (with having to compute them)
    Corrolary of Gershgorin circle theorem - see https://math.stackexchange.com/questions/2005314/link-between-the-largest-eigenvalue-and-the-largest-entry-of-a-symmetric-matrix
    See also Corollary 2.4 and 2.5 (+ maybe other parts in Ch.2) from Mooij's PhD thesis for additional criteria
    """
    assert M.shape[0] == M.shape[1] #check that M is a square matrix
    M1 = np.abs(M.copy())
    for i in range(M.shape[0]):
        M1[i,i] = M[i,i]
    upper_bound_1 = np.max(np.sum(M1, axis=1))
    if upper_bound_1 < 1:
        print("Re(largest_eigenvalue)<1")
    #weaker result than above
    upper_bound_2 = M.shape[0]
    if upper_bound_2 < 1:
        print("Re(largest_eigenvalue)<1")
    
    
def lower_bound_largest_eigenvalue(M):
    """
    Gives quickly a lower bound of the largest eigenvalue (with having to compute them)
    Corrolary of Gershgorin circle theorem - see https://math.stackexchange.com/questions/2005314/link-between-the-largest-eigenvalue-and-the-largest-entry-of-a-symmetric-matrix
    See also Corollary 2.4 and 2.5 (+ maybe other parts in Ch.2) from Mooij's PhD thesis for additional criteria
    """
    lower_bound = np.max(np.diagonal(M))
    if lower_bound > 1:
        print("Re(largest_eigenvalue)>1")
    M1 = - np.abs(M.copy())
    for i in range(M.shape[0]):
        M1[i,i] = M[i,i]
    lower_bound_1 = np.min(np.sum(M1, axis=1))
    if lower_bound_1 > 1:
        print("Re(largest_eigenvalue)>1")
    