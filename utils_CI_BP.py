# from jax import jit #I removed that as it is not speeding up the computations
# import jax.numpy as np #I removed that as it is not speeding up the computations
# import numpy as onp
import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
from operator import xor


def F(x, w):
    print("function F is deprecated: use F_w (or F_f) instead")
    return F_w(x, w)

#@jit
def F_w(x, w): #previously called F
    """
    Compute F_x = F(x, w)
    MAYBE INSTEAD I CAN TAKE PRE-BUILT FUNCTIONS (E.G. SIGMOID) HERE: https://github.com/google/jax/blob/master/jax/experimental/stax.py (it could be faster)
    Note that we have: F(x) = 2 * artanh( (2*w_ij-1) * tanh(x/2) )
    """
#     exp_x = np.exp(x)
#     F_x = np.log((w*exp_x + 1 - w) / ((1-w)*exp_x + w))
#     return F_x
    return 2 * np.arctanh( (2*w-1) * np.tanh(x/2) ) #a bit faster (by 30% compared to the lines above)

def F_f(x, factor):
    """
    Compute F_x = F(x, factor)
    MAYBE INSTEAD I CAN TAKE PRE-BUILT FUNCTIONS (E.G. SIGMOID) HERE: https://github.com/google/jax/blob/master/jax/experimental/stax.py (it could be faster)
    """
    exp_x = np.exp(x)
#     F_x = np.log((factor[1,1]*exp_x + factor[0,1]) / (factor[1,0]*exp_x + factor[0,0]))
    return F_x
#     try:
#         F_x = np.log((factor[1,1]*exp_x + factor[0,1]) / (factor[1,0]*exp_x + factor[0,0]))
#         return F_x
#     except RuntimeWarning:
#         print(x, factor)

def F_w_approx_tanh(x, w):
    """
    Linearizes F_w(x) ~ 2 * (2*w-1) * tanh(x/2)
    (the approximation is really good for w<0.7, and even for w~0.7 it starts to fail only for x>2)
    """
    return 2 * (2*w-1) * np.tanh(x/2)


def F_w_approx(x, w):
    """
    Approximation of F_x = F(x, w)
    """
    eps_1 = w - 0.5
    eps_2 = -(w - 0.5)
    eps_3 = -(w - 0.5)
    eps_4 = w - 0.5
    p = 1 / (1 + np.exp(-x)) #p = sig(x)
    
    #Approx where bounds are fitted but the approx around x=0 is bad (because the arbitrary sigmoid function is fitted based on the values at -inf and +inf). Arbitrary sigmoid function which is designed to fit well the curve for x=-inf and +inf. But this function often doesn't fit well for x close to 0
#     low = np.log((0.5+eps_2) / (0.5+eps_1))
#     high = np.log((0.5+eps_4) / (0.5+eps_3))
#     return low + (high-low)*p
    #Approx which fits well for low x but not well for the bounds (except if all eps_i are small). Based on the Taylor expansion for low eps_i. This function often doesn't fit well for strong |x| (unless all |eps_i|<0.05 for instance)
    return 2*(eps_2-eps_1) + p*2*(eps_4-eps_2-eps_3+eps_1) #= - 4*eps_1 + p*8*eps_1

def F_f_approx(x, f):
    """
    Approximation of F_x = F(x, factor)
    """
    eps_1 = f[0,0]-0.5
    eps_2 = f[0,1]-0.5
    eps_3 = f[1,0]-0.5
    eps_4 = f[1,1]-0.5
    p = 1 / (1 + np.exp(-x)) #p = sig(x)
    
    #Approx where bounds are fitted but the approx around x=0 is bad (because the arbitrary sigmoid function is fitted based on the values at -inf and +inf)
#     low = np.log((0.5+eps_2) / (0.5+eps_1))
#     high = np.log((0.5+eps_4) / (0.5+eps_3))
#     return low + (high-low)*p
    #Approx which fits well for low x but not well for the bounds (except if all eps_i are small)
    return 2*(eps_2-eps_1) + p*2*(eps_4-eps_2-eps_3+eps_1)


# class Graph:
#     def __init__(self, graph_G):
#         super().__init__()
        
# # #         self.nodes = np.unique(list(graph.keys())) #with numpy
# #         self.nodes = set(list(itertools.chain.from_iterable(graph.keys()))) #with jax.numpy
        
#         self.n = len(self.nodes)
        
# # #         self.edges = []
# # #         for (i, j) in graph.keys():
# # #             self.edges.append((i, j))
# #         #better (?) to have a dictionnary of all possible pairs, with True if connection
        
# #         self.neighbors = {}
# #         for i in self.nodes:
# #             self.neighbors[i] = [j for j in self.nodes if (((i,j) in graph) or ((j,i) in graph))]
 

def generalized_xor(*args):
    """
    Returns True if and if only there is exactly one argument equal to True (and all the others = False)
    It helps to check that one of the options (and only one) is selected
    For 2 arguments, it is the normal XOR = operator.xor function
    """
    return np.sum([int(arg) for arg in args]) == 1


# def one_among(*args):
#     """
#     Returns True if and if only there is exactly one argument different from None (and all the others = None)
#     It helps to check that one of the options (and only one) is selected
#     """
    

def create_alpha_dict(graph_G, 
                      alpha=None, alpha_c=None, alpha_d=None, dict_alpha_impaired=None):
    """
    Creates a dictionnary with all the alpha_ij (for all directed edge (i,j))
    It also checks that exactly one among {alpha; (alpha_c,alpha_d); dict_alpha_impaired} is defined
    dict_alpha_impaired is defined for any oriented edge (i,j)
    The information of graph_G is only useful to know the orientation of edges (= whether it is going down or up)
    """
    #do some checks
#     assert xor(not(alpha_c is None) and not(alpha_d is None), not(dict_alpha_impaired is None)) #exclusive or
    assert generalized_xor((alpha_c is not None) and (alpha_d is not None),
                           dict_alpha_impaired is not None,
                           alpha is not None) #exclusive or (True if and if only one of the arguments is True)
    dict_alpha = {}
    
    if alpha is not None:
#         alpha_c = alpha
#         alpha_d = alpha
#         return {key: alpha for key in graph_G.edges}
        for (node1, node2) in graph_G.edges:
            dict_alpha[node1, node2] = alpha
            dict_alpha[node2, node1] = alpha #because dict_alpha should have both alpha_ij and alpha_ji
        return dict_alpha
    
    elif (alpha_c is not None) and (alpha_d is not None): #alpha_c and alpha_d are uniform i.e. identical for each edge
        dict_alpha_edge = {'down': alpha_c,
                           'up': alpha_d} #because M_ij = F(B_i - alpha_ij M_ji)
        for node1, node2, val in graph_G.edges(data=True): #we do not count the same edge twice because graph_G only has (node1,node2), not both (node1,node2) and (node2,node1)
            type_edge = val['orientation']
            type_edge_opposite = 'up' if type_edge == 'down' else 'down'
            dict_alpha[node1, node2] = dict_alpha_edge[type_edge]
            dict_alpha[node2, node1] = dict_alpha_edge[type_edge_opposite]
            
    else: #non-uniformity of alpha, indicated by dict_alpha_impaired
        for edge in graph_G.edges:
            for edge_dir in [(edge[0], edge[1]), (edge[1],edge[0])]: #to take both directions
                node1, node2 = edge_dir
                if (node1, node2) in dict_alpha_impaired.keys():
                    dict_alpha[node1, node2] = dict_alpha_impaired[node1, node2]
                else:
                    dict_alpha[node1, node2] = 1

    return dict_alpha


class Network:
    
    def __init__(self, graph_G, M_ext, 
                 alpha=None, alpha_c=None, alpha_d=None, dict_alpha_impaired=None,
                 damping=0, keep_history=False, with_factors=False,
                 which_CI='CI'
                ):
        """
        damping=0 corresponds to BP/CI : M_new = (1-damping)*F(M_old) + damping*M_old
        graph_G is undirected (has type Graph) but has information about the directionality ('up' or 'down' associated to each undirected edge (node1,node2), i.e. (node2,node1) does not exist in graph_G)
        
        Potentially one could include non-uniform alpha for edges (add field 'alpha' to the dict which is associated to each edge), or even different alpha_c and alpha_d (fields 'alpha_c' and 'alpha_d'):
        input "dict_alpha_impaired" should have the form {edge:alpha} (or otherwise {edge:(alpha_c,alpha_d)} if we want to specify alpha_c=alpha_d) (= dict with all edges for which alpha is impaired, and the associated value of alpha). If an edge isn't indicated in the dictionnary but exists in the graph, it means alpha=1 for this edge
        """
        super().__init__()
        
        #do some checks
        assert generalized_xor(alpha is not None, 
                               (alpha_c is not None) and (alpha_d is not None), 
                               dict_alpha_impaired is not None)
        
        self.graph_G = graph_G #Graph(graph_G)
        self.damping = damping
        
        #initiate the external messages
        ex_Mext = M_ext[list(M_ext.keys())[0]]
        if isinstance(ex_Mext, int) + isinstance(ex_Mext, float): #case where M_ext is constant over the whole simulation
#             print("M_ext is constant")
            self.T = max(len(self.graph_G) * 2, 50) #this is arbitrary - it is possible that BP/CI does not have time to fully converge with that number of iterations
            self.M_ext = {node : ([M_ext[node]]*self.T if node in M_ext else [0]*self.T) for node in self.graph_G.nodes}
        else:
#             print("M_ext is variable")
            self.T = len(M_ext[list(M_ext.keys())[0]])
            self.M_ext = {node : (M_ext[node] if node in M_ext else [0]*self.T) for node in self.graph_G.nodes} #because M_ext only has keys corresponding to stimulated nodes only (if for instance n_stimulated_nodes=1)
#         print(self.M_ext)
        
        #define the connections weights
        self.with_factors = with_factors
        if with_factors == False:
            self.w = {}
            for node1, node2, val in graph_G.edges(data=True): #for (i, j), val in graph_G.items():
                w = val['weight'] #val[0]
                self.w[node1, node2] = w
                self.w[node2 ,node1] = w
        else:
            self.factor = {}
            for node1, node2, val in graph_G.edges(data=True): #for (i, j), val in graph.items():
                factor = val[0]
                self.factor[node1, node2] = factor
                self.factor[node2, node1] = factor.T
        
        #define the alpha (for each each edge). Convention: to compute M_ij, alpha[i,j] is considered (not alpha[j,i]) 
        #i.e. M_ij = F(B_i - alpha_ij M_ji)
        # print("dict_alpha_impaired", dict_alpha_impaired)
        if which_CI == 'CI':
            self.alpha = create_alpha_dict(graph_G, 
                                           alpha=alpha, alpha_c=alpha_c, alpha_d=alpha_d, 
                                           dict_alpha_impaired=dict_alpha_impaired)
#             print("self.alpha for a given node = ")
#             if type(self.alpha[list(self.alpha)[0]]) == int:
#                 print(self.alpha)
#             plt.plot(self.alpha[list(self.alpha)[0]])
#             plt.show()
        elif which_CI == 'CIbeliefs':
            self.alpha = dict_alpha_impaired
        # print("self.alpha", self.alpha)
        
        #initiate the messages
        if which_CI == 'CI':
            self.M = {}
            for (node1, node2) in graph_G.edges:
                self.M[node1, node2] = 0
                self.M[node2, node1] = 0
        
        #initiate B_history (history of the beliefs)
        if keep_history or which_CI == 'CIbeliefs':
            self.B_history = {}
            for node in self.graph_G.nodes:
                self.B_history[node] = []
        # print("self.B_history", self.B_history)
        
        if np.sum([not(isinstance(val, int) or isinstance(val, float)) for val in self.alpha.values()]) != 0:
            self.temporal_alpha = True
        else:
            self.temporal_alpha = False #False if (isinstance(alpha_c, int) + isinstance(alpha_c, float)) else True #does not deal with the cases where only one out of (alpha_c, alpha_d) is constant and the other one varies with time
#         print("self.temporal_alpha", self.temporal_alpha)
        
        #initiate the beliefs
#         self.B = {i: 0 for i in self.graph_G.nodes}
        
        #create useful variables from all the ones above
        self.neighbors = {i: self.get_neighbors(i) for i in self.graph_G.nodes} #{i: [j for j in self.graph_G.nodes if (j, i) in self.graph_G] for i in ....}   #compute it once for all
        
    def get_neighbors(self, node):
        return list(self.graph_G.neighbors(node)) #ok because graph_G is undirected; otherwise use nx.all_neighbors(node) #self.graph.neighbors[node]
        
    def step_message_passing(self, t, keep_history=False):
        """
        TODO: vectorize on all edges?? (using Mooij's PhD thesis). That will probably be faster
        Careful with the convention: M_ij = F(B_i - alpha_ij M_ji)
        i.e. alpha_ij appears in the computation of M_ij, but is in front of the term M_ji
        """
#         M_old = M.copy()
#         B_old = 
        sum_M = self.compute_beliefs(t-1)
#         print("sum_M = {}".format(sum_M))
        
        #copy self.M
        M_old = copy.copy(self.M) #shallow/deep copy?
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                for (i,j) in self.M:
                    self.M[i, j] = F_w(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j])
            else:
                for (i,j) in self.M:
                    self.M[i, j] = F_f(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j])
        else: #default
            if self.with_factors == False:
#                 for (i,j) in self.M:
#                     self.M[i, j] = F_w(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j])
                compute_message = lambda edge: F_w(sum_M[edge[0]] - self.alpha[edge] * M_old[edge[1], edge[0]], self.w[edge])
                self.M = dict(zip(self.M.keys(), map(compute_message, self.M.keys())))
            else:
                for (i,j) in self.M:
                    self.M[i, j] = F_f(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j])
#         print({key: np.array(val).item() for key, val in self.M.items()})
        del M_old
        
#         print(self.M)
#         print()
    
        if keep_history:
            B_current = self.compute_beliefs(t)
            for node in self.graph_G.nodes:
                self.B_history[node].append(B_current[node])
                
    #@jit  
    def step_message_passing_CIbeliefs(self, t):
        """
        Uses the following approximation:
        $$B_j^t = \sum\limits_{i \in N(j)} F_{ij}(B_i^{t-1}) - \sum\limits_{k=2}^{k_{max}} \alpha_{j,k} B_j^{t-k}$$
        (alpha are associated to nodes j, and not to edges as in CI)
    
        keep_history is not used (the history of the beliefs is needed in the equation)
    
        Here alpha are in the opposite order: alpha_kmax, alpha_kmax-1, ..., alpha_3, alpha_2
    
        TODO: Try to write it in a vectorized way
        """
        if self.temporal_alpha: #alpha varies with time
            print("not implemented")
            sys.exit()
        else: #default
            if self.with_factors == False:
                for j in self.B_history:
#                     print("alpha[j]", self.alpha[j])
                    F_wij_t_minus_1 = lambda i: F_w(self.B_history[i][t-1], self.w[i,j])
                    neighbors_contrib = np.sum(list(map(F_wij_t_minus_1 , self.neighbors[j]))) #np.sum([F_w(self.B_history[i][t-1], self.w[i,j]) for i in self.neighbors[j]])
                    self_inhib = np.dot(self.alpha[j][max(len(self.alpha[j])-len(self.B_history[j][:-1]),0):], 
                                        np.array(self.B_history[j])[-len(self.alpha[j]) - 1: -1]) #self.alpha[j][-len(self.B_history[j][:-1]):] (without the "len(self.alpha[j])") does not work: for instance [1,2,3][-0:] = [1,2,3] and not [] as we would like   #added the np.array(list) in order to use jax.numpy (the input must be of type array in function dot)
#                     print("neighbors_contrib", neighbors_contrib)
#                     print("self_inhib", self_inhib)
#                     print("self.M_ext[j][t-1]", self.M_ext[j][t-1])
                    self.B_history[j].append(neighbors_contrib - self_inhib + self.M_ext[j][t-1]) #self.alpha[j] = ... , alpha_{j,3}, alpha_{j,2}.  #self.alpha[j][-len(self.B_history[j]):] because at the beginning of the simulation the list self.alpha[j] has a bigger size than self.B_history[j]
            else:
                for j in self.B_history:
                    self.B_history[j].append(
                        np.sum([F_f(self.B_history[i][t-1], self.factor[i,j]) for i in self.neighbors[j]]) - 
                        np.dot(self.alpha[j][-len(self.B_history[j][:-1]):], 
                               np.array(self.B_history[j])[-len(self.alpha[j]) - 1: -1]) +
                        self.M_ext[j][t-1]
                    ) #added the np.array(list) in order to use jax.numpy (the input must be of type array in function dot)
                
                
    def step_message_passing_approx(self, t, keep_history=False):
        sum_M = self.compute_beliefs(t-1)
        
        #copy self.M
        M_old = copy.copy(self.M) #shallow/deep copy?
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                for (i,j) in self.M:
                    self.M[i, j] = F_w_approx(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j])
            else:
                for (i,j) in self.M:
                    self.M[i, j] = F_f_approx(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j])
        else: #default
            if self.with_factors == False:
                for (i,j) in self.M:
                    self.M[i, j] = F_w_approx(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j])
            else:
                for (i,j) in self.M:
                    self.M[i, j] = F_f_approx(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j])
        del M_old
        
        if keep_history:
            B_current = self.compute_beliefs(t)
            for node in self.graph_G.nodes:
                self.B_history[node].append(B_current[node])
                
                
    def step_message_passing_approx_tanh(self, t, keep_history=False):
        sum_M = self.compute_beliefs(t-1)
        
        #copy self.M
        M_old = copy.copy(self.M) #shallow/deep copy?

        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                for (i,j) in self.M:
                    self.M[i, j] = (1-self.damping) * F_w_approx_tanh(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j]) + self.damping * M_old[i, j]
#                     self.M[i, j] = F_w_approx_tanh(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j])
            else:
                for (i,j) in self.M:
                    self.M[i, j] = (1-self.damping) * F_f_approx_tanh(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j]) + self.damping * M_old[i, j]
#                     self.M[i, j] = F_f_approx_tanh(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j])
        else: #default
            if self.with_factors == False:
                for (i,j) in self.M:
                    self.M[i, j] = (1-self.damping) * F_w_approx_tanh(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j]) + self.damping * M_old[i, j]
#                     self.M[i, j] = F_w_approx_tanh(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j])
            else:
                for (i,j) in self.M:
                    self.M[i, j] = (1-self.damping) * F_f_approx_tanh(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j]) + self.damping * M_old[i, j]
#                     self.M[i, j] = F_f_approx_tanh(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j])
        del M_old
        
        if keep_history:
            B_current = self.compute_beliefs(t)
            for node in self.graph_G.nodes:
                self.B_history[node].append(B_current[node])
#             print(self.B_history)
#             print({key:val[-1] for key,val in self.B_history.items()})
            
            
    def compute_beliefs(self, t):
        B = {i: self.M_ext[i][t] + np.sum([self.M[j, i] for j in self.graph_G.neighbors(i)]) #self.graph_G.neighbors[i]])
             for i in self.graph_G.nodes
            } #should it be M_ext[i][t] or M_ext[i][t-1]??
        return B
    
            
    def run_CI(self, verbose=False, keep_history=False):
        if keep_history:
            for node in self.graph_G.nodes:
                self.B_history[node].append(0)
            B_current = self.compute_beliefs(0) #0th iteration (just taking into account the external messages
            for node in self.graph_G.nodes:
                self.B_history[node].append(B_current[node]) #this line is not mandatory (starting at 0 is enough) #--> remove?
        for t in range(self.T):
            self.step_message_passing(t, keep_history)
        B = self.compute_beliefs(self.T - 1) #final belief
        if verbose:
            print("self.M")
            print(self.M)
        return B
    
    #@jit
    def run_CIbeliefs(self, verbose=False):
        for node in self.graph_G.nodes:
            self.B_history[node].append(0)
        # print("self.B_history", self.B_history)
        for t in range(1, self.T+1): #starts at 1!!!
            self.step_message_passing_CIbeliefs(t)
        B = {node: B_history_node[-1] for node, B_history_node in self.B_history.items()} #final belief
        if verbose:
            print("self.B")
            print(self.B)
        return B
    
    def run_CI_approx(self, verbose=False, keep_history=False):
        if keep_history:
            for node in self.graph_G.nodes:
                self.B_history[node].append(0)
            B_current = self.compute_beliefs(0) #0th iteration (just taking into account the external messages
            for node in self.graph_G.nodes:
                self.B_history[node].append(B_current[node]) #this line is not mandatory (starting at 0 is enough)
        for t in range(self.T):
            self.step_message_passing_approx(t, keep_history)
        B = self.compute_beliefs(self.T - 1) #final belief
        if verbose:
            print("self.M")
            print(self.M)
        return B
    
    def run_CI_approx_tanh(self, verbose=False, keep_history=False):
        """
        "linearized" CI (using the linearization artanh(x)~x --> same equation as Srdjan's work, with a tanh)
        To be more precise, we use artanh(J*tanh(x))~J*tanh(x)
        So the "linearized" CI is not fully fully linearized but has a tanh
        """
        if keep_history:
            for node in self.graph_G.nodes:
                self.B_history[node].append(0)
            B_current = self.compute_beliefs(0) #0th iteration (just taking into account the external messages
            for node in self.graph_G.nodes:
                self.B_history[node].append(B_current[node]) #this line is not mandatory (starting at 0 is enough)
        for t in range(self.T):
#             print("t = {}".format(t))
#             print("M_ext = {}".format(M_ext[t]))
            self.step_message_passing_approx_tanh(t, keep_history)
        B = self.compute_beliefs(self.T - 1) #final belief
        if verbose:
            print("self.M")
            print(self.M)
        return B


def run_CI(graph_G, M_ext, 
           alpha_c=None, alpha_d=None, dict_alpha_impaired=None,
           damping=0, verbose=False, keep_history=False, with_factors=False):
    assert xor(not(alpha_c is None) and not(alpha_d is None), not(dict_alpha_impaired is None)) #exclusive or
    net = Network(graph_G, M_ext, alpha_c=alpha_c, alpha_d=alpha_d, dict_alpha_impaired=dict_alpha_impaired, 
                  damping=damping, keep_history=keep_history, with_factors=with_factors)
    B_final = net.run_CI(verbose=verbose, keep_history=keep_history)
    if keep_history:
        net.B_history = {key:np.array(val) for key,val in net.B_history.items()} #need to transform the lists into numpy arrays
        return B_final, net.B_history
    else:
        return B_final, None #return B_final
    #or simply in all cases "return net"?


def run_BP(graph_G, M_ext, verbose=False, damping=0, keep_history=False, with_factors=False):
    net = Network(graph_G, M_ext, alpha=1, 
                  damping=damping, keep_history=keep_history, with_factors=with_factors)
    B_final = net.run_CI(verbose=verbose, keep_history=keep_history)
    if keep_history:
        net.B_history = {key:np.array(val) for key,val in net.B_history.items()}
        return B_final, {key:np.array(val) for key,val in net.B_history.items()} #need to transform the lists into numpy arrays
    else:
        return B_final, None #return B_final
    #or simply in all cases "return net"?
    
def run_CIbeliefs(graph_G, M_ext, 
                  alpha_c=None, alpha_d=None, dict_alpha_impaired=None,
                  damping=0, verbose=False, keep_history=False, with_factors=False):
    """
    Uses the following approximation:
    $$B_j^t = \sum\limits_{i \in N(j)} F_{ij}(B_i^{t-1}) - \sum\limits_{k=2}^{k_{max}} \alpha_{jk} B_j^{t-k}$$
    (alpha are associated to nodes, and not to edges as in CI)
    
    There is no reason to use alpha_c and alpha_d, but instead alpha_dict (keys = all nodes)
    
    keep_history is not used in functon Network and function self.run_CIbeliefs
    """
    assert damping == 0 #what would damping be here?
    assert xor(not(alpha_c is None) and not(alpha_d is None), not(dict_alpha_impaired is None)) #exclusive or
    net = Network(graph_G, M_ext, alpha_c=alpha_c, alpha_d=alpha_d, dict_alpha_impaired=dict_alpha_impaired, 
                  damping=damping, keep_history=keep_history, with_factors=with_factors, which_CI='CIbeliefs')
    B_final = net.run_CIbeliefs(verbose=verbose)
    if keep_history:
        net.B_history = {key:np.array(val) for key,val in net.B_history.items()} #need to transform the lists into numpy arrays
        return B_final, net.B_history
    else:
        return B_final, None #return B_final
    #or simply in all cases "return net"?
    
def run_CI_approx(graph_G, M_ext, 
                  alpha_c=None, alpha_d=None, dict_alpha_impaired=None,
                  damping=0, verbose=False, keep_history=False, with_factors=False):
    assert xor(not(alpha_c is None) and not(alpha_d is None), not(dict_alpha_impaired is None)) #exclusive or
    net = Network(graph_G, M_ext, alpha_c=alpha_c, alpha_d=alpha_d, dict_alpha_impaired=dict_alpha_impaired, 
                  damping=damping, keep_history=keep_history, with_factors=with_factors)
    B_final = net.run_CI_approx(verbose=verbose, keep_history=keep_history)
    if keep_history:
        net.B_history = {key:np.array(val) for key,val in net.B_history.items()}
        return B_final, net.B_history
    else:
        return B_final, None #return B_final
    #or simply in all cases "return net"?
    
def run_BP_approx(graph_G, M_ext, damping=0, verbose=False, keep_history=False, with_factors=False):
#     alpha_c, alpha_d = 1, 1
#     net = Network(graph_G, M_ext, alpha_c=alpha_c, alpha_d=alpha_d,
    net = Network(graph_G, M_ext, alpha=1,
                  damping=damping, keep_history=keep_history, with_factors=with_factors)
    B_final = net.run_CI_approx(verbose=verbose, keep_history=keep_history)
    if keep_history:
        net.B_history = {key:np.array(val) for key,val in net.B_history.items()}
        return B_final, net.B_history
    else:
        return B_final, None #return B_final
    #or simply in all cases "return net"?
    
    
def run_CI_approx_tanh(graph_G, M_ext, 
                       alpha_c=None, alpha_d=None, dict_alpha_impaired=None,
                       damping=0, verbose=False, keep_history=False, with_factors=False):
    assert xor(not(alpha_c is None) and not(alpha_d is None), not(dict_alpha_impaired is None)) #exclusive or
    net = Network(graph_G, M_ext, alpha_c=alpha_c, alpha_d=alpha_d, dict_alpha_impaired=dict_alpha_impaired, 
                  damping=damping, keep_history=keep_history, with_factors=with_factors)
    B_final = net.run_CI_approx_tanh(verbose=verbose, keep_history=keep_history)
    if keep_history:
        net.B_history = {key:np.array(val) for key,val in net.B_history.items()}
        return B_final, net.B_history
    else:
        return B_final, None #return B_final
    #or simply in all cases "return net"?
    
def run_BP_approx_tanh(graph_G, M_ext, damping=0, verbose=False, keep_history=False, with_factors=False):
#     alpha_c, alpha_d = 1, 1
#     net = Network(graph_G, M_ext, alpha_c=alpha_c, alpha_d=alpha_d,
    net = Network(graph_G, M_ext, alpha=1,
                  damping=damping, keep_history=keep_history, with_factors=with_factors)
    B_final = net.run_CI_approx_tanh(verbose=verbose, keep_history=keep_history)
    if keep_history:
        net.B_history = {key:np.array(val) for key,val in net.B_history.items()}
        return B_final, net.B_history
    else:
        return B_final, None #return B_final
    #or simply in all cases "return net"?
    
    
    
def sum_squared_updates(updates, begin=0):
    return {key: np.sum(val[begin:]**2) for key, val in updates.items()}

def sum_abs_updates(updates, begin=0):
    return {key: np.sum(np.abs(val[begin:])) for key, val in updates.items()}

def get_activations(B_history, method='square_updates_B', k=None):
    assert method in ['square_updates_B', 'leaky_belief']
    if method == 'square_updates_B':
#         updates = {key: B[1:] - B[:-1] for key, B in B_history.items()}
#         squared_updates = {key:val**2 for key,val in updates.items()}        
#         sum_squared_updates = sum_squared_updates(updates, begin=begin)
#         sum_squared_updates = np.array([sum_squared_updates[node] for node in G.nodes])
#         sum_abs_updates = sum_abs_updates(updates, begin=begin)
#         sum_abs_updates = np.array([sum_abs_updates[node] for node in G.nodes])
#         activations_history = {key: val**2 for key, val in updates.items()}
        activations_history = {key: (B[1:] - B[:-1])**2 for key, B in B_history.items()}
    elif method == 'leaky_belief':
        assert k != None
        if k == np.inf: #k=+inf means in fact that activity=|B| (instead of |dB/dt + k*B|) --> I use that to look at overconfidence
            activations_history = {key: np.abs(B)
                                   for key, B in B_history.items()
                                  }
        else: #default
            activations_history = {key: np.abs(B[1:] - B[:-1] + k * B[:-1])
                                   for key, B in B_history.items()
                                  }  #abs(updates + k*B)
    return activations_history


def get_total_activation(activations_history, begin=0):
    return {key: np.sum(val[begin:]) for key, val in activations_history.items()}


def detect_bistability_dict(B_history_CI, verbose=False):
    """
    Returns True is one of the nodes has some bistable behavior
    """
    dict_is_bistable = {key: detect_bistability(val) for key,val in B_history_CI.items()}
    list_bistable = list(dict_is_bistable.values())
    exists_bistable_node = list_bistable.count(True) >= 1
    if verbose and exists_bistable_node:
        for node,val in B_history_CI.items():
            if dict_is_bistable[node] == True:
                plt.plot(val)
                break
        plt.title('Example of node for which there is bistability')
        plt.show()
    return exists_bistable_node


def detect_bistability(B_history_CI):
    """
    B_history_CI is an array, not a dict: only one node
    TODO: possibly add a criterion: that the value of B at the end of each period is less than 1 in absolute value
    """
    hist, bin_edges = np.histogram(B_history_CI, bins=np.linspace(-6,6,40))
    # plt.plot(hist)
    # plt.show()
    # print(bin_edges)
    mean_bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2
#     plt.plot(mean_bin_edges, hist)
#     plt.show()
    d = dict(zip(mean_bin_edges, hist))
    # print(d)
    key_highest_val = list(d.keys())[np.argmax(list(d.values()))]
    # print(key_highest_val)
    if np.abs(key_highest_val) > 1:
#         print("bistability")
        return True
    else:
        return False

    
def detect_frustration_dict(B_history_CI, begin=0):
    """
    TODO: update this function (see low frustration detection in function load_simulated_data from analyze_effects_CI_vs_BP.py)
    """
    squared_updates_CI = {node:((val[1:] - val[:-1])**2)[begin:] for node,val in B_history_CI.items()}
    res = np.max(np.array(list(squared_updates_CI.values()))) > 3 #> 1
    return res


def test_convergence(arr):
    """
    Very simple convergence test: that the last update is smaller than 0.1
    """
    return np.abs(arr[-1] - arr[-2]) < 0.1


def test_convergence_dict(B_history_CI):
    for val in B_history_CI.values():
        if test_convergence(val) == False:
            return False
    return True