from compute_effects_CI_vs_BP import *
from graph_generator import generate_graph
from utils_stability_analysis import *
import numpy as np
from utils_CI_BP import *
# from utils_graph_rendering import *
from pprint import pprint
import networkx as nx
import bct
from utils_plot_dict import *

#I removed entirely variable graph and kept G (but I renamed G into graph_G for now, TODO = rename into graph when the code runs ok)

class Simulate:
    def __init__(self,
                 
                 type_graph=None, graph_G=None, which_graph_G=None,
                 method_weighting='normal_J', sigma_weighting=1, w_uniform=None,
                 remove_cerebellum_and_vermis=False, remove_ofc=False, 
                 binarize_realistic_connectome=False,
                 
                 type_M_ext=None, M_ext=None, n_periods=4, stimulated_nodes=None, n_stimulated_nodes=None, 
                 variance_Mext=None, T_period=500, mean_Mext=0,
                 
                 alpha_c=None, alpha_d=None, list_alphac_alphad=None, dict_alpha_impaired=None, #+ list_dict_alpha_impaired?
                 run_also_BP=True,
                 keep_history=True, save_last=False,
                 
                 begin=0,
                 look_for_frustration_before_simulation=False, 
                 check_frustration_and_bistability=True, predict_frustration_and_bistability=True,
                 print_advancement=False,
                 which_CI='CI'
                ):
        """
        run_also_BP: we have to test BP and CI on the same graphs. Indeed, some graphs for which CI is frustrated can have BP converge, and it would then bias the results to save the results of BP but not CI.
        By default, give type_graph and type_M_ext and alpha_c and alpha_d
        """
        super().__init__()
        # print("type_M_ext", type_M_ext)
#         print("type_graph", type_graph)
        
        #####################  Running some checks with the inputs  ########################
        #checking that exactly one among (alpha_c,alpha_d), list_alphac_alphad and dict_alpha_impaired is defined
        assert generalized_xor(
#             alpha is not None,
            (alpha_c is not None) and (alpha_d is not None),
            dict_alpha_impaired is not None,
            list_alphac_alphad is not None
        )
        #checking that if list_alphac_alphad is given, then run_also_BP=False (i.e. run_also_BP is not True) but (1,1) is included in list_alphac_alphad
        assert not((list_alphac_alphad is not None) and (run_also_BP==True))
#         print("list_alphac_alphad = {}".format(list_alphac_alphad))
        assert not((list_alphac_alphad is not None) and ((1,1) not in list_alphac_alphad)) #list_alphac_alphad should contain BP (alpha_c=alpha_d=1)
        #checking that if type_graph is not given, then graph_G is given and which_graph_G is not None
        assert not((type_graph is None) and (graph_G is None)) and not((type_graph is not None) and (graph_G is not None))
        #checking that if type_M_ext is not given, then M_ext is given
        assert not((type_M_ext is None) and (M_ext is None)) and not((type_M_ext is not None) and (M_ext is not None))
        assert not((graph_G is not None) and (type_graph is not None)) #assert not((which_graph_G is not None) and (type_graph is not None)) #i.e. if graph_G is given as input to Simulate, then type_graph is not used but instead directly inferred from graph_G
                
        assert which_CI in ['CI', 'CIbeliefs']
        dict_fun = {'CI': run_CI, 'CIbeliefs': run_CIbeliefs}
        fun_run_CI = dict_fun[which_CI]
        
        ############################  Define the graph  ####################################
        #1. Generate an unoriented graph (= create graph structure (without the weights))
        if graph_G is None: #no graph provided (default case) --> generate it
            assert which_graph_G is None
            graph_G = generate_graph(type_graph, 
                                     remove_cerebellum_and_vermis=remove_cerebellum_and_vermis,
                                     remove_ofc=remove_ofc,
                                     binarize_realistic_connectome=binarize_realistic_connectome
                                    ) #before: G, graph_array = generate_graph(...)
            #note that graph_G might have unconnected nodes.
#             #Ensure that the graph does not have unconnected nodes
#             while nx.is_connected(G.to_undirected())==False: #redo if connection missing
#                 graph_G = generate_graph(type_graph) #before: G, graph = generate_graph(type_graph)
#             Check G
#             if nx.is_connected(G.to_undirected()) == False:
#                 print("skip: nodes of G are not all connected")
#                 continue
#         else:
#             G = G_from_graph(graph).to_undirected() #goal = recover the edges (below we orient them randomly)
        else:
            assert which_graph_G in ['structure', 'weighted'] #is graph_G provided in input weighted (final graph_G) or not (structure of the graph = intermediate graph_G)?
    
        #2. Assign weights to the graph (= Set the strength of the connections + Orient the edges randomly)
        if which_graph_G != 'weighted':
            graph_G = weighting_graph(graph_G, method_weighting=method_weighting,
                                      sigma_weighting=sigma_weighting, w_uniform=w_uniform)
        if print_advancement:
            print("Finished generating and weighting the graph")
        ######################################################################################################

        ##################  Predicting frustration or bistability  #########################################
#         plot_eigenvalues_from_graph(graph_G, alpha=alpha_c, order='exact')
#         plot_eigenvalues_from_graph(graph_G, alpha=1, order='exact')
        if predict_frustration_and_bistability:
            # print("checking the stability of CI")
            if xor((alpha_c is not None) and (alpha_d is not None), dict_alpha_impaired is not None):
                alpha_dict = {'CI': {'alpha_c': alpha_c, 'alpha_d':alpha_d, 
                                     'dict_alpha_impaired': dict_alpha_impaired}
                             } #some of these are None
            else:
#                 print("Not implemented (predict_frustration_and_bistability=True and list_alphac_alphad) - see code")
#                 sys.exit()
                alpha_dict = {'CI ({}, {})'.format(alpha_c, alpha_d): {'alpha_c':alpha_c, 'alpha_d':alpha_d} 
                              for (alpha_c, alpha_d) in list_alphac_alphad}
            if run_also_BP: #also add something for BP
                alpha_dict['BP'] = {'alpha': 1}
            self.skipping = False
            for which, alpha_which in alpha_dict.items():
#                 print(alpha_which)
                S = get_stability_matrix(graph_G, **alpha_which)
                #print("got S (stability matrix)")
                eig_values = np.linalg.eig(S)[0]
                eig_values_real = eig_values.real
                #eig_values_imag = eig_values.imag
                # plot_eigenvalues(eig_values=eig_values)
                if np.max(np.abs(eig_values)) >= 1:
                    print("stability problem ({})".format(which))
                    # plot_eigenvalues(eig_values=eig_values)
#                     if not((np.min(eig_values_real) <= -1) or (np.max(eig_values_real) >= 1)):
#                         print("probable problem? - plotting the eigenvalues") #I actually think that it's not a problem
#                         print("Unusual case (the discrete system is unstable, but all the real parts of the eigenvalues are between -1 and 1)")
#                         plot_eigenvalues(eig_values=eig_values)
#                         sys.exit()
#                     if np.min(eig_values_real) <= -1:
#                         print("predicts frustration ({})".format(which))
#                     elif np.max(eig_values_real) >= 1:
#                         print("predicts bistability ({})".format(which))
                    self.skipping = True
                    break
#             if print_advancement:
#                 print("Finished computing the eigenvalues of the stability matrix")
            if self.skipping == True:
                print("Skipping (stability matrix predicted frustration or bistability)")
#                 sys.exit()
                return #instead of sys.exit()
        ######################################################################################################
        
        ##################  Generate external messages ###################################################
        if M_ext is None:
            M_ext = generate_M_ext(type_M_ext, graph_G, n_stimulated_nodes=n_stimulated_nodes, 
                                   stimulated_nodes=stimulated_nodes, variance_Mext=variance_Mext, mean_Mext=mean_Mext,
                                   n_periods=n_periods, type_graph=type_graph, T_period=T_period)
#             plt.figure(figsize=(14,6))
#             for node,M_ext_node in M_ext.items():
#                 plt.plot(M_ext_node)
#             plt.show()
#             T = len(M_ext[list(M_ext.keys())[0]])
#             print({key:np.mean(val) for key,val in M_ext.items()})
            if print_advancement:
                print("Finished generating the external messages")
        ######################################################################################################
        
        #################  Look for frustration by simulating on the 1st period only     #####################
        ################# Quick check to look for frustration in the first period for BP #####################
        if look_for_frustration_before_simulation and (len(graph_G) > 50) and check_frustration_and_bistability:
            B_BP, B_history_BP = run_BP(graph_G, 
                                        {node: val[:500] for node, val in M_ext.items()}, 
                                        keep_history=keep_history)
            # B_BP, B_history_BP = run_BP(graph_G, M_ext, keep_history=keep_history)
            B_history_BP = {key: np.array(val) for key,val in B_history_BP.items()}
            frustration = detect_frustration_dict(B_history_BP, begin=begin)
            if frustration:
                print("Skipping (Frustration detected on BP on a shorter simulation)")
                self.skipping = True
#                 sys.exit()
                return #instead of sys.exit()
            print("The initial test (BP on the 1st period) gave no frustration")
            print("Running the whole simulation...")
        ########################################################################################

        if ((alpha_c is not None) and (alpha_d is not None)) or (dict_alpha_impaired is not None):
        
            self.skipping = False
        
            #Run CI
            B_CI, B_history_CI = fun_run_CI(graph_G, M_ext, 
                                            alpha_c=alpha_c, alpha_d=alpha_d, 
                                            dict_alpha_impaired=dict_alpha_impaired,
                                            keep_history=keep_history)
            # updates_CI = {key:val[1:]-val[:-1] for key,val in B_history_CI.items()}
            # average_B_CI = {key: np.mean(np.abs(val)) for key,val in B_history_CI.items()}
            # activations_history_CI = get_activations(B_history_CI, method='leaky_belief', k=k)
            # total_activation_CI = compute_total_activation(activations_history_CI, begin=begin)
            # sum_squared_updates_CI = sum_squared_updates(updates_CI, begin=begin)
            # sum_abs_updates_CI = sum_abs_updates(updates_CI, begin=begin)
            # squared_updates_CI = {key:val**2 for key,val in updates_CI.items()}  
            if print_advancement:
                print("Finished running CI")

            if check_frustration_and_bistability:
                
                #Filtering files for which there is frustration
                frustration = detect_frustration_dict(B_history_CI, begin=begin)
                if frustration:
                    if dict_alpha_impaired is None:
                        print("Frustration detected (pb for alpha_c = {}, alpha_d = {}) - Skipping this graph"
                              .format(alpha_c, alpha_d))
                    else:
                        print("Frustration detected (pb for CI)")
                    self.skipping = True
    #                 break

                #Filtering files for which there is bistability
                bistability = detect_bistability_dict(B_history_CI)
                if bistability:
                    if dict_alpha_impaired is None:
                        print("Bistability detected (pb for alpha_c = {}, alpha_d = {}) - Skipping this graph"
                              .format(alpha_c, alpha_d))
                    else:
                        print("Bistability detected (pb for CI)")
                    self.skipping = True
    #                 break
    
                if print_advancement:
                    print("Finished looking for frustration/bistability for CI")
    
            #Run BP
            if run_also_BP:
                B_BP, B_history_BP = run_BP(graph_G, M_ext, keep_history=keep_history)
                # updates_BP = {key:val[1:]-val[:-1] for key,val in B_history_BP.items()}
                # average_B_BP = {key: np.mean(np.abs(val)) for key,val in B_history_BP.items()}
                # activations_history_BP = get_activations(B_history_BP, method='leaky_belief', k=k)
                # total_activation_BP = compute_total_activation(activations_history_BP, begin=begin)
                # sum_squared_updates_BP = sum_squared_updates(updates_BP, begin=begin)
                # sum_abs_updates_BP = sum_abs_updates(updates_BP, begin=begin)
                # squared_updates_BP = {key:val**2 for key,val in updates_BP.items()}
                if print_advancement:
                    print("Finished running BP")

                if check_frustration_and_bistability:
                
                    #Filtering files for which there is frustration
                    frustration = detect_frustration_dict(B_history_BP, begin=begin)
                    if frustration:
                        print("Frustration detected (pb for alpha_c = {}, alpha_d = {}) - Skipping this graph".format(1, 1))
                        self.skipping = True
    #                     break

                    #Filtering files for which there is bistability
                    bistability = detect_bistability_dict(B_history_BP)
                    if bistability:
                        print("Bistability detected (pb for alpha_c = {}, alpha_d = {}) - Skipping this graph".format(1, 1))
                        self.skipping = True
    #                     break

                    if print_advancement:
                        print("Finished looking for frustration/bistability for BP")

        elif list_alphac_alphad is not None:
            self.skipping = False
            B_history_CI_all = {}
            if save_last:
                B_CI_all = {}
            for (alpha_c, alpha_d) in list_alphac_alphad: #list_alphac_alphad should be sorted (alpha from small to 1)
#                 print("alpha_c = {}, alpha_d = {}".format(alpha_c, alpha_d))
                
                #CI
                B_CI, B_history_CI = fun_run_CI(graph_G, M_ext, alpha_c, alpha_d, keep_history=keep_history)
                
                if check_frustration_and_bistability:
#                     print("alpha_c = {}, alpha_d = {}".format(alpha_c, alpha_d))
                    
                    #Filtering files for which there is frustration
                    frustration = detect_frustration_dict(B_history_CI, begin=begin)
                    if frustration:
                        print("Frustration detected (pb for alpha_c = {}, alpha_d = {}) - Skipping this graph"
                              .format(alpha_c, alpha_d))
    #                     if (alpha_c, alpha_d) != list_alphac_alphad[0]:
    #                         print("problem in the code: there was no frustration for (alpha_c,alpha_d)=({},{}) but there is for (alpha_c,alpha_d)=({},{})".format(list_alphac_alphad[0][0], list_alphac_alphad[0][1], alpha_c, alpha_d))
    #                         print("---> TODO: change the code") #----> ok: the code is fine (just check that skipping=True before saving the otherwise potentially incomplete B_history_CI_all
    #                         sys.exit()
                        if predict_frustration_and_bistability:
                            print("This behaviour was not predicted by the stability matrix...")
                            S = get_stability_matrix(graph_G, alpha_c=alpha_c, alpha_d=alpha_d)
                            eig_values = np.linalg.eig(S)[0]
                            plot_eigenvalues(eig_values=eig_values)
                            plot_B_history_CI(B_history_CI)
                        self.skipping = True
                        break

                    #Filtering files for which there is bistability
                    bistability = detect_bistability_dict(B_history_CI)
                    if bistability:
                        print("Bistability detected (pb for alpha_c = {}, alpha_d = {}) - Skipping this graph"
                              .format(alpha_c, alpha_d))
                        self.skipping = True
                        if predict_frustration_and_bistability:
                            print("This behaviour was not predicted by the stability matrix...")
                            S = get_stability_matrix(graph_G, alpha_c=alpha_c, alpha_d=alpha_d)
                            eig_values = np.linalg.eig(S)[0]
                            plot_eigenvalues(eig_values=eig_values)
                            plot_B_history_CI(B_history_CI)
                        break
                
                B_history_CI_all[alpha_c, alpha_d] = B_history_CI
                if save_last:
                    B_CI_all[alpha_c, alpha_d] = B_CI
        
#         self.G = G
#         self.graph = graph
        self.graph_G = graph_G
        self.M_ext = M_ext
        
        if list_alphac_alphad is None:
            self.B_history_CI = B_history_CI
            if save_last:
                self.B_CI = B_CI
#             self.activations_history_CI = activations_history_CI
#             self.total_activation_CI = total_activation_CI
            if run_also_BP:
                self.B_history_BP = B_history_BP
                if save_last:
                    self.B_BP = B_BP
#                 self.activations_history_BP = activations_history_BP
#                 self.total_activation_BP = total_activation_BP
        else:
            self.B_history_CI_all = B_history_CI_all
            if save_last:
                self.B_CI_all = B_CI_all

    def compute_updates_CI(self, which='CI'):
        if which == 'CI':
            B_history_CI = self.B_history_CI
        elif which == 'BP':
            B_history_CI = self.B_history_BP
        updates_CI = {key: val[1:] - val[:-1] for key,val in B_history_CI.items()}
        return updates_CI

    def compute_updates_BP(self):
        return self.compute_updates_CI(which='BP')
    
    def compute_average_B_CI(self, which='CI'):
        if which == 'CI':
            B_history_CI = self.B_history_CI
        elif which == 'BP':
            B_history_CI = self.B_history_BP
        average_B_CI = {key: np.mean(np.abs(val)) for key,val in B_history_CI.items()}        
        return average_B_CI

    def compute_average_B_BP(self):
        return self.compute_average_B_CI(which='BP')
    
    def compute_activations_history_CI(self, method='leaky_belief', k=0.05, which='CI'):
        if which == 'CI':
            B_history_CI = self.B_history_CI
        elif which == 'BP':
            B_history_CI = self.B_history_BP
        activations_history_CI = get_activations(B_history_CI, method=method, k=k)
        if which == 'CI':
            self.activations_history_CI = activations_history_CI
        elif which == 'BP':
            self.activations_history_BP = activations_history_CI
        return activations_history_CI
    
    def compute_activations_history_BP(self, method='leaky_belief', k=0.05):
        return self.compute_activations_history_CI(method=method, k=k, which='BP')
    
    def compute_confidence_history_CI(self, method='leaky_belief', which='CI'):
        """
        Here "confidence" is not B, but refers to |B| (i.e. activation = |dB/dt + k*B| with k=inf)
        """
        if which == 'CI':
            B_history_CI = self.B_history_CI
        elif which == 'BP':
            B_history_CI = self.B_history_BP
        confidence_history_CI = get_activations(B_history_CI, method=method, k=np.inf)
        if which == 'CI':
            self.confidence_history_CI = confidence_history_CI
        elif which == 'BP':
            self.confidence_history_BP = confidence_history_CI
        return confidence_history_CI
    
    def compute_confidence_history_BP(self, method='leaky_belief'):
#         return self.compute_activations_history_CI(method=method, which='BP') #wrong!!! (k != np.inf; changes activations_history_BP...)
        return self.compute_confidence_history_CI(method=method, which='BP')
    
    def compute_total_activation_CI(self, method='leaky_belief', k=0.05, begin=0, which='CI'):
        if not(hasattr(self, 'activations_history_'+str(which))):
            self.compute_activations_history_CI(method=method, k=k, which=which)
            if not(hasattr(self, 'activations_history_'+str(which))):
                print("pb")
        if which == 'CI':
            activations_history_CI = self.activations_history_CI
        elif which == 'BP':
            activations_history_CI = self.activations_history_BP
        total_activation_CI = get_total_activation(activations_history_CI, begin=begin)     
        if which == 'CI':
            self.total_activation_CI = total_activation_CI
        elif which == 'BP':
            self.total_activation_BP = total_activation_CI
        return total_activation_CI
        
    def compute_total_activation_BP(self, method='leaky_belief', k=0.05, begin=0):
        return self.compute_total_activation_CI(method=method, k=k, begin=begin, which='BP')
    
    def compute_total_confidence_CI(self, method='leaky_belief', begin=0, which='CI'):
        if not(hasattr(self, 'confidence_history_'+str(which))):
            self.compute_confidence_history_CI(method=method, which=which)
            if not(hasattr(self, 'confidence_history_'+str(which))):
                print("pb")
        if which == 'CI':
            confidence_history_CI = self.confidence_history_CI
        elif which == 'BP':
            confidence_history_CI = self.confidence_history_BP
        total_confidence_CI = get_total_activation(confidence_history_CI, begin=begin)     
        if which == 'CI':
            self.total_confidence_CI = total_confidence_CI
        elif which == 'BP':
            self.total_confidence_BP = total_confidence_CI
        return total_confidence_CI
        
    def compute_total_confidence_BP(self, method='leaky_belief', begin=0):
        return self.compute_total_confidence_CI(method=method, begin=begin, which='BP')
    
    def compute_overactivation(self, method='leaky_belief', k=0.05, begin=0, which='percent'):
        assert which in ['percent', 'diff']
        if not(hasattr(self, 'total_activation_BP')):
            self.compute_total_activation_BP(method=method, k=k, begin=begin)
        if not(hasattr(self, 'total_activation_CI')):
            self.compute_total_activation_CI(method=method, k=k, begin=begin)
        y_CI = self.total_activation_CI
        y_BP = self.total_activation_BP
        if which == 'diff': #overactivation
            d_overactivation = {node: y_CI[node] - y_BP[node] for node in y_BP.keys()}
        if which == 'percent': #overactivation in %
            d_overactivation = {node: 100 * (y_CI[node] - y_BP[node]) / y_BP[node] for node in y_BP.keys()}
        self.d_overactivation = d_overactivation
        return d_overactivation
    
    def compute_overconfidence(self, method='leaky_belief', begin=0, which='percent'):
        assert which in ['percent', 'diff']
        if not(hasattr(self, 'total_confidence_BP')):
            self.compute_total_confidence_BP(method=method, begin=begin)
        if not(hasattr(self, 'total_confidence_CI')):
            self.compute_total_confidence_CI(method=method, begin=begin)
        y_CI = self.total_confidence_CI
        y_BP = self.total_confidence_BP
        if which == 'diff': #overconfidence
            d_overconfidence = {node: y_CI[node] - y_BP[node] for node in y_BP.keys()}
        if which == 'percent': #overconfidence in %
            d_overconfidence = {node: 100 * (y_CI[node] - y_BP[node]) / y_BP[node] for node in y_BP.keys()}
        self.d_overconfidence = d_overconfidence
        return d_overconfidence
    
    def check_frustration_CI(self, begin=0, plot=False):
        """
        To filter files for which there is frustration, use this function
        """
        if hasattr(self, 'B_history_CI_all'): #the simulation was run for list_alphac_alphad
            res = False
            for alpha_CI, B_history_CI in self.B_history_CI_all.items():
                alpha_c, alpha_d = alpha_CI
                if detect_frustration_dict(B_history_CI, begin=begin):
                    print("Frustration detected (alpha_c = {}, alpha_d = {})".format(alpha_c, alpha_d))
                    res = True
                    break
            return res
        B_history_CI = self.B_history_CI
        res = detect_frustration_dict(B_history_CI, begin=begin)
        if res:
            print("Frustration detected (CI)")
            if plot:
                plot_B_history_CI(B_history_CI)
        return res
    
    def check_frustration_BP(self, begin=0, plot=False):
        B_history_BP = self.B_history_BP
        res = detect_frustration_dict(B_history_BP, begin=begin)
        if res:
            print("Frustration detected (BP)")
            if plot:
                plot_B_history_CI(B_history_BP)
        return res

    def check_bistability_CI(self, verbose=False, plot=False):
        """
        To filter files for which there is bistability of the belief, use this function
        """
        if hasattr(self, 'B_history_CI_all'): #the simulation was run for list_alphac_alphad
            res = False
            for alpha_CI, B_history_CI in self.B_history_CI_all.items():
                alpha_c, alpha_d = alpha_CI
                if detect_bistability_dict(B_history_CI, verbose=verbose):
                    print("Bistability detected (alpha_c = {}, alpha_d = {})".format(alpha_c, alpha_d))
                    res = True
                    break
            return res
        B_history_CI = self.B_history_CI
        res = detect_bistability_dict(B_history_CI)
        if res:
            print("Bistability detected (CI)")
            if plot:
                plot_B_history_CI(B_history_CI)
#             print("Bistability detected (alpha_c = {}, alpha_d = {})".format(alpha_c, alpha_d))
        return res

    def check_bistability_BP(self, verbose=False, plot=False):
        B_history_BP = self.B_history_BP
        res = detect_bistability_dict(B_history_BP)
        if res:
            print("Bistability detected (BP)")
            if plot:
                plot_B_history_CI(B_history_BP)
        return res