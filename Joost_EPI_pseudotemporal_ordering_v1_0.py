################################################################################
################## STRT-EPIDERMIS -- PSEUDOTEMPORAL ORDERING ###################
################################################################################

"""
Scripts for the pseudotemporal ordering of cells based on the PQ-Tree approach
introduced by Magwene et al. and Trapnell et al.
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import itertools

from EPI_misc_scripts_v1_1 import *
from EPI_affinity_propagation_v1_0 import *
from EPI_neg_binom_regression_v1_1 import *

from rpy2 import robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

rbase = importr('base')
rvgam = importr('VGAM', robject_translations = {"nvar.vlm": "nvar__vlm"})

################################################################################
################################ MAIN FUNCTIONS ################################
################################################################################

def PTO_create_MST(dataset, prog = 'neato'):
    
    """
    Creates a MST (Prim's algorithm) based on the pearson correlation of a sample x gene matrix.
    -----
    dataset: pd.DataFrame of m samples x n genes
    prog: graphviz prog; default = 'neato'
    -----
    returns MST as networkx Graph and spatial positions from graphviz layout as dictionairy
    """
    
    adj_mat = 1 - dataset.corr()
    
    G = nx.Graph()
    
    G.add_nodes_from([node for node in adj_mat.index])
    
    for node in G.nodes():
    
        tmp_edges = [(node, index, {'weight': adj_mat[node].ix[index]}) for index in adj_mat[node].index]
        G.add_edges_from(tmp_edges)
    
    MST = nx.minimum_spanning_tree(G)
    
    MST_pos = nx.graphviz_layout(MST, prog=prog)
    
    return MST, MST_pos
    
################################################################################

def PTO_create_MST_2d(dist_mat, prog = 'neato'):
    
    """
    Creates a MST (Prim's algorithm) based on the euclidean distance of a sample x sample distance matrix.
    -----
    dataset: distance matrix as pd.DataFrame of m samples x m samples
    prog: graphviz prog; default = 'neato'
    -----
    returns MST as networkx Graph and spatial positions from graphviz layout as dictionairy
    """
    
    G = nx.Graph()
    
    G.add_nodes_from([node for node in dist_mat.index])

    for node in G.nodes():
        
        tmp_edges = [(node, index, {'weight': dist_mat[node].ix[index]}) for index in dist_mat[node].index]
        G.add_edges_from(tmp_edges)
    
    MST = nx.minimum_spanning_tree(G)
    
    MST_pos = nx.graphviz_layout(MST, prog=prog)
    
    return MST, MST_pos
    
################################################################################   
    
def PTO_diameter_path(MST, return_edges = False):
    
    """
    Finds the diameter path of an MST using Dijkstras algorithm.
    ---
    MST: minimum spanning tree as networkx Graph
    ---
    returns diameter path as list and diameter edges as networkx Graph
    """
    
    if len(MST.nodes()) == 1:
        
        return [node for node in MST.nodes()]
    
    #1. Calculate node degree and find terminal nodes (degree == 1):
    
    node_degr = MST.degree()
    node_term = [node for node in node_degr if node_degr[node] == 1]
    
    #2. Find shortest paths between all nodes (Dijkstra):
    
    dijkstra_all = nx.all_pairs_dijkstra_path_length(MST, weight = 1)
    
    #3. Find longest path between all combinations of terminal nodes
    
    path_max = ''
    len_path_max = 0

    for node_pair in itertools.combinations(node_term, 2): 
        len_path_tmp = dijkstra_all[node_pair[0]][node_pair[1]]
    
        if len_path_tmp > len_path_max: 
            len_path_max = len_path_tmp
            path_max = node_pair
    
    #4. Return path
    
    diam_path = nx.dijkstra_path(MST, path_max[0], path_max[1])
    
    diam_edges = nx.Graph()
    diam_edges.add_nodes_from(diam_path)
    diam_edges.add_edges_from([(diam_path[pos],diam_path[pos+1]) for pos in range(len(diam_path) - 1)])
    
    if return_edges == True:
        
        print 'Diameter path between %s and %s with lenght %s' % (path_max[0], path_max[1], len_path_max)
        return diam_edges
    
    if return_edges == False:
        
        return diam_path
        
################################################################################
    
    
################################################################################

def PTO_create_pseudotemporal_ordering(dataset, sample_groups, return_min = 50, return_path = False):
    
    """
    Creates pseudotemporal ordering of cells by drawing MST and finding shorted permutation in subsequent PQ tree (see Magwene
    et al. 2003)
    -----
    dataset: pd.DataFrame for m samples x n genes.
    sample_groups: pd.Series of sample groups from AP clustering.
    return_min: [int] - Number of best node permutations evaluated for every node; default = 50.
    return_path: if True, returns networkx Graph of PTO path; default = False
    -----
    returns PTO_coords: pd.Series with pseudotemporal coordinates [float] for every cell
    (returns PTO_path: networkx Graph of PTO path for drawing)
    """
    
    print '\nCreating MST\n'
    
    MST, MST_pos = PTO_create_MST(dataset)
    
    diam_edges = PTO_diameter_path(MST, return_edges = True)
    
    PTO_draw_MST_groups(MST, MST_pos, diam_edges, sample_groups)
    
    MST_ = MST.copy() #Full cell labels
    
    MST, node_dict = PTO_relabel_nodes(MST)
    
    dist_int = PTO_distance_matrix(dataset, node_dict)
    
    print '\nCreating PQ-Tree\n'
    
    PQ = PTO_PQ_tree(MST)
    
    print 'Finding permutations\n'
    
    PTO = PTO_PQ_simple_permutations(PQ, dist_int, return_min = return_min)
        
    PTO_coords = PTO_ordered_coordinates(PTO, node_dict, dist_int)
    
    PTO_path = PTO_ordered_path(PTO, node_dict)
    
    PTO_draw_MST_groups(MST_, MST_pos, PTO_path, sample_groups)
    
    print '\nReturning coordinates\n'

    if return_path == True:
        
        return PTO_coords, PTO_path
    
    else:
        
        return PTO_coords
        
################################################################################

def PTO_create_pseudotemporal_ordering_2d(dist_mat, sample_groups, return_min = 50, return_path = False):
    
    """
    Creates pseudotemporal ordering of cells by drawing MST and finding shorted permutation in subsequent PQ tree (see Magwene
    et al. 2003). Uses distance matrix of euclidean distance in 2d space as basis.
    -----
    dist_mat: distance matrix of pd.DataFrame for m samples x m samples.
    sample_groups: pd.Series of sample groups from AP clustering.
    return_min: [int] - Number of best node permutations evaluated for every node; default = 50.
    return_path: if True, returns networkx Graph of PTO path; default = False
    -----
    returns PTO_coords: pd.Series with pseudotemporal coordinates [float] for every cell
    (returns PTO_path: networkx Graph of PTO path for drawing)
    """
    
    print '\nCreating MST\n'
    
    MST, MST_pos = PTO_create_MST_2d(dist_mat)
    
    diam_edges = PTO_diameter_path(MST, return_edges = True)
    
    PTO_draw_MST_groups(MST, MST_pos, diam_edges, sample_groups)
    
    MST_ = MST.copy() #Full cell labels
    
    MST, node_dict = PTO_relabel_nodes(MST)
    
    dist_int = PTO_distance_matrix_2d(dist_mat, node_dict)
    
    print '\nCreating PQ-Tree\n'
    
    PQ = PTO_PQ_tree(MST)
    
    print 'Finding permutations\n'
    
    PTO = PTO_PQ_simple_permutations(PQ, dist_int, return_min = return_min)
        
    PTO_coords = PTO_ordered_coordinates(PTO, node_dict, dist_int)
    
    PTO_path = PTO_ordered_path(PTO, node_dict)
    
    PTO_draw_MST_groups(MST_, MST_pos, PTO_path, sample_groups)
    
    print '\nReturning coordinates\n'

    if return_path == True:
        
        return PTO_coords, PTO_path
    
    else:
        
        return PTO_coords
            
################################################################################

def fit_vgam(dataset, PTO_coords, genes, df):
    
    """
    Fits a VGAM cubic spline model with formula "Y ~ s(X)" where X is pseudotime to all selected genes and returns the fitted
    values as well as the chi2 test statistic as well as the p-value of the fit compared to the restricted model (no pseudo-
    time dependency)
    -----
    dataset: [pd.DataFrame] containing m cells x n genes. Preferentially log2 transformed.
    PTO_coords: [pd.Series] containing cell IDs and pseudotime positions.
    genes: [list] of genes for which the model should be fitted.
    df: effective degrees of freedom [int] used for cubic spline fit.
    -----
    Returns:
    [pd.DataFrame] containing the fitted values for every selected gene.
    [pd.DataFrame] containing the test statstic and p-values for every selected gene.
    """
    
    #initialize output formats
    
    predict_x = np.arange(0, np.max(PTO_coords), 1)
    
    fitted = pd.DataFrame(index = genes, columns = predict_x)
    
    stats = pd.DataFrame(index = genes, columns = ['Chisq', 'Pr(>Chisq)'])
    
    #iterate through genes
    
    for g in genes:
        
        #define values

        X = robjects.IntVector(list(PTO_coords.values))
        Y = robjects.IntVector(list(dataset.ix[g, PTO_coords.index]))
        DF = robjects.DataFrame({'X':X, 'Y':Y})

        #fit full model

        fmla_full = robjects.Formula("Y ~ s(X, df = %s)" % df)
        fit_full = rvgam.vgam(fmla_full, rvgam.negbinomial, data = DF)

        #predict values
        
        X_pred = robjects.IntVector(predict_x)
        DF_pred = robjects.DataFrame({'X':X_pred})
        predicted = rvgam.predict(fit_full, newdata = DF_pred, type = 'response')
        
        fitted.ix[g] = list(predicted)
        
        #perform LR test
        
        lr_stats = rvgam.lrtest(fit_full, "s(X, df = %s)" % df)
        
        stats.ix[g, 'Chisq'] = lr_stats.do_slot('Body')[3][1]
        stats.ix[g, 'Pr(>Chisq)'] = lr_stats.do_slot('Body')[4][1]
        
    return fitted, stats
    
################################################################################
################################### DRAWING ####################################
################################################################################
    
def PTO_draw_MST_groups(MST, MST_pos, diam_edges, sample_groups, cmap = plt.cm.jet, node_size = 100, 
                        linewidths = 1.0, width = 0.5, width_diam = 5.0, edge_color = 'grey', edges_off = False):
    
    """
    Draws MST with group specific colormap and highlighted diameter path.
    -----
    MST: MST networkx Graph
    MST_pos: networkx position dict for MST
    diamedges: networkx Graph containing nodes and edges of MST diameter path
    sample_groups: pd.Series containing sample groups from AP clustering
    """
    
    gr_max = float(np.max(sample_groups.values))
    clist = [cmap(sample_groups[node] / gr_max) for node in MST.nodes()]
    
    plt.figure(facecolor = 'w', figsize = (20,20))
    ax = plt.axes()
    
    x_pos = [pos[0] for pos in MST_pos.values()]
    y_pos = [pos[1] for pos in MST_pos.values()]
    
    ax.set_xlim(min(x_pos) * 1.1, max(x_pos) * 1.1)
    ax.set_ylim(min(y_pos) * 1.1, max(y_pos) * 1.1)
    
    nx.draw_networkx(MST, pos = MST_pos, ax = ax, with_labels = False, node_size = node_size, linewidths = linewidths, 
                     width = width, edge_color = edge_color, node_color = clist, vmin = 0, vmax = 1)
    
    nx.draw_networkx_edges(MST, pos = MST_pos, ax = ax, edgelist = diam_edges.edges(), width=width_diam)
    
################################################################################
              
def PTO_plot_gene(dataset, PTO_coords, sample_groups, gene, fit = None, cmap = plt.cm.jet):
    
    """
    Plots gene expression in all cells according to pseudotemporal order of cells
    -----
    dataset: pd.DataFrame for m samples x n genes.
    PTO_coords: pd.Series with pseudotemporal coordinates [float] for every cell.
    sample_groups: pd.Series of sample groups from AP clustering.
    gene: Selected gene [str]
    fit: pd.DataFrame for m samples x n genes with PTO fitted data (e.g. Lowess smoothing)
    """
    
    gr_max = float(np.max(sample_groups.values))
    clist = [cmap(gr / gr_max) for gr in sample_groups[PTO_coords.index].values]

    x_pos = PTO_coords.values
    y_pos = dataset[PTO_coords.index].ix[gene]
    
    plt.figure(facecolor = 'w', figsize = (15,10))
    ax = plt.axes()
    
    ax.set_xlim(0, np.max(x_pos))
    ax.set_ylim(0, np.max(y_pos) * 1.1)

    ax.scatter(x_pos, y_pos, c = clist, cmap = cmap, s = 50, linewidths=0.75)
    
    if fit is not None:
        
        ax.plot(fit.index, fit[gene], color = 'black', linewidth = 3)
        
################################################################################
        
def PTO_subtract(dataset, PTO_fitted, corr_max, set_null=True):
    
    """
    For each cell, uses the pseudotime correlation of said cell, to subtract the fitted pseudotime signature.
    ----------
    dataset: pd.DataFrame m cells x n genes. Must be of the same data format (e.g. log2) as PTO_fitted.
    PTO_fitted: pd.DataFrame containing the spline-fitted data along p pseudotime points for n genes.
    corr_max: pd.Series containing the maximal pseudotime correlation for m cells.
    set_null: set negative values to zero. Default: True.
    ----------
    returns pd.DataFrame of m cells x n genes where the pseudotime signal has been subtracted.
    """
    
    #generate congruent dataframes for substraction
    
    dataset = dataset[corr_max.index]
    dataset_new = dataset.copy()
    
    PTO_fitted.columns = PTO_fitted.columns.astype(float)
    PTO_fitted = PTO_fitted[corr_max.values]
    PTO_fitted.columns = corr_max.index
    
    #substract
            
    dataset_new.ix[PTO_fitted.index] = dataset.ix[PTO_fitted.index] - PTO_fitted
    
    if set_null==True:
        
        dataset_new[dataset_new < 0] = 0
    
    return dataset_new

################################################################################
################### SUBFUNCTIONS FOR PQTREE AND PERMUTATIONS ###################
################################################################################

def PTO_indecisive_backbone(MST, diam_path):
    
    """
    Finds indecisive backbone (between first and last node with degree >= 3 in diameter path)
    -----
    MST: minimum spanning tree as networkx Graph
    diam_path: diameter path as list of nodes
    -----
    returns indecisive backbone as list of nodes
    """
    
    #1. Calculate node degrees in MST
    
    MST_degr = MST.degree()
    
    #2. Find index of first indecisive node
    
    ix1 = None
    for ix in range(len(diam_path)): 
        if MST_degr[diam_path[ix]] >= 3:
            ix1 = ix
            break
            
    if ix1 is None:
        return None
            
    #3. Find index of last indecisive node
    
    for ix in range(len(diam_path))[::-1]:
        if MST_degr[diam_path[ix]] >= 3:
            ix2 = ix
            break
    
    #4. Return indecisive backbone
    
    return diam_path[ix1 : ix2+1]
    
################################################################################
    
def PTO_diam_path_branches(MST, diam_path):
    
    #1. Find branches
    
    MST_copy = MST.copy()
    branches = {}
    for node in diam_path:
        branches[node] = []
        for subnode in MST.neighbors(node):
            if subnode not in diam_path:
                MST_copy.remove_edge(node, subnode)
                branches[node].append(nx.single_source_dijkstra_path(MST_copy, subnode).keys())
                
    #2. Find subgraphs
                
    subgraphs = {}
    for root in branches.keys():
        subgraphs[root] = []
        for branch in branches[root]:
            subg_tmp = MST.subgraph(branch)
            subgraphs[root].append(subg_tmp)
            
    #3. Return results
    
    class Results:
        def __init__(self, branches, subgraphs):
            self.branches, self.subgraphs = branches, subgraphs
            
    return Results(branches, subgraphs)
    
################################################################################ 
    
class Qnode(list):
    def __repr__(self):
        return str(tuple(self))

class Pnode(list):
    pass
    
################################################################################

def PTO_relabel_nodes(MST):
        
    node_dict = {}
    ix = 0

    for node in MST.nodes():
        node_dict[node] = ix
        ix += 1
    
    MST = nx.relabel_nodes(MST, node_dict)
    
    return MST, node_dict
    
################################################################################

def PTO_PQ_tree(MST):
    
    diam_path = PTO_diameter_path(MST)
    
    backbone = PTO_indecisive_backbone(MST, diam_path)
    
    if backbone is None:
        return Qnode(diam_path)
    
    branches = PTO_diam_path_branches(MST, backbone)

    pqtree = []
    
    for root in backbone:
        pqtree.append(Pnode([Qnode((root,))] + \
                 Pnode([PTO_PQ_tree(s) for s in branches.subgraphs[root]]) ))

    return Qnode(pqtree)

################################################################################

def PTO_distance_matrix(dataset, node_dict):
    
    dist_str = 1 - dataset.corr()
    
    dist_int = pd.DataFrame(columns = node_dict.values(), index = node_dict.values())
    
    for node1 in node_dict:
        for node2 in node_dict:
            
            dist_int.ix[node_dict[node1], node_dict[node2]] = dist_str.ix[node1, node2]
            
    return dist_int
    
################################################################################

def PTO_distance_matrix_2d(dist_mat, node_dict):
    
    dist_int = dist_mat
    
    dist_int.index = [node_dict[ix] for ix in dist_mat.index]
    dist_int.columns = [node_dict[ix] for ix in dist_mat.columns]
    
    return dist_int

################################################################################  
    
def PTO_calculate_distance(permutation, dist_int):
    
    dist = 0
    
    for pos in range(len(permutation) - 1):
        dist += dist_int.ix[permutation[pos], permutation[pos + 1]]
        
    return dist  
    
################################################################################

def PTO_flatten_list(l):
    
    for i in l:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in PTO_flatten_list(i):
                yield j
        else:
            yield i
   
################################################################################   
    
def PTO_PQ_simple_permutations(pq, dist_int, return_min = 100):
    
    perms = PTO_PQ_simple_permutation_node(pq[0], dist_int, return_min = 100)
    
    for node in range(1, len(pq)):
        
        print node
                
        perms_growing = pd.Series()
        
        if len(pq[node]) > 7:
            
            perms_new = PTO_PQ_simple_permutation_node(pq[node], dist_int, return_min = return_min, permutation = False)
        
        else:
            
            perms_new = PTO_PQ_simple_permutation_node(pq[node], dist_int, return_min = return_min, permutation = True)
        
        for i in itertools.product([eval(ix) for ix in perms.index], [eval(ix) for ix in perms_new.index]):
    
            perms_growing[str(i[0] + i[1])] = perms[str(i[0])] + perms_new[str(i[1])] + dist_int.ix[i[0][-1], i[1][0]]
            
        perms = perms_growing.order()[:return_min]

    return eval(perms.index[0])

################################################################################

def PTO_PQ_simple_permutation_node(pq_sub, dist_int, return_min = 6, permutation = True):
    
    products_inp = 'itertools.product('

    for n in pq_sub:
        if type(n) == Qnode and len(n) > 1:
            inp_tmp = [list(PTO_flatten_list(n)), list(PTO_flatten_list(n[::-1]))] 
        else:
            inp_tmp = [list(PTO_flatten_list(n))]
        products_inp += '%s, ' % (inp_tmp)
    
    products_inp = products_inp[:-2] + ')'
    
    perm = pd.Series()
    
    for product in eval(products_inp):
        
        if permutation == True:
            
            for i in itertools.permutations(product):
                i = list(PTO_flatten_list(i))
                perm[str(i)] = PTO_calculate_distance(i, dist_int)
                
        elif permutation == False:
            
            i = list(PTO_flatten_list(product))
            perm[str(i)] = PTO_calculate_distance(i, dist_int)
            i = list(PTO_flatten_list(product))[::-1]
            perm[str(i)] = PTO_calculate_distance(i, dist_int)

    return perm.order()[:return_min]
    
################################################################################
    
def PTO_ordered_path(PTO, node_dict):
    
    node_dict_inv = {v:k for k, v in node_dict.items()}
    
    G = nx.Graph()
    
    G.add_nodes_from([node_dict_inv[node] for node in PTO])
    
    G.add_edges_from([(node_dict_inv[PTO[pos]], node_dict_inv[PTO[pos+1]]) for pos in range(len(PTO) - 1)])
    
    return G
    
################################################################################
    
def PTO_ordered_coordinates(PTO, node_dict, dist_int, reverse = False):
    
    if reverse == True:
        
        PTO = PTO[::-1]
        
    node_dict_inv = {v:k for k, v in node_dict.items()}
        
    coord = pd.Series()
        
    for pos in range(len(PTO)):
        
        if pos == 0:
            
            dist = 0
            
        else:
            
            dist += dist_int.ix[PTO[pos -1], PTO[pos]]
            
        coord[node_dict_inv[PTO[pos]]] = dist
        
    return coord
    
################################################################################

def pairwise_distance_2d(dataset):
    
    dist_mat = pd.DataFrame(columns = dataset.index, index = dataset.index)
    
    for col, ix in itertools.combinations_with_replacement(dataset.index, r = 2):
        
        dist_tmp = np.sqrt(np.abs(dataset.ix[col, 0] - dataset.ix[ix, 0])**2 + np.abs(dataset.ix[col, 1] - dataset.ix[ix, 1])**2)
        
        dist_mat.ix[col, ix], dist_mat.ix[ix, col] = dist_tmp, dist_tmp
        
    return dist_mat
    
################################################################################
################################## CORRELATION #################################
################################################################################

def PTO_correlate(dataset, PTO_fitted, cells, genes, return_p=False):
    
    """
    Correlates every cells to every pseudotime point based on cubic spline fitted expression data.
    ----------
    dataset: DataFrame of m cells x n genes. Must be of the same data format (e.g. log2 transformed as PTO_fitted)
    PTO_fitted: DataFrame of m points in pseudotime x n genes fitted with cubic splines.
    cells: list of cells to be correlated.
    genes: list of genes to be used in the correlation analysis.
    ----------
    returns DataFrame containing the Pearson correlation (and p-value) of m cells x p pseudotime points.
    
    """
    
    #define dataset
    
    dataset = dataset.ix[genes, cells]
    PTO_fitted = PTO_fitted.ix[genes]
    
    #define output

    corr_r = pd.DataFrame(index = cells, columns = PTO_fitted.columns)
    corr_p = pd.DataFrame(index = cells, columns = PTO_fitted.columns)

    for m in cells:

        x_tmp = dataset[m]

        for p in PTO_fitted.columns:

            y_tmp = PTO_fitted[p]

            corr_tmp = scipy.stats.pearsonr(x_tmp, y_tmp)

            corr_r.ix[m, p] = corr_tmp[0]
            corr_p.ix[m, p] = corr_tmp[1]
            
    if return_p == True:
        
        return corr_r, corr_p
    
    else:
        
        return corr_r
        
################################################################################

def PTO_correlate_find_max(corr_r, return_r = False):
    
    """
    For each cell in corr_r, returns the pseudotime with maximal correlation.
    """

    corr_max = pd.Series(index = corr_r.index)
    corr_max_r = pd.Series(index = corr_r.index)
    
    for ix in corr_r.index:
    
        corr_max[ix] = float(corr_r.ix[ix].order().index[-1])
        corr_max_r[ix] = corr_r.ix[ix].order()[-1]
    
    if return_r == True:
        return corr_max, corr_max_r
    else:
        return corr_max

################################################################################
    
################################################################################
################################## VALIDATION ##################################
################################################################################

def PTO_robustness_tsne(dataset, iterations, dview, modus, resampling=0.2, return_min = 50,
                        perplexity=30.0, early_exaggeration=2.0, learning_rate=1000.0, n_iter=1000):
    
    """
    Performs pseudotemporal ordering on a specified number of t-SNE plots to evaluate
    whether it is robust independent of random t-SNE plot seed. Offers the following
    modi:    (1) Iterations: all cells are considered.
             (2) Resampling: a fraction of cells is randomly removed.
             (3) Shuffling: cell IDs are shuffled.
    ----------
    dataset: [pd.DataFrame] of m cells x n genes.
    iterations: number [int] of iterations.
    dview: Ipython DirectView Instance for parallel computing.
    modus: 'tSNE_iterations', 'tSNE_resampling' or 'tSNE_shuffling'.
    resampling: fraction of cells removed from the dataset. Default = 0.2.
    ----------
    return [pd.DataFrame] containing pseudotemporal coordinates over the number
    of iterations
    """
    
    ####################
    
    def PTO_robustness_helper(dataset, tsne, modus, iteration, resampling=0.2, return_min = 50):
        """
        Helper function for PTO_robustness
        """

        #modify dataset

        if modus == 'tSNE_iterations':

            pass

        elif modus == 'tSNE_resampling':

            size = int(len(dataset.columns) * (1 - resampling))

            dataset = dataset[np.random.choice(dataset.columns, size=size, replace=False)]

        elif modus == 'tSNE_shuffling':

            dataset.columns = np.random.choice(dataset.columns, size=len(dataset.columns), replace=False)

        #create distance matrix

        dist_mat_input = 1 - dataset.corr()

        tsne_tmp = pd.DataFrame(tsne.fit_transform(dist_mat_input), index = dist_mat_input.index, columns = ['x', 'y'])

        dist_mat = pairwise_distance_2d(tsne_tmp)

        #create PTO

        MST, MST_pos = PTO_create_MST_2d(dist_mat)

        diam_edges = PTO_diameter_path(MST, return_edges = True)

        MST_ = MST.copy() #Full cell labels

        MST, node_dict = PTO_relabel_nodes(MST)

        dist_int = PTO_distance_matrix_2d(dist_mat, node_dict)

        PQ = PTO_PQ_tree(MST)

        PTO = PTO_PQ_simple_permutations(PQ, dist_int, return_min = return_min)

        PTO_coords = PTO_ordered_coordinates(PTO, node_dict, dist_int)

        return PTO_coords
    
    ####################
    
    #define tSNE parameters
    
    tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, 
                learning_rate=learning_rate, n_iter=n_iter, metric='precomputed', init='random', 
                verbose=0, random_state=None)
    
    #define output dataframe
    
    output = pd.DataFrame(index = range(iterations), columns = dataset.columns)
    
    #parallel process iterations
    
    tmp = dview.map_sync(PTO_robustness_helper,
                         [dataset] * iterations,
                         [tsne] * iterations,
                         [modus] * iterations,
                         range(iterations),
                         [resampling] * iterations, 
                         [return_min] * iterations)
    
    #update output DataFrame
    
    for ix, data in enumerate(tmp):
        
        output.ix[ix, data.index] = data
        
    return output
    
################################################################################

def PTO_correlation_robustness_v1P(dataset, PTO_fitted, cells, genes, dview, iterations, resampling=0.5):
    
    """
    Calculates the robustness of the correlation of a cell to a spatial or temporal signature by
    randomly removing part of the signature.
    ----------
    dataset: [pd.DataFrame] of m cells x n genes.
    PTO_fitted: [pd.DataFrame] containing the cubic splines over the range of the fitted axis.
    cells: [list] of cell ID to be considered.
    genes: [list] of genes to be considered.
    iterations: number [int] of interations.
    resampling: fraction of genes used. Default = 0.5.
    ----------
    returns [pd.DataFrame] containing positions for all specified cells over all iterations.
    """
    
    ##########################################################
    
    def PTO_correlation_robustness_helper(dataset, PTO_fitted, cells, genes):
        
        #define dataset
    
        dataset = dataset.ix[genes, cells]
        PTO_fitted = PTO_fitted.ix[genes]
    
        #define output

        corr_r = pd.DataFrame(index = cells, columns = PTO_fitted.columns)

        for m in cells:

            x_tmp = dataset[m]

            for p in PTO_fitted.columns:

                y_tmp = PTO_fitted[p]

                corr_tmp = scipy.stats.pearsonr(x_tmp, y_tmp)

                corr_r.ix[m, p] = corr_tmp[0]
                
        #find position with maximum correlation
                
        corr_max = pd.Series(index = corr_r.index)
    
        for ix in corr_r.index:
    
            corr_max[ix] = float(corr_r.ix[ix].order().index[-1])
    
        return corr_max
        
    ##########################################################
    
    it = iterations
    n_genes = int(len(genes) * resampling)
    
    #define output
    
    output = pd.DataFrame(index = range(iterations), columns = cells)
    
    #perform test against baseline in parallel
    
    output_tmp = dview.map_sync(PTO_correlation_robustness_helper,
                                [dataset] * it,
                                [PTO_fitted] * it,
                                [cells] * it,
                                [np.random.choice(genes, n_genes, replace = False) for i in range(it)])
    
    #fuse data and return
    
    for pos, tmp in enumerate(output_tmp):
        
        output.ix[pos] = tmp
    
    return output 
    
################################################################################

def PTO_correlation_robustness_distance(corr_max, robustness):
    
    """
    Calculates the distance between the position of a cell with or without resampling. It
    is assumed that cells with greater average distance upon removal of parts of the gene signature
    are considered less robust in their signature.
    ----------
    corr_max: [pd.Series] containing the position of m cells
    robustness: [pd.DataFrame] containing the positions of m cells after resampling over n iterations.
    ----------
    return [pd.DataFrame] containing the distances to corr_max of m cells after resampling over n iterations.
    """
    
    #define output
    
    output = pd.DataFrame(index = robustness.index, columns = robustness.columns)
    
    #iterate over columns
    
    for c in robustness.columns:
        
        output[c] = np.abs(robustness[c] - corr_max[c])
        
    return output
    
################################################################################
############################# ROLLING WAVE PLOT ################################
################################################################################

def PTO_define_peak(dataset, cutoff = 0.5):
    
    """
    Finds the peak of spline-fitted gene expression as well as the points where the expression
    exceeds and undercuts a certain cutoff.
    ----------
    dataset: [pd.DataFrame] of cubic-spline fitted expression of n genes over m points
    cutoff: relative expression [float of 0.0 - 1.0] level where 'induction' and 'deactivation' points are to be set.
    ----------
    returns [pd.DataFrame] containing the axis point of 'peak', 'induction' and 'deactivation' of n genes.
    """
    
    gene_InPe = pd.DataFrame(index = dataset.index, columns = ['induction', 'peak', 'deactivation'])
    
    for gene in dataset.index:
        
        gene_InPe.ix[gene, 'peak'] = dataset.ix[gene][dataset.ix[gene] == np.max(dataset.ix[gene])].index[0]
        
        if dataset.ix[gene, 0] >= cutoff:
            
            induced = True
        
        elif dataset.ix[gene, 0] < cutoff:
            
            induced = False
        
        for pos in dataset.columns:
            
            if induced == False and dataset.ix[gene, pos] >= cutoff:
                
                gene_InPe.ix[gene, 'induction'] = pos
                
                induced = True
                
            if induced == True and dataset.ix[gene, pos] <= cutoff:
                
                gene_InPe.ix[gene, 'deactivation'] = pos
                
                break
                
        gene_InPe.ix[gene] = gene_InPe.ix[gene].fillna(gene_InPe.ix[gene, 'peak'])
        
    return gene_InPe.astype(float)
    
################################################################################

def PTO_order_groups_InPe(gene_groups, in_pe_de, sort = ['induction','peak','deactivation']):
    
    """
    Orders clustered genes according to induction < peak > deactivation.
    ----------
    gene_groups: Series of n genes clustered according to expression over pseudotime.
    in_pe_de: DataFrame containing pseudotime-point of induction, peak and deaction.
    sort: sort order defining whether 'induction, 'peak', or 'deactivation' should have primacy.
    ----------
    return Series of n reordered genes
    """
    
    order_new = []
    
    #iterate over groups
    
    in_pe_de_mid = in_pe_de['induction'].astype(float) + (0.5 * (in_pe_de['deactivation'].astype(float) - in_pe_de['induction'].astype(float)))
        
    for gr in return_unique(gene_groups):
        
        ix_tmp = gene_groups[gene_groups==gr].index
        
        if type(sort) == list:
        
            sort_tmp = list(in_pe_de.ix[ix_tmp].sort(sort).index)
            
        if sort == 'middle':
            
            sort_tmp = list(in_pe_de_mid.ix[ix_tmp].order().index)
            
        order_new += sort_tmp
        
    return gene_groups[order_new]
    
################################################################################
############################### MISCELLANEOUS ##################################
################################################################################

def PTO_bin_axis(data, n_bins, bin_range, bin_names):
    
    """
    Divides a pseudotemporal axis into bins.
    -----------
    data: [pd.Series] containing the axis coordinates of m cells.
    n_bins: number [int] of bins.
    bin_range: [tuple] of beginning and end of bin range.
    bin_names: [list] of names for each bin.
    -----------
    returns
    """
    
    #get bin edges
    
    bins = np.histogram(data, n_bins, bin_range)[1]
    
    #digitize
    
    dig = np.digitize(data, bins)
    
    #transform to pd.Series with bin names
    
    output = pd.Series(index = data.index)
    
    for pos,ix in enumerate(data.index):
        
        output.ix[ix] = bin_names[dig[pos]-1]
        
    return bins, output