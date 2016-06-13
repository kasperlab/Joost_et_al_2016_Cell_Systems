################################################################################
#################### STRT-EPIDERMIS -- AFFINITY PROPAGATION ####################
################################################################################

"""
Scripts for affinity propagation and screening for AP parameter by using information
criteria
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import random, itertools
from collections import Counter
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AffinityPropagation
from EPI_misc_scripts_v1_1 import *

################################################################################
################################# CLUSTERING ###################################
################################################################################

def AP_clustering_v3P(dataset, aff_matrix, axis, affinity, preference, damping, path, linkage = 'Ward', max_iter=200, 
                     convergence_iter=15, copy=True, verbose=False, nearest_neighbour = False, ):
      
    """
    Defines clusters along either axis of the expression matrix using the affinity propagation algorithm 
    (Frey and Dueck, Science 2007). The cells (axis = 0) or genes (axis = 1) within the clusters are 
    subsequently ordered by Wards linkage clustering while the clusters themselves are ordered by Wards linkage 
    clustering based on either (1) averaged expression or (2) nearest neighbour cluster distances.
    
    (1) Average distance: distance between two clusters is calculated according to the averaged expression values
    of each datapoint with the clusters.
    (2) NN distance: distance between two clusters is calculated based on the smallest distance between two datapoints
    with each cluster.
    
    The scikit-learn implementation is used for AP clustering while Wards linkage is based on the SciPy implementation.
    
    -----
    dataset: [pd.DataFrame] of m samples x n genes.
    
    aff_matrix: precomputed affinity matrix [pd.DataFrame] of samples or genes. Must be the same length as either 
    columns (axis = 0) or indices (axis = 1) of the dataset. 'None' if affinity == 'euclidean'.
    
    axis: 0 (cells) or 1 (genes)
    
    affinity: 'euclidean' for negative euclidean distance, 'precomputed' for custom distance (e.g. Pearsson correlation)
    
    preference: AP preference* [float]. If preference = None, the median affinity is used as preference.
        
    damping: AP damping* [float between 0.5 and 1.0].
    
    path: path specifying location of EPI_affinity_propagation_v1 custom scripts.
    
    linkage: in-cluster linkage. Either 'Ward' or 'single'. Default = 'Ward'

    nearest_neighbour: if False (default), cluster average is used to calculate the group order. Else, nearest neighbour 
    distance is used for initial group ordering
    
    * and additional function arguments are specified in 
    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation
    -----
    returns: sorted pd.Series containing pd Axis indices (cell or gene names) with associated group number
    """
    
    ### 1. Perform AP clustering
    
    #affinity is euclidean (sklearn default, dataset is passed and euclidean affinity matrix is calculated)
    
    if affinity == 'euclidean':

        af = affinity_propagation(dataset=dataset, axis=axis, affinity=affinity, preference=preference,
                                  damping=damping, max_iter=max_iter, convergence_iter=convergence_iter, copy=copy, 
                                  verbose=verbose)
    
    #affinity is precomputed (aff_matrix is passed)
    
    elif affinity == 'precomputed':
        
        af = affinity_propagation(dataset=aff_matrix, axis=axis, affinity=affinity, preference=preference,
                                  damping=damping, max_iter=max_iter, convergence_iter=convergence_iter, copy=copy, 
                                  verbose=verbose)
        
    #label AP output
    
    if axis == 0:
        labels = pd.Series(af.labels_, index = dataset.columns)
        
    elif axis == 1:
        labels = pd.Series(af.labels_, index = dataset.index)
            
    ### 2. Perform in-group Wards linkage
    
    #create pd.DataFrame for mean expression (not used when nearest_neighbour == True)
    
    cluster_mean = pd.DataFrame() 
    
    #create dict to be filled with clustered indices for datapoints in each cluster
    
    cluster_sub = {} 
    
    #iterate through clusters
    
    for cluster in set(labels):
        tmp_ix = labels[labels == cluster].index #tmp indices for datapoints within cluster
        
        # a. Cells
        
        if axis == 0:
            
            #calculate cluster average
            
            tmp_data = dataset[tmp_ix]
            cluster_mean[cluster] = tmp_data.mean(axis = 1)
            
            #continue without Wards linkage if cluster contains just a single cell
            
            if len(tmp_data.columns) == 1:
                 cluster_sub[cluster] = [column for column in tmp_data.columns]
                 continue
            
            #linkage of cells within cluster
            
            tmp_dist = 1 - tmp_data.corr() #linkage at the moment only implemented with Pearson distance
            
            if linkage == 'Ward':
                tmp_Z = sch.ward(tmp_dist)
                
            elif linkage == 'single':
                tmp_Z = sch.single(tmp_dist)
                
            tmp_leaves = sch.dendrogram(tmp_Z, no_plot = True)['leaves']
            tmp_sorted = [column for column in tmp_dist.ix[tmp_leaves,tmp_leaves].columns]
            cluster_sub[cluster] = tmp_sorted
        
        # b. Genes
        
        elif axis == 1:
            
            #calculate cluster average

            tmp_data = dataset.ix[tmp_ix]
            cluster_mean[cluster] = tmp_data.mean(axis = 0)
            
            #continue without Wards linkage if cluster contains just a single gene
            
            if len(tmp_data.index) == 1:
                 cluster_sub[cluster] = [index for index in tmp_data.index]
                 continue
            
            #linkage of genes within cluster
            
            tmp_dist = 1 - tmp_data.T.corr() #linkage at the moment only implemented with Pearson distance
            
            if linkage == 'Ward':
                tmp_Z = sch.ward(tmp_dist)
                
            elif linkage == 'single':
                tmp_Z = sch.single(tmp_dist)
            
            tmp_leaves = sch.dendrogram(tmp_Z, no_plot = True)['leaves']
            tmp_sorted = [index for index in tmp_dist.ix[tmp_leaves,tmp_leaves].index]
            cluster_sub[cluster] = tmp_sorted
    
    ### 3. Perform inter-group Wards linkage
    
    #Calculate correlation distance of cluster averages
    
    if nearest_neighbour == False:
        
        cluster_dist = cluster_mean.corr()
    
    #Calculate correlation distance of cluster nearest neighbors
    
    elif nearest_neighbour == True:
        
        if axis == 0:
            dist_mat = 1 - dataset.corr()
            
        elif axis == 1:
            dist_mat = 1 - dataset.T.corr()
            
        cluster_dist = ap_group_distance_matrix(dist_mat, labels, return_edges = False)
        
    #Perform Wards linkage
        
    Z_cluster = sch.ward(cluster_dist)
    cluster_leaves = sch.dendrogram(Z_cluster, no_plot = True)['leaves']
    
    ### 4. Sort and return sorted labels
    
    af_sorted = []
    for cluster in cluster_leaves:
        af_sorted += cluster_sub[cluster]
        
    return labels[af_sorted]

################################################################################

def affinity_propagation(dataset, axis, preference, affinity, damping=0.5, max_iter=200, convergence_iter=15, 
                         copy=True, verbose=False):
    
    """
    Helper around sk-learn AffinityPropagation function.
    """
    
    af = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter,
    copy=copy, preference=preference, affinity=affinity, verbose=verbose)
    
    if axis == 0:
        af.fit(dataset.T)
    elif axis == 1:
        af.fit(dataset)
    
    return af
    
################################################################################

def ap_group_distance_matrix(dist_mat, groups, return_edges = True):
    
    """
    Helper function to calculate nearest neighbor distance between clusters.
    ----------
    dist_mat: distance matrix [pd.DataFrame], e.g. Pearsson distance
    groups: pd.Series containing AP group membership for elements in dist_mat
    return_edges: if True, returns indices of nearest neighbors for each group
    ----------
    returns pd.DataFrame containing nearest neighbors distances (and nearest neighbor indices)
    """
    
    #initialise pd.DataFrames for NN distances and NN indices
    
    dist_mat_groups = pd.DataFrame(columns = set(groups), index = set(groups))
    
    group_edges = pd.DataFrame(columns = set(groups), index = set(groups))
    
    #iterate through clusters
    
    for ix, col in itertools.combinations_with_replacement(set(groups), 2):
    
            #define cluster indices
            
            tmp_col = groups[groups == col].index
            tmp_ix = groups[groups == ix].index
            
            #use indices to restrict dist_mat 
            
            tmp_dist = dist_mat.ix[tmp_ix, tmp_col]
            
            #search restricted dist_mat for minimal distance (and corresponding indices)
            
            tmp_min = tmp_dist.min().min()
            tmp_edges = (tmp_dist.min(axis = 0).argmin(), tmp_dist.min(axis = 1).argmin())
                        
            dist_mat_groups.ix[ix, col], dist_mat_groups.ix[col, ix] = float(tmp_min), float(tmp_min)
            group_edges.ix[ix, col], group_edges.ix[col, ix] = tmp_edges, tmp_edges
    
    if return_edges == True:
    
        return dist_mat_groups.astype(float), group_edges
        
    else:
        
        return dist_mat_groups.astype(float)

################################################################################
############################ PARAMETER SELECTION ###############################
################################################################################

def AP_IC_v3P(dataset, aff_matrix, axis, affinity, preference, damping, path, dview, linkage='Ward',
              max_iter=200, convergence_iter=15, copy=True, verbose=False, nearest_neighbour=False,
              criterion='BIC'):
    
    """
    Calculates the IC values (AIC or BIC) for affinity propagation clustering in the specified range of preference and damping values.
    -----
    dataset: pd.DataFrame containing m samples x n genes.
    aff_matrix: precomputed affinity matrix [pd.DataFrame] of samples or genes. Must be the same length as either 
    columns (axis = 0) or indices (axis = 1) of the dataset. 'None' if affinity == 'euclidean'.
    axis: 0 for samples, 1 for genes. Redundant when using 'precomputed'.
    affinity: 'euclidean' for negative euclidean distance, 'precomputed' for custom distance (e.g. Pearsson).
    preference: np.arange specififying range of preference values to test.
    damping: np.arange specififying range of damping values to test.
    criterion: 'AIC' or 'BIC'. Default = 'BIC'.
    dview: name of Ipython DirectView Instance for parallel computing.
    -----
    returns pd.DataFrames containing IC values (IC) and number of groups (Ng) for preference / damping pairs.
    """

    #initialize output DataFrame
    
    IC = pd.DataFrame(columns = preference, index = damping)
    Ng = pd.DataFrame(columns = preference, index = damping)
    
    #define preference and damping parameters
    
    pref, damp = zip(*[x for x in itertools.product(preference, damping)])
    
    l_map = len(pref)
    
    #do AP clustering in parallel
    
    ap = dview.map_sync(AP_clustering_v3P, 
                        [dataset] * l_map, 
                        [aff_matrix] * l_map, 
                        [axis] * l_map,
                        [affinity] * l_map, 
                        pref, 
                        damp,
                        [path] * l_map)
    
    #calculate IC value
    
    ic = dview.map_sync(calculateIC_v1P,
                        [dataset] * l_map, 
                        ap,
                        [axis] * l_map,
                        [criterion] * l_map)
    
    #update output DataFrames
    
    for P, D, A, I in zip(pref, damp, ap, ic):
        
        IC.ix[D, P] = I
        Ng.ix[D, P] = len(set(A))
    
    return IC, Ng
    
################################################################################

def calculateIC_v1P(dataset, groups, axis, criterion):
    
    """
    Calculates the Aikike (AIC) or Bayesian information criterion (BIC) using a formula delineated in
    http://en.wikipedia.org/wiki/Bayesian_information_criterion
    -----
    dataset: pd.DataFrame of m samples x n genes.
    groups: pd.Series containing group identity (int) for each sample or gene in dataframe.
    axis: 0 for samples, 1 for genes.
    criterion: 'AIC' or 'BIC'.
    """
    
    #for parallel processing, import modules and helper functions to engine namespace
    
    import numpy as np
    import pandas as pd
    from collections import Counter
    
    # main formula: BIC = N * ln (Vc) + K * ln (N)
    # main formula: AIC = 2 * N * ln(Vc) + 2 * K
    # Vc = error variance
    # n = number of data points
    # k = number of free parameters
    
    if axis == 0:
        
        X = dataset
        
    elif axis == 1:
        
        X = dataset.T
    
    Y = groups
    
    N = len(X.columns)
    
    K = len(set(Y))
    
    #1. Compute pd.Series Kl containing cluster lengths
    
    Kl = pd.Series(index = set(Y))
    Kl_dict = Counter(Y)
    
    for cluster in set(Y):
        Kl[cluster] = Kl_dict[cluster]
        
    #2. Compute pd.DataFrame Vc containing variances by cluster
    
    Vc = pd.DataFrame(index = X.index, columns = set(Y))
    
    for cluster in set(Y):
        
        tmp_ix = Y[Y == cluster].index
        tmp_X_var = X[tmp_ix].var(axis = 1) + 0.05 #to avoid -inf values
        Vc[cluster] = tmp_X_var
        
    #3. Calculate the mean variance for each cluster
    
    Vc = Vc.mean(axis = 0)
    
    #4. Calculate the ln of the mean variance
    
    Vc = np.log(Vc)
    
    #5. Multiply Vc by group size Kl
    
    Vc = Vc * Kl
    
    #6. Calculate accumulative error variance
    
    Vc = Vc.sum()
    
    #7a. Calculate BIC
    
    BIC = Vc + K * np.log(N)
    
    #7b. Calculate AIC
    
    AIC = 2 * Vc + 2 * K
    
    #8. Return AIC or BIC value
    
        
    if criterion == 'BIC':
        
        return BIC
    
    if criterion == 'AIC':
        
        return AIC
        
################################################################################
        
def AP_IC_findmin(IC):
    
    """
    Returns column- (AP preference) and row-index (AP damping) of minimal IC value.
    -----
    IC: pd.DataFrame of IC values according to AP preference (columns) and AP damping (rows).
    -----
    returns preference and damping of minimal IC value in IC.
    """
    
    IC_min = IC.min(axis = 0).min()
    
    preference = list(IC.min(axis = 0)[IC.min(axis = 0) == IC_min].index)
    
    if len(preference) > 1:
        
        preference = preference[(len(preference) / 2 + len(preference) % 2) - 1]
        
    else:
        
        preference = preference[0]
        
    preference = float(preference)
        
    damping = list(IC[preference][IC[preference] == IC_min].index)
    
    if len(damping) > 1:
        
        damping = damping[(len(damping) / 2 + len(damping) % 2) - 1]
        
    else:
        
        damping = damping[0]
        
    damping = float(damping)
        
    return preference, damping
        
################################################################################
############################# MANUAL PROCESSING ################################
################################################################################

def AP_fuse_clusters(dataset, groups, axis, to_fuse, linkage='Ward'):
    
    """
    Fuses two AP clusters and performs new linkage analysis on the fused cluster.
    ----------
    dataset: pd.DataFrame of m cells x n genes (should be the same as used for the initial group-file generation.
    groups: pd.Series with ordered cluster identity of m cells or n genes.
    axis: 0 (cells) or 1 (genes).
    to_fuse: list of group number to fuse.
    linkage: 'Ward' or 'single' linkage. Default: 'Ward'.
    ----------
    returns group file with fused groups
    """
    
    #save cluster order
    
    order = return_unique(groups)
    
    #give new cluster the ID of the fused clusters with the lowest ID
    
    id_min = np.min(to_fuse)
    
    for c in to_fuse:
        groups[groups == c] = id_min
        
    #decrement indices to fill gaps
        
    to_fuse.remove(id_min)
    to_fuse = sorted(to_fuse)[::-1]
    
    for c in to_fuse:
        groups[groups > c] = groups[groups > c].values - 1
        
    #perform new linkage analysis
    
    tmp_ix = groups[groups == id_min].index #tmp indices for datapoints within cluster
        
    # a. Cells
        
    if axis == 0:
            
        tmp_data = dataset[tmp_ix]
        tmp_dist = 1 - tmp_data.corr() #linkage at the moment only implemented with Pearson distance
            
        if linkage == 'Ward':
            tmp_Z = sch.ward(tmp_dist)
                
        elif linkage == 'single':
            tmp_Z = sch.single(tmp_dist)
                
        tmp_leaves = sch.dendrogram(tmp_Z, no_plot = True)['leaves']
        tmp_sorted = [column for column in tmp_dist.ix[tmp_leaves,tmp_leaves].columns]
        
    # b. Genes
        
    elif axis == 1:
            
        tmp_data = dataset.ix[tmp_ix]
        tmp_dist = 1 - tmp_data.T.corr() #linkage at the moment only implemented with Pearson distance
            
        if linkage == 'Ward':
            tmp_Z = sch.ward(tmp_dist)
                
        elif linkage == 'single':
            tmp_Z = sch.single(tmp_dist)
            
        tmp_leaves = sch.dendrogram(tmp_Z, no_plot = True)['leaves']
        tmp_sorted = [index for index in tmp_dist.ix[tmp_leaves,tmp_leaves].index]
        
    #create new incides
    
    ix_new = []
    
    for c in order:
        
        if c == id_min:
            
            ix_new += tmp_sorted
            
        else:
            
            ix_new += list(groups[groups == c].index)
            
    return groups[ix_new]
    
################################################################################

def AP_invert_index(group_file, group):
    
    """
    Inverts indices of a group in file of cluster groups.
    ----------
    group_file: pd.Series with ordered cluster identity of m cells or n genes.
    group: ID (int) of group to be inverted.
    ----------
    returns group file with inverted groups.
    """
    
    ix_new = []
    
    for gr_tmp in return_unique(group_file):
        
        if gr_tmp == group:
            
            ix_tmp = list(group_file[group_file == gr_tmp].index)[::-1]
            
        else:
            
            ix_tmp = list(group_file[group_file == gr_tmp].index)
            
        ix_new += ix_tmp
        
    group_file_new = group_file[ix_new]
    
    return group_file_new

################################################################################

def AP_groups_reorder(groups, order, link_to = None):
    
    """
    Reorders the groups in an sample or gene group Series either completely or partially
    -----
    groups: pd.Series of either samples (Cell ID) or gene (gene ID) linked to groups (int)
    order: list containing either complete or partial new order of groups
    link_to: defines which group position is retained when groups are reorded partially; default == None, groups are linked to
    first group in 'order'
    -----
    returns reordered group Series
    """
    
    # (1) Define new group order
    
    if set(order) == set(groups):
        order_new = order
        
    else:
        
        order_new = return_unique(groups, drop_zero = False)
        
        if link_to in order:
            link = link_to
        
        elif link_to not in order or link_to == None:
            link = order[0]
            
        order.remove(link)
        
        for group in order:
            
            order_new.remove(group)
            ins_ix = order_new.index(link) + 1
            gr_ix = order.index(group)
            order_new.insert(ins_ix + gr_ix, group)
            
    # (2) Reorder groups
    
    groups_new = pd.Series()
    
    for group in order_new:
        
        groups_new = groups_new.append(groups[groups == group])
        
    groups_new = groups_new.astype(np.int64)
    
    return groups_new

################################################################################
################################# ROBUSTNESS ###################################
################################################################################

def robustness_AP_v1P(dataset, groups, genes, resampling, iterations, preference, damping, dview):
    
    """
    Test the robustness of the (cell group!) AP clustering by removing a subset of the dataset,
    reclustering the dataset using the same parameters and determining the maximum number of cells
    from each initial group which end up back together. NB: only implemented for AP clustering using Pearson
    correlation as affinity. 
    -----------
    dataset: [pd.DataFrame] of m samples x n genes.
    groups: [pd.Series] of AP-clustered cell groups.
    genes: [list] of genes used for initial clustering.
    resampling: [float] fraction of cells to be randomly removed from the dataset.
    iterations: [int] number of reclustering repeats
    preference: AP preference [float]. If preference = None, the median affinity is used as preference.
    damping: AP damping [float between 0.5 and 1.0].
    -----------
    return pd.DataFrame containing the correctly reclustered fractions of cells in each group (and a null distribution)
    """
    
    ####################
    
    def calculate_robustness(dataset, aff_matrix, groups, resampling, iterations, preference, damping):
    
        """
        Helper function for parallel computing.
        """
        
        #initialize temporary output
        
        robustness_tmp = pd.Series(index = [str(x) for x in set(groups)])
        robustness_null_tmp = pd.Series(index = [str(x) for x in set(groups)])

        #selected subset of full dataset

        ix_sel = random.sample(groups.index, int(len(groups.index) * resampling))
        
        #AP cluster restricted dataset using same parameters for preference and damping

        groups_tmp = AP_clustering_v3P(dataset[ix_sel], 
                                       aff_matrix.ix[ix_sel, ix_sel], 
                                       0, 
                                       'precomputed', 
                                       preference, 
                                       damping,
                                       None,
                                       linkage = 'single')
        #score

        for gr in set(groups):
            ix_tmp = groups[ix_sel][groups[ix_sel] == gr].index
            robustness_tmp.ix[gr] = np.max(Counter(groups_tmp[ix_tmp]).values()) / float(len(ix_tmp))
            
        #score null (= shuffled indices)

        groups_shuffle = groups.copy()
        np.random.shuffle(groups_shuffle)

        for gr in set(groups_shuffle):
            ix_tmp = groups_shuffle[ix_sel][groups_shuffle[ix_sel] == gr].index
            robustness_null_tmp.ix[gr] = np.max(Counter(groups_tmp[ix_tmp]).values()) / float(len(ix_tmp))
            
        return robustness_tmp, robustness_null_tmp
    
    ####################
    
    #create output DataFrames

    robustness = pd.DataFrame(columns = [str(x) for x in set(groups)], index = range(iterations))

    robustness_null = pd.DataFrame(columns = [str(x) for x in set(groups)], index = range(iterations))

    #create correlation data

    aff_matrix = dataset.ix[genes, groups.index].corr()
    
    #parallel compute data
    
    data_tmp = dview.map_sync(calculate_robustness,
                              [dataset] * iterations, 
                              [aff_matrix] * iterations, 
                              [groups] * iterations, 
                              [resampling] * iterations, 
                              range(iterations), 
                              [preference] * iterations, 
                              [damping] * iterations)

    #combine
    
    #return data_tmp
    
    for it, data in enumerate(data_tmp):
        
        robustness.ix[it] = data[0]
        robustness_null.ix[it] = data[1]

    return robustness, robustness_null

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################