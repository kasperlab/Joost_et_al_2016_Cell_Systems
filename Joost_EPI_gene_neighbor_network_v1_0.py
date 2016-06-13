################################################################################
################### STRT-EPIDERMIS -- GENE NEIGHBOR NETWORK ####################
################################################################################

"""
Scripts for inferring putative gene regulatory relationships based on correlation
and shared neighbors.
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import numpy as np
import pandas as pd
import itertools
from collections import Counter
import pystan
import networkx as nx

from EPI_misc_scripts_v1 import *

################################################################################
############################ NETWORK CONSTRUCTION ##############################
################################################################################

def GRN_CLR(corr_mat, method = 'normal'):
    
    """
    Calculates context likelihood of relatedness (Faith et al. 2007).
    ----------
    corr_mat: DataFrame containing the similarity (e.g. Pearson correlation) of n x n genes.
    method: CLR method. 'normal'.
    ----------
    returns DataFrame of CLR-corrected distance of n x n genes.
    """
    
    #get zscores along both axes
           
    Z0 = corr_mat.apply(lambda x: (x - np.mean(x)) / np.std(x), axis = 0)
    Z1 = corr_mat.apply(lambda x: (x - np.mean(x)) / np.std(x), axis = 1)
    
    #combine zscores uing unweighted Stouffer method
        
    Z = (Z0 + Z1) / np.sqrt(2)
    
    return Z
    
################################################################################

def GRN_shared_NN(corr_mat, k, c):
    
    """
    Constructs gene-gene network based on shared neighbors. The k nearest neighbors
    are determined for each gene and an edge between to genes is drawn if those genes
    share at least c nearest neighbors.
    ----------
    corr_mat: DataFrame of n x n genes containing correlations or affinities between genes.
    k: number [int] of neighbors for each gene.
    c: number [int] of shared nearest neighbors required for edge between genes.
    ----------
    return DataFrame of n x n genes where edges are marked as 1.
    """
    
    #get k nearest neighbours for each gene
    
    print '\nFind %s nearest neighbors for each gene' % (k)
    
    kNN = {}
    
    for g in corr_mat.index:
        
        kNN[g] = corr_mat.ix[g].order().index[-k:]
                
    #get shared nearest neighbours
    
    print '\nFind shared nearest neighbors'
    
    shNN = pd.DataFrame(index = corr_mat.index, columns = corr_mat.index)
    
    for g1, g2 in itertools.combinations_with_replacement(corr_mat.index, 2):
        
        shNN_tmp = len(set(kNN[g1] & kNN[g2]))
        
        shNN.ix[g1,g2], shNN.ix[g2,g1] = shNN_tmp, shNN_tmp
        
    #establish edges between genes with more than c shNN
    
    print '\nDrop all edges between genes with less than %s shared nearest neighbors' % (c)
    
    shNN_bin = pd.DataFrame(0, index = corr_mat.index, columns = corr_mat.index)
    
    shNN_bin[shNN >= c] = 1
    
    return shNN_bin
    
################################################################################
################################ VISUALIZATION #################################
################################################################################
    
def GRN_get_jaccard_dist(bin_mat):
    
    """
    Calculates Jaccard distance of genes from binarized matrix.
    ----------
    bin_mat: DataFrame of n(col) x n(ix) genes where edges are marked with 1.
    ----------
    return DataFrame of n(ix) x n(ix) genes containing the Jaccard distance between the genes.
    """
    
    bin_mat_jaccard = pd.DataFrame(0, index = bin_mat.index, columns = bin_mat.index)
    
    for g1, g2 in itertools.combinations_with_replacement(bin_mat.index, 2):
        
        g1_tmp = bin_mat.ix[g1][bin_mat.ix[g1]==1].index
        g2_tmp = bin_mat.ix[g2][bin_mat.ix[g2]==1].index
                        
        jaccard_tmp = float(len(set(g1_tmp & g2_tmp))) / float(len(set(g1_tmp | g2_tmp)))
        
        bin_mat_jaccard.ix[g1,g2], bin_mat_jaccard.ix[g2,g1] = jaccard_tmp, jaccard_tmp
        
    return bin_mat_jaccard
    
################################################################################
    
def GRN_create_nx(bin_mat, drop_alone=False, prog='neato'):
    
    """
    Generates network/Graph from binarized gene-gene matrix.
    ----------
    bin_mat: DataFrame of n x n genes where edges are marked with 1.
    drop_alone: Whether to drop nodes without edges. Default: False.
    prog: GraphViz prog. Default:'neato'.
    ----------
    returns nx Graph and nx position dict.
    """
    
    #add nodes
    
    G = nx.Graph()
    
    if drop_alone==True:
        G.add_nodes_from(bin_mat.sum(axis=1)[bin_mat.sum(axis=1)>1].index)
        
    else:
        G.add_nodes_from(bin_mat.index)
        
    #add edges
    
    for n1, n2 in itertools.combinations(G.nodes(),2):
        
        if bin_mat.ix[n1,n2] == 1:
            
            G.add_edge(n1,n2)
            
    #draw network
    
    pos = nx.graphviz_layout(G, prog=prog)
    
    return G, pos

################################################################################
