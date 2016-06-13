################################################################################
################### STRT-EPIDERMIS -- MISCELLANEOUS SCRIPTS ####################
################################################################################

"""
A variety of smaller scripts for data input, data wrangling and transformation,
data plotting and data analysis. Scripts were usually tested interactively in Ipython Notebook
before being transferred here
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import os, math, datetime, random, itertools
from collections import Counter
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import fmin
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

################################################################################
################################# DATA INPUT ###################################
################################################################################

def create_ID():
    
    "Creates experiment ID (YmdHm) to identify output"
    
    exp_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
    
    print "\nThe experiment ID is %s" % (exp_id)
    
    return exp_id
    
################################################################################

def setWDandMaster(path, master_inp):
    
    """
    Sets the working directory (path [str]) for the following functions and imports the master file [str]. The working directory requires subfolders containing
    'Sequencing' data, 'Metadata', 'Capturing' data and 'Scepter' data identifiable by IFC barcode. The master file is located in the working 
    directory and contains all IFC barcodes to be included in the analysis divided by linebreaks. Returns the contents of the master file.
    """ 
    
    ### select working directory for master file
    
    os.chdir(path)
               
    ### import master file       
           
    master_file = open('Input/%s' % (master_inp), 'r').read().split()
    print '\nThe master file contains the following %s libraries: % s' % (len(master_file), master_file)
    
    return master_file 
    
################################################################################

def ImportandMerge(master_file):
      
    """
    Iterates through the master file and imports sequencing and metadata for every specified IFC / library. Transforms sequencing data 
    into DataFrame with Cells as columns and Genes as rows. Merges general metadata and capturing metadata sets and adds quantitative
    data about sequencing results (number of genes, transcripts etc.). Every cell is uniquely defined using the following format:
    '[IFC-barcode]-[#IFC-row][#IFC-column]. Merged sequencing (= 'seq') and metadata DataFrames (= 'meta') are returned.    
    """   
    
    ### check whether master file is imported
    
    try:
        master_file
    except:
        print 'Master file not imported!'
        
    ###i terate through library / IFC barcodes in master file
        
    for barcode in master_file:
        
        ### import and format well exclusion data
        
        excl = []
                
        for line in open('Capturing/wells_to_exclude_%s.txt' % (barcode), 'r').read().split('\n')[1:][:-1]:
            if line.split('\t')[1] == '10' or line.split('\t')[1] == '11' or line.split('\t')[1] == '12':
                excl.append('%s-%s%s' % (barcode, line.split('\t')[0], line.split('\t')[1]))
            else:
                excl.append('%s-%s0%s' % (barcode, line.split('\t')[0], line.split('\t')[1]))
                   
        ###import and format sequencing data per barcode
        
        seq_tmp = pd.read_table('Sequencing/C1-%s-%s_expression.tab' % (barcode[:7], barcode[7:]), sep = '\t', header = 2, index_col = 0, low_memory = False)
        seq_tmp = seq_tmp.ix['ERCC-00002':, 'A01':].astype(int)
        seq_tmp.columns = ['%s-%s' % (barcode, column) for column in seq_tmp.columns]
        seq_tmp = seq_tmp.drop(excl, axis = 1)
        seq_tmp.index.name = 'Gene'; seq_tmp.columns.name = 'Cell'

        ###compile metadata per barcode
                
        #import and transform general (IFC-specific) metadata
        
        meta_gen_tmp = pd.read_table('Metadata/C1_metadata_%s.txt' % (barcode), sep = '\t', header = None, index_col = 0)
        meta_gen_tmp = meta_gen_tmp.reindex(columns = range(1, len(seq_tmp.columns) + 1))
        meta_gen_tmp = meta_gen_tmp.fillna(method = 'ffill', axis = 1)
        meta_gen_tmp.columns = seq_tmp.columns
        
        #import and transform mouse metadata
        
        meta_mouse_tmp = pd.read_table('Metadata/mice_metadata_%s.txt' % (barcode), sep = '\t', header = None, index_col = 0)
        meta_mouse_tmp = meta_mouse_tmp.reindex(columns = range(1, len(seq_tmp.columns) + 1))
        meta_mouse_tmp = meta_mouse_tmp.fillna(method = 'ffill', axis = 1)
        meta_mouse_tmp.columns = seq_tmp.columns     
        
        #import and transform capturing metadata
        
        meta_cap_tmp = pd.read_table('Capturing/capture_rep_image_%s.txt' % (barcode), sep = '\t', header = 0, index_col = None)
        meta_cap_tmp = meta_cap_tmp.T        
        meta_cap_tmp = meta_cap_tmp.drop(['row','col','capture_site'])
        meta_cap_tmp.columns = ['%s-%s' % (barcode, well) for well in open('/Users/sijo/Documents/Epidermis map/Scripts/96well.txt','r').read().split()]
        meta_cap_tmp = meta_cap_tmp[seq_tmp.columns]
        
        #join datasets
        
        meta_tmp = meta_gen_tmp.append([meta_mouse_tmp, meta_cap_tmp])

        #update Sca1 fluorescence data 
        
        Sca1 = []
                
        for line in open('Capturing/wells_positive_green_%s.txt' % (barcode), 'r').read().split('\n')[1:][:-1]:
            if line.split('\t')[1] == '10' or line.split('\t')[1] == '11' or line.split('\t')[1] == '12':
                Sca1.append('%s-%s%s' % (barcode, line.split('\t')[0], line.split('\t')[1]))
            else:
                Sca1.append('%s-%s0%s' % (barcode, line.split('\t')[0], line.split('\t')[1]))
                
        Sca1 = list(set(Sca1).difference(set(excl)))
        
        meta_tmp.ix['green_flag', Sca1] = 1
        
        #calculate volume
        
        meta_tmp_volume = pd.Series((((meta_tmp.ix['diameter(um)'] / 2) ** 3) * math.pi * 4/3), name = 'volume(um^3)')
        meta_tmp = meta_tmp.append(meta_tmp_volume)
        
        #calculate spikes and repeats
        
        spikes = open('/Users/sijo/Documents/Epidermis map/Scripts/spikes.txt', 'r').read().split('\n')[:-1]
        repeats = open('/Users/sijo/Documents/Epidermis map/Scripts/repeats.txt','r').read().split('\n')[:-1]
        
        if barcode in open('/Users/sijo/Documents/Epidermis map/Scripts/spikes_double.txt', 'r').read().split():
            meta_tmp_spikes = pd.Series(seq_tmp.ix[spikes].sum(axis = 0) / 2, name = 'sum_spikes')
            
        else:
            meta_tmp_spikes = pd.Series(seq_tmp.ix[spikes].sum(axis = 0), name = 'sum_spikes')
            
        meta_tmp_repeats = pd.Series(seq_tmp.ix[repeats].sum(axis = 0), name = 'sum_repeats')
        meta_tmp = meta_tmp.append([meta_tmp_spikes, meta_tmp_repeats])
        
        #calculate transcripts and genes
        
        meta_tmp_transcripts = pd.Series(seq_tmp.ix['Xkr4':'Erdr1'].sum(axis = 0), name = 'sum_transcripts')
        meta_tmp_genes = pd.Series(seq_tmp.ix['Xkr4':'Erdr1'].apply(lambda x: x > 0).sum(axis = 0), name = 'sum_genes')
        meta_tmp = meta_tmp.append([meta_tmp_transcripts, meta_tmp_genes])
        
        #change metadata to correct datatype
        
        meta_tmp.ix[['area(um^2)','diameter(um)','volume(um^3)']] = meta_tmp.ix[['area(um^2)','diameter(um)','volume(um^3)']].astype(float)
                
        ###join datasets
        
        print 'Adding %s to dataset' % (barcode)
        
        if master_file.index(barcode) == 0:
            seq = seq_tmp
            seq_ix = seq.index
            meta = meta_tmp
            
        else:
            seq = pd.concat([seq, seq_tmp], axis = 1)
            meta = pd.concat([meta, meta_tmp], axis = 1)
            
    seq = seq.ix[seq_ix]  #to prevent lexicographical ordering of big dataset
    
    return seq, meta
    
################################################################################

def saveData_v1(dataset, path, id_, name):
    
    """
    Saves pd.DataFrames or pd.Series to csv.
    ----------
    dataset: [pd.DataFrame] or [pd.Series].
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    """
    
    #print '\nSaving data as .txt'
        
    dataset.to_csv('%s/%s_%s.txt' % (path, id_, name), sep = '\t')
    
################################################################################

def saveData_to_pickle_v1(dataset, path, id_, name):
    
    """
    Saves pd.DataFrames or pd.Series to pickle.
    ----------
    dataset: [pd.DataFrame] or [pd.Series].
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    """
    
    #print '\nSaving data as .txt'
        
    dataset.to_pickle('%s/%s_%s.txt' % (path, id_, name))

################################################################################

def loadData_v1(path, id_, name, datatype):
    
    """
    loads pd.DataFrames or pd.Series from csv.
    ----------
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    datatype: 'DataFrame' or 'Series'.
    ----------
    returns [pd.DataFrame] or [pd.Series]
    """
    
    if datatype == 'DataFrame':
        
        dataset = pd.DataFrame.from_csv('%s/%s_%s.txt' % (path, id_, name), sep = '\t', header = 0, index_col = 0)
    
    elif datatype == 'Series':
        
        dataset = pd.Series.from_csv('%s/%s_%s.txt' % (path, id_, name), sep = '\t', header = None, index_col = 0)
        
    return dataset
    
################################################################################
    
def loadData_from_pickle_v1(path, id_, name):
    
    """
    loads pd.DataFrames or pd.Series from csv.
    ----------
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    ----------
    returns [pd.DataFrame] or [pd.Series]
    """
    
    return pd.read_pickle('%s/%s_%s.txt' % (path, id_, name))

################################################################################
################# DATA TRANSFORMATION AND FEATURE SELECTION ####################
################################################################################

def dropNull(dataset, path, drop_spikes = True, drop_repeats = True, cutoff_mean = 0):

    """
    Takes the merged sequencing dataset, drops spike values (based on 'spike.txt' containing all current spike index names) unless 
    set False and removes all unexpressed genes (sum == 0 over the whole dataset)
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes.
    path: path where txt. files specifying spikes and repeats are stored.
    drop_spikes: [bool] indicating whether ERCC spikes are removed from the dataset.
    drop_repeats: [bool] indicating whethere repeats are removed from the dataset.
    cutoff_mean: Average expression count [float] at which a gene is dropped from the dataset.
    ----------
    returns seq dataset [pd.DataFrame] containing m cells x n genes.
    """
      
    ###drop spike and repeat indices unless specifically disabled
    
    spikes = open('%s/spikes.txt' % (path),'r').read().split('\n')[:-1]
    repeats = open('%s/repeats.txt' % (path),'r').read().split('\n')[:-1]
    
    if drop_spikes == True:
        print '\nDropping spikes from dataset'
        dataset = dataset.drop(spikes)
        
    if drop_repeats == True:
        print '\nDropping repeats from dataset'
        dataset = dataset.drop(repeats)   
    
        
    ###drop rows of non-expressed genes
    
    print '\nDropping unexpressed genes from dataset'
    dataset = dataset.drop(dataset[dataset.mean(axis = 1) <= cutoff_mean].index)
    
    return dataset

################################################################################

def cellCutoff(dataset, cutoff):
    
    """
    Removes all observations / cells whose total number of transcripts lies below a
    specified cutoff.
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes.
    cutoff: number of total transcript / molecule count [int] under which a cell is dropped from the dataset.
    ----------
    returns seq dataset [pd.DataFrame] containing m cells x n genes.
    """
    
    print '\nRemoving cells with less than %s transcripts' % (cutoff)
    
    dataset = dataset[dataset.columns[dataset.sum() >= cutoff]]
    
    return dataset

################################################################################

def log2Transform(dataset, add = 1):
    
    """
    Calculates the binary logarithm (log2(x + y)) for every molecule count / cell x in dataset. 
    Unless specified differently, y = 1.
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes.
    add: y [float or int] in (log2(x + y)).
    ----------
    returns seq dataset [pd.DataFrame] containing m cells x n genes.
    """
    
    print '\nCalculating binary logarithm of x + %s' % (add)
    dataset = np.log2(dataset + add)
    
    return dataset
          
################################################################################

def gene_correlation_v1(dataset, method = 'pearson', percentile = 95):
    
    """
    Calcutes the number of highly (>= xth percentile of overall correlation) correlated neighbours for each gene.
    -----
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes.
    method: correlation method ['pearson' or 'spearman']. Default = 'pearson'
    percentile: percentile of overall intercorrelation which defines highly correlated genes (default = 95).
    -----
    pd.Series showing number of highly correlated neighbours [int] for every gene in dataset
    """
    
    corr_mat = np.abs(dataset.T.corr(method = method))   
    corr_filt_thresh = np.percentile(corr_mat, percentile)
    corr_filt = corr_mat.apply(lambda x: x >= corr_filt_thresh).sum(axis = 1).order()
    
    return corr_filt


################################################################################

def log2_cv_fit(dataset):
    
    """
    Fits noise model: log2(CV) = log2(mean^alpha + k).
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes. NB: Must be untransformed!
    ----------
    returns pd.Series containing difference between measured and expected cv for every gene and fitted values for [alpha, k].
    """
    
    #########################
    
    def min_cvfit(log2_mean, log2_cv, x0):
        
        """
        Helper function for log_cv_fit
        """
    
        nestedfun = lambda x: np.sum(np.abs(np.log2 ((2 ** log2_mean) ** x[0] + x[1]) - log2_cv))  
        xopt0, xopt1  = fmin(nestedfun, x0 = x0)
        
        return xopt0, xopt1
    
    #########################

    data_mean = dataset.mean(axis = 1)
    data_cv = dataset.std(axis = 1) / dataset.mean(axis = 1)

    log2_mean = np.log2(data_mean)
    log2_cv = np.log2(data_cv)

    x0 = [-0.5, 0.5]
    xopt0, xopt1 = min_cvfit(log2_mean, log2_cv, x0)
    xopt = [xopt0, xopt1]
    
    log2_cv_fit = np.log2(( 2 ** log2_mean) ** xopt[0] + xopt[1])
    log2_cv_diff = log2_cv - log2_cv_fit

    return log2_cv_diff, xopt

    
################################################################################

def select_features_v1(dataset, cutoff_mean, cutoff_corr, nr_features, path_input, percentile=95, return_all=False, drop_spikes = True, drop_repeats = True):

    """
    Helper function to compress the feature selection workflow.
    ----------
    dataset: [DataFrame] of m cells x n genes. Must be non-transformed.
    cutoff_mean: gene expression cutoff [float]. Average expression required for a gene to be retained
    in the dataset.
    cutoff_corr: correlation cutoff [int]. Number of highly correlated partner genes required for a gene to be retained
    in the dataset.
    nr_features: number of features to select [int]. Features are ordered according to log2_cv_diff.
    path_input: name of path leading to input files including spike and repeat lists
    percentile: correlation percentile at which two genes are considered highly correlated. Default: 95.
    return_all: if True, additionally returns corr_filt, log2_cv_diff and x_opt. Default: False
    ----------
    return log2-transformed DataFrame of m cells and n selected features. If return_all is True, additionally 
    returns corr_filt, log2_cv_diff and x_opt
    """

    dataset = dropNull(dataset, path_input, drop_spikes = drop_spikes, drop_repeats = drop_repeats, cutoff_mean=cutoff_mean)
    
    print "\nAfter mean expression cutoff of %s, %s genes remain" % (cutoff_mean, len(dataset.index))
    
    corr_filt = gene_correlation_v1(dataset, method ='pearson', percentile=percentile)
    
    genes_sel_1 = corr_filt[corr_filt >= cutoff_corr].index 
    
    print "\nAfter correlation cutoff of %s, %s genes remain\n" % (cutoff_corr, len(genes_sel_1))
    
    log2_cv_diff, xopt = log2_cv_fit(dataset.ix[genes_sel_1])
    
    genes_sel_2 = log2_cv_diff.order()[-nr_features:].index
    
    draw_log2_cv_diff(dataset.ix[genes_sel_1], log2_cv_diff, xopt, selected=genes_sel_2)
    
    print "\nAfter high variance feature selection, %s genes remain" % (len(genes_sel_2))
    
    dataset = dataset.ix[genes_sel_2]
    
    dataset = log2Transform(dataset)
    
    if return_all==True:
    
        return dataset, corr_filt, log2_cv_diff, xopt
    
    else:
        
        return dataset
    
################################################################################

def find_tSNE_v1P(dist_mat, cell_groups, iterations, dview, perplexity=30.0, early_exaggeration=2.0, learning_rate=1000.0, 
              n_iter=1000, init='random', verbose=0, random_state=None):
    
    """
    Function to screen for appropiate tSNE representation of dataset.
    ----------
    dist_mat: distance matrix [DataFrame] of m x m cells
    cell_groups: Series of cell group identity.
    iterations: number [int] of tSNE representations produced and plotted.
    dview: name of Ipython DirectView Instance for parallel computing.
    ----------
    draws number of tSNE plot and returns coordinates for selection.
    """
    ##########################################################
    
    def create_tSNE(tSNE, dist_mat):
        
        """
        Helper function for parallelization of find_tSNE.
        """
        
        return pd.DataFrame(tsne.fit_transform(dist_mat), index = dist_mat.index, columns = ['x', 'y'])
    
    ##########################################################
    
    #define TSNE parameters
    
    tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, 
                learning_rate=learning_rate, n_iter=n_iter, metric='precomputed', init=init, 
                verbose=verbose, random_state=random_state)
    
    #create tSNE coords in paralell
    
    tsne_coords = dview.map_sync(create_tSNE, 
                              [tsne] * iterations, 
                              [dist_mat] * iterations)
    
    #draw tSNE plots
    
    for ix, coords in enumerate(tsne_coords):
    
        draw_tSNE(coords, cell_groups, ix, cmap=plt.cm.jet)
    
    return tsne_coords
    
################################################################################

def dist_mat_dim_reduc(dataset, dim=50, method='TruncatedSVD'):
    
    """
    Returns a correlation distance matrix after dimensionality reduction with either PCA or Truncated SVD.
    ----------
    dataset: DataFrame of m cells x n genes
    dim: number of dimensions. Default: 50.
    method. Dimensionality reduction method. Either 'PCA' or 'TruncatedSVD'. Default: 'TruncatedSVD'.
    ----------
    returns DataFrame containing the correlation distances of m x m cells.
    """
    
    if method == 'PCA':
        
        pca = PCA(n_components=dim, copy=True, whiten=False)
        
        data_tmp = pd.DataFrame(pca.inverse_transform(pca.fit_transform(dataset.T)).T, index = dataset.index, columns = dataset.columns)
        
    if method == 'TruncatedSVD':
        
        tSVD = TruncatedSVD(n_components=dim, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
        
        data_tmp =  pd.DataFrame(tSVD.inverse_transform(tSVD.fit_transform(dataset.T)).T, index = dataset.index, columns = dataset.columns)
        
    return 1 - data_tmp.corr()

################################################################################
################################## PLOTTING ####################################
################################################################################

def draw_log2_cv_diff(dataset, log2_cv_diff, xopt, selected = None):
    
    """
    Plots the average gene expression versus the coefficient of variation of log2 normalized expression data and overlays
    the expected cv per expression level from the the fitted noise model.
    -----
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes. NB: Must be untransformed!
    log2_cv_diff: pd.Series containing difference between measured and expected cv for every gene.
    xopt: fitted parameters of noise model.
    selected: list or pd Index file containing the indices of selected cells. Default: none.
    """
    
    data_mean = dataset.mean(axis = 1)
    data_cv = dataset.std(axis = 1) / data_mean

    log2_mean = np.log2(data_mean)
    log2_cv = np.log2(data_cv)
    
    line_x = np.arange(log2_mean.min(), log2_mean.max(), 0.01)
    line_y = np.log2(( 2 ** line_x) ** xopt[0] + xopt[1])
    
    clist = pd.Series('blue', index = log2_mean.index)
    clist[log2_cv_diff[log2_cv_diff > 0].index] = 'red'
    
    if np.all(selected != None):
        clist[selected] = 'green'

    fig = plt.figure(figsize = [10,10], facecolor = 'w')
    ax0 = plt.axes()
    
    ax0.set_xlabel('Average number of transcripts [log2]')
    ax0.set_ylabel('Coefficient of variation (CV) [log2]')
    
    ax0.set_xlim(log2_mean.min() - 0.5, log2_mean.max() + 0.5)
    ax0.set_ylim(log2_cv.min() - 0.5, log2_cv.max() + 0.5)
    
    ax0.scatter(log2_mean, log2_cv, c = clist, linewidth = 0,)
    ax0.plot(line_x, line_y, c = 'k', linewidth = 3)
    
################################################################################

def draw_heatmap(dataset, cell_groups, gene_groups, cmap = plt.cm.jet):
    
    """
    Draw heatmap showing gene expression ordered according to cell_groups and
    gene_groups Series (e.g. AP clustering). Cell and gene groups membership is 
    visualized in two additional panels:
    ----------
    dataset: pd.DataFrame of m cells * n genes.
    cell_groups: pd.Series with ordered cluster identity of m cells.
    gene_groups: pd.Series with ordered cluster identity of n genes.
    cmap: matplotlib color map. Default: plt.cm.jet.
    """
    
    dataset = dataset.ix[gene_groups.index, cell_groups.index]
    dataset = dataset.apply(lambda x: x / max(x), axis = 1)

    plt.figure(figsize=(20,20), facecolor = 'w')
    
    #draw heatmap

    axSPIN1 = plt.axes()
    axSPIN1.set_position([0.05, 0.05, 0.9, 0.9])
    
    axSPIN1.imshow(dataset, aspect = 'auto', interpolation = 'nearest')
    
    remove_ticks(axSPIN1)
    
    #draw genes bar

    divider = make_axes_locatable(axSPIN1)

    axGene_gr = divider.append_axes("right", size= 0.5, pad=0.05)

    axGene_gr.set_xlim(-0.5,0.5)
    axGene_gr.imshow(np.matrix(gene_groups).T, aspect = 'auto')
    
    remove_ticks(axGene_gr)
    
    #draw genes bar ticks
    
    gene_groups_ticks = pd.Series(index = set(gene_groups))
    
    for gr in gene_groups_ticks.index:
                
        first_ix = list(gene_groups.values).index(gr)
        last_ix = len(gene_groups) - list(gene_groups.values)[::-1].index(gr)
        gene_groups_ticks[gr] = first_ix + ((last_ix - first_ix) / 2.0)
        
    axGene_gr.set_yticks(gene_groups_ticks.values)
    axGene_gr.set_yticklabels(gene_groups_ticks.index)
    axGene_gr.yaxis.set_ticks_position('right')
    
    #draw cells bar
    
    axCell_gr = divider.append_axes("bottom", size= 0.5, pad=0.05)

    axCell_gr.set_ylim(-0.5, 0.5)
    axCell_gr.imshow(np.matrix(cell_groups), aspect = 'auto')
    
    remove_ticks(axCell_gr)
    
    #draw cells bar ticks
    
    cell_groups_ticks = pd.Series(index = set(cell_groups))
        
    for gr in cell_groups_ticks.index:
                
        first_ix = list(cell_groups.values).index(gr)
        last_ix = len(cell_groups) - list(cell_groups.values)[::-1].index(gr)
        cell_groups_ticks[gr] = first_ix + ((last_ix - first_ix) / 2.0)
        
    axCell_gr.set_xticks(cell_groups_ticks.values)
    axCell_gr.set_xticklabels(cell_groups_ticks.index)
    axCell_gr.xaxis.set_ticks_position('bottom')

################################################################################

def draw_AP_dist_mat(dist_mat, groups):
    
    """
    Draws distance matrices of either m cells or n genes randomly shuffled and
    ordered according to group Series (e.g. AP clustering).
    ----------
    dist_mat: pd.DataFrame with distances of either m x m cells or n x n genes.
    groups: pd.Series with ordered cluster identity of m cells or n genes.
    """
    
    plt.figure(figsize = [20,10], facecolor = 'w')

    ax0 = plt.subplot(121)

    tmp_ix = list(dist_mat.index)
    random.shuffle(tmp_ix)

    ax0.matshow(dist_mat.ix[tmp_ix, tmp_ix], vmin = 0, vmax = 1)

    ax1 = plt.subplot(122)

    ax1.matshow(dist_mat.ix[groups.index, groups.index], vmin = 0, vmax = 1)
    
################################################################################
    
def draw_tSNE(tsne_coords, cell_groups, number = None, cmap=plt.cm.jet):
    
    """
    Function to draw tSNE plots.
    ----------
    tsne_coords: DataFrame of tSNE coordinates in two dimensions.
    cell_groups: Series of cell group identity.
    number: int for indentification of plot in tSNE screen.
    """
    
    #initialize figure

    height = 14
    width = 14

    plt.figure(facecolor = 'w', figsize = (width, height))

    #define x- and y-limits

    x_min, x_max = np.min(tsne_coords['x']), np.max(tsne_coords['x'])
    y_min, y_max = np.min(tsne_coords['y']), np.max(tsne_coords['y'])
    x_diff, y_diff = x_max - x_min, y_max - y_min

    pad = 2.0

    if x_diff > y_diff:
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_min * (x_diff / y_diff) - pad, y_max * (x_diff / y_diff) + pad)

    if x_diff < y_diff:
        xlim = (x_min * (y_diff/x_diff) - pad, x_max * (y_diff/x_diff) + pad)
        ylim = (y_min - pad, y_max + pad)

    text_pad = 2
    
    #define x- and y-axes

    ax1 = plt.subplot()

    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])

    remove_ticks(ax1)
    
    #define colormap
        
    gr_max = float(np.max(cell_groups.values))
    clist_tsne = [gr / gr_max for gr in cell_groups.values]

    ax1.scatter(tsne_coords.ix[cell_groups.index, 'x'],
                tsne_coords.ix[cell_groups.index, 'y'], 
                s = 100,
                linewidth = 0.0,
                c = clist_tsne)
    
    #draw number
    
    ax1.text(xlim[1] * 0.9, ylim[1] * 0.9, number, family = 'Arial', fontsize = 25)
    
################################################################################

def draw_barplots(dataset, cell_groups, genes, cmap = plt.cm.jet):
    
    """
    draws expression of selected genes in order barplot with juxtaposed group identity
    dataset: pd.DataFrame of n samples over m genes
    sample_group_labels: ordered (!) pd.Series showing sample specific group indentity 
    list_of_genes: list of selected genes
    color: matplotlib cmap
    """
    
    # set figure framework
    
    plt.figure(facecolor = 'w', figsize = (21, len(genes) * 3 + 1))
        
    gs0 = plt.GridSpec(1,1, left = 0.05, right = 0.95, top = 1 - 0.05 / len(genes),
                       bottom = 1 - 0.15 / len(genes), hspace = 0.0, wspace = 0.0)
    
    gs1 = plt.GridSpec(len(genes), 1, hspace = 0.05, wspace = 0.0, left = 0.05, right = 0.95, 
                       top = 1 - 0.2 / len(genes) , bottom = 0.05)
    
    #define dataset
    
    dataset = dataset.ix[genes, cell_groups.index]
    
    #define max group ID for color definition
    
    val_max = float(np.max(cell_groups.values))
    
    #draw genes
    
    for ix, g in enumerate(genes):
    
        ax = plt.subplot(gs1[ix])
        ax.set_xlim(left = 0, right = (len(dataset.columns)))
                     
        if g != genes[-1]:
            ax.xaxis.set_ticklabels([])
        
        elif g == genes[-1]:
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
                
        ax.set_ylabel(g, fontsize = 25)
        ax.yaxis.labelpad = 10
        
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
            tick.label.set_label(10)
            
        ax.bar(np.arange(0, len(dataset.columns),1), 
               dataset.ix[g],
               width=1,
               color=[cmap(val/val_max) for val in cell_groups.values],
               linewidth=0)
    
    #create groups bar
    
    ax_bar = plt.subplot(gs0[0])
    
    ax_bar.set_xlim(left = 0, right = (len(dataset.columns)))
    ax_bar.set_ylim(bottom = 0, top = 1)
    
    for ix, val in enumerate(cell_groups.values):
        
        ax_bar.axvspan(ix,
                       ix+1,
                       ymin=0,
                       ymax=1, 
                       color = cmap(val / val_max))
    
    remove_ticks(ax_bar)
    
################################################################################
############################### MISCELLANEOUS ##################################
################################################################################

def create_gene_list_for_IPA(dataset, groups):
    
    """
    Returns a binarized gene list as input for IPA.
    ----------
    dataset: DataFrame of m cells x n genes.
    groups: Series of gene group IDs.
    ----------
    """
    
    output = pd.DataFrame(0, index = dataset.index, columns = return_unique(groups))
    
    for gr in return_unique(groups):
        
        genes_tmp = groups[groups==gr].index
        
        output.ix[genes_tmp, gr] = 1
        
    return output
    
################################################################################
############################## HELPER FUNCTIONS ################################
################################################################################

def chunks(l, n):
    """ 
    Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

################################################################################

def remove_ticks(axes, linewidth = 0.5):
    """
    Removes ticks from matplotlib Axes instance
    """
    axes.set_xticklabels([]), axes.set_yticklabels([])
    axes.set_xticks([]), axes.set_yticks([])
    for axis in ['top','bottom','left','right']:
        axes.spines[axis].set_linewidth(linewidth)

################################################################################

def return_unique(groups, drop_zero = False):
    """
    Returns unique instances from a list (e.g. an AP cluster Series) in order 
    of appearance.
    """
    unique = []
    
    for element in groups.values:
        if element not in unique:
            unique.append(element)
            
    if drop_zero == True:
        unique.remove(0)
        
    return unique

################################################################################

def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

################################################################################

################################################################################

def get_viridis():
    
    """
    Compiles viridis as matplotlib cmap
    """
    c_viridis = [[ 0.26700401,  0.00487433,  0.32941519],
       [ 0.26851048,  0.00960483,  0.33542652],
       [ 0.26994384,  0.01462494,  0.34137895],
       [ 0.27130489,  0.01994186,  0.34726862],
       [ 0.27259384,  0.02556309,  0.35309303],
       [ 0.27380934,  0.03149748,  0.35885256],
       [ 0.27495242,  0.03775181,  0.36454323],
       [ 0.27602238,  0.04416723,  0.37016418],
       [ 0.2770184 ,  0.05034437,  0.37571452],
       [ 0.27794143,  0.05632444,  0.38119074],
       [ 0.27879067,  0.06214536,  0.38659204],
       [ 0.2795655 ,  0.06783587,  0.39191723],
       [ 0.28026658,  0.07341724,  0.39716349],
       [ 0.28089358,  0.07890703,  0.40232944],
       [ 0.28144581,  0.0843197 ,  0.40741404],
       [ 0.28192358,  0.08966622,  0.41241521],
       [ 0.28232739,  0.09495545,  0.41733086],
       [ 0.28265633,  0.10019576,  0.42216032],
       [ 0.28291049,  0.10539345,  0.42690202],
       [ 0.28309095,  0.11055307,  0.43155375],
       [ 0.28319704,  0.11567966,  0.43611482],
       [ 0.28322882,  0.12077701,  0.44058404],
       [ 0.28318684,  0.12584799,  0.44496   ],
       [ 0.283072  ,  0.13089477,  0.44924127],
       [ 0.28288389,  0.13592005,  0.45342734],
       [ 0.28262297,  0.14092556,  0.45751726],
       [ 0.28229037,  0.14591233,  0.46150995],
       [ 0.28188676,  0.15088147,  0.46540474],
       [ 0.28141228,  0.15583425,  0.46920128],
       [ 0.28086773,  0.16077132,  0.47289909],
       [ 0.28025468,  0.16569272,  0.47649762],
       [ 0.27957399,  0.17059884,  0.47999675],
       [ 0.27882618,  0.1754902 ,  0.48339654],
       [ 0.27801236,  0.18036684,  0.48669702],
       [ 0.27713437,  0.18522836,  0.48989831],
       [ 0.27619376,  0.19007447,  0.49300074],
       [ 0.27519116,  0.1949054 ,  0.49600488],
       [ 0.27412802,  0.19972086,  0.49891131],
       [ 0.27300596,  0.20452049,  0.50172076],
       [ 0.27182812,  0.20930306,  0.50443413],
       [ 0.27059473,  0.21406899,  0.50705243],
       [ 0.26930756,  0.21881782,  0.50957678],
       [ 0.26796846,  0.22354911,  0.5120084 ],
       [ 0.26657984,  0.2282621 ,  0.5143487 ],
       [ 0.2651445 ,  0.23295593,  0.5165993 ],
       [ 0.2636632 ,  0.23763078,  0.51876163],
       [ 0.26213801,  0.24228619,  0.52083736],
       [ 0.26057103,  0.2469217 ,  0.52282822],
       [ 0.25896451,  0.25153685,  0.52473609],
       [ 0.25732244,  0.2561304 ,  0.52656332],
       [ 0.25564519,  0.26070284,  0.52831152],
       [ 0.25393498,  0.26525384,  0.52998273],
       [ 0.25219404,  0.26978306,  0.53157905],
       [ 0.25042462,  0.27429024,  0.53310261],
       [ 0.24862899,  0.27877509,  0.53455561],
       [ 0.2468114 ,  0.28323662,  0.53594093],
       [ 0.24497208,  0.28767547,  0.53726018],
       [ 0.24311324,  0.29209154,  0.53851561],
       [ 0.24123708,  0.29648471,  0.53970946],
       [ 0.23934575,  0.30085494,  0.54084398],
       [ 0.23744138,  0.30520222,  0.5419214 ],
       [ 0.23552606,  0.30952657,  0.54294396],
       [ 0.23360277,  0.31382773,  0.54391424],
       [ 0.2316735 ,  0.3181058 ,  0.54483444],
       [ 0.22973926,  0.32236127,  0.54570633],
       [ 0.22780192,  0.32659432,  0.546532  ],
       [ 0.2258633 ,  0.33080515,  0.54731353],
       [ 0.22392515,  0.334994  ,  0.54805291],
       [ 0.22198915,  0.33916114,  0.54875211],
       [ 0.22005691,  0.34330688,  0.54941304],
       [ 0.21812995,  0.34743154,  0.55003755],
       [ 0.21620971,  0.35153548,  0.55062743],
       [ 0.21429757,  0.35561907,  0.5511844 ],
       [ 0.21239477,  0.35968273,  0.55171011],
       [ 0.2105031 ,  0.36372671,  0.55220646],
       [ 0.20862342,  0.36775151,  0.55267486],
       [ 0.20675628,  0.37175775,  0.55311653],
       [ 0.20490257,  0.37574589,  0.55353282],
       [ 0.20306309,  0.37971644,  0.55392505],
       [ 0.20123854,  0.38366989,  0.55429441],
       [ 0.1994295 ,  0.38760678,  0.55464205],
       [ 0.1976365 ,  0.39152762,  0.55496905],
       [ 0.19585993,  0.39543297,  0.55527637],
       [ 0.19410009,  0.39932336,  0.55556494],
       [ 0.19235719,  0.40319934,  0.55583559],
       [ 0.19063135,  0.40706148,  0.55608907],
       [ 0.18892259,  0.41091033,  0.55632606],
       [ 0.18723083,  0.41474645,  0.55654717],
       [ 0.18555593,  0.4185704 ,  0.55675292],
       [ 0.18389763,  0.42238275,  0.55694377],
       [ 0.18225561,  0.42618405,  0.5571201 ],
       [ 0.18062949,  0.42997486,  0.55728221],
       [ 0.17901879,  0.43375572,  0.55743035],
       [ 0.17742298,  0.4375272 ,  0.55756466],
       [ 0.17584148,  0.44128981,  0.55768526],
       [ 0.17427363,  0.4450441 ,  0.55779216],
       [ 0.17271876,  0.4487906 ,  0.55788532],
       [ 0.17117615,  0.4525298 ,  0.55796464],
       [ 0.16964573,  0.45626209,  0.55803034],
       [ 0.16812641,  0.45998802,  0.55808199],
       [ 0.1666171 ,  0.46370813,  0.55811913],
       [ 0.16511703,  0.4674229 ,  0.55814141],
       [ 0.16362543,  0.47113278,  0.55814842],
       [ 0.16214155,  0.47483821,  0.55813967],
       [ 0.16066467,  0.47853961,  0.55811466],
       [ 0.15919413,  0.4822374 ,  0.5580728 ],
       [ 0.15772933,  0.48593197,  0.55801347],
       [ 0.15626973,  0.4896237 ,  0.557936  ],
       [ 0.15481488,  0.49331293,  0.55783967],
       [ 0.15336445,  0.49700003,  0.55772371],
       [ 0.1519182 ,  0.50068529,  0.55758733],
       [ 0.15047605,  0.50436904,  0.55742968],
       [ 0.14903918,  0.50805136,  0.5572505 ],
       [ 0.14760731,  0.51173263,  0.55704861],
       [ 0.14618026,  0.51541316,  0.55682271],
       [ 0.14475863,  0.51909319,  0.55657181],
       [ 0.14334327,  0.52277292,  0.55629491],
       [ 0.14193527,  0.52645254,  0.55599097],
       [ 0.14053599,  0.53013219,  0.55565893],
       [ 0.13914708,  0.53381201,  0.55529773],
       [ 0.13777048,  0.53749213,  0.55490625],
       [ 0.1364085 ,  0.54117264,  0.55448339],
       [ 0.13506561,  0.54485335,  0.55402906],
       [ 0.13374299,  0.54853458,  0.55354108],
       [ 0.13244401,  0.55221637,  0.55301828],
       [ 0.13117249,  0.55589872,  0.55245948],
       [ 0.1299327 ,  0.55958162,  0.55186354],
       [ 0.12872938,  0.56326503,  0.55122927],
       [ 0.12756771,  0.56694891,  0.55055551],
       [ 0.12645338,  0.57063316,  0.5498411 ],
       [ 0.12539383,  0.57431754,  0.54908564],
       [ 0.12439474,  0.57800205,  0.5482874 ],
       [ 0.12346281,  0.58168661,  0.54744498],
       [ 0.12260562,  0.58537105,  0.54655722],
       [ 0.12183122,  0.58905521,  0.54562298],
       [ 0.12114807,  0.59273889,  0.54464114],
       [ 0.12056501,  0.59642187,  0.54361058],
       [ 0.12009154,  0.60010387,  0.54253043],
       [ 0.11973756,  0.60378459,  0.54139999],
       [ 0.11951163,  0.60746388,  0.54021751],
       [ 0.11942341,  0.61114146,  0.53898192],
       [ 0.11948255,  0.61481702,  0.53769219],
       [ 0.11969858,  0.61849025,  0.53634733],
       [ 0.12008079,  0.62216081,  0.53494633],
       [ 0.12063824,  0.62582833,  0.53348834],
       [ 0.12137972,  0.62949242,  0.53197275],
       [ 0.12231244,  0.63315277,  0.53039808],
       [ 0.12344358,  0.63680899,  0.52876343],
       [ 0.12477953,  0.64046069,  0.52706792],
       [ 0.12632581,  0.64410744,  0.52531069],
       [ 0.12808703,  0.64774881,  0.52349092],
       [ 0.13006688,  0.65138436,  0.52160791],
       [ 0.13226797,  0.65501363,  0.51966086],
       [ 0.13469183,  0.65863619,  0.5176488 ],
       [ 0.13733921,  0.66225157,  0.51557101],
       [ 0.14020991,  0.66585927,  0.5134268 ],
       [ 0.14330291,  0.66945881,  0.51121549],
       [ 0.1466164 ,  0.67304968,  0.50893644],
       [ 0.15014782,  0.67663139,  0.5065889 ],
       [ 0.15389405,  0.68020343,  0.50417217],
       [ 0.15785146,  0.68376525,  0.50168574],
       [ 0.16201598,  0.68731632,  0.49912906],
       [ 0.1663832 ,  0.69085611,  0.49650163],
       [ 0.1709484 ,  0.69438405,  0.49380294],
       [ 0.17570671,  0.6978996 ,  0.49103252],
       [ 0.18065314,  0.70140222,  0.48818938],
       [ 0.18578266,  0.70489133,  0.48527326],
       [ 0.19109018,  0.70836635,  0.48228395],
       [ 0.19657063,  0.71182668,  0.47922108],
       [ 0.20221902,  0.71527175,  0.47608431],
       [ 0.20803045,  0.71870095,  0.4728733 ],
       [ 0.21400015,  0.72211371,  0.46958774],
       [ 0.22012381,  0.72550945,  0.46622638],
       [ 0.2263969 ,  0.72888753,  0.46278934],
       [ 0.23281498,  0.73224735,  0.45927675],
       [ 0.2393739 ,  0.73558828,  0.45568838],
       [ 0.24606968,  0.73890972,  0.45202405],
       [ 0.25289851,  0.74221104,  0.44828355],
       [ 0.25985676,  0.74549162,  0.44446673],
       [ 0.26694127,  0.74875084,  0.44057284],
       [ 0.27414922,  0.75198807,  0.4366009 ],
       [ 0.28147681,  0.75520266,  0.43255207],
       [ 0.28892102,  0.75839399,  0.42842626],
       [ 0.29647899,  0.76156142,  0.42422341],
       [ 0.30414796,  0.76470433,  0.41994346],
       [ 0.31192534,  0.76782207,  0.41558638],
       [ 0.3198086 ,  0.77091403,  0.41115215],
       [ 0.3277958 ,  0.77397953,  0.40664011],
       [ 0.33588539,  0.7770179 ,  0.40204917],
       [ 0.34407411,  0.78002855,  0.39738103],
       [ 0.35235985,  0.78301086,  0.39263579],
       [ 0.36074053,  0.78596419,  0.38781353],
       [ 0.3692142 ,  0.78888793,  0.38291438],
       [ 0.37777892,  0.79178146,  0.3779385 ],
       [ 0.38643282,  0.79464415,  0.37288606],
       [ 0.39517408,  0.79747541,  0.36775726],
       [ 0.40400101,  0.80027461,  0.36255223],
       [ 0.4129135 ,  0.80304099,  0.35726893],
       [ 0.42190813,  0.80577412,  0.35191009],
       [ 0.43098317,  0.80847343,  0.34647607],
       [ 0.44013691,  0.81113836,  0.3409673 ],
       [ 0.44936763,  0.81376835,  0.33538426],
       [ 0.45867362,  0.81636288,  0.32972749],
       [ 0.46805314,  0.81892143,  0.32399761],
       [ 0.47750446,  0.82144351,  0.31819529],
       [ 0.4870258 ,  0.82392862,  0.31232133],
       [ 0.49661536,  0.82637633,  0.30637661],
       [ 0.5062713 ,  0.82878621,  0.30036211],
       [ 0.51599182,  0.83115784,  0.29427888],
       [ 0.52577622,  0.83349064,  0.2881265 ],
       [ 0.5356211 ,  0.83578452,  0.28190832],
       [ 0.5455244 ,  0.83803918,  0.27562602],
       [ 0.55548397,  0.84025437,  0.26928147],
       [ 0.5654976 ,  0.8424299 ,  0.26287683],
       [ 0.57556297,  0.84456561,  0.25641457],
       [ 0.58567772,  0.84666139,  0.24989748],
       [ 0.59583934,  0.84871722,  0.24332878],
       [ 0.60604528,  0.8507331 ,  0.23671214],
       [ 0.61629283,  0.85270912,  0.23005179],
       [ 0.62657923,  0.85464543,  0.22335258],
       [ 0.63690157,  0.85654226,  0.21662012],
       [ 0.64725685,  0.85839991,  0.20986086],
       [ 0.65764197,  0.86021878,  0.20308229],
       [ 0.66805369,  0.86199932,  0.19629307],
       [ 0.67848868,  0.86374211,  0.18950326],
       [ 0.68894351,  0.86544779,  0.18272455],
       [ 0.69941463,  0.86711711,  0.17597055],
       [ 0.70989842,  0.86875092,  0.16925712],
       [ 0.72039115,  0.87035015,  0.16260273],
       [ 0.73088902,  0.87191584,  0.15602894],
       [ 0.74138803,  0.87344918,  0.14956101],
       [ 0.75188414,  0.87495143,  0.14322828],
       [ 0.76237342,  0.87642392,  0.13706449],
       [ 0.77285183,  0.87786808,  0.13110864],
       [ 0.78331535,  0.87928545,  0.12540538],
       [ 0.79375994,  0.88067763,  0.12000532],
       [ 0.80418159,  0.88204632,  0.11496505],
       [ 0.81457634,  0.88339329,  0.11034678],
       [ 0.82494028,  0.88472036,  0.10621724],
       [ 0.83526959,  0.88602943,  0.1026459 ],
       [ 0.84556056,  0.88732243,  0.09970219],
       [ 0.8558096 ,  0.88860134,  0.09745186],
       [ 0.86601325,  0.88986815,  0.09595277],
       [ 0.87616824,  0.89112487,  0.09525046],
       [ 0.88627146,  0.89237353,  0.09537439],
       [ 0.89632002,  0.89361614,  0.09633538],
       [ 0.90631121,  0.89485467,  0.09812496],
       [ 0.91624212,  0.89609127,  0.1007168 ],
       [ 0.92610579,  0.89732977,  0.10407067],
       [ 0.93590444,  0.8985704 ,  0.10813094],
       [ 0.94563626,  0.899815  ,  0.11283773],
       [ 0.95529972,  0.90106534,  0.11812832],
       [ 0.96489353,  0.90232311,  0.12394051],
       [ 0.97441665,  0.90358991,  0.13021494],
       [ 0.98386829,  0.90486726,  0.13689671],
       [ 0.99324789,  0.90615657,  0.1439362 ]]
       
    cm_viridis = mpl.colors.ListedColormap(c_viridis)
       
    return cm_viridis