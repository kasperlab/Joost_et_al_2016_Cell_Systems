################################################################################
########## STRT-EPIDERMIS -- BAYESIAN NEGATIVE BINOMINAL REGRESSION ############
################################################################################

"""
Scripts for Bayesian negative binominal regression and interpretation/inference
from the posterior probabilities (traces).
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import numpy as np
import pandas as pd
import itertools
from collections import Counter
import pystan

from EPI_misc_scripts_v1_1 import *

################################################################################
############################## FEATURE SELECTION ###############################
################################################################################

def NBR_select_genes_mean(dataset, sample_groups, cutoff = 0.1):
    
    """
    Selects genes for negative binominal regression based on group-wise mean expression. A gene is
    selected if its expression exceeds a cutoff in at least on group
    ----------
    dataset: [pd.DataFrame] of m cells / samples x n genes.
    sample_groups: [pd.Series] containing the group membership of m cells.
    cutoff: mean expression at which a gene is included.
    ----------
    returns list of genes to be used for NBR
    """
    
    #get mean expression of genes in each group
    
    group_mean = pd.DataFrame(index = dataset.index, columns = set(sample_groups))
    
    for gr in set(sample_groups):
        
        ix_tmp = sample_groups[sample_groups == gr].index
        
        group_mean[gr] = dataset[ix_tmp].mean(axis=1)
        
    #return genes with expression >= cutoff
    
    return list(group_mean.max(axis=1)[group_mean.max(axis=1)>=cutoff].index)

################################################################################
############################### MODEL FITTING ##################################
################################################################################

def neg_binom_regression_v2P(dataset, cell_groups, genes, path, id_, name, chunk_size, dview, iter=275, chains=4, 
                            warmup=25, n_jobs=4):
    
    """
    Models group-specific gene expression using Bayesian negative binomial generalized linear regression.
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes. Must be untransformed and contain all genes (for Baseline)!
    cell_groups: [pd.Series] specifying group/cluster membership of m x cells in dataset.
    genes: [list] of genes to be fitted.
    path: path to output file.
    id_: experiment ID for output file
    name: name for output file. Format will be: path/ID_name
    chunk_size: number of genes to be fitted in parallel.
    dview: name of Ipython DirectView Instance for parallel computing.
    iter: number of MCMC iterations to perform in each chain. Default: 275.*
    chains: number of MCM chains. Default: 4.*
    warmup: number of warmup/burnin iterations. Not used for inference. Default: 25.*
    n_jobs: number of engines used for MCMC of single gene. Default: 4.
    
    * Default parameters yield 4 * 275 - 4 * 25 = 1000 traces per group per gene. 
    ----------
    returns DataFrame containing traces of the modeled coefficient 'beta' for each gene and each group (+ Baseline)
    """
    
    #define model
    
    model_code = """
    data {
        int<lower=0> N; # number of outcomes
        int<lower=0> K; # number of predictors
        matrix<lower=0>[N,K] x; # predictor matrix
        int y[N]; # outcomes
    }

    parameters {
        vector<lower=1>[K] beta; # coefficients
        real<lower=0.001> r; # overdispersion
    }

    model {
        vector[N] mu;
        vector[N] rv; 
        
        # priors
        r ~ cauchy(0,1);
        beta ~ pareto(1,1.5);

        # vectorize the overdispersion
        for (n in 1:N) {
                rv[n] <- square(r + 1) - 1;
        }

        #regression
        mu <- x * (beta - 1) + 0.001;
        y ~ neg_binomial(mu ./ rv, 1 / rv[1]);
    }
    """
    
    #number of outcomes
    
    N = len(cell_groups)
    
    #number of predictors
    
    K = (1 + len(set(cell_groups)))
    
    #predictor matrix 
    
    NxK = pd.DataFrame(index = list(set(cell_groups)) + ['Baseline'], columns = cell_groups.index).fillna(0)
    
    NxK.ix['Baseline'] = dataset[NxK.columns].sum(axis = 0) / dataset[NxK.columns].sum(axis = 0).mean()

    for group in set(cell_groups):
        
        NxK.ix[group, cell_groups[cell_groups == group].index] = 1
        
    #run model
    
    traces = neg_binom_regression_free_input_v2P(model_code, dataset, genes, N, K, NxK, path, id_, name, chunk_size, dview, iter=275, chains=4, warmup=25, n_jobs=4)
        
    return traces
    
################################################################################

def neg_binom_regression_free_input_v2P(model_code, dataset, genes, N, K, NxK, path, id_, name, chunk_size, dview, iter=275, chains=4, 
                            warmup=25, n_jobs=4):
                            
    """
    Models gene expression using Bayesian negative binomial generalized linear regression with free input of parameters.
    ----------
    model_code: code of model to be used in Stan language.
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes. Must be untransformed and contain all genes (for Baseline)!
    genes: [list] of genes to be fitted.
    N: number [int] of outcomes.
    K: number [int] of predictors.
    NxK: [pd.DataFrame] containing predictors over outcomes.
    path: path to output file.
    id_: experiment ID for output file
    name: name for output file. Format will be: path/ID_name
    chunk_size: number of genes to be fitted in parallel.
    dview: name of Ipython DirectView Instance for parallel computing.
    iter: number of MCMC iterations to perform in each chain. Default: 275.*
    chains: number of MCM chains. Default: 4.*
    warmup: number of warmup/burnin iterations. Not used for inference. Default: 25.*
    n_jobs: number of engines used for MCMC of single gene. Default: 4.
    
    * Default parameters yield 4 * 275 - 4 * 25 = 1000 traces per group per gene. 
    ----------
    returns DataFrame containing traces of the modeled coefficient 'beta' for each gene and each group (+ Baseline)
    """
    
    #########################
    
    def neg_binom_fit_gene_free_input_v2(model, dataset, gene, N, K, NxK, iter, chains, warmup, n_jobs):
    
        """
        Helper function for neg_binom_regression_v1P
        """

        #bring data into right format

        data_tmp = {'N': N,'K': K,'x': np.matrix(NxK.T),'y': [int(val) for val in dataset.ix[gene, NxK.columns]]}

        #fit model

        fit = model.sampling(data=data_tmp, iter=iter, chains=chains, warmup=warmup, n_jobs=n_jobs)
        
        #extract posterior

        post_tmp = fit.extract()

        #combine all traces in pd.DataFrame

        traces = pd.DataFrame(index = [gene], columns = [str(x) for x in NxK.index])

        for ix, group in enumerate(NxK.index):

            traces.ix[gene,str(group)] = post_tmp['beta'][:,ix]

        return traces

    #########################
        
    #establish or load DataFrame containing traces
    
    try:
        traces = loadData_from_pickle_v1(path, id_, name)
        
    except:
        traces = pd.DataFrame(columns = [str(x) for x in NxK.index])
        
    #establish total gene count (before gene list is changed based on already modeled genes)
    
    genes_all = len(genes)
            
    #remove genes which have already been modeled from list of genes
    
    genes = list(set(genes).difference(set(traces.index)))
    
    #initialize counter
    
    print "%s / %s" % (len(traces.index), genes_all)
    
    #initialize pystan model
    
    model = pystan.StanModel(model_code=model_code)
    
    #iterate through chunks
        
    for gene_chunk in chunks(genes, chunk_size):
        
        l = len(gene_chunk)
        
        #fit in parallel
        
        traces_tmp = dview.map_sync(neg_binom_fit_gene_free_input_v2,
                                     [model] * l, 
                                     [dataset] * l, 
                                     gene_chunk, 
                                     [N] * l, 
                                     [K] * l, 
                                     [NxK] * l, 
                                     [iter] * l, 
                                     [chains] * l, 
                                     [warmup] * l,
                                     [n_jobs] * l)
                        
        #append traces DataFrame
        
        traces = traces.append(traces_tmp)
        
        #update counter
        
        print "%s / %s" % (len(traces.index), genes_all)
        
        #save progress
        
        saveData_to_pickle_v1(traces, path, id_, name)
                        
    saveData_to_pickle_v1(traces, path, id_, name)
        
    return traces
    
################################################################################
################################## ANALYSIS ####################################
################################################################################

def neg_binom_summary_stats_v1P(traces, dview):
    
    """
    Calculates the summary statistics of the coefficient 'beta' of n genes in m groups modeled with an 
    Bayesian negative binominal regression model.
    ----------
    traces: DataFrame containing the traces [list] of coefficient 'beta' in m groups over n genes.
    dview: name of Ipython DirectView Instance for parallel computing.
    ----------
    returns DataFrame containing the summary statistics [dict] of 'beta' in m groups over n genes.
    Summary statistics include: 'mean','median','std', 'max', 'min', 'Q5', 'Q25', 'Q75', 'Q95'.
    """
    
    ##########################################################
    
    def neg_binom_get_summary(gene, traces):
        """
        Helper function for neg_binom_summary_stats_v1P.
        """
    
        #define output file

        multi_ix = [np.array([gene,gene,gene,gene,gene,gene,gene,gene,gene]),
                    np.array(['mean','median','std','min','max','Q5','Q25','Q75','Q95'])]

        output_tmp = pd.DataFrame(index = multi_ix, columns = traces.index)

        #iterate through groups and calculate summary statistics

        for gr in traces.index:

            data_tmp = traces.ix[gr]

            output_tmp.ix[gene,'mean'][gr] = np.mean(data_tmp)
            output_tmp.ix[gene,'median'][gr] = np.median(data_tmp)
            output_tmp.ix[gene,'std'][gr] = np.std(data_tmp)
            output_tmp.ix[gene,'max'][gr] = np.max(data_tmp)
            output_tmp.ix[gene,'min'][gr] = np.min(data_tmp)
            output_tmp.ix[gene,'Q5'][gr] = np.percentile(data_tmp, 5)
            output_tmp.ix[gene,'Q25'][gr] = np.percentile(data_tmp, 25)
            output_tmp.ix[gene,'Q75'][gr] = np.percentile(data_tmp, 75)
            output_tmp.ix[gene,'Q95'][gr] = np.percentile(data_tmp, 95)

        return output_tmp

    ##########################################################
    
    #calculate gene wise summary statistics in parallel
    
    output_tmp = dview.map_sync(neg_binom_get_summary,
                                traces.index,
                                [traces.ix[gene] for gene in traces.index])
    
    #fuse data and return
        
    output = pd.concat(output_tmp, axis = 0)
    
    return output
    
################################################################################

def neg_binom_vs_zero_v1P(summary, dview, PP='99.9', zero = 1.05):
    
    """
    For each group, tests the posterior probability of 'beta' against zero and determines that a gene is additionally 
    expressed in a group if 'beta' exceeds zero with a posterior probability of either 99.9% or 95%. For each active gene,
    returns different measures of the effect size - i.e. the number of molecules additionally expressed within a group.
    ----------
    summary: DataFrame with summary data of the negative binominal regression traces.
    dview: name of Ipython DirectView Instance for parallel computing.
    PP: posterior probability cutoff. '99.9' or '95'. Default: '99.9'
    zero: absolute zero value. Default: 1.05.
    ----------
    returns 'bin' - a DataFrame indicating whether genes are considered 'on'/1 or 'off' in a population - and 'size' -
    a DataFrame with three measures of effect sizes.
    """
    ##########################################################
        
    def neg_binom_test_vs_zero(gene, summary, PP, zero):
        
        """
        Helper function for neg_binom_test_vs_zero_v1P.
        """
        
        #define output files
        
        bin_tmp = pd.DataFrame(index = [gene], columns = summary.columns)
        
        multi_ix = [np.array([gene,gene,gene,gene,gene]),
                    np.array(['mean','median','min','percentile - 25','percentile - 5'])]
        
        size_tmp = pd.DataFrame(index = multi_ix, columns = summary.columns)
        
        #iterate through groups and score
        
        for gr in summary.columns:
            
            if PP == '99.9':
            
                if summary.ix['min', gr] > zero:
                
                    bin_tmp.ix[gene, gr] = 1
                                                                        
                    size_tmp.ix[gene, 'mean'][gr] = summary.ix['mean', gr] - zero
                    size_tmp.ix[gene, 'median'][gr] = summary.ix['median', gr] - zero
                    size_tmp.ix[gene, 'percentile - 25'][gr] = summary.ix['Q25', gr] - zero
                    size_tmp.ix[gene, 'percentile - 5'][gr] = summary.ix['Q5', gr] - zero
                    size_tmp.ix[gene, 'min'][gr] = summary.ix['min', gr] - zero
                
                else:
                
                    bin_tmp.ix[gene, gr] = 0
                
                    size_tmp.ix[gene, 'mean'][gr] = 'n.s'
                    size_tmp.ix[gene, 'median'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 25'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 5'][gr] = 'n.s'
                    size_tmp.ix[gene, 'min'][gr] = 'n.s'
                    
            elif PP == '95':
                
                if summary.ix['Q5', gr] > zero:
                    
                    bin_tmp.ix[gene, gr] = 1
                    
                    size_tmp.ix[gene, 'mean'][gr] = summary.ix['mean', gr] - zero
                    size_tmp.ix[gene, 'median'][gr] = summary.ix['median', gr] - zero
                    size_tmp.ix[gene, 'percentile - 25'][gr] = summary.ix['Q25', gr] - zero
                    size_tmp.ix[gene, 'percentile - 5'][gr] = summary.ix['Q5', gr] - zero
                    size_tmp.ix[gene, 'min'][gr] = summary.ix['min', gr] - zero
                
                else:
                
                    bin_tmp.ix[gene, gr] = 0
                
                    size_tmp.ix[gene, 'mean'][gr] = 'n.s'
                    size_tmp.ix[gene, 'median'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 25'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 5'][gr] = 'n.s'
                    size_tmp.ix[gene, 'min'][gr] = 'n.s'
                    
                
        return bin_tmp, size_tmp
    
    ##########################################################
    
    #perform test against zero in parallel
    
    output_tmp = dview.map_sync(neg_binom_test_vs_zero,
                                [gene for gene in summary.index.levels[0]],
                                [summary.ix[gene] for gene in summary.index.levels[0]],
                                [PP for gene in summary.index.levels[0]],
                                [zero for gene in summary.index.levels[0]])
    
    #fuse data and return
        
    bin_ = pd.concat([x[0] for x in output_tmp], axis = 0)
    size = pd.concat([x[1] for x in output_tmp], axis = 0)
    
    return bin_, size

################################################################################

def neg_binom_vs_baseline_v1P(summary, dview, PP='99.9'):
    
    """
    For each group, tests the posterior probability of 'beta' against the baseline and calls a gene active in a group
    if 'beta' exceeds the baseline with a specified posterior probability (99.9%, 95% or 90%). For each active gene, 
    returns different measures of the effect size - i.e. the number of molecules additionally expressed within a group compared to the baseline.
    ----------
    summary: DataFrame with summary data of the negative binominal regression traces.
    dview: name of Ipython DirectView Instance for parallel computing.
    PP: posterior probability cutoff. '99.9', '95' or '90'. Default: '99.9'
    ----------
    returns 'bin' - a DataFrame indicating whether genes are considered 'on'/1 or 'off' in a population - and 'size' -
    a DataFrame with three measures of effect sizes.
    """
    ##########################################################
        
    def neg_binom_test_vs_baseline(gene, summary, PP):
        
        """
        Helper function for neg_binom_test_vs_baseline_v1P.
        """
        
        #define output files
        
        bin_tmp = pd.DataFrame(index = [gene], columns = summary.columns[:-1])
        
        multi_ix = [np.array([gene,gene,gene,gene,gene]),
                    np.array(['mean','median','min','percentile - 25','percentile - 5'])]
        
        size_tmp = pd.DataFrame(index = multi_ix, columns = summary.columns[:-1])
        
        #iterate through groups and score
        
        for gr in summary.columns[:-1]:
            
            if PP == '99.9':
            
                if summary.ix['min', gr] > summary.ix['max', 'Baseline']:
                
                    bin_tmp.ix[gene, gr] = 1
                                                                        
                    size_tmp.ix[gene, 'mean'][gr] = summary.ix['mean', gr] - summary.ix['mean', 'Baseline']
                    size_tmp.ix[gene, 'median'][gr] = summary.ix['median', gr] - summary.ix['median', 'Baseline']
                    size_tmp.ix[gene, 'percentile - 25'][gr] = summary.ix['Q25', gr] - summary.ix['Q75', 'Baseline']
                    size_tmp.ix[gene, 'percentile - 5'][gr] = summary.ix['Q5', gr] - summary.ix['Q95', 'Baseline']
                    size_tmp.ix[gene, 'min'][gr] = summary.ix['min', gr] - summary.ix['max', 'Baseline']
                
                else:
                
                    bin_tmp.ix[gene, gr] = 0
                
                    size_tmp.ix[gene, 'mean'][gr] = 'n.s'
                    size_tmp.ix[gene, 'median'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 25'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 5'][gr] = 'n.s'
                    size_tmp.ix[gene, 'min'][gr] = 'n.s'
                    
            elif PP == '95':
                
                if summary.ix['min', gr] > summary.ix['Q95', 'Baseline'] or summary.ix['Q5', gr] > summary.ix['max', 'Baseline']:
                    
                    bin_tmp.ix[gene, gr] = 1
                    
                    size_tmp.ix[gene, 'mean'][gr] = summary.ix['mean', gr] - summary.ix['mean', 'Baseline']
                    size_tmp.ix[gene, 'median'][gr] = summary.ix['median', gr] - summary.ix['median', 'Baseline']
                    size_tmp.ix[gene, 'percentile - 25'][gr] = summary.ix['Q25', gr] - summary.ix['Q75', 'Baseline']
                    size_tmp.ix[gene, 'percentile - 5'][gr] = summary.ix['Q5', gr] - summary.ix['Q95', 'Baseline']
                    size_tmp.ix[gene, 'min'][gr] = summary.ix['min', gr] - summary.ix['max', 'Baseline']
                
                else:
                
                    bin_tmp.ix[gene, gr] = 0
                
                    size_tmp.ix[gene, 'mean'][gr] = 'n.s'
                    size_tmp.ix[gene, 'median'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 25'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 5'][gr] = 'n.s'
                    size_tmp.ix[gene, 'min'][gr] = 'n.s'
                    
            elif PP == '90':
                
                if summary.ix['Q5', gr] > summary.ix['Q95', 'Baseline']:
                    
                    bin_tmp.ix[gene, gr] = 1
                    
                    size_tmp.ix[gene, 'mean'][gr] = summary.ix['mean', gr] - summary.ix['mean', 'Baseline']
                    size_tmp.ix[gene, 'median'][gr] = summary.ix['median', gr] - summary.ix['median', 'Baseline']
                    size_tmp.ix[gene, 'percentile - 25'][gr] = summary.ix['Q25', gr] - summary.ix['Q75', 'Baseline']
                    size_tmp.ix[gene, 'percentile - 5'][gr] = summary.ix['Q5', gr] - summary.ix['Q95', 'Baseline']
                    size_tmp.ix[gene, 'min'][gr] = summary.ix['min', gr] - summary.ix['max', 'Baseline']
                
                else:
                
                    bin_tmp.ix[gene, gr] = 0
                
                    size_tmp.ix[gene, 'mean'][gr] = 'n.s'
                    size_tmp.ix[gene, 'median'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 25'][gr] = 'n.s'
                    size_tmp.ix[gene, 'percentile - 5'][gr] = 'n.s'
                    size_tmp.ix[gene, 'min'][gr] = 'n.s'
                    
                
        return bin_tmp, size_tmp
    
    ##########################################################
    
    #perform test against baseline in parallel
    
    output_tmp = dview.map_sync(neg_binom_test_vs_baseline,
                                [gene for gene in summary.index.levels[0]],
                                [summary.ix[gene] for gene in summary.index.levels[0]],
                                [PP for gene in summary.index.levels[0]])
    
    #fuse data and return
        
    bin_ = pd.concat([x[0] for x in output_tmp], axis = 0)
    size = pd.concat([x[1] for x in output_tmp], axis = 0)
    
    return bin_, size

################################################################################

def neg_binom_vs_groups_v1P(summary, dview, PP='99.9'):
    
    """
    Test whether a gene qualifies as a unique marker for a group based on negative binominal regression modelling. 
    A gene n is considered a unique marker for group G, if its coefficient 'beta'(G) is above 'beta'(baseline) with 
    a specified posterior probability (99.9%, 95% or 90%) and if 'beta'(G) is above the coefficient 'beta' of the second highest expressing 
    group with a specified posterior probability. To achieve this, for each gene, the highest 'beta' is compared to the
    second highest 'beta'. If the highest 'beta' is not the baseline value and if the highest 'beta' is above the second
    'highest' with a specified posterior probability, the group with the highest 'beta' is set to 1 in the 'bin' file and several
    measures of the effect size between the highest and second highest expressing groups are returned.
    ----------
    summary: DataFrame with summary data of the negative binominal regression traces.
    dview: name of Ipython DirectView Instance for parallel computing.
    PP: posterior probability cutoff. '99.9', '95' or '90. Default: '99.9'
    ----------
    returns 'bin' - a DataFrame indicating whether genes are considered 'unique'/1 or 'not unique' for a population - and 'size' -
    a DataFrame with three measures of effect sizes between the highest and second highest expressing group.
    """
    ##########################################################
        
    def neg_binom_test_vs_groups(gene, summary, PP):
        
        """
        Helper function for neg_binom_vs_groups_v1P.
        """
        
        #define output files
        
        bin_tmp = pd.DataFrame(index = [gene], columns = summary.columns[:-1])
        
        multi_ix = [np.array([gene,gene,gene,gene,gene]),
                    np.array(['mean','median','min','percentile - 25','percentile - 5'])]
        
        size_tmp = pd.DataFrame(index = multi_ix, columns = summary.columns[:-1])
        
        #find groups with the highest and second highest mean value for 'beta'
        
        first, second = summary.ix['mean'].order().index[-1], summary.ix['mean'].order().index[-2]
        
        #set all bin values to 0 and all size values to n.s.
        
        bin_tmp[summary.columns[:-1]] = 0
            
        size_tmp.ix[gene, 'mean'][summary.columns[:-1]] = 'n.s'
        size_tmp.ix[gene, 'median'][summary.columns[:-1]] = 'n.s'
        size_tmp.ix[gene, 'percentile - 25'][summary.columns[:-1]] = 'n.s'
        size_tmp.ix[gene, 'percentile - 5'][summary.columns[:-1]] = 'n.s'
        size_tmp.ix[gene, 'min'][summary.columns[:-1]] = 'n.s'
        
        #if first is 'Baseline', no group has a 'beta' beyond the 'Baseline' value
        
        if first != 'Baseline':
            
        #if first and second are significantly different, the gene is useful as unique marker and set to 1 in
        #the highest expressing population for which the effects size compared to second is reported
        
            if PP == '99.9':
        
                if summary.ix['min', first] > summary.ix['max', second]:
                
                    bin_tmp[first] = 1
                
                    size_tmp.ix[gene, 'mean'][first] = summary.ix['mean', first] - summary.ix['mean', second]
                    size_tmp.ix[gene, 'median'][first] = summary.ix['median', first] - summary.ix['median', second]
                    size_tmp.ix[gene, 'percentile - 25'][first] = summary.ix['Q25', first] - summary.ix['Q75', second]
                    size_tmp.ix[gene, 'percentile - 5'][first] = summary.ix['Q5', first] - summary.ix['Q95', second]
                    size_tmp.ix[gene, 'min'][first] = summary.ix['min', first] - summary.ix['max', second]
                    
            elif PP == '95':
        
                if summary.ix['Q5', first] > summary.ix['max', second] or summary.ix['min', first] > summary.ix['Q95', second]:
                
                    bin_tmp[first] = 1
                
                    size_tmp.ix[gene, 'mean'][first] = summary.ix['mean', first] - summary.ix['mean', second]
                    size_tmp.ix[gene, 'median'][first] = summary.ix['median', first] - summary.ix['median', second]
                    size_tmp.ix[gene, 'percentile - 25'][first] = summary.ix['Q25', first] - summary.ix['Q75', second]
                    size_tmp.ix[gene, 'percentile - 5'][first] = summary.ix['Q5', first] - summary.ix['Q95', second]
                    size_tmp.ix[gene, 'min'][first] = summary.ix['min', first] - summary.ix['max', second]
                    
            elif PP == '90':
        
                if summary.ix['Q5', first] > summary.ix['Q95', second]:
                
                    bin_tmp[first] = 1
                
                    size_tmp.ix[gene, 'mean'][first] = summary.ix['mean', first] - summary.ix['mean', second]
                    size_tmp.ix[gene, 'median'][first] = summary.ix['median', first] - summary.ix['median', second]
                    size_tmp.ix[gene, 'percentile - 25'][first] = summary.ix['Q25', first] - summary.ix['Q75', second]
                    size_tmp.ix[gene, 'percentile - 5'][first] = summary.ix['Q5', first] - summary.ix['Q95', second]
                    size_tmp.ix[gene, 'min'][first] = summary.ix['min', first] - summary.ix['max', second]
                
        return bin_tmp, size_tmp
    
    ##########################################################
    
    #perform test against baseline in parallel
    
    output_tmp = dview.map_sync(neg_binom_test_vs_groups,
                                [gene for gene in summary.index.levels[0]],
                                [summary.ix[gene] for gene in summary.index.levels[0]],
                                [PP for gene in summary.index.levels[0]])
    
    #fuse data and return
        
    bin_ = pd.concat([x[0] for x in output_tmp], axis = 0)
    size = pd.concat([x[1] for x in output_tmp], axis = 0)
    
    return bin_, size
    
################################################################################

def neg_binom_compare_groups_v1P(summary, gr1, gr2, dview, PP='99.9'):
    
    """
    Test whether a gene is differentially expressed between two groups / predictors based on negative binominal regression modelling. 
    A gene n is considered differentially expressed in group G vis-a-vis groups H, if its coefficient 'beta'(G) is above 'beta'(H) or 
    vice versa with a specified posterior probability (99.9%, 95% or 90%). If - for instance - 'beta'(G) is above 'beta'(H) with the specified
    PP, group G is set to 1 for n in the 'bin' file and several
    measures of the effect size between 'beta'(G) and 'beta'(H) are returned.
    ----------
    summary: DataFrame with summary data of the negative binominal regression traces.
    gr1: Name of the first group.
    gr2: Name of the second group.
    dview: name of Ipython DirectView Instance for parallel computing.
    PP: posterior probability cutoff. '99.9', '95' or '90. Default: '99.9'
    ----------
    returns 'bin' - a DataFrame indicating whether genes are considered 'differentially expressed'/1 or 'not differentially expressed' 
    for a population - and 'size' - a DataFrame with three measures of effect sizes between the highest and second highest expressing group.
    """
    ##########################################################
        
    def neg_binom_compare_groups_helper(gene, summary, gr1, gr2, PP):
        
        """
        Helper function for neg_binom_compare_groups_v1P.
        """
        
        #define output files
        
        bin_tmp = pd.DataFrame(index = [gene], columns = [gr1, gr2])
        
        multi_ix = [np.array([gene,gene,gene,gene,gene]),
                    np.array(['mean','median','min','percentile - 25','percentile - 5'])]
        
        size_tmp = pd.DataFrame(index = multi_ix, columns = [gr1, gr2])
        
        #set all bin values to 0 and all size values to n.s.
        
        bin_tmp.ix[gene,[gr1, gr2]] = 0
            
        size_tmp.ix[gene, 'mean'][[gr1, gr2]] = 'n.s'
        size_tmp.ix[gene, 'median'][[gr1, gr2]] = 'n.s'
        size_tmp.ix[gene, 'percentile - 25'][[gr1, gr2]] = 'n.s'
        size_tmp.ix[gene, 'percentile - 5'][[gr1, gr2]] = 'n.s'
        size_tmp.ix[gene, 'min'][[gr1, gr2]] = 'n.s'

        
        if PP == '99.9':
        
            if summary.ix['min', gr1] > summary.ix['max', gr2]:
                
                    bin_tmp[gr1] = 1
                
                    size_tmp.ix[gene, 'mean'][gr1] = summary.ix['mean', gr1] - summary.ix['mean', gr2]
                    size_tmp.ix[gene, 'median'][gr1] = summary.ix['median', gr1] - summary.ix['median', gr2]
                    size_tmp.ix[gene, 'percentile - 25'][gr1] = summary.ix['Q25', gr1] - summary.ix['Q75', gr2]
                    size_tmp.ix[gene, 'percentile - 5'][gr1] = summary.ix['Q5', gr1] - summary.ix['Q95', gr2]
                    size_tmp.ix[gene, 'min'][gr1] = summary.ix['min', gr1] - summary.ix['max', gr2]
                    
            elif summary.ix['min', gr2] > summary.ix['max', gr1]:
                
                    bin_tmp[gr2] = 1
                
                    size_tmp.ix[gene, 'mean'][gr2] = summary.ix['mean', gr2] - summary.ix['mean', gr1]
                    size_tmp.ix[gene, 'median'][gr2] = summary.ix['median', gr2] - summary.ix['median', gr1]
                    size_tmp.ix[gene, 'percentile - 25'][gr2] = summary.ix['Q25', gr2] - summary.ix['Q75', gr1]
                    size_tmp.ix[gene, 'percentile - 5'][gr2] = summary.ix['Q5', gr2] - summary.ix['Q95', gr1]
                    size_tmp.ix[gene, 'min'][gr2] = summary.ix['min', gr2] - summary.ix['max', gr1]
                    
        elif PP == '95':
        
            if summary.ix['Q5', gr1] > summary.ix['max', gr2] or summary.ix['min', gr1] > summary.ix['Q95', gr2]:
                
                    bin_tmp[gr1] = 1
                
                    size_tmp.ix[gene, 'mean'][gr1] = summary.ix['mean', gr1] - summary.ix['mean', gr2]
                    size_tmp.ix[gene, 'median'][gr1] = summary.ix['median', gr1] - summary.ix['median', gr2]
                    size_tmp.ix[gene, 'percentile - 25'][gr1] = summary.ix['Q25', gr1] - summary.ix['Q75', gr2]
                    size_tmp.ix[gene, 'percentile - 5'][gr1] = summary.ix['Q5', gr1] - summary.ix['Q95', gr2]
                    size_tmp.ix[gene, 'min'][gr1] = summary.ix['min', gr1] - summary.ix['max', gr2]
                    
            elif summary.ix['Q5', gr2] > summary.ix['max', gr1] or summary.ix['min', gr2] > summary.ix['Q95', gr1]:
                
                    bin_tmp[gr2] = 1
                
                    size_tmp.ix[gene, 'mean'][gr2] = summary.ix['mean', gr2] - summary.ix['mean', gr1]
                    size_tmp.ix[gene, 'median'][gr2] = summary.ix['median', gr2] - summary.ix['median', gr1]
                    size_tmp.ix[gene, 'percentile - 25'][gr2] = summary.ix['Q25', gr2] - summary.ix['Q75', gr1]
                    size_tmp.ix[gene, 'percentile - 5'][gr2] = summary.ix['Q5', gr2] - summary.ix['Q95', gr1]
                    size_tmp.ix[gene, 'min'][gr2] = summary.ix['min', gr2] - summary.ix['max', gr1]
                    
        elif PP == '90':
        
            if summary.ix['Q5', gr1] > summary.ix['Q95', gr2]:
                
                    bin_tmp[gr1] = 1
                
                    size_tmp.ix[gene, 'mean'][gr1] = summary.ix['mean', gr1] - summary.ix['mean', gr2]
                    size_tmp.ix[gene, 'median'][gr1] = summary.ix['median', gr1] - summary.ix['median', gr2]
                    size_tmp.ix[gene, 'percentile - 25'][gr1] = summary.ix['Q25', gr1] - summary.ix['Q75', gr2]
                    size_tmp.ix[gene, 'percentile - 5'][gr1] = summary.ix['Q5', gr1] - summary.ix['Q95', gr2]
                    size_tmp.ix[gene, 'min'][gr1] = summary.ix['min', gr1] - summary.ix['max', gr2]
                    
            elif summary.ix['Q5', gr2] > summary.ix['Q95', gr1]:
                
                    bin_tmp[gr2] = 1
                
                    size_tmp.ix[gene, 'mean'][gr2] = summary.ix['mean', gr2] - summary.ix['mean', gr1]
                    size_tmp.ix[gene, 'median'][gr2] = summary.ix['median', gr2] - summary.ix['median', gr1]
                    size_tmp.ix[gene, 'percentile - 25'][gr2] = summary.ix['Q25', gr2] - summary.ix['Q75', gr1]
                    size_tmp.ix[gene, 'percentile - 5'][gr2] = summary.ix['Q5', gr2] - summary.ix['Q95', gr1]
                    size_tmp.ix[gene, 'min'][gr2] = summary.ix['min', gr2] - summary.ix['max', gr1]
                
        return bin_tmp, size_tmp
    
    ##########################################################
    
    #perform test against baseline in parallel
    
    output_tmp = dview.map_sync(neg_binom_compare_groups_helper,
                                [gene for gene in summary.index.levels[0]],
                                [summary.ix[gene] for gene in summary.index.levels[0]],
                                [gr1 for gene in summary.index.levels[0]],
                                [gr2 for gene in summary.index.levels[0]],
                                [PP for gene in summary.index.levels[0]])
    
    #fuse data and return
        
    bin_ = pd.concat([x[0] for x in output_tmp], axis = 0)
    size = pd.concat([x[1] for x in output_tmp], axis = 0)
    
    return bin_, size

################################################################################

def bin_above_baseline(dataset, cell_groups, neg_binom_summary, genes, factor=2):
    
    """
    Binarizes a gene expression matrix by examining whether a gene is expressed in a particular cell above baseline
    (99.9% posterior probability) * factor.
    ----------
    dataset: DataFrame of m cells x n genes. NB: since the total number of mRNA is considered as baseline variable, the
    dataset should contain all genes which have gone into the calculation of that variable during the modeling.
    cell_groups: m cells clustered into groups (e.g. by AP).
    neg_binom_summary: summary file of negative binominal regression traces.
    genes: list of genes to consider.
    factor: a gene is called expressed in a cell if expression > baseline(99.9%) * variable * factor.
    ----------
    return DataFrama indicating whether a gene can be called expressed (1) in a certain cell.
    """
    
    #calculate baseline variables
    
    var_baseline = dataset[cell_groups.index].sum(axis = 0) / dataset[cell_groups.index].sum(axis = 0).mean()
    
    #calculate baseline threshold
    
    thresh_baseline = pd.DataFrame(index = genes, columns = cell_groups.index)
    
    for g in genes:
        thresh_baseline.ix[g] = (var_baseline * neg_binom_summary.ix[g].ix['max','Baseline']) * factor
        
    #evaluate whethere observed expression is above threshold
    
    data_bin = pd.DataFrame(0, index = genes, columns = cell_groups.index)
    data_bin[dataset.ix[genes, cell_groups.index] >  thresh_baseline] = 1
    
    return data_bin
    
################################################################################

def neg_binom_bin_cell_above_baseline(dataset, NBR_summary, genes, factor = 1.0):
    
    """
    Binarizes gene expression on a cell-to-cell basis by comparing the observed gene expression of a cell to a threshold
    derived from beta_baseline and x_baseline(PP = 99.9%).
    ----------
    dataset: DataFrame of m cells x n genes. Must be the same format also used to derive the baseline during neg. binom. regression.
    NBR_summary: Summary file of neg. binon. regression over x groups and y genes.
    genes: list of genes to be considered.
    factor: factor to multiply with baseline treshold. Default: 1)
    ----------
    return DataFrame with the binarized expression of y genes over m cells.
    """
    
    #trim input
    
    NBR_summary = NBR_summary.ix[genes]
    
    #define output
    
    output = pd.DataFrame(0, index = genes, columns = dataset.columns)
        
    #define baseline variable
    
    x_baseline = dataset.sum(axis = 0) / dataset.sum(axis = 0).mean()
    x_baseline_df = pd.DataFrame(index = NBR_summary.index.levels[0], columns = dataset.columns).fillna(x_baseline)
        
    #define maximal value under baseline
    
    beta_baseline = NBR_summary.swaplevel(0,1).ix['max']['Baseline'] - 1
    beta_baseline_df = pd.DataFrame(columns = NBR_summary.index.levels[0], index = dataset.columns).fillna(beta_baseline).T

    baseline_df = x_baseline_df * beta_baseline_df * factor
        
    #compare whether observed molecules are over baseline
    
    output[dataset.ix[baseline_df.index, baseline_df.columns] > baseline_df] = 1
    
    return output
    
################################################################################

def neg_binom_extract_genes(data, score = 'percentile - 5', number = 10):
    
    """
    For each group, returns the genes with the highest difference to either the baseline or other groups.
    ----------
    data: dataset containing the effects sizes of gene expression compared to either the baseline or all other groups.
    score: scoring method ['percentile - 25', 'percentile - 5','mean' or 'min] used to calculate effect size. 
    Default:'percentile - 5'.
    number: number of genes to be extracted. Default: 10.
    ----------
    returns DataFrame containing the extracted gene symbols for every group
    """
    
    #create output
    
    output = pd.DataFrame(index = range(number), columns = data.columns)
    
    #iterate over groups
    
    for gr in data.columns:
        
        #subselect data
        
        data_tmp = data.swaplevel(0,1,0).ix[score][gr]
        
        #remove n.s. genes
        
        data_tmp = data_tmp[data_tmp != 'n.s']
        
        #select genes
        
        genes_sel = data_tmp.order()[-number:].index[::-1]
        
        #update output
        
        for ix, gene in enumerate(genes_sel):
            output.ix[ix, gr] = gene
    
    return output
    
################################################################################
############################ SAMPLING AND SCORING ##############################
################################################################################

def neg_binom_simulate_data_v3(traces, NxK, sample_groups, coeff_dict, repeats=100):
    
    """
    Samples values for the coefficients in NxK from NBR traces a defined number 
    of times and generates a simulated gene expression matrices based on the 
    median expression of each gene in each cell over each sampling iteration.
    ----------
    traces: traces of n genes modeled by NBR.
    NxK: predictor matrix over m cells used to generate the NBR.
    sample_groups: [pd.Series] containing the group membership of cells.
    coeff_dict: [dict] of coefficients to be simulated together.
    repeats: number [int] of times coefficients are to be sampled.
    ----------
    returns three [pd.DataFrames]:
        output_cells: simulated data of n genes in m single cells with all coefficients /predictors considered.
        output_groups: simulated data of n genes in g groups with all coefficients /predictors considered.
        output_coeffs: simulated data of n genes over the whole dataset stratified according to groups of coefficents.
    """
    
    #define outputs
    
    output_cells = pd.DataFrame(index = traces.index, columns = NxK.columns)
    
    output_groups = pd.DataFrame(index = traces.index, columns = set(sample_groups))
    
    output_coeffs = pd.DataFrame(index = traces.index, columns = coeff_dict.keys())
    
    #iterate over genes
    
    for g in traces.index:
        
        #sample coeffs [repeats]-times and calculate median
        
        coeffs_tmp = pd.Series([np.median(np.random.choice(traces.ix[g,gr], size = repeats, replace = True) - 1) for gr in traces.columns],
                               index = traces.columns)
                
        #simulate data
        
        data_full_tmp = (coeffs_tmp * NxK.T).T
        
        data_cell_tmp = data_full_tmp.sum(axis = 0)
        
        data_coeff_tmp = data_full_tmp.sum(axis = 1)
                        
        #generate per cell data 
        
        output_cells.ix[g] = data_cell_tmp[output_cells.columns]
        
        #generate per group data
                    
        output_groups.ix[g] = [data_cell_tmp[sample_groups[sample_groups==gr].index].sum() for gr in output_groups.columns]
        
        #generate per coeff data
        
        output_coeffs.ix[g] = [data_coeff_tmp[coeff_dict[k]].sum() for k in output_coeffs.columns]
                
    return output_cells, output_groups, output_coeffs

################################################################################

def neg_binom_simulate_score_per_cell_v3(data_obs, data_sim):
    
    """
    Compares the simulated data to the observed data on a cell to cell basis.
    ----------
    data_obs: [pd.DataFrame] of empirically observed data.
    data_sim: [pd.DataFrame] of data simulated by sampling from NBR traces.
    ----------
    returns for each gene in each cell, the number of molecules explained (found in both
    observed in simulated), underexplained (found in obsereved but not accounted for in simulated)
    and overexplained (found in simulated but not in observed).
    """
    
    #trim obs
    
    data_obs = data_obs.ix[data_sim.index, data_sim.columns]
    
    #prepare output DataFrame

    output = pd.DataFrame(index = ['explained','overexplained','underexplained'], columns = data_obs.columns)
            
    #calculate difference

    data_diff = data_sim - data_obs

    #get overexplained data

    data_over = data_diff.copy()
    data_over[data_over <0] = 0
        
    output.ix['overexplained'] = data_over.sum(axis = 0)

    #get underexplained data

    data_under = data_diff.copy()
    data_under[data_under >0] = 0

    output.ix['underexplained'] = np.abs(data_under.sum(axis = 0))

    #get explained

    data_exp = data_obs + data_under

    output.ix['explained'] = data_exp.sum(axis = 0)

    return output

################################################################################

def neg_binom_simulate_score_per_group_v3(data_obs, data_sim, sample_groups):
    
    """
    Compares the simulated data to the observed data on a group basis.
    ----------
    data_obs: [pd.DataFrame] of empirically observed data.
    data_sim: [pd.DataFrame] of data simulated by sampling from NBR traces.
    sample_groups: [pd.Series] containing group membership for each cell.
    ----------
    returns for each gene pooled for each group, the number of molecules explained (found in both
    observed in simulated), underexplained (found in obsereved but not accounted for in simulated)
    and overexplained (found in simulated but not in observed).
    """ 
    
    data_obs_ = data_obs.copy()
    
    #prepare output DataFrame

    output = pd.DataFrame(index = ['explained','overexplained','underexplained'], columns = set(sample_groups))
        
    #bring obs data into right format
    
    data_obs = pd.DataFrame(index = data_sim.index, columns = set(sample_groups))
    
    for gr in set(sample_groups):
        
        ix_tmp = sample_groups[sample_groups==gr].index
        
        data_obs[gr] = data_obs_.ix[data_sim.index, ix_tmp].sum(axis = 1)
            
    #calculate difference

    data_diff = data_sim - data_obs

    #get overexplained data

    data_over = data_diff.copy()
    data_over[data_over <0] = 0
        
    output.ix['overexplained'] = data_over.sum(axis = 0)

    #get underexplained data

    data_under = data_diff.copy()
    data_under[data_under >0] = 0

    output.ix['underexplained'] = np.abs(data_under.sum(axis = 0))

    #get explained

    data_exp = data_obs + data_under

    output.ix['explained'] = data_exp.sum(axis = 0)

    return output

################################################################################

def neg_binom_simulate_score_per_coeff_v3(data_obs, data_sim):
    
    """
    Compares the simulated data to the observed data on a signature / coefficient basis.
    ----------
    data_obs: [pd.DataFrame] of empirically observed data.
    data_sim: [pd.DataFrame] of data simulated by sampling from NBR traces.
    sample_groups: [pd.Series] containing group membership for each cell.
    ----------
    returns for each gene pooled for over the whole dataset, the number of molecules explained (found in both
    observed in simulated), underexplained (found in obsereved but not accounted for in simulated)
    and overexplained (found in simulated but not in observed) based on the coefficients specified in coeff_dict.
    """ 
        
    data_obs_ = data_obs.copy().sum(axis=1)
    
    #prepare output DataFrame

    output = pd.DataFrame(index = ['explained','overexplained','underexplained'], columns = data_sim.columns)
        
    #bring obs data into right format

    data_obs = pd.DataFrame(index = data_sim.index, columns = data_sim.columns)
    
    for col in data_sim.columns:
                
        data_obs[col] = data_obs_.ix[data_sim.index]
            
    #calculate difference

    data_diff = data_sim - data_obs

    #get overexplained data

    data_over = data_diff.copy()
    data_over[data_over <0] = 0
        
    output.ix['overexplained'] = data_over.sum(axis = 0)

    #get underexplained data

    data_under = data_diff.copy()
    data_under[data_under >0] = 0

    output.ix['underexplained'] = np.abs(data_under.sum(axis = 0))

    #get explained

    data_exp = data_obs + data_under

    output.ix['explained'] = data_exp.sum(axis = 0)

    return output