#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Oct Apr 21 11:19:02 2021

@author: jhyun95

"""

from __future__ import print_function
import os, collections
import numpy as np
import pandas as pd
import scipy.sparse, scipy.optimize, scipy.stats
import statsmodels.stats
import subprocess as sp
import multiprocessing as mp
import pangenomix.sparse_utils

def compute_bernoulli_grid_core_genome(df_genes_dense, 
    prob_bounds=(0.8,0.99999999), init_capture_prob=0.9999, 
    init_gene_freqs=None):
    '''
    Models each element of a gene table as a Bernoulli random variable
    to estimate the true frequency of each gene in a pangenome.
    Assumes each gene i has a true frequency p_i, each genome j has
    a gene capture rate of q_j, and the occurence of gene i in genome j
    follows Bernoulli(p_i,q_j). Computes p and q to maximize likelihood
    through coordinate descent (repeatedly optimize each p, then each q)
    
    Parameters
    ----------
    df_genes_dense : pd.DataFrame
        Dense binary gene x genome DataFrame, i.e. of candidate core genes.
    prob_bounds : 2-tuple
        Bounds for true frequencies, default assumes that candidate core
        genes are provided (default (0.8,1.0))
    init_capture_prob : float
        Initial guess for genome gene capture rate q, default assumes near
        perfect gene capture. (default 0.999999)
    init_gene_freqs : list
        Initial guess for genome true frequencies. If None, uses observed
        frequencies (default None)
        
    Returns
    -------
    df_opt : pd.Series
        Returns initial and final loglikelihood and P,Q estimates
    res : scipy.optimize.optimize.OptimizeResult
        Optimization results object from Scipy
    '''
    
    ''' Set up optimization and guesses '''
    n_genes, n_genomes = df_genes_dense.shape
    X = df_genes_dense.values
    if init_gene_freqs is None:
        raw_gene_counts = X.sum(axis=1) / float(n_genomes)
        P_guess = np.array(raw_gene_counts)
    else:
        P_guess = np.array(init_gene_freqs)
    Q_guess = init_capture_prob*np.ones(n_genomes)
    PQ_guess = np.concatenate((P_guess,Q_guess))
    PQ_guess = np.clip(PQ_guess, prob_bounds[0], prob_bounds[1])
    init_ll = __bernoulli_grid_loglikelihood__(X, PQ_guess[:n_genes], PQ_guess[n_genes:])
    print('Initial loglikelihood:', init_ll)
    
    ''' Generate initial guess output '''
    gene_prob_labels = ['p_' + x for x in df_genes_dense.index]
    genome_prob_labels = ['q_' + x for x in df_genes_dense.columns]
    labels = ['Loglikelihood'] + gene_prob_labels + genome_prob_labels
    df_init = pd.Series(data=[init_ll] + PQ_guess.tolist(), index=labels)
    df_init.name = 'initial'
    
    ''' Solve optimization and format output '''
    neg_ll = lambda PQ: -__bernoulli_grid_loglikelihood__(X, PQ[:n_genes], PQ[n_genes:])
    neg_ll_grad = lambda PQ: -__bernoulli_grid_loglikelihood_gradient__(X, PQ[:n_genes], PQ[n_genes:])
    bounds = [prob_bounds] * len(PQ_guess)
    res = scipy.optimize.minimize(neg_ll, PQ_guess, method='L-BFGS-B', jac=neg_ll_grad,
                                  bounds=bounds, options={'disp':True})
    print('Final loglikelihood:', -res.fun)
    output = [-res.fun] + res.x.tolist()
    df_opt = pd.Series(data=output, index=labels)
    df_opt.name = 'optimum'
    df_opt_full = pd.concat([df_init, df_opt], axis=1)
    return df_opt_full, res
    

def compute_bernoulli_grid_core_genome_cd(df_genes_dense, 
    n_iterations=10, prob_bounds=(0.8,0.99999999), init_capture_prob=0.9999, 
    init_gene_freqs=None, use_logs=False):
    '''
    DON'T USE THIS, Coordinate descent optimizes poorly compared to 
    L-BFGS-B with defined gradient.
    
    See compute_bernoulli_grid_core_genome for details. Computes p and q 
    to maximize likelihood through coordinate descent instead of L-BFGS-B
    (repeatedly optimize each p, then each q).
    '''
    ''' Initialize output '''
    iteration = 0
    n_genes, n_genomes = df_genes_dense.shape
    df_opt_out = np.zeros((1+n_genes+n_genomes,n_iterations+1))
    gene_mat = df_genes_dense.values
    if init_gene_freqs is None:
        raw_gene_counts = gene_mat.sum(axis=1) / float(n_genomes)
        P_guess = np.array(raw_gene_counts)
    else:
        P_guess = np.array(init_gene_freqs)
    P_guess = np.clip(P_guess, prob_bounds[0], prob_bounds[1])
    
    ''' Initialize guesses and initial likelihood '''
    if use_logs:
        bounds = np.log(np.array(prob_bounds))
        LP = np.log(P_guess)
        LQ = np.log(init_capture_prob)*np.ones(n_genomes)
        LL = __bernoulli_grid_loglikelihood_from_logs__(gene_mat,LP,LQ)
        df_opt_out[0,iteration] = LL
        df_opt_out[1:1+n_genes,iteration] = np.exp(LP)
        df_opt_out[1+n_genes:,iteration] = np.exp(LQ)
        print('Initial Loglikelihood:', LL)

        ''' Maximize loglikelihood '''
        for iteration in range(1,n_iterations+1):
            print('Iteration:', iteration)
            for i in np.arange(n_genes):
                LP[i] = __bernoulli_grid_coordinate_descent_from_logs__(gene_mat[i,:], LQ, LP[i], bounds)
            for j in np.arange(n_genomes):
                LQ[j] = __bernoulli_grid_coordinate_descent_from_logs__(gene_mat[:,j], LP, LQ[j], bounds)
            LL = __bernoulli_grid_loglikelihood_from_logs__(gene_mat, LP, LQ)
            df_opt_out[0,iteration] = LL
            df_opt_out[1:1+n_genes,iteration] = np.exp(LP)
            df_opt_out[1+n_genes:,iteration] = np.exp(LQ)
            print('Loglikelihood:', LL)
    else:
        bounds = np.array(prob_bounds)
        P = P_guess
        Q = init_capture_prob*np.ones(n_genomes)
        LL = __bernoulli_grid_loglikelihood__(gene_mat, P, Q)
        df_opt_out[0,iteration] = LL
        df_opt_out[1:1+n_genes,iteration] = P
        df_opt_out[1+n_genes:,iteration] = Q
        print('Loglikelihood:', LL)

        ''' Maximize loglikelihood '''
        for iteration in range(1,n_iterations+1):
            print('Iteration:', iteration)
            for i in np.arange(n_genes):
                P[i] = __bernoulli_grid_coordinate_descent__(gene_mat[i,:], Q, P[i], bounds=prob_bounds)
            for j in np.arange(n_genomes):
                Q[j] = __bernoulli_grid_coordinate_descent__(gene_mat[:,j], P, Q[j], bounds=prob_bounds)
            LL = __bernoulli_grid_loglikelihood__(gene_mat, P, Q)
            df_opt_out[0,iteration] = LL
            df_opt_out[1:1+n_genes,iteration] = P
            df_opt_out[1+n_genes:,iteration] = Q
            print('Loglikelihood:', LL)
        
    ''' Format output '''
    gene_prob_labels = ['p_' + x for x in df_genes_dense.index]
    genome_prob_labels = ['q_' + x for x in df_genes_dense.columns]
    labels = ['Loglikelihood'] + gene_prob_labels + genome_prob_labels
    return pd.DataFrame(data=df_opt_out, index=labels, columns=np.arange(n_iterations+1))

def __bernoulli_grid_loglikelihood__(X,P,Q):
    ''' Computes LL of observed gene table X for given true gene frequencies 
        P and genome-specific gene capture rates Q '''
    probs = np.outer(P,Q)
    lls = np.multiply(X,np.log(probs)) + np.multiply(1.0-X,np.log(1.0-probs))
    return lls.sum()

def __bernoulli_grid_loglikelihood_from_logs__(X,LP,LQ):
    ''' See __bernoulli_grid_loglikelihood__, LP = log(P), LQ = log(Q) '''
    logprobs = np.add.outer(LP,LQ)
    lls = np.multiply(X,logprobs) + np.multiply(1.0-X,np.log(-np.expm1(logprobs)))
    return lls.sum()

def __bernoulli_grid_loglikelihood_gradient__(X,P,Q):
    ''' Computes the gradient of the LL of observed gene table X for given 
        true gene frequencies P and genome-specific gene capture rates '''
    m,n = X.shape
    nprobs = 1.0 - np.outer(P,Q)
    dLdp = np.divide(X.sum(axis=1), P)
    dLdp -= np.divide(((1.0 - X) * np.tile(Q,(m,1))), nprobs).sum(axis=1)
    dLdq = np.divide(X.sum(axis=0), Q)
    dLdq -= np.divide(((1.0 - X) * np.tile(P,(n,1)).T), nprobs).sum(axis=0)
    return np.concatenate((dLdp, dLdq))

def __bernoulli_grid_coordinate_descent_from_logs__(Xk, LQ, last, logbounds):
    ''' See __bernoulli_grid_optimize__, LQ = log(Q), solves for log(p_k) '''
    Xksum = Xk.sum()
    numer = np.multiply(1.0-Xk,np.exp(LQ))
    f = lambda lp: Xksum*np.exp(-lp) - np.divide(numer,-np.expm1(lp+LQ)).sum()
    if f(logbounds[0]) * f(logbounds[1]) >= 0: # boundary
        if np.abs(last - logbounds[0]) < np.abs(last - logbounds[1]):
            return logbounds[0]
        else:
            return logbounds[1]
    return scipy.optimize.root_scalar(f, bracket=logbounds, method='brentq').root

def __bernoulli_grid_coordinate_descent__(Xk, Q, last, bounds):
    ''' Optimize a single true gene frequency p_k under fixed
        genome-specific gene capture rates Q, within bounds.
        Uses last value for edge cases. '''
    Xksum = Xk.sum()
    numer = np.multiply(1.0-Xk,Q)
    f = lambda p: Xksum/p - np.divide(numer,(1.0 - p*Q)).sum()
    if f(bounds[0]) * f(bounds[1]) >= 0: # boundary
        if np.abs(last - bounds[0]) < np.abs(last - bounds[1]):
            return bounds[0]
        else:
            return bounds[1]
    return scipy.optimize.root_scalar(f, bracket=bounds, method='brentq').root    
    

def compute_beta_binomial_core_genome(df_genes, frac_recovered=0.999, df_counts=None,
                                    num_points=100, ks_iter=1000):
    '''
    Estimates the frequency threshold for core genes with an error rate model.
    
    This approach models the probability that a core gene is missing in a genome 
    (i.e. due to sequencing/assembly artifacts) with a Beta distribution Beta(a,b). 
    Correspondingly, given G genomes, the distribution of core gene misses (number 
    of core genes missing in 0,1,2,... genomes) is modeled as BetaBinomial(G,a,b),
    assuming that true core genes >> true near-core genes.

    MLE estimates for a and b are computed, and the CDF of the fitted distribution
    is used to determine the frequency threshold for capturing the desired 
    fraction of the core genome. Also computes fit quality metrics mean absolute error, 
    Monte-Carlo Kolmogorov Smirnov test p-value (goodness of fit), Shapiro-Wilk p-value 
    (residual normality) and Durbin-Watson statistic (residual autocorrelation).
    
    Parameters
    ----------
    df_genes : pd.DataFrame
        Sparse binary gene x genome DataFrame. Ignored if df_counts is not None
    frac_recovered : float
        Fraction of core genomes to be recovered
    df_counts : pd.Series or None
        Pre-computed frequency distribution {gene freq:num genes}, ignores df_genes if 
        not None and assumes n_genomes = maximum frequency observed (default None).
    num_points : int or list
        Number of points to fit. If list, computes fits and quality metrics for all
        number of points. More points = larger range to fit, but more noise from
        true near-core genes (default 100)
    ks_iter : int
        Number of iterations for estimating MC-KS test p-value (default 1000).
        Takes about 1 second per 500 iterations per num_points.
        
    Returns
    -------
    output : pd.Series or pd.DataFrame
        If num_points is a single value, returns a Series with these values
            alpha : MLE for alpha
            beta : MLE for beta
            cutoff : Maximum number of misses to accept a gene as core, 
                     corresponding to the core genome recovery fraction.
            mae : Mean absolute error 
            kolmogorov_smirnov_pvalue: Monte-Carlo KS test p-value for goodness of fit
            shapiro_wilk_pvalue: Shapiro-Wilk p-value for residual normality
            durbin_watson_stat : Durbin-Watson stat for residual autocorrelation
        If num_points is a list, returns of DataFrame with these columns,
        indexed by the number of points.
    '''
    ''' Get gene frequency distribution '''
    if df_counts is None:
        n_genes, n_genomes = df_genes.shape
        gene_mat = pangenomix.sparse_utils.sparse_arrays_to_sparse_matrix(df_genes)
        df_counts = pd.Series(collections.Counter(np.array(gene_mat.sum(axis=1))[:,0]))
    else:
        n_genomes = max(df_counts.index)
    
    df_fit_results = {}
    fit_points = num_points if type(num_points) != int else [num_points]
    for n_points in fit_points:
        ''' Convert gene frequencies to gene misses '''
        df = df_counts.reindex(np.arange(1,n_genomes+1)).fillna(0) # fill in any 0s        
        df = df_counts.iloc[-n_points:]
        df.index = n_genomes - df.index # convert counts to misses
        df = df.reindex(reversed(df.index))

        ''' Compute MLE for BetaBinomial alpha and beta '''
        X = df.index # number of times a core gene is missing (unit: genomes)
        Y = df.values # number of genes at each missing rate (unit: genes)
        loglikelihood = lambda ab: -np.dot(Y, betabin_logpmf(X,n_genomes,ab[0],ab[1]))
        res = scipy.optimize.minimize(loglikelihood, x0=(1,100), method='Nelder-Mead')
        a, b = res.x

        ''' Compute frequency threshold '''
        cutoff = 0
        cdf = np.exp(betabin_logpmf(cutoff,n_genomes,a,b))
        while cdf < frac_recovered:
            cutoff += 1
            cdf += np.exp(betabin_logpmf(cutoff,n_genomes,a,b))
        
        ''' Compute fit quality statistics '''
        Yhat = Y.sum() * np.exp(betabin_logpmf(X, n_genomes, a, b))
        residuals = (Y - Yhat).values
        mae = np.abs(residuals).mean()
        stat, sw_pvalue = scipy.stats.shapiro(residuals)
        dwstat = statsmodels.stats.stattools.durbin_watson(residuals)
        
        model_pmf = np.exp(betabin_logpmf(np.arange(n_genomes), n_genomes, a, b))
        model_cdf = np.cumsum(model_pmf)
        err = 1 - model_cdf
        sim_limit = np.where(err < 1e-8)[0][0] # simulate up to tolerance 1e-8
        if sim_limit > 0:
            ks_pvalue, ks_stat, ks_sim = ks_montecarlo_bbn(
                df, n_genomes, a, b, iterations=ks_iter, sim_limit=sim_limit)
        else:
            ks_pvalue = np.nan
        df_fit_results[n_points] = pd.Series({
            'alpha':a, 'beta':b, 'cutoff':cutoff, 'mae':mae,
            'kolmogorov_smirnov_pvalue': ks_pvalue,
            'shapiro_wilk_pvalue': sw_pvalue,
            'durbin_watson_stat': dwstat
        })
    df_fit_results = pd.DataFrame.from_dict(df_fit_results, orient='index')
    if df_fit_results.shape[0] == 1:
        return df_fit_results.iloc[0,:]
    return df_fit_results


def run_mlst(fna_paths, output_file, n_jobs=1, mlst_path='../tools/mlst/bin/mlst', env=None):
    '''
    Wrapper for https://github.com/tseemann/mlst, allowing parallel jobs
    
    Parameters
    ----------
    fna_paths : list
        List of paths to genome assemblies to run MLST on
    output_file : str
        Path to output combined raw MLST results
    n_jobs : int
        Number of parallel jobs (default 1)
    mlst_path : str
        Path of MLST program (default '../tools/mlst/bin/mlst')
    env : dict
        Environment to spawn MLST processes, such as that generated
        by os.environ.copy(), or None for default (default None).
    '''
    mlst_wrapper = lambda x: run_mlst_single(x, mlst_path, env)
    with open(output_file, 'w+') as f:
        if n_jobs > 1: # parallel jobs
            p = mp.Pool(processes=n_jobs)
            outputs = p.map(mlst_wrapper, fna_paths)
            for output in outputs:
                f.write(output)
        else: # single job
            for fna_path in fna_paths:
                f.write(mlst_wrapper(fna_path))

####################################################################################

def ks_montecarlo_bbn(Ycounts, n, a, b, iterations=100, sim_limit=1000):
    ''' 
    Monte-Carlo Kolmogorov-Smirnov for a beta-binomial. Standard KS-test does not
    apply for discrete distributions or distributions with estimated parameters,
    distribution of the KS stat must be computed by simulation.
    '''
    ''' Compute CDF of estimated distribution '''
    Xrange = np.arange(sim_limit)
    model_pmf = np.exp(betabin_logpmf(Xrange, n, a, b))
    model_cdf = np.cumsum(model_pmf)

    ''' Compute observed eCDF and KS stat '''
    ecdf = ecdf_from_counts(Ycounts.index, Ycounts.values, sim_limit)
    ks_stat = np.max(np.abs(ecdf - model_cdf))

    ''' Compute simulated eCDFs and KS stats'''
    n_samples = Ycounts.sum()
    bbn_draws = draw_bbn(n, a, b, size=n_samples*iterations, sim_limit=sim_limit)
    bbn_draws = bbn_draws.reshape(iterations,n_samples)
    ks_sim = np.zeros(iterations)
    for i in np.arange(iterations):
        unique_vals, counts = np.unique(bbn_draws[i,:], return_counts=True)
        ecdf_sim = ecdf_from_counts(unique_vals, counts, sim_limit)
        ks_sim[i] = np.max(np.abs(ecdf_sim - model_cdf))
    pvalue = (ks_stat < ks_sim).sum()/float(iterations)
    return pvalue, ks_stat, ks_sim

def draw_bbn(n, a, b, size, sim_limit=1000):
    ''' 
    Randomly draws <size> values from a specified beta-binomial distribution BBN(n,a,b).
    sim_limit determines the maximum value to be simulated, larger = slower, more accurate.
    '''
    Xs = np.arange(sim_limit) # range to simulate (larger = more accurate CDF, slower)
    probs = np.exp(betabin_logpmf(Xs, n, a, b))
    probs /= probs.sum() # should already be very close to 1 but just to make np happy
    return np.random.choice(Xs, size=size, p=probs)

def ecdf_from_counts(vals, counts, limit):
    ''' Computes eCDF from unique values and counts for x = np.arange(limit) '''
    pmf = np.zeros(limit)
    for i in np.arange(len(vals)):
        pmf[vals[i]] += counts[i]
    return np.cumsum(pmf) / pmf.sum()

from scipy.special import betaln
def betabin_logpmf(x, n, a, b):
    ''' 
    Beta-binomial log-PMF, used in compute_error_based_core_genome(). From:
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/_discrete_distns.py
    '''
    k = np.floor(x)
    combiln = -np.log(n + 1) - betaln(n - k + 1, k + 1)
    return combiln + betaln(k + a, n - k + b) - betaln(a, b)

def run_mlst_single(fna_path, mlst_path='../tools/mlst/bin/mlst', env=None):
    ''' Single thread of running MLST, see run_mlst() '''
    p = sp.Popen(['./' + mlst_path, fna_path], stdout=sp.PIPE, stderr=sp.PIPE, env=env)
    output, err = p.communicate()
    return output