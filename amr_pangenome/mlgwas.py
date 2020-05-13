#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 3 18:49:02 2020

@author: jhyun95

"""

import time
import collections, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats
import sklearn.base, sklearn.svm, sklearn.linear_model
import sklearn.metrics, sklearn.model_selection
import imblearn.ensemble
import xgboost as xgb

import mlrse


def plot_grid_search_performance(df_reports, params, grouping='case', figsize=None):
    '''
    Generate seaborn stripplots showing average MCC and GWAS scores for 
    different hyperparameter values.
    
    Parameters
    ----------
    df_reports : pd.DataFrame
        DataFrame with gridsearch results. Needs columns for all params, 
        grouping, 'MCC avg', and 'GWAS score',.
    params : list
        Names of hyperparameters (must be columns in df_reports)
    grouping : str
        Name of column by which to group performance scores (default 'case')
    figsize : tuple
        Figure size in inches, as in plt.subplots. If None, uses 
        (12, len(params*3). (default None)
    '''
    
    if figsize is None:
        figsize = (6*2, len(params)*3)
    fig, axs = plt.subplots(len(params), 2, figsize=figsize)
    for p,param in enumerate(params):
        param_values = df_reports[param].unique()
        ax = axs[p][0]
        sns.stripplot(x=grouping, y='MCC avg', hue=param, data=df_reports, ax=axs[p][0], dodge=True, size=3)
        sns.stripplot(x=grouping, y='GWAS score', hue=param, data=df_reports, ax=axs[p][1], dodge=True, size=3)
        axs[p][0].set_title('MCC avg vs ' + param)
        axs[p][1].set_title('GWAS score vs ' + param)
        axs[p][0].get_legend().remove()
        axs[p][1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if p == len(params)-1:
            axs[p][0].set_xticklabels(axs[p][0].get_xticklabels(), rotation=90)
            axs[p][1].set_xticklabels(axs[p][0].get_xticklabels(), rotation=90)
        else:
            axs[p][0].set_xticklabels([])
            axs[p][1].set_xticklabels([])
    plt.tight_layout()
    return fig, axs


def grid_search_svm_rse(df_features, df_labels, df_aro, drug,
    parameter_sweep={'n_estimators': [25,50,100,200],
                     'max_features': [1.0,0.75,0.5,0.25],
                     'max_samples': [1.0,0.75,0.5,0.25],
                     'SVM_C': [0.1,1,10,100]},                    
    fixed_parameters={'bootstrap':True, 'bootstrap_features':False,
                      'SVM_penalty':'l1', 'SVM_loss':'squared_hinge',
                      'SVM_dual':False, 'SVM_class_weight':'balanced'},
    compress_redundant_features=False, n_jobs=1, cv_jobs=1, seed=1):
    '''
    Grid search for SVM-RSE. For the specified hyperparameter space, 
    computes the GWAS score, ranks of known AMR genes, test MCC from 
    5-fold cross validation, and run times.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Binary samples x features DataFrame. Assumes all values are 0 or 1.
        Will automatically transpose if necessary.
    df_labels : pd.Series
        Binary Series corresponding to samples in df_features. 
    df_aro : pd.DataFrame
        Processed CARD output with ARO allele vs. ARO ID vs. drug
        data for an organism. Usually ends in "_allele_by_aro.csv" 
    drug : str
        Name of drug to evaluate AMR alleles against. Uses all ARO hits
        across all drugs if None.
    parameter_sweep : dict
        Dictionary mapping parameter names to lists of values. 
        Keys are either kwargs for sklearn.ensemble.BaggingClassifier, 
        or if preceded by "SVM_", kwargs for sklearn.svm.LinearSVC.
    fixed_parameters : dict
        Dictionary mapping parameters to values to be held constant.
        Keys are either kwargs for sklearn.ensemble.BaggingClassifier, 
        or if preceded by "SVM_", kwargs for sklearn.svm.LinearSVC.
    compress_redundant_features : bool
        If True, combines features that are statistically identical into 
        feature "blocks". After training, all features get the weight of their
        parent block, which is used for GWAS evaluation (default False).
    n_jobs : int
        Number of jobs per model, overrides parameter dictionaries (default 1)
    cv_jobs : int
        Number of jobs per cv-fold, total maximum threads will be 
        n_jobs * cv_jobs. Best values are 1 or 5 (default 5).
    seed : int
        Randomization seed, overrides parameter dictionaries (default 1)
        
    Returns
    -------
    df_report : pd.DataFrame
        DataFrame with model data for all hyperparameter combinations.
    '''
    def split_params(params):
        ''' Split ensemble-level and model-level parameters '''
        ensemble_params = {}; svm_params = {}
        for param, value in params.items():
            if param[:4] == 'SVM_': # model-level param
                svm_params[param[4:]] = value
            else: # ensemble-level parm
                ensemble_params[param] = value
        return ensemble_params, svm_params
    
    if compress_redundant_features: # combining identical features
        df_feat, block_features = compress_features(df_features)
        print 'Compressed features:', df_feat.shape
    else: # otherwise use original features directly
        df_feat = df_features; block_features = None
        
    report_values = []
    param_columns = list(parameter_sweep.keys())
    columns = param_columns + ['GWAS score', 'GWAS ranks', 'MCC avg', 'MCC 5CV', 'GWAS time', '5CV time']
    round3 = lambda x: round(x,3)
    mcc_scorer = sklearn.metrics.make_scorer(sklearn.metrics.matthews_corrcoef)
    rse_fixed_params, svm_fixed_params = split_params(fixed_parameters)
    for variable_parameters in sklearn.model_selection.ParameterGrid(parameter_sweep):
        print variable_parameters
        model_params = dict(fixed_parameters)
        model_params.update(variable_parameters)
        model_params['n_jobs'] = n_jobs
        model_params['random_state'] = seed
        model_params['SVM_random_state'] = seed
        rse_params, svm_params = split_params(model_params)
        rse_params.update(rse_fixed_params)
        svm_params.update(svm_fixed_params)
        
        ''' Train model on full data for weights -> ranks -> GWAS score '''
        print '\tTraining full (GWAS)...',
        t = time.time()
        base_clf = sklearn.svm.LinearSVC(**svm_params)
        df_weights, clf, X, y = gwas_rse(df_feat, df_labels, null_shuffle=False,
             base_model=base_clf, rse_kwargs=rse_params, return_matrices=True)
        if compress_redundant_features:
            df_weights = expand_block_features(df_weights, block_features)
            
        aro_hit_count = df_aro[pd.notnull(df_aro.loc[:,drug])].shape[0]
        if aro_hit_count == 0: # no reference hits, cannot score based on GWAS
            gwas_score = np.nan; gwas_ranks = [np.nan]
        else: # reference hits available
            df_eval = evaluate_gwas(df_weights, df_aro, drug, signed_weights=True, block_features=block_features)
            gwas_score, gwas_ranks = score_ranking(df_eval, signed_weights=True)
        time_gwas = time.time() - t
        print round3(time_gwas)
        
        ''' Evaluate model prediction performance with 5-fold CV '''
        print '\tTraining 5CV (MCC)...',
        t = time.time()
        mcc_scores = sklearn.model_selection.cross_val_score(
            clf, X=X, y=y, scoring=mcc_scorer, cv=5, n_jobs=cv_jobs)
        mcc_avg = np.mean(mcc_scores)
        time_5cv = time.time() - t
        print round3(time_5cv) 

        ''' Report scores '''
        print '\tSelected features:', df_weights.shape[0]
        print '\tGWAS Score:', round3(gwas_score), '\t', map(round3, gwas_ranks)
        print '\tPred Score:', round3(mcc_avg), '\t', map(round3, mcc_scores)
        param_set = list(map(lambda x: variable_parameters[x], param_columns))
        report_values.append(param_set + [gwas_score, gwas_ranks, 
            mcc_avg, mcc_scores, time_gwas, time_5cv])
        
    df_report = pd.DataFrame(columns=columns, data=report_values)
    return df_report


def score_ranking(df_eval, signed_weights=True):
    '''
    Scores a ranking table from evaluate_gwas with the following formula:
        gwas_score = sum[0.5 ^ (min(res_rank, sus_rank) - 1)/10]
    Exponential rank score makes the value of a true hit decrease by 50%
    for every 10 ranks, i.e. 1st = 1, 11th = 0.5, 21st = 0.25, etc.
    Uses "true hits from df_eval table returned by evaluate_gwas. 
    Intended for hyperparameter optimization.
    
    Parameters
    ----------
    df_eval : pd.DataFrame
        DataFrame with ranks of known AMR hits from evaluate_gwas()
    signed_weights : bool
        If True, assumes weights are signed (as in SVMs) and uses the
        minimum of res_rank and sus_rank for scoring. If False, assumes
        weights are positive (as in tree-based classifiers) (default True)
        
    Returns
    -------
    gwas_score : float
        Rank-weighted GWAS score as described earlier
    min_ranks : np.array
        Ranks used for computing GWAS score
        
    '''
    if df_eval.shape[0] == 0: # no hits
        return 0.0, []
    else: # at least one hit
        if signed_weights:
            ranks = df_eval.loc[:,['sus_rank','res_rank']].values
            min_ranks = np.amin(ranks, axis=1)
        else:
            min_ranks = df_eval.loc[:,'rank'].values
        #scores = 1.0 / np.log(1.0 + min_ranks)
        adj_ranks = (min_ranks - 1.0) / 10.0
        scores = np.exp2(-adj_ranks).sum()
        gwas_score = scores.sum()
        return gwas_score, min_ranks
    

def evaluate_gwas(df_weights, df_aro, drug, signed_weights=True, block_features=None):
    '''
    Compares the genetic features detected by a statistical method 
    to those detected by sequence homology to known AMR genes (CARD/RGI).
    For each detected feature reports:
        Weight: Raw weight reported by df_weights
        R rank: Rank sorted descending (resistance score)
        S rank: Rank sorted ascending (susceptibility score)
        Gene: Gene name for feature
        Feature Type: Gene, Allele, or Upstream
        Hit: Allele name for ARO hit
        ARO: ARO ID of the ARO hit
        Hit Type: Exact or same gene
    
    Parameters
    ----------
    df_weights : pd.Series
        Features mapping to weights, usually a single column of
        df_weights returned by gwas methods.
    df_aro : pd.DataFrame
        Processed CARD output with ARO allele vs. ARO ID vs. drug
        data for an organism. Usually ends in "_allele_by_aro.csv" 
    drug : str
        Name of drug to evaluate AMR alleles against. Uses all ARO hits
        across all drugs if None.
    signed_weights : bool
        If True, assumes weights are signed (as in SVMs) and produces 
        both res_rank and sus_rank columns. If False, assumes weights are
        strictly positive (as in tree-based classifiers) and produced a 
        single rank column (default True).
    block_features : list
        If redundant features were compressed using compress_features(),
        uses block_features to derive weights for original features.
        Otherwise, set to None (default None).
        
    Returns
    -------
    df_eval : pd.DataFrame
        DataFrame describing statistically detected AMR hits that
        were also detected by CARD/RGI as defined by df_aro.
    '''
    
    def feature_to_gene(feature):
        ''' Returns the gene and feature type (C, A, or U) of a genetic feature '''
        last_letter_match = list(re.finditer(r'[A-Z]', feature, re.I))[-1]
        last_letter = last_letter_match.group()
        if last_letter == 'C': # cluster/gene feature
            gene = feature
        else: # allele or upstream feature
            gene = feature[:last_letter_match.end()-1]
        return gene, {'C':'gene', 'A':'allele', 'U':'upstream', 'D':'downstream'}[last_letter]  
            # TODO: reference pangenome.py for feature type abbreviations
    
    ''' Extract alleles and genes relevant to drug '''
    df_drug = df_aro[pd.notnull(df_aro[drug])] if not drug is None else df_aro
    aro_alleles = df_drug.index.tolist()
    aro_genes, _ = zip(*map(feature_to_gene, aro_alleles))
    
    ''' Get feature rankings '''
    df_eval = pd.DataFrame(index=df_weights.index)
    raw_weights = np.nanmean(df_weights.values, axis=1) 
    df_eval['weight'] = raw_weights
    if signed_weights: # positive and negative weights
        df_eval['sus_rank'] = scipy.stats.rankdata(raw_weights) # smallest = 1
        df_eval['res_rank'] = scipy.stats.rankdata(-raw_weights) # largest = 1
    else: # strictly positive weights 
        df_eval['rank'] = scipy.stats.rankdata(-raw_weights) 
    
    ''' Reduce down to just ARO hits '''
    feature_genes, feature_types = zip(*map(feature_to_gene, df_eval.index))
    df_eval['gene'] = feature_genes
    df_eval['feature_type'] = feature_types
    hits = filter(lambda x: df_eval.loc[x,'gene'] in aro_genes, df_eval.index)
    df_eval = df_eval.reindex(index=hits)
    
    ''' Add in ARO hit information '''
    hit_aros = []; hit_aro_alleles = []; hit_types = []
    for hit in hits:
        if hit in df_drug.index: # exact match
            hit_type = 'Exact'
            hit_aro_allele = hit
        else: # hit of related gene
            hit_type = 'Same gene'
            hit_gene = df_eval.loc[hit,'gene']
            for i in range(len(aro_alleles)):
                aro_allele = aro_alleles[i]; aro_gene = aro_genes[i]
                if aro_gene == hit_gene:
                    hit_aro_allele = aro_allele; break
        hit_aro = df_drug.loc[hit_aro_allele,'ARO']
        hit_aros.append(hit_aro)
        hit_aro_alleles.append(hit_aro_allele)
        hit_types.append(hit_type)
    df_eval['ARO_allele'] = hit_aro_alleles
    df_eval['ARO'] = hit_aros
    df_eval['hit_type'] = hit_types
    return df_eval
    

def gwas_rse_boruta(df_features, df_labels, base_model=None,
                    n_estimators=100, max_samples=0.8, max_features=0.5, preshuffle=True):
    '''
    Runs a random subspace ensemble with feature selection by Boruta.
    For each iteration, randomly selects samples (w/ replacement) and features
    (w/o replacement), as in RSE. Then, for each selected features, a shuffled
    null feature is created. The combined features + shuffled features are 
    used to fit a classifier. After all models are trained, the pair tests
    are done between each feature weight vs. its corresponding null feature weight.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Binary samples x features DataFrame. Assumes all values are 0 or 1.
        Will automatically transpose if necessary.
    df_labels : pd.Series
        Binary Series corresponding to samples in df_features. 
    base_model : sklearn classifier
        Any sklearn classifier compatible with BaggingClassifier. 
        If None, uses sklearn.svm.LinearSVC, see get_default_svm() (default None)
    n_estimators : int
        Number of models to train in ensemble (default 100)
    max_samples : float
        Fraction of samples to use per model (default 0.8)
    max_features : float
        Fraction of features to use per model (default 0.5)
    preshuffle : bool
        If True, speeds up shuffling by precomputing shuffles for Boruta, and
        applying those shuffles in a random order to the features for each 
        model iteration. This is in contrast with generating a brand new
        set of shuffles for each model iteration, which is much slower (default True)
    '''
    
    ''' Set up data and model '''
    X, y = setup_Xy(df_features, df_labels, null_shuffle=False)
    n_samples, n_features = X.shape
    if base_model:
        base_clf = sklearn.base.clone(base_model)
    else: # defaulting to L1-SVM with class balance
        base_clf = get_default_svm()
        
    ''' Train each model with selected features + corresponding shuffled features '''
    all_feat_weights = np.empty((n_features, n_estimators))
    all_shuf_weights = np.empty((n_features, n_estimators))
    all_feat_weights[:] = np.nan
    all_shuf_weights[:] = np.nan
    sample_limit = int(max_samples * n_samples)
    feature_limit = int(max_features * n_features)
    
    ''' Optionally, pre-compute shuffles for faster data generation '''
    if preshuffle:
        print 'Precomputing shuffles...'
        shuffler = np.tile(np.arange(sample_limit), (feature_limit,1))
        [np.random.shuffle(x) for x in shuffler]
    
    for est_i in range(n_estimators):
        ''' Randomly select features and samples '''
        print 'Model', est_i+1
        clf = sklearn.base.clone(base_clf)
        if max_samples == 1.0:
                selected_samples = np.arange(n_samples)
        else:
            ysel = np.array([0]); # dummy start
            while ysel.sum() == 0 or ysel.sum() == ysel.shape[0]: # if bootstrap, don't bootstrap into just one class
                selected_samples = np.random.choice(np.arange(n_samples), sample_limit, replace=True)
                ysel = y[selected_samples]
                
        if max_features == 1.0:
            selected_features = np.arange(n_features)
        else: 
            selected_features = np.random.choice(np.arange(n_features), feature_limit, replace=False)
        Xsel = X[selected_samples,:][:,selected_features]
            
        ''' Create corresponding shuffled features '''
        Xshuf = Xsel.copy()
        if preshuffle: # reuse shuffle orders
            np.random.shuffle(shuffler)
            for i in range(Xshuf.shape[1]):
                Xshuf[:,i] = Xshuf[shuffler[i],i]
        else: # generate new shuffle orders, slow
            for i in range(Xshuf.shape[1]):
                np.random.shuffle(Xshuf[:,i])
        Xboruta = np.concatenate([Xsel, Xshuf], axis=1)
        
        ''' Train and extract feature weights '''
        clf.fit(Xboruta, ysel)
        all_feat_weights[selected_features, est_i] = clf.coef_[:,:feature_limit] # weights of actual features
        all_shuf_weights[selected_features, est_i] = clf.coef_[:,feature_limit:] # weights of shuffled features
        
    ''' Examine features '''
    df_feat_weights = pd.DataFrame(data=all_feat_weights, index=df_features.index, 
                         columns=map(lambda x: 'M' + str(x), range(n_estimators)))
    df_shuf_weights = pd.DataFrame(data=all_shuf_weights, index=df_features.index, 
                         columns=map(lambda x: 'M' + str(x) + '*', range(n_estimators)))
    
    ''' Filter out features with 0-weight (unshuffled) '''
    df_avg_weights = df_feat_weights.mean(axis=1, skipna=True)
    df_avg_weights = df_avg_weights[pd.notnull(df_avg_weights)] # exclude unused features
    df_avg_weights = df_avg_weights[df_avg_weights != 0.0] # exclude 0-weight features
    df_feat_weights = df_feat_weights.reindex(df_avg_weights.index)
    df_shuf_weights = df_shuf_weights.loc[df_feat_weights.index,:]
    df_weights = pd.concat([df_feat_weights, df_shuf_weights], axis=1)
    print df_weights.shape

    ''' Compare feature weight distributions (Wilcoxon signed rank) '''
    df_scores = pd.DataFrame(index=df_weights.index, columns=['models', 'avg_weight', 'stat', 'pvalue'])
    for i, feature in enumerate(df_feat_weights.index):
        df_weight_pairs = pd.DataFrame(index=df_feat_weights.columns[:n_estimators], columns=['feat','shuf'])
        df_weight_pairs['feat'] = df_weights.iloc[i,:n_estimators].values
        df_weight_pairs['shuf'] = df_weights.iloc[i,n_estimators:].values
        df_weight_pairs.dropna(how='all', inplace=True) # ignore models where feature wasn't included
        w1 = df_weight_pairs.values[:,0]
        w2 = df_weight_pairs.values[:,1]
        #stat, p = scipy.stats.wilcoxon(w1, w2)
        stat, p = scipy.stats.ttest_rel(w1, w2)
        m = df_weight_pairs.shape[0]
        avg_weight = df_avg_weights.loc[feature]
        df_scores.loc[feature,:] = (m, avg_weight, stat, p)
    return df_weights, df_scores
    

def gwas_rse(df_features, df_labels, null_shuffle=False, base_model=None, 
             rse_kwargs={'n_estimators':100, 'max_samples':0.8, 'max_features':0.5,
                         'bootstrap':True, 'bootstrap_features':False}, return_matrices=False):
    '''
    Runs a random subspace ensemble using BaggingClassifier.
    Can specify a base_model, or defaults to a L1-normalized, class weighted SVM.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Binary samples x features DataFrame. Assumes all values are 0 or 1.
        Will automatically transpose if necessary.
    df_labels : pd.Series
        Binary Series corresponding to samples in df_features. 
    null_shuffle : bool
        If True, randomly shuffles labels for null model testing (default False)
    base_model : sklearn classifier
        Any sklearn classifier compatible with BaggingClassifier. 
        If None, uses sklearn.svm.LinearSVC, see get_default_svm() (default None)
    rse_kwargs : dict
        Keyword arguments to pass to BaggingClassifier.
        (default {'n_estimators':100, 'max_samples':0.8, 'max_features':0.5,
        'bootstrap':True, 'bootstrap_features':False})
    return_matrics : bool
        If True, returns X and y used for training (default False)
                
    Returns
    -------
    df_weights : pd.DataFrame
        A feature x model DataFrame containing RSE weights for each model.
        Features excluded by feature bootstrapping have weight np.nan, compared
        to features included but unused which have weight 0. Only includes
        features that have a non-zero weight at least once.
    rse : CarefulBaggingClassifier
        Fitted ensemble (functionally identical to sklearn.ensemble.BaggingClassifier)
    X : np.ndarray
        Feature matrix used for training (if return_matrices is True)
    y : np.ndarray
        Label vector used for training (if return_matrics is True)
    '''
    X, y = setup_Xy(df_features, df_labels, null_shuffle)
    
    ''' Set up classifier '''
    if base_model:
        base_clf = sklearn.base.clone(base_model)
    else: # defaulting to L1-SVM with class balance
        base_clf = get_default_svm()
    rse_clf = mlrse.CarefulBaggingClassifier(base_clf, **rse_kwargs)
    
    ''' Train classifier '''
    rse_clf.fit(X,y)
    
    ''' Extract parameters '''
    n_samples, n_features = X.shape
    n_estimators = len(rse_clf.estimators_)
    weights = np.empty((n_features, n_estimators))
    weights[:] = np.nan
    for i in range(n_estimators):
        clf_coef = rse_clf.estimators_[i].coef_ # for BaggingClassifier
        # clf_coef = rse_clf.estimators_[i].named_steps.classifier.coef_ # for BalancedBaggingClassifier
        clf_feat = rse_clf.estimators_features_[i]
        weights[clf_feat,i] = clf_coef
    df_weights = pd.DataFrame(data=weights, index=df_features.index, 
                              columns=map(lambda x: 'M' + str(x), range(n_estimators)))
    
    ''' Filter out 0-weight parameters from weight table '''
    df_avg_weights = df_weights.mean(axis=1, skipna=True)
    df_weights = df_weights[df_avg_weights != 0.0]
    df_weights.dropna(axis=0, how='all', inplace=True)
    if return_matrices:
        return df_weights, rse_clf, X, y
    else:
        return df_weights, rse_clf
    

def gwas_fisher_exact(df_features, df_labels, null_shuffle=False, report_rate=10000):
    ''' 
    Applies Fisher Exact test between each feature the label. 
    Returns a dataframe with TPs/FPs/FNs/TNs/signed pvalues/oddsratios
    Adapted from scripts/microbial_gwas.py 
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Binary samples x features DataFrame. Assumes all values are 0 or 1.
        Will automatically transpose if necessary.
    df_labels : pd.Series
        Binary Series corresponding to samples in df_features. 
    null_shuffle : bool
        If True, randomly shuffles labels for null model testing (default False)
    report_rate : int
        Prints message each time this many features are processed (default 10000)
    '''
    def __batch_contingency__(X, y):
        ''' Creates a 3D table of contingency tables such that
            Table[i] = [[a,b],[c,d]] = contigency table for ith entry
                     = [[TP,FP],[FN,TN]] '''
        Y = np.broadcast_to(y, reversed(X.shape)).T
        Xb = X.astype(bool); Yb = Y.astype(bool)
        contingency = np.zeros(shape=(Xb.shape[1],2,2))
        contingency[:,0,0] = np.sum(np.logical_and(Xb,Yb), axis=0) # true positives
        contingency[:,0,1] = np.sum(np.logical_and(Xb,1-Yb), axis=0) # false positives
        contingency[:,1,0] = np.sum(np.logical_and(1-Xb,Yb), axis=0) # false negatives
        contingency[:,1,1] = np.sum(np.logical_and(1-Xb,1-Yb), axis=0) # true negatives
        return contingency
        
    ''' Shuffle labels for null model '''
    X, y = setup_Xy(df_features, df_labels, null_shuffle)
    
    ''' Compute contingency tables per feature '''
    contingency = __batch_contingency__(X, y)
    print 'Computed contingency tables:', contingency.shape
    
    ''' Apply Fisher's exact tests and compute ORs per feature '''
    df_output = pd.DataFrame(index=df_features.index if df_features.shape[1] == y.shape[0] else df_features.columns,
                     columns=['TPs','FPs','FNs','TNs'], data=contingency.reshape(-1,4))
    pvalues = np.zeros(X.shape[1])
    oddsratios = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        if (i % report_rate) == 0:
            print 'Feature', i
        oddsratio, pvalue = scipy.stats.fisher_exact(contingency[i,:,:])
        pvalue = -pvalue if oddsratio < 1 else pvalue # signed pvalue by direction
        pvalues[i] = pvalue
        oddsratios[i] = oddsratio
    df_output['pvalue'] = pvalues
    df_output['oddsratio'] = oddsratios
    return df_output


def expand_block_features(df_weights, block_features):
    '''
    Expands a block x column DataFrame to a feature x column DataFrame, 
    using block_features originally generated by compress_features().
    '''
    expanded_values = []; expanded_features = []
    for b, block in enumerate(df_weights.index):
        block_index = int(block[1:])
        block_specific_values = df_weights.values[b,:]
        block_specific_features = block_features[block_index]
        expanded_values += [block_specific_values] * len(block_specific_features)
        expanded_features += block_specific_features
    df_expanded = pd.DataFrame(index=expanded_features, 
       columns=df_weights.columns, data=expanded_values)
    return df_expanded
    

def compress_features(df_features):
    '''
    Takes a binary feature x sample DataFrame and combines features 
    that occur in the identical set of samples. Yields features named 'B#'.
    
    1) Converts SparseArray columns to COO matrix
    2) Converts to LIL for direct access to row data for sorting
    3) Iterates through sorted rows to combine statistically identical features
    4) Creates COO matrix from feature blocks
    5) Converts COO matrix -> CSR matrix -> SparseArray DataFrame
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Binary feature x sample DataFrame. Supports SparseArray columns.
        
    Returns
    -------
    df_block : pd.DataFrame
        Binary feature block x sample DataFrame with SparseArray columns
    block_features : list
        List of lists with names of features in each block.
        For example block_features[3] = features in block "B3"
    '''
    
    ''' Sort features by occurence in samples '''
    is_sparse = reduce(lambda x,y: x and y, map(lambda x: 'sparse' in x, df_features.ftypes))
    if is_sparse: # sorting rows of sparse matrix
        spdata = sparse_arrays_to_sparse_matrix(df_features).tolil()
        dfsg = pd.Series(index=df_features.index, data=spdata.rows).sort_values()
        del spdata
    else: # sorting rows of dense matrix: convert to sparse
        return compress_features(sparsify_dataframe(df_features), feature_block_file)

    ''' Combine features into blocks with identical occurence '''
    block_features = [] # maps feature block to related features
    row_indices = []; col_indices = []; last_occurence = None
    for feat, occur in dfsg.iteritems(): 
        if occur != last_occurence: # new feature by occurence
            last_occurence = occur
            col_indices.append(np.array(occur))
            row_indices.append(np.full(len(occur), len(block_features)))
            block_features.append([feat])
        else: # feature with same occurence
            block_features[-1].append(feat)
    del dfsg
    row_indices = np.hstack(row_indices)
    col_indices = np.hstack(col_indices)
    data = np.ones(row_indices.shape)
    blockdata = scipy.sparse.coo_matrix((data, (row_indices, col_indices))).tocsc()
    del row_indices, col_indices, data

    ''' Convert CSC block x sample matrix to DataFrame of SparseArrays '''
    sparse_vals = {}
    for c, sample in enumerate(df_features.columns):
        sample_data = blockdata[:,c].toarray().reshape(-1)
        sparse_vals[sample] = pd.SparseArray(sample_data, fill_value=0)
        sparse_vals[sample].fill_value = np.nan
    del blockdata
    blocks = map(lambda x: 'B' + str(x), range(len(block_features)))
    df_block = pd.DataFrame(sparse_vals, index=blocks)
    return df_block, block_features


def check_feature_blocks(df_block, block_features, df_features, reporting=100):
    ''' 
    Verifies that feature blocks generated by compress_features() 
    are consistent with the original feature matrix.
    '''
    errors = 0
    for b, features in enumerate(block_features):
        if b % reporting == 0:
            print 'Checking B' + str(b), '| Cumulative errors:', errors
        block_name = 'B' + str(b)
        block_occur = df_block.loc[block_name,:].fillna(0).values
        for feature in features:
            feature_occur = df_features.loc[feature,:].fillna(0).values
            matched = np.array_equal(feature_occur, block_occur)
            if not matched:
                print feature, block_name
                print '\t', block_occur
                print '\t', feature_occur
                errors += 1
    print 'Rows with errors:', errors
        

def make_amr_gwas_data_generator(df_features, df_amr, max_imbalance=0.95, min_variation=2, 
                                 binarize=True, drugs=None):
    '''
    Creates a generator that yields (drug, df_features, df_phenotype) 
    tuples for each drug that is sufficiently balanced from full 
    feature and phenotype matrices. For imbalanced cases, yields the tuple
    (drug, None, df_phenotype).
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Binary feature x sample DataFrame
    df_amr : pd.DataFrame
        SIR sample x drug DataFrame
    max_imbalance : float
        Maximum class imbalance to allowed to generate data (default 0.95)
    min_variation : int
        Minimum variation in feature to include, see filter_nonvariable()
        If set to 0, all features are used each time (default 2)
    binarize : bool
        If True, runs binarize_amr_table on df_amr (default True)
    drugs : list
        Only generate data for these drugs, or use all drugs if None (default None)
    
    Returns
    -------
    feature_gen : generator
        Yields (drug, df_features, df_phenotype) tuples for each balanced
        drug cases, or (drug, None, df_phenotype) if case is too imbalanced.
    '''
    df_amr_bin, sir_counts = binarize_amr_table(df_amr) if binarize else df_amr
    
    def amr_gwas_generator():
        for drug in df_amr_bin.columns:
            ''' Get drug-specific data '''
            df_phenotype = df_amr_bin[drug]
            df_phenotype = df_phenotype.dropna().astype(int)
            relevant_genomes = df_phenotype.index

            ''' Check extent of class imbalance '''
            res_rate = df_phenotype.sum() / float(df_phenotype.shape[0])
            sus_rate = 1.0 - res_rate
            is_imbalanced = res_rate > max_imbalance or sus_rate > max_imbalance
            
            ''' Optionally, check if drug is of interest '''
            is_relevant_drug = (drugs is None) or (drug in drugs)
            
            if is_imbalanced or not is_relevant_drug:
                ''' Do not process features for imbalanced cases or irrelevant cases '''
                yield (drug, None, df_phenotype)
            else:
                ''' Filter out features missing in the drug-specific dataset,
                    then features with low variability '''
                df_drug_features = df_features.loc[:,relevant_genomes]
                df_feat_counts = df_drug_features.fillna(0).sum(axis=1)
                missing = df_feat_counts[df_feat_counts < 1].index
                df_drug_features = df_drug_features.drop(missing)
                df_drug_features = filter_nonvariable(df_drug_features, min_variation=min_variation)
                yield (drug, df_drug_features, df_phenotype)
                
    return amr_gwas_generator()


def binarize_amr_table(df_amr):
    ''' 
    Converts SIR values in a pandas DataFrame or Series into binary phenotypes 
        - Susceptible (0): "Susceptible", "Susceptible-dose dependent" 
        - Non-Susceptible (1): "Resistant", "Intermediate", "Non-susceptible"
        - Missing (np.nan): np.nan, "Not defined", anything else
        
    Parameters 
    ----------
    df_amr : pd.DataFrame or pd.Series
        Dataframe with raw SIR values
        
    Returns
    -------
    df_pheno : pd.DataFrame or pd.Series
        Dataframe with binarized SIR values
    sir_counts : dict
        Dictionary mapping counts of original SIRs (for Series), or nested
        dictionary mapping drug:original SIR:count (for DataFrame)
        
    '''
    def binarize_sir(sir):
        ''' Binarizes a single SIR value '''
        if sir in ['Susceptible', 'Susceptible-dose dependent']:
            return 0
        elif sir in ['Resistant', 'Intermediate', 'Non-susceptible']:
            return 1
        elif sir in [np.nan, 'Not defined']:
            return np.nan
        else:
            print 'Unknown SIR:', sir
            return np.nan
        
    ''' Process input column-wise '''
    if type(df_amr) == pd.Series: # single column
        sir_counts = dict(collections.Counter(df_amr.values))
        df_pheno = df_amr.map(binarize_sir)
        return df_pheno, sir_counts
    elif type(df_amr) == pd.DataFrame: # process DataFrame column-by-column
        processed = map(lambda col: binarize_amr_table(df_amr[col]), df_amr.columns)
        processed_columns, sir_counters = zip(*processed)
        sir_counts = {df_amr.columns[i]:sir_counters[i] for i in range(df_amr.shape[1])}
        df_pheno = pd.concat(processed_columns, axis=1)
        return df_pheno, sir_counts
    else: # unknown format
        print 'Input not Series or DataFrame'
        return df_amr, None

    
def setup_Xy(df_features, df_labels, null_shuffle=False):
    ''' 
    Takes feature DataFrame and label Series and returns the 
    X and y numpy arrays to use with sklearn. Returns inputs directly
    if np arrays are provided. Takes into account shuffling for null 
    models, and potential need to transpose. 
    '''
    
    ''' Process feature table '''
    is_sparse = reduce(lambda x,y: x and y, map(lambda x: 'sparse' in x, df_features.ftypes))
    if type(df_features) == pd.core.frame.DataFrame: # normal DataFrame provided
        if is_sparse: # columns are SparseArray
            X = sparse_arrays_to_sparse_matrix(df_features)
        else: # fully dense DataFrame 
            X = df_features.values
    elif type(df_features) == pd.core.sparse.frame.SparseDataFrame: # SparseDataFrame provided
        X = df_features.to_coo() # sparse data structures sometimes faster to train 
    else: # otherwise, assume arrays provided
        X = df_features.copy()
        
    ''' Process label table  '''
    if type(df_labels) == pd.core.frame.DataFrame: # normal DataFrame provided
        y = df_labels.values
    else: # otherwise, assumes arrays provided
        y = df_labels.copy()

    ''' Check for transposed tables '''
    n = y.shape[0]
    if X.shape[0] != n: # dimension mismatch
        if X.shape[1] == n: # transpose provided
            X = X.T
        else:
            print 'Incompatible dimensions:', df_features.shape, df_labels.shape
            return None
    if null_shuffle:
        y = np.random.shuffle(y)
    return X, y


def filter_nonvariable(df_features, min_variation=2):
    ''' 
    Removes features that have very low variability. For example,
    if min_variation=3, removes features that present in only 1 or 2
    features, or features missing in only 0, 1, or 2 features. 
    Creates a copy of df_features, intended to speed up ML. 
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Feature matrix as features (rows) x samples (cols)
    min_variation : int
        Minimum variation to keep a feature, see explanation above (default 2)
        
    Returns
    -------
    df : pd.DataFrame
        Copy of df_features with low variation features removed
    '''
    if min_variation == 0: # no filtering
        return df_features
    df_counts = df_features.sum(axis=1)
    num_features, num_samples = df_features.shape
    min_count = min_variation
    max_count = num_samples - min_variation # inclusive count bounds
    selected_features = df_counts[(df_counts >= min_count) & (df_counts <= max_count)].index
    return df_features.loc[selected_features,:]


def sparse_arrays_to_sparse_matrix(dfs):
    '''
    Converts a binary DataFrame with SparseArray columns into a
    scipy.sparse.coo_matrix.
    '''
    num_entries = int(dfs.fillna(0).sum().sum())
    positions = np.empty((2,num_entries), dtype='int')
    fill_values = np.empty(num_entries)
    current = 0
    for i in range(dfs.shape[1]):
        col_entries = dfs.iloc[:,i].values
        col_num_entries = col_entries.sp_index.npoints
        positions[0, current:current+col_num_entries] = col_entries.sp_index.indices
        positions[1, current:current+col_num_entries] = i
        fill_values[current:current+col_num_entries] = col_entries.sp_values
        current += col_num_entries
    spdata = scipy.sparse.coo_matrix((fill_values, positions), shape=dfs.shape)
    return spdata


def sparsify_dataframe(df_dense):
    '''
    Creates a DataFrame with SparseArray columns from a dense DataFrame 
    '''
    spdata = {}
    for col in df_dense.columns:
        sample = df_dense[col]
        spdata[col] = pd.SparseArray(sample, fill_value=np.nan)
    return pd.DataFrame(spdata, index=df_dense.index)


def concatenate_sparsearray_dataframes(dfs):
    '''
    Concatenates a list of DataFrames with SparseArray columns.
    Acheives the same results as pd.concat(dfs, axis=0), but
    retains SparseArray structure rather than reverting to
    defunct SparseDataFrame. Assumes columns are consistent.
    '''
    concat_data = {}
    concat_index = np.hstack(map(lambda x: x.index.values, dfs))
    for col in dfs[0].columns:
        col_values = map(lambda x: x[col].values, dfs)
        col_values = np.hstack(col_values)
        concat_data[col] = pd.SparseArray(col_values, fill_value=np.nan)
    return pd.DataFrame(concat_data, index=concat_index)


def get_default_svm(seed=None):
    ''' Default sklearn-compatible SVM classifier '''
    base_clf = sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', 
       dual=False, class_weight='balanced', random_state=seed)
    return base_clf