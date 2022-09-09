#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 21:43:22 2022

Adapted from scripts/bICA.py

@author: jhyun95
"""

import numpy as np
import pandas as pd
import seaborn as sns

def formal_concept_decomposition(S, limit=None, sort_components=True,
    overlap=False, dim_balance=False, seed=None, verbose=False):
    ''' Algorithm 2 from: https://doi.org/10.1016/j.jcss.2009.05.002
        Same as formal_concept_decomposition, but more than
        10 times faster with better use of numpy. Produces a slightly
        different factorization than the original method. Can also provide
        a seed to shuffle/unshuffle rows and cols to produce different 
        factorizations.
        
        Args:
        S : 2D array
            A binary 2D numpy array to be decomposed
        limit : int
            Maximum number of concepts to limit run time, use None 
            to find a complete decomposition (default None)
        sort_components : boolean
            Sort the components by size, largest first. Can disable this 
            to return the original order of components discovered by the
            greedy algorithm (default True)
        overlap : boolean
            Whether or not to allow concepts to overlap. Decomposition
            is different and slower when allowed. Finally, the factorization
            W*H != S, rather np.nonzero(W*H) = S. (default False)
        dim_balance : boolean
            Whether or not to balance the propensity of expanding a block
            in either dimension, for highly non-square input. Only used
            when overlap=False (default False).
        seed : int
            Seed for shuffle rows/cols to find a different factorization.
            If None, does not apply any shuffling (default None)
    
        Returns:
            W : 2D array
                A binary 2D numpy array with factor loadings 
            H : 2D array
                A binary 2D numpy array with object encodings
            F : list
                A list of pairs of tuples that define the concepts. Each
                concept is saved as ( (rows in concept), (cols in concept) )
    '''
    
    ''' Shuffle rows and columns randomly if seed is provided '''
    S_total = np.sum(S)
    if not seed is None: # if seed is provided, shuffle rows and columns before factorizing
        np.random.seed(seed)
        num_rows, num_cols = S.shape
        row_shuffle = np.arange(num_rows); np.random.shuffle(row_shuffle) # shuffle rows
        col_shuffle = np.arange(num_cols); np.random.shuffle(col_shuffle) # shuffle columns
        U = S[row_shuffle,:][:,col_shuffle]; F = []
    else: # if no seed, do not shuffle rows/columns
        U = np.copy(S); F = []
        
    if limit is None: # if no limit on number of factors, set at maximum
        limit = S.shape[0] * S.shape[1]
    
    dim_coeff = np.log(U.shape[0]) / np.log(U.shape[1]) # rows / columns
    ''' Greedily discover largest blocks to cover the matrix '''
    while np.sum(U) > 0 and len(F) < limit: # unassigned relations exist
        accessible_rows = np.nonzero(np.sum(U, axis=1))[0].tolist()
        accessible_cols = np.nonzero(np.sum(U, axis=0))[0].tolist()
        concept_columns = []
        
        ''' Incrementally create the best next factor'''
        can_expand = True; current_score = 0
        while can_expand and len(accessible_rows) > 0 and len(accessible_cols) > 0:
        
            accessible_block_U = U[np.ix_(accessible_rows,accessible_cols)]
            col_sums_U = np.sum(accessible_block_U, axis=0)
            
            if overlap: # if allowing 1s to be covered multiple times
                ''' Neither penalty nor score for repeating a cell'''
                accessible_block_S = S[np.ix_(accessible_rows,accessible_cols)]
                last_block = U[np.ix_(accessible_rows, concept_columns)] # the current block
                last_row_scores = np.sum(last_block, axis=1) # score due to each row in current block
                new_col_scores = accessible_block_S * last_row_scores[np.newaxis].T # score = shared rows + new cells
                merge_scores = np.sum(new_col_scores, axis=0) + col_sums_U
            else: # if only allowing 1s to be covered once
                if dim_balance: # try to balance propensity to expand block in either dimension
                    merge_scores = ( (len(concept_columns) + 1)**dim_coeff ) * col_sums_U
                else: # no balancing dimensions
                    merge_scores = (len(concept_columns) + 1) * col_sums_U
                
            next_merge = np.argmax(merge_scores) # position of best column to add
            next_score = merge_scores[next_merge]
            
            if next_score > current_score:
                ''' Add the column to the current concept '''
                actual_next_merge = accessible_cols[next_merge]
                concept_columns.append(actual_next_merge)
                accessible_cols.remove(actual_next_merge)
                ''' Reduce the active block to rows shared in the concept columns'''
                if overlap: # if allowing 1s to be covered multiple times
                    next_rows = np.nonzero(accessible_block_S[:,next_merge])[0]
                else: # if only allowing cells to be covered once
                    next_rows = np.nonzero(accessible_block_U[:,next_merge])[0]
                actual_next_rows = [accessible_rows[x] for x in next_rows]
                accessible_rows = actual_next_rows
                ''' Update the concept score '''
                current_score = next_score
            else:
                can_expand = False
                
        if current_score > 0 : # a new block was found
            ''' Add the newly discovered concept '''
            concept = (tuple(accessible_rows), tuple(concept_columns))
            F.append(concept) # update concept list
            U[np.ix_(concept[0],concept[1])] = 0 # remove assigned relations
        if verbose: # print progress
            print('Components found:', len(F), '|', 'Coverage:', 1.0 - np.sum(U) / float(S_total))
            
    ''' Unshuffle the concepts if seed was provided '''
    if not seed is None:
        F_unshuffle = []
        for x_terms, y_terms in F:
            x_terms_unshuffle = [row_shuffle[x] for x in x_terms] 
            y_terms_unshuffle = [col_shuffle[y] for y in y_terms] 
            F_unshuffle.append((x_terms_unshuffle, y_terms_unshuffle))
        F = F_unshuffle

    ''' Construct the factorization '''
    if sort_components:
        F = sort_concepts_by_size(F)
    W,H = decompose_from_concepts(S,F)
    return W,H,F


def decompose_from_concepts(S,F):
    ''' Uses a set of concepts to decompose the original matrix,
        use with concepts generated by formal_concept_decomposition '''
    m, n = S.shape
    f = len(F)
    W = np.zeros((m,f), dtype=int)
    H = np.zeros((f,n), dtype=int)
    for i, concept in enumerate(F):
        x_terms, y_terms = concept
        W[[[x] for x in x_terms],i] = 1
        H[i,y_terms] = 1
    return W, H


def encode_from_concepts(F):
    ''' Uses a set if concepts to create an encoding of a matrix, 
        without actually loading the matrix. Will return the H matrix
        from decompose_from_concepts, without using S '''
    f = len(F); n = 0
    for concept in F: # infer number of strains
        n = max(max(concept[1]), n)
    H = np.zeros(shape=(f,n+1), dtype=int) # create H matrix
    for i, concept in enumerate(F): 
        H[i,concept[1]] = 1
    return H
   
    
def compute_concept_list_similarity(F1,F2,S):
    ''' 
    Examines how similar two FCD factorizations by greedily pairing
    up FCs from each factorization to maximize the total number of
    1s in the original matrix covered by both FCs in each pair. The score 
    is the total number of such 1s / total 1s in original matrix. 
    '''
    def find_overlap(C1, C2):
        ''' Find number of overlapping cells between two concept blocks'''
        X1 = set(C1[0]); Y1 = set(C1[1])
        X2 = set(C2[0]); Y2 = set(C2[1])
        return len(X1.intersection(X2)) * len(Y1.intersection(Y2))
                 
    f1 = len(F1); f2 = len(F2)
    i = 0 # current concept in first set
    unmatched = list(range(f2)) # unmatched concepts in second set
    total_overlap = 0
    while len(unmatched) > 0 and i < f1:
        best_match = None; best_overlap = -1
        for j in unmatched: # iterate through unmatched concepts in second set
            overlap = find_overlap(F1[i],F2[j]) 
            if overlap > best_overlap: # better overlap between this pair of components
                 best_overlap = overlap
                 best_match = j
        unmatched.remove(best_match) 
        total_overlap += best_overlap
        i += 1 # move to next concept in first set
    score = total_overlap / float(np.sum(S))
    return score
    

def compute_concept_coverage(S, F, plot=False, log_rate=50):
    ''' Examine the fraction of relationships covered by top X components '''
    total_relations = float(S.sum())
    uncovered_relations = total_relations
    uncovered = S.astype(bool)
    coverage = np.zeros(len(F)+1)
    m,n = S.shape; f = len(F)
    for i, concept in enumerate(F):
        if log_rate > 0 and (i+1) % log_rate == 0:
            print('Factor', i+1, 'of', len(F))
        x_terms, y_terms = concept
        uncovered_relations -= uncovered[np.ix_(x_terms, y_terms)].sum()
        uncovered[np.ix_(x_terms, y_terms)] = False
        coverage[i+1] = 1.0 - uncovered_relations / total_relations
    if plot:
        ax = sns.tsplot(coverage, np.arange(len(F)+1))
        ax.axhline(y=1.0, color='k', linestyle='--')
        ax.set_title('Relationships covered vs. Number of binary ICs')
        ax.set_xlabel('# binary ICs')
        ax.set_ylabel('fraction of relationships covered')
    return coverage


def sort_concepts_by_size(F):
    ''' Sorts a list of concepts by size, largest first  '''
    return sorted(F, key=lambda f: len(f[0]) * len(f[1]), reverse=True)


def load_formal_concepts(path, sort_components=False):
    ''' Loads the concept list F from file. Inverse of save_formal_concepts'''
    F = []
    with open(path, 'r') as f:
        for line in f:
            i, x_out, y_out = line.split('|')
            x_terms = map(int, x_out.split(','))
            y_terms = map(int, y_out.split(','))
            F.append( (tuple(x_terms), tuple(y_terms)) )
        if sort_components: # optionally sort by component size, largest first
            F = sort_concepts_by_size(F)
    return F


def save_formal_concepts(F, path):
    ''' Saves the concepts found formal concept decomposition to file.
        Note: Does not save the identity of each index '''
    output = []
    for i, concept in enumerate(F):
        x_terms, y_terms = concept
        x_out = ','.join(map(str, x_terms))
        y_out = ','.join(map(str, y_terms))
        line_out = str(i) + '|' + x_out + '|' + y_out
        output.append(line_out)
    with open(path, 'w+') as f:
        f.write('\n'.join(output))
    
    
def save_formal_concepts_full(F, path_W, path_H, path_F, ref_table):
    ''' Saves the concepts, as well as both the encoding and loading
        matrices with labels, for a formal concept decomposition. '''
    W, H = decompose_from_concepts(ref_table.values, F)
    allele_names = ref_table.index
    strain_names = ref_table.columns
    FCD_labels = ['FCD_' + str(x) for x in range(len(H))]
    df_fcd_loadings = pd.DataFrame(W, index=allele_names, columns=FCD_labels).replace(0,np.nan)
    df_fcd_encodings = pd.DataFrame(H, index=FCD_labels, columns=strain_names).replace(0,np.nan)
    df_fcd_loadings.to_csv(path_W)
    df_fcd_encodings.to_csv(path_H)
    save_formal_concepts(F, path_F)
    