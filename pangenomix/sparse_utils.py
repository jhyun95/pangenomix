#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 22:20:07 2021

@author: jhyun95

Auxillary functions for working with sparse data, primarily
LightSparseDataFrame, a wrapper around scipy.sparse.spmatrix
with some pandas DataFrame-like functions.
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import scipy.sparse

def read_lsdf(npz_file, label_file=None):
    '''
    Creates a LightSparseDataFrame from file.
    
    Parameters
    ----------
    npz_file : str
        Path to npz file with scipy.sparse matrix
    label_file : str
        Path to text file with index and column names. 
        If None, uses <npz_file>.labels.txt (default None)
        
    Returns
    -------
    lsdf : LightSparseDataFrame
        LightSparseDataFrame constucted from files
    '''
    data = scipy.sparse.load_npz(npz_file)
    label_path = npz_file + '.labels.txt' if label_file is None else label_file
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(line.strip())
    n_rows, n_cols = data.shape
    return LightSparseDataFrame(labels[:n_rows], labels[n_rows:], data)


def compress_rows(lsdf):
    '''
    Efficiently compresses identical rows in a LightSparseDataFrame.
    Intended for use with sparse binary matrices (checks if elements 
    are non-zero, without checking actual values).
    
    Parameters
    ----------
    lsdf : LightSparseDataFrame
        LightSparseDataFrame with rows to compress
    
    Returns
    -------
    lsdf_block : LightSparseDataFrame
        Similarly formatted LSDF with non-redundant rows. Rows are 
        labeled "B<row number>".
    block_definitions : list
        List indexed such that block_definitions[i] has the row labels
        of the original rows identical to the ith row in lsdf_block.
    '''
    spblock, block_indices = compress_rows_spmatrix(lsdf.data)
    num_blocks = spblock.shape[0]
    index_labels = ['B'+str(x) for x in np.arange(num_blocks)]
    lsdf_block = LightSparseDataFrame(index=index_labels, columns=lsdf.columns, data=spblock)
    block_definitions = [lsdf.index[x] for x in block_indices]
    return lsdf_block, block_definitions


def compress_rows_spmatrix(spmat):
    '''
    Efficiently compresses identical rows in a scipy.sparse.spmatrix.
    Intended for use with sparse binary matrices (checks if elements 
    are non-zero, without checking actual values).
    
    Parameters
    ----------
    spmat : scipy.sparse.spmatrix
        Scipy sparse matrix, must have tocsr() method
        
    Returns
    -------
    spblock : scipy.sparse.csr_matrix
        CSR formatted matrix with non-redundant rows
    block_definitions : list
        List indexed such that block_definitions[i] has the indices
        of the original rows identical to the ith row in spblock.
    '''
    spdata = spmat.tocsr()
    row_sizes = spdata.getnnz(axis=1)
    indices_to_block = {} # maps shared row indices to block number
    block_representatives = [] # indexed such that ith position = representative for ith block
    block_definitions = [] # indexed such that ith position = rows for ith block
    indptr = 0
    for row_pos in np.arange(spdata.shape[0]):
        row_indices = tuple(spdata.indices[indptr:indptr+row_sizes[row_pos]])
        if not row_indices in indices_to_block:
            indices_to_block[row_indices] = len(indices_to_block)
            block_representatives.append(row_pos)
            block_definitions.append([])
        block_id = indices_to_block[row_indices]
        block_definitions[block_id].append(row_pos)
        indptr += row_sizes[row_pos]
        
    spblock = spdata[block_representatives,:]
    return spblock, block_definitions


def sparse_arrays_to_lsdf(dfs):
    '''
    Converts a binary DataFrame with SparseArray columns into a
    LightSparseDataFrame
    '''
    spdata = sparse_arrays_to_spmatrix(dfs)
    return LightSparseDataFrame(index=dfs.index, columns=dfs.columns, data=spdata)


def sparse_arrays_to_spmatrix(dfs):
    '''
    Converts a binary DataFrame with SparseArray columns into a
    scipy.sparse.coo_matrix.
    '''
    positions = []
    fill_values = []
    for i in range(dfs.shape[1]):
        col_entries = dfs.iloc[:,i].values
        col_num_entries = col_entries.sp_index.npoints
        col_positions = np.empty((2,col_num_entries), dtype='int')
        col_positions[0,:] = col_entries.sp_index.indices
        col_positions[1,:] = i
        col_fill_values = col_entries.sp_values
        positions.append(col_positions)
        fill_values.append(col_fill_values)
    positions = np.concatenate(positions, axis=1)
    fill_values = np.concatenate(fill_values)
    spdata = scipy.sparse.coo_matrix((fill_values, positions), shape=dfs.shape)
    return spdata


def labelslice_sparse_arrays(dfs, spmat=None, indices=None, columns=None):
    '''
    Efficiently slices a binary DataFrame with SparseArray columns by label, similar to .loc
    '''
    i_indices = None; i_columns = None
    if not (indices is None):
        index_to_pos = {label:i for i,label in enumerate(dfs.index)}
        i_indices = [index_to_pos[x] for x in indices]
    if not (columns is None):
        columns_to_pos = {label:i for i,label in enumerate(dfs.columns)}
        i_columns = [columns_to_pos[x] for x in columns]
    return islice_sparse_arrays(dfs, spmat, i_indices, i_columns)


def islice_sparse_arrays(dfs, spmat=None, i_indices=None, i_columns=None):
    '''
    Efficiently slices a binary DataFrame with SparseArray columns by position, similar to .iloc 
    '''
    if spmat is None: # no pre-converted matrix
        X = sparse_arrays_to_spmatrix(dfs)
    else:
        X = spmat
        
    if not (i_indices is None):
        X = X.tocsr()[i_indices,:]
    if not (i_columns is None):
        X = X.tocsc()[:,i_columns]
    X = X.tocsc()
    new_index = dfs.index if i_indices is None else dfs.index[i_indices]
    new_columns = dfs.columns if i_columns is None else dfs.columns[i_columns]
    dfs_new = {}
    for i, column in enumerate(new_columns):
        col_data = X[:,i].toarray()[:,0]
        dfs_new[column] = pd.SparseArray(col_data)
    dfs_new = pd.DataFrame.from_dict(dfs_new)
    dfs_new.index = new_index
    return dfs_new, X

    
class LightSparseDataFrame:
    
    def __init__(self, index, columns, data):
        '''
        Parameters
        ----------
        index : list
            List of objects to use as index labels
        columns : list
            List of objects to use as column labels
        data : scipy.sparse.spmatrix
            Table data in scipy sparse matrix format
        '''
        try:
            self.data = data.tocoo()
        except: 
            print('ERROR: Could not convert data to COO format')
            self.data = np.nan
        self.index = np.array(index)
        self.columns = np.array(columns)
        self.shape = self.data.shape
        self.index_map = {index[i]:i for i in range(len(index))}
        self.column_map = {columns[i]:i for i in range(len(columns))}
        if len(index) != data.shape[0]:
            print('ERROR: Index length does not match data')
        if len(columns) != data.shape[1]:
            print('ERROR: Column length does no match data')
            
            
    def transpose(self):
        ''' Returns transposed table '''
        return LightSparseDataFrame(index=self.columns, columns=self.index, 
                                    data=self.data.transpose())
            
            
    def labelslice(self, indices=None, columns=None):
        '''
        Returns a dataframe slice by index and/or column labels.
        
        Parameters
        ----------
        indices : list
            Index labels of rows to select, or None if selecting all (default None)
        columns : list
            Column labels of columns to select, or None if selecting all (default None)
        
        Returns
        -------
        lsdf : LightSparseDataFrame
            LightSparseDataFrame with selected index and column labels
        '''
        i_indices = None if indices is None else [self.index_map[x] for x in indices]
        i_columns = None if columns is None else [self.column_map[x] for x in columns]
        return self.islice(i_indices, i_columns)
    

    def islice(self, i_indices=None, i_columns=None):
        '''
        Returns a dataframe slice by index and/or column positions.
        
        Parameters
        ----------
        indices : list of ints
            Index positions of rows to select, or None if selecting all (default None)
        columns : list of ints
            Column positions of columns to select, or None if selecting all (default None)
        
        Returns
        -------
        lsdf : LightSparseDataFrame
            LightSparseDataFrame with selected index and column positions
        '''
        if (i_indices is None) and not (i_columns is None): # column slice
            new_index = self.index
            new_columns = self.columns[i_columns]
            new_data = self.data.tocsc()[:,i_columns]
        elif not (i_indices is None) and (i_columns is None): # row slice
            new_index = self.index[i_indices]
            new_columns = self.columns
            new_data = self.data.tocsr()[i_indices,:]
        elif not(i_indices is None) and not (i_columns is None): # row/col slice
            new_index = self.index[i_indices]
            new_columns = self.columns[i_columns]
            new_data = self.data.tocsc()[:,i_columns].tocsr()[i_indices,:]
        else:
            print('No indices or columns selected')
            return None
        return LightSparseDataFrame(new_index, new_columns, new_data)
        
    
    def drop_empty(self, axis='index'):
        '''
        Returns a copy with empty rows (axis='index' or 0) or columns (axis='columns' or 1) removed.
        '''
        if axis == 'index' or axis == 0: # remove empty rows
            i_indices = np.where(np.array(self.data.sum(axis=1))[:,0] > 0)[0]
            return self.islice(i_indices=i_indices)
        elif axis == 'columns' or axis == 1: # remove empty columns
            i_columns = np.where(np.array(self.data.sum(axis=0))[0,:] > 0)[0]
            return self.islice(i_columns=i_columns)
        
        
    def sum(self, axis='index'):
        '''
        Computes row (axis='index' or 0) or column (axis='columns' or 1) sums. 
        Returns a dense array.
        '''
        if axis == 'index' or axis == 0: # row sums
            return np.array(self.data.sum(axis=1))[:,0]
        elif axis == 'columns' or axis == 1: # column sums
            return np.array(self.data.sum(axis=0))[0,:]
        
    
    def to_npz(self, npz_file, label_file=None):
        '''
        Save data to three files.
        
        Parameters
        ----------
        npz_file : str
            Path to output npz file with scipy.sparse matrix
        label_file : str
            Path output text file with index and column names. 
            Writes indices first, then columns, one per line.
            If None, uses <npz_file>.labels.txt (default None)
        '''
        label_path = npz_file + '.labels.txt' if label_file is None else label_file
        with open(label_path, 'w+') as f:
            for index_val in self.index:
                f.write(index_val + '\n')
            for column_val in self.columns:
                f.write(column_val + '\n')
        scipy.sparse.save_npz(npz_file, self.data.tocoo())

        
    def to_sparse_arrays(self):
        '''
        Converts to pd.DataFrame with pd.SparseArray columns (old format)
        '''
        sp_arrays = {}
        data_csc = self.data.tocsc()
        for c, column in enumerate(self.columns):
            # CSC column -> dense column -> SparseArray
            col_vector = data_csc[:,c].toarray()[:,0]
            sp_arrays[column] = pd.SparseArray(col_vector)
            sp_arrays[column].fill_value = np.nan
        return pd.DataFrame(data=sp_arrays, index=self.index)
        