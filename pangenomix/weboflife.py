#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 10 04:10:34 2022

@author: jhyun95

"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_node_gene_content(G, df_gene_pa, mrca_to_species, root='N3623'):
    ''' 
    Computes the fraction of child species with the gene for nonterminal nodes.
    For terminal nodes, returns 1 if present, 0 otherwise.
    '''
    node_bfs = __get_bfs_traversal__(G, root)
    node_content = {}
    for node in reversed(node_bfs):
        node_content[node] = np.zeros(2)
        if node in mrca_to_species:
            species = mrca_to_species[node]
            has_gene = df_gene_pa.loc[species]
            node_content[node][int(has_gene)] = 1
        else:
            for child in G[node]:
                node_content[node] += node_content[child]
    for node in node_content:
        presence_rate = float(node_content[node][1]) / node_content[node].sum()
        node_content[node] = presence_rate
    return node_content


def draw_nx_dendrogram(G, root, node_colors=None, ax=None, length_attr='len', return_coords=False):
    '''
    Generates a circular dendrogram from a networkx defined phylogeny.
    
    Parameters
    ----------
    G : nx.DiGraph
        Graph with edges going root -> terminals
    root : str
        Name of node to treet as root
    node_colors : dict, color, or None
        If dict, maps node names to colors. If None, all nodes are black.
        Otherwise, assigns the values to all nodes (default None)
    ax : AxesSubplot
        Subplot to plot onto (default None)
    length_attr : str
        Name of attribute to use to extract edge lengths (default 'len')
    return_coords : bool
        If True, returns both the subplot and node coordinates (default False)
    '''
    connector_color = 'black'
    default_node_color = 'black'
    whitespace_scaling = 1.05
    node_size = 30
    
    ''' Compute depths/radius of each node '''
    node_bfs = __get_bfs_traversal__(G, root)
    depths = {root:0.0}
    for node in node_bfs:
        for child in G[node]:
            child_depth = depths[node] + G[node][child][length_attr]
            depths[child] = round(child_depth, 8)

    ''' Generate terminal node order with depth-first search '''
    dfs = list(nx.algorithms.traversal.depth_first_search.dfs_preorder_nodes(G, source=root))
    terminal_order = [node for node in dfs if len(G[node]) == 0]
    terminal_set = set(terminal_order)
    n_terminals = len(terminal_set)

    ''' Compute node angles '''
    angles = {}
    terminal_angles = np.arange(0,1.0,1.0/n_terminals) * 2 * np.pi
    for node in reversed(node_bfs):   
        if node in terminal_set: # terminal node, evenly spaced in order previously calculated
            angle = terminal_angles[terminal_order.index(node)]
            angles[node] = {'mean':angle, 'max':angle, 'min':angle}
        else: # intermediate node, take average of immediate children
            for child in G[node]:
                if not child in angles:
                    angles = get_angles(child, angles)
            child_angles = [angles[child]['mean'] for child in G[node]]
            angles[node] = {'mean': np.mean(child_angles),
                'max': np.max(child_angles), 'min': np.min(child_angles)}

    ''' Compute node final positions and colors '''
    node_xy = np.zeros((len(G),2))
    for i,node in enumerate(G):
        node_xy[i][0] = depths[node] * np.cos(angles[node]['mean'])
        node_xy[i][1] = depths[node] * np.sin(angles[node]['mean'])
    if type(node_colors) == dict: # individually defined colors
        node_color_order = [node_colors[node] for node in G.nodes]
    elif node_colors == None: # if nothing, default black
        node_color_order = [default_node_color] * len(G.nodes)
    else: # otherwise, pass directly to each node
        node_color_order = [node_colors] * len(G.nodes)
        
    ''' Generate connecting lines '''
    if ax is None:
        fig, ax = plt.subplots(1,1)
    for parent in node_bfs:
        if len(G[parent]) > 0: # non-terminal
            ''' Draw arcs'''
            diameter = 2.0 * depths[parent]
            arc = mpl.patches.Arc(xy=(0,0), width=diameter, height=diameter, 
                theta1=angles[parent]['min'] * 180.0 / np.pi, 
                theta2=angles[parent]['max'] * 180.0 / np.pi,
                linewidth=1, fill=False, color=connector_color)
            ax.add_patch(arc)

            ''' Draw child branches '''
            for child in G[parent]:
                x1 = depths[child] * np.cos(angles[child]['mean'])
                y1 = depths[child] * np.sin(angles[child]['mean'])
                x2 = depths[parent] * np.cos(angles[child]['mean'])
                y2 = depths[parent] * np.sin(angles[child]['mean'])
                ax.plot([x1,x2],[y1,y2], color=connector_color)

    ''' Draw nodes, formatting '''
    ax.scatter(node_xy[:,0], node_xy[:,1], s=node_size, c=node_color_order, zorder=10)
    max_radius = max(depths.values())
    ax.set_xlim([-whitespace_scaling*max_radius, whitespace_scaling*max_radius])
    ax.set_ylim([-whitespace_scaling*max_radius, whitespace_scaling*max_radius])
    if return_coords:
        return ax, node_xy
    return ax


def __get_bfs_traversal__(G, root):
    ''' Generates a BFS traversal of the graph as a list of nodes, including terminals '''
    bfs = list(nx.algorithms.traversal.breadth_first_search.bfs_successors(G, source=root))
    node_bfs = [x[0] for x in bfs] # sequence of non-terminal nodes, breadth-first
    for node, successors in bfs: # appending terminal nodes
        for successor in successors:
            if len(G[successor]) == 0: #terminal
                node_bfs.append(successor)
    return node_bfs
    