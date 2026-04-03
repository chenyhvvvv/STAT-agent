import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.distance import cosine
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from numba import njit, prange
from collections import Counter


def Delaunay_adjacency_mtx(adata, spatial_key='spatial', cut_percentage=99, return_adata=False, verbose=True):

    changed = 0
    
    if return_adata:
        adata_copy = adata.copy()
    else:
        adata_copy = adata

    if not isinstance(adata_copy, list):
        changed = 1
        adata_copy = [adata_copy]
    
    if verbose:
        print(f'Generating Delaunay neighbor graph...')
        iterator = tqdm(range(len(adata_copy)))
    else:
        iterator = range(len(adata_copy))
    
    for k in iterator:

        # Delauney triangulation
        coords = adata_copy[k].obsm[spatial_key].copy()
        tri = Delaunay(coords)

        # edge set
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                edges.add(edge)
        edges = np.array(list(edges))

        # filter out long edges
        edge_lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
        threshold = np.percentile(edge_lengths, cut_percentage)
        filtered_edges = edges[edge_lengths <= threshold]
        
        # construct neighbor graph
        n_cells = adata_copy[k].shape[0]
        data = [1] * filtered_edges.shape[0]
        adj_matrix = sp.csr_matrix((data, filtered_edges.T), shape=(n_cells, n_cells), dtype=int)
        adj_matrix = adj_matrix + adj_matrix.T

        adata_copy[k].obsp['delaunay_adj_mtx'] = adj_matrix
    
    if verbose:
        print(f'All done!\n')
    
    if changed:
        adata_copy = adata_copy[0]

    if return_adata:
        return adata_copy


def knn_adjacency_matrix(adata, spatial_key='spatial', n_neighbors=20, return_adata=False, verbose=True):

    changed = 0

    if return_adata:
        adata_copy = adata.copy()
    else:
        adata_copy = adata

    if not isinstance(adata_copy, list):
        changed = 1
        adata_copy = [adata_copy]

    if verbose:
        print(f'Generating KNN neighbor graph...')
        iterator = tqdm(range(len(adata_copy)))
    else:
        iterator = range(len(adata_copy))
    
    for k in iterator:

        # find knn
        coords = adata_copy[k].obsm[spatial_key].copy()

        adj_matrix = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='connectivity', 
                                      metric='minkowski', p=2, include_self=False)
        adj_matrix = sp.csr_matrix(adj_matrix, dtype=int)

        adata_copy[k].obsp[f'knn_adj_mtx_{n_neighbors}'] = adj_matrix
    
    if verbose:
        print(f'All done!\n')
    
    if changed:
        adata_copy = adata_copy[0]

    if return_adata:
        return adata_copy
    

def joint_adjacency_matrix(adata, spatial_key='spatial', cut_percentage=99, n_step=3, n_neighbors=20, 
                           return_adata=False, verbose=True):
    
    if return_adata:
        adata_copy = adata.copy()
        adata_copy = Delaunay_adjacency_mtx(adata_copy,
                                            spatial_key=spatial_key, 
                                            cut_percentage=cut_percentage, 
                                            return_adata=return_adata,
                                            verbose=verbose,
                                            )
    else:
        adata_copy = adata
        Delaunay_adjacency_mtx(adata_copy,
                               spatial_key=spatial_key, 
                               cut_percentage=cut_percentage, 
                               return_adata=return_adata,
                               verbose=verbose,
                               )

    changed = 0
    if not isinstance(adata_copy, list):
        changed = 1
        adata_copy = [adata_copy]
    
    if verbose:
        print(f'Performing graph completion...')
        iterator = tqdm(range(len(adata_copy)))
    else:
        iterator = range(len(adata_copy))
    
    for k in iterator:
        
        coords = adata_copy[k].obsm[spatial_key].copy()

        delaunay_adj_mtx = adata_copy[k].obsp['delaunay_adj_mtx']
        delaunay_adj_mtx = delaunay_adj_mtx + sp.eye(delaunay_adj_mtx.shape[0])
        delaunay_adj_mtx = sp.linalg.matrix_power(delaunay_adj_mtx, n_step)
        delaunay_adj_mtx = (delaunay_adj_mtx > 0).astype(int)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='minkowski', p=2) 
        nbrs.fit(coords)
        indices = nbrs.kneighbors(coords, return_distance=False)

        add_rows = []
        add_cols = []
        # add_data = []

        for i in range(delaunay_adj_mtx.shape[0]):

            current_neighbors = delaunay_adj_mtx[i].nonzero()[1]
            n_to_add = n_neighbors - len(current_neighbors)
        
            if n_to_add > 0: 
                
                potential_neighbors = [j for j in indices[i] if j not in current_neighbors]
                add_rows += [i] * n_to_add
                add_cols += potential_neighbors[:n_to_add]
                # add_data += [1.] * n_to_add

        joint_adj_mtx = sp.csr_matrix(([1.]*len(add_rows), (add_rows, add_cols)), shape=delaunay_adj_mtx.shape)
        joint_adj_mtx += delaunay_adj_mtx
                
        adata_copy[k].obsp[f'joint_adj_mtx_{n_neighbors}'] = joint_adj_mtx
    
    if verbose:
        print(f'All done!\n')
    
    if changed:
        adata_copy = adata_copy[0]

    if return_adata:
        return adata_copy


def index2onehot(indices, n_cols=None, sparse=True):

    idx_list = list(indices)

    n_rows = len(idx_list)
    if n_cols is None:
        n_cols = len(list(set(idx_list)))
    
    rows = list(range(n_rows))
    data = [1.] * n_rows
    
    # generate onehot
    onehot = sp.csr_matrix((data, (rows, idx_list)), shape=(n_rows, n_cols))
    
    if sparse:
        return onehot
    else:
        return onehot.toarray()


def label2onehot(labels, n_cols=None, label_summary=None, change2str=True, sparse=True):
    
    # get the labels
    if label_summary is None:
        label_summary = set(labels)
    if change2str:
        label_summary = [str(label) for label in list(label_summary)]
    label_summary = sorted(label_summary)

    if n_cols is None:
        n_cols = len(label_summary)
    
    if change2str:
        label_idx = [label_summary.index(str(label)) for label in labels]
    else:
        label_idx = [label_summary.index(label) for label in labels]
    
    # generate onehot
    onehot = index2onehot(label_idx, n_cols=n_cols, sparse=sparse)
    
    return onehot


def label2onehot_anndata(adata, ct_key, return_adata=False, sparse=True, verbose=True):

    changed = 0

    if return_adata:
        adata_copy = adata.copy()
    else:
        adata_copy = adata

    if not isinstance(adata_copy, list):
        changed = 1
        adata_copy = [adata_copy]
    
    # get the labels
    ct_summary = set()
    for k in range(len(adata_copy)):
        ct_summary.update(set(adata_copy[k].obs[ct_key].copy().astype(str)))
    ct_summary = [str(ct) for ct in list(ct_summary)]
    ct_summary = sorted(ct_summary)
    ct_to_idx = {str(ct): str(idx) for idx, ct in enumerate(ct_summary)}
    idx_to_ct = {str(idx): str(ct) for idx, ct in enumerate(ct_summary)}

    if verbose:
        print(f'The cell types of interest are:')
        for i in range(len(ct_summary)):
            print(str(ct_summary[i]))
        print('')
    
    # generate onehot

    if verbose:
        print(f'Generating one-hot matrix...')
        iterator = tqdm(range(len(adata_copy)))
    else:
        iterator = range(len(adata_copy))
    
    for k in iterator:
        
        celltypes = adata_copy[k].obs[ct_key].copy().astype(str)
        ct_idx = [ct_summary.index(ct) for ct in celltypes]

        onehot = label2onehot(celltypes, n_cols=len(ct_summary), label_summary=ct_summary, sparse=sparse)

        adata_copy[k].obs['celltype_idx'] = ct_idx
        adata_copy[k].obsm['onehot'] = onehot
        adata_copy[k].uns['ct2idx'] = ct_to_idx
        adata_copy[k].uns['idx2ct'] = idx_to_ct
    
    if verbose:
        print(f'All done!\n')

    if changed:
        adata_copy = adata_copy[0]

    if return_adata:
        return adata_copy


def calculate_distribution(cn_labels, ct_indices, label_summary=None, n_niches=None, 
                           n_celltypes=None, change2str=True, sparse=True):
    
    if isinstance(cn_labels, list):
        if label_summary is None:
            label_summary = set(cn_labels)
        if change2str:
            label_summary = [str(label) for label in list(label_summary)]
        label_summary = sorted(label_summary)
        cn_onehot = label2onehot(cn_labels, n_cols=n_niches, label_summary=label_summary, change2str=change2str, sparse=sparse)
    else:
        cn_onehot = cn_labels

    if isinstance(ct_indices, list):
        ct_onehot = index2onehot(ct_indices, n_cols=n_celltypes, sparse=sparse)
    else:
        ct_onehot = ct_indices
    
    cell_count_niche = cn_onehot.T.sum(axis=1)
    dist = (cn_onehot.T @ ct_onehot) / cell_count_niche
    cell_count_niche = np.array(cell_count_niche).flatten()
    
    return dist, cell_count_niche


def edge_cutting(adj_mtx, labels):

    rows, cols = adj_mtx.nonzero()
    if isinstance(labels[0], list):
        mask = np.array([bool(set(labels[i]) & set(labels[j])) for i, j in zip(rows, cols)])
    else:
        mask = np.array([labels[i] == labels[j] for i, j in zip(rows, cols)])

    row_filtered = rows[mask]
    col_filtered = cols[mask]
    data = [1.] * row_filtered.shape[0]

    new_adj_mtx = sp.csr_matrix((data, (row_filtered, col_filtered)), shape=adj_mtx.shape)

    return new_adj_mtx


def update_microenvironment(adj_mtx, ct_onehot, n_celltypes, n_slices=None, cut_edge=True, cn_labels=None):

    if not isinstance(adj_mtx, list):
        adj_mtx_list = [adj_mtx.copy()]
    else:
        adj_mtx_list = adj_mtx.copy()
    
    if n_slices is None:
        n_slices = len(adj_mtx_list)

    ct_onehot_list = []
    if isinstance(ct_onehot, list):
        # index list for each slice
        if isinstance(ct_onehot[0], list):
            for i in range(len(ct_onehot)):
                ct_onehot_list.append(index2onehot(ct_onehot, n_cols=n_celltypes, sparse=True))
        # concatenated index list for all slices
        elif isinstance(ct_onehot[0], (int, float)):
            ct_onehot_mtx = index2onehot(ct_onehot, n_cols=n_celltypes, sparse=True)
            n = 0
            for i in range(n_slices):
                n_cells = adj_mtx_list[i].shape[0]
                ct_onehot_list.append(ct_onehot_mtx[n:n+n_cells])
                n += n_cells
        # one hot matrix list for each slice
        else:
            ct_onehot_list = ct_onehot
    # concatenated one hot matrix for all slices
    else:
        n = 0
        for i in range(n_slices):
            n_cells = adj_mtx_list[i].shape[0]
            ct_onehot_list.append(ct_onehot[n:n+n_cells])
            n += n_cells
    
    if cut_edge and cn_labels is None:
        raise ValueError('Please provide cell niche label for each cell/microenvironment!')
    
    n_neighbor_list = []
    micro_dist_list = []

    n = 0
    for i in range(n_slices):

        if cut_edge:
            n_cells = adj_mtx_list[i].shape[0]
            adj_mtx_list[i] = edge_cutting(adj_mtx_list[i], 
                                           cn_labels[n:n+n_cells],
                                           )
            n += n_cells

        n_neighbors = adj_mtx_list[i].sum(axis=1)
        micro_dist_mtx = (adj_mtx_list[i] @ ct_onehot_list[i]) / n_neighbors
        micro_dist_list.append(micro_dist_mtx)
        n_neighbor_list.append(list(np.array(n_neighbors, dtype=float).flatten()))
    
    return micro_dist_list, n_neighbor_list


def pca(mtx, explained_var=None, n_components=None, n_components_max=None, standardize=True, verbose=True):
    
    if sp.issparse(mtx):
        X_origin = mtx.copy().toarray()
    else:
        X_origin = mtx.copy()

    n_samples, n_features = X_origin.shape
    if n_components_max is None:
        n_components_max = min(n_samples, n_features)
    else:
        n_components_max = min(n_samples, n_features, n_components_max)

    X_centered = (X_origin - np.mean(X_origin, axis=0))
    if standardize:
        std = np.std(X_origin, axis=0)
        std[std == 0] = 1
        X_centered = X_centered / std

    _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    explained_variance = (S ** 2) / (n_samples - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var

    if explained_var is not None:
        explained_var = np.clip(explained_var, None, 1.)
        cumulative_ratio = np.cumsum(explained_variance_ratio)
        n_components = np.searchsorted(cumulative_ratio, explained_var) + 1
        if n_components > n_components_max:
            n_components = n_components_max
            if verbose:
                print(f'{n_components} principal components selected according to the upper bound of PCs amount, ' 
                          f'{100*cumulative_ratio[n_components-1]:.2f}% of variance explained.\n')
        elif verbose:
            print(f'{n_components} principal components selected according to the explained variance.\n')

    elif n_components is not None:
        if n_components > n_components_max:
            n_components = n_components_max
            if verbose:
                print(f'Returning {n_components} principal components since n_components is too large!\n')
        elif verbose:
            print(f'{n_components} principal components selected.\n')

    else:
        n_components = n_components_max
        if verbose:
            print(f'Returning {n_components} principal components.\n')

    X_pca = np.dot(X_centered, Vt[:n_components].T)
    components = Vt[:n_components]

    return X_pca, S[:n_components], components


def measure_distribution_gap(mtx1, mtx2, metric='jsd', precompute=None, weight=None, parallel=True, eps=1e-10):

    """
    mtx1: cell niche number * cell type number
    mtx2: cell number * cell type number
    """

    if sp.issparse(mtx1):
        m1 = mtx1.toarray()
    else:
        m1 = mtx1
    if sp.issparse(mtx2):
        m2 = mtx2.toarray()
    else:
        m2 = mtx2

    if m1.ndim == 1:
        m1 = np.expand_dims(m1, axis=0)
    if m2.ndim == 1:
        m2 = np.expand_dims(m2, axis=0)
    if weight is not None:
        if weight.ndim == 1:
            weight = np.expand_dims(weight, axis=1)
    

    if metric.lower() == 'jsd':
        if parallel and weight is not None:
            if precompute is None:
                precompute = cal_log(m2) 
            gap = gap_wjsd(m1, m2, precompute, weight, eps=eps)
        elif parallel:
            if precompute is None:
                precompute = cal_log(m2)
            gap = gap_jsd(m1, m2, precompute, eps=eps)
        else:
            log_m1 = np.log(np.clip(m1, eps, 1))
            if precompute is not None:
                log_m2 = precompute
            else:
                log_m2 = np.log(np.clip(m2, eps, 1))
            gap_list = []
            for d in range(m1.shape[0]):
                if weight is not None:  
                    # calculate weighted average distributions
                    avg_dist = m1[d] * (1 - weight[:, d][:, np.newaxis]) + m2 * weight[:, d][:, np.newaxis]
                else:  
                    # calculate average distributions
                    avg_dist = (m1[d] + m2) / 2
                log_avg = np.log(np.clip(avg_dist, eps, 1))  # N*K
                gap1 = -np.dot(log_avg, m1[d].T) + np.sum(m1[d] * log_m1[d])  # N*
                gap2 = np.sum(m2 * (log_m2 - log_avg), axis=1)  # N*
                if weight is not None:
                    gap_list.append((1 - weight[:, d]) * gap1 +  weight[:, d] * gap2)
                else:
                    gap_list.append((gap1 + gap2)/2)
            gap = np.array(gap_list).T

    elif metric.lower() == 'crossentropy' or metric.lower() == 'ce':
        if parallel:
            gap = gap_ce(m1, m2, eps=eps)
        else:
            log_m1 = np.log(np.clip(m1, eps, 1))  
            gap = -np.dot(m2, log_m1.T)
    
    elif metric.lower() == 'kld':  # KL(mtx1||mtx2)
        if parallel:
            if precompute is None:
                precompute = cal_log(m2)
            gap = gap_kld(m1, m2, precompute, eps=eps)
        else:
            log_m1 = np.log(np.clip(m1, eps, 1))
            if precompute is not None:
                log_m2 = precompute
            else:
                log_m2 = np.log(np.clip(m2, eps, 1))
            gap = -np.dot(log_m2, m1.T) + np.sum(m1 * log_m1, axis=1)
    
    elif metric.lower() == 'kld_reverse':  # KL(mtx2||mtx1)
        if parallel:
            if precompute is None:
                precompute = cal_log(m2)
            gap = gap_kld_reverse(m1, m2, precompute, eps=eps)
        else:
            log_m1 = np.log(np.clip(m1, eps, 1))
            if precompute is not None:
                log_m2 = precompute
            else:
                log_m2 = np.log(np.clip(m2, eps, 1))
            gap = (-np.dot(log_m1, m2.T) + np.sum(m2 * log_m2, axis=1)).T
    
    elif metric.lower() == 'euclidean':
        if parallel:
            gap = gap_euclidean(m1, m2)
        else:
            gap = np.linalg.norm(m2[:, np.newaxis] - m1, axis=2)
    
    elif metric.lower() == 'manhattan':
        if parallel:
            gap = gap_manhattan(m1, m2)
        else:
            gap = np.sum(np.abs(m2[:, np.newaxis] - m1), axis=2)

    elif metric.lower() == 'cosine':  # 1 - cosine similarity
        if parallel:
            precompute = cal_norm(m2)
            gap = gap_cosine(m1, m2, precompute)
        else:
            m1_norm = np.linalg.norm(m1, axis=1, keepdims=True) 
            if precompute is not None:
                m2_norm = precompute
            else:
                m2_norm = np.linalg.norm(m2, axis=1, keepdims=True) 
            gap = 1 - np.dot(m2, m1.T) / (m2_norm * m1_norm.T)
    
    else:
        raise ValueError(f"Unknown metric {metric}. Supported metrics are: \n"
                          "'jsd', 'crossentropy', 'kld', 'kld_reverse', 'euclidean', 'manhattan', 'cosine'.")

    return gap


@njit(parallel=True)
def cal_log(mtx, eps=1e-10):

    log_mtx = np.empty((mtx.shape[0], mtx.shape[1]), dtype=np.float32)
    for i in prange(mtx.shape[0]):
        for k in range(mtx.shape[1]):
            v = mtx[i, k]
            if v < eps:
                v = eps
            log_mtx[i, k] = np.log(v)

    return log_mtx


@njit(parallel=True)
def cal_norm(mtx):

    norm = np.empty(mtx.shape[0], dtype=np.float32)
    for i in prange(mtx.shape[0]):
        s2 = 0.0
        for k in range(mtx.shape[1]):
            s2 += mtx[i, k] * mtx[i, k]
        norm[i] = np.sqrt(s2)
    
    return norm


@njit(parallel=True)
def gap_jsd(m1, m2, precompute, eps=1e-10):

    n_niche, n_ct = m1.shape
    n_cell = m2.shape[0]

    gap = np.empty((n_cell, n_niche), dtype=np.float32)

    log_m1 = np.empty_like(m1, dtype=np.float32)
    for j in prange(n_niche):
        for k in range(n_ct):
            v = m1[j, k]
            if v < eps:
                v = eps
            log_m1[j, k] = np.log(v)

    # log_m2 = np.empty_like(m2, dtype=np.float32)
    # for i in prange(n_cell):
    #     for k in range(n_ct):
    #         v = m2[i, k]
    #         if v < eps:
    #             v = eps
    #         log_m2[i, k] = np.log(v)

    for i in prange(n_cell):
        for j in range(n_niche):
            s = 0.0
            for k in range(n_ct):
                p = m1[j, k]
                q = m2[i, k]
                m = 0.5 * (p + q)
                if p > 0:
                    s += 0.5 * p * (log_m1[j, k] - np.log(m))
                if q > 0:
                    s += 0.5 * q * (precompute[i, k] - np.log(m))
            gap[i, j] = s
    
    return gap


@njit(parallel=True)
def gap_wjsd(m1, m2, precompute, weight, eps=1e-10):

    n_niche, n_ct = m1.shape
    n_cell = m2.shape[0]

    gap = np.empty((n_cell, n_niche), dtype=np.float32)

    log_m1 = np.empty_like(m1, dtype=np.float32)
    for j in prange(n_niche):
        for k in range(n_ct):
            v = m1[j, k]
            if v < eps:
                v = eps
            log_m1[j, k] = np.log(v)

    # log_m2 = np.empty_like(m2, dtype=np.float32)
    # for i in prange(n_cell):
    #     for k in range(n_ct):
    #         v = m2[i, k]
    #         if v < eps:
    #             v = eps
    #         log_m2[i, k] = np.log(v)

    for i in prange(n_cell):
        for j in range(n_niche):
            w = weight[i, j]
            s = 0.0
            for k in range(n_ct):
                p = m1[j, k]
                q = m2[i, k]
                m = p * (1 - w) + q * w
                if p > 0:
                    s += (1 - w) * p * (log_m1[j, k] - np.log(m))
                if q > 0:
                    s += w * q * (precompute[i, k] - np.log(m))
            gap[i, j] = s
    
    return gap


@njit(parallel=True)
def gap_ce(m1, m2, eps=1e-10):

    n_niche, n_ct = m1.shape
    n_cell = m2.shape[0]

    gap = np.empty((n_cell, n_niche), dtype=np.float32)

    log_m1 = np.empty_like(m1, dtype=np.float32)
    for j in prange(n_niche):
        for k in range(n_ct):
            v = m1[j, k]
            if v < eps:
                v = eps
            log_m1[j, k] = np.log(v)

    for i in prange(n_cell):
        for j in range(n_niche):
            s = 0.0
            for k in range(n_ct):
                q = m2[i, k]
                if q > 0:
                    s -= q * log_m1[j, k]
            gap[i, j] = s
    
    return gap


@njit(parallel=True)
def gap_kld(m1, m2, precompute, eps=1e-10):

    n_niche, n_ct = m1.shape
    n_cell = m2.shape[0]

    gap = np.empty((n_cell, n_niche), dtype=np.float32)

    log_m1 = np.empty_like(m1, dtype=np.float32)
    for j in prange(n_niche):
        for k in range(n_ct):
            v = m1[j, k]
            if v < eps:
                v = eps
            log_m1[j, k] = np.log(v)

    # log_m2 = np.empty_like(m2, dtype=np.float32)
    # for i in prange(n_cell):
    #     for k in range(n_ct):
    #         v = m2[i, k]
    #         if v < eps:
    #             v = eps
    #         log_m2[i, k] = np.log(v)

    for i in prange(n_cell):
        for j in range(n_niche):
            s = 0.0
            for k in range(n_ct):
                p = m1[j, k]
                if p > 0:
                    s += p * (log_m1[j, k] - precompute[i, k])
            gap[i, j] = s
    
    return gap


@njit(parallel=True)
def gap_kld_reverse(m1, m2, precompute, eps=1e-10):

    n_niche, n_ct = m1.shape
    n_cell = m2.shape[0]

    gap = np.empty((n_cell, n_niche), dtype=np.float32)

    log_m1 = np.empty_like(m1, dtype=np.float32)
    for j in prange(n_niche):
        for k in range(n_ct):
            v = m1[j, k]
            if v < eps:
                v = eps
            log_m1[j, k] = np.log(v)

    # log_m2 = np.empty_like(m2, dtype=np.float32)
    # for i in prange(n_cell):
    #     for k in range(n_ct):
    #         v = m2[i, k]
    #         if v < eps:
    #             v = eps
    #         log_m2[i, k] = np.log(v)

    for i in prange(n_cell):
        for j in range(n_niche):
            s = 0.0
            for k in range(n_ct):
                q = m2[i, k]
                if q > 0:
                    s += q * (precompute[i, k] - log_m1[j, k])
            gap[i, j] = s 
    
    return gap


@njit(parallel=True)
def gap_euclidean(m1, m2):

    n_niche, n_ct = m1.shape
    n_cell = m2.shape[0]

    gap = np.empty((n_cell, n_niche), dtype=np.float32)

    for i in prange(n_cell):
        for j in range(n_niche):
            s = 0.0
            for k in range(n_ct):
                p = m1[j, k]
                q = m2[i, k]
                s += (p - q) ** 2
            gap[i, j] = np.sqrt(s)
    
    return gap


@njit(parallel=True)
def gap_manhattan(m1, m2):

    n_niche, n_ct = m1.shape
    n_cell = m2.shape[0]

    gap = np.empty((n_cell, n_niche), dtype=np.float32)

    for i in prange(n_cell):
        for j in range(n_niche):
            s = 0.0
            for k in range(n_ct):
                p = m1[j, k]
                q = m2[i, k]
                s += np.abs(p - q)
            gap[i, j] = s
    
    return gap


@njit(parallel=True)
def gap_cosine(m1, m2, precompute):

    n_niche, n_ct = m1.shape
    n_cell = m2.shape[0]

    gap = np.empty((n_cell, n_niche), dtype=np.float32)

    m1_norm = np.empty(n_niche, dtype=np.float32)
    for j in prange(n_niche):
        s1 = 0.0
        for k in range(n_ct):
            s1 += m1[j, k] * m1[j, k]
        m1_norm[j] = np.sqrt(s1)

    # m2_norm = np.empty(n_cell, dtype=np.float32)
    # for i in prange(n_cell):
    #     s2 = 0.0
    #     for k in range(n_ct):
    #         s2 += m2[i, k] * m2[i, k]
    #     m2_norm[i] = np.sqrt(s2)

    for i in prange(n_cell):
        for j in range(n_niche):
            s = 0.0
            for k in range(n_ct):
                p = m1[j, k]
                q = m2[i, k]
                s += p * q
            gap[i, j] = 1.0 - s / (m1_norm[j] * precompute[i])

    return gap


def cell2cellniche(cn_labels, ct_onehot, micro_dist, precompute=None, label_summary=None, n_celltypes=None, 
                   metric='jsd', max_iters=100, tol=1e-4, change2str=False, refine_k=0,
                   parallel=True, sparse=True, verbose=True):

    current_labels = cn_labels.copy()
    if label_summary is None:
        label_summary = sorted(set(current_labels))
    n_cells = len(current_labels)
    low_reassigned_rate_iter = 0
    history_changes = []
    MAX_HISTORY = 10

    cn_dist, cell_count_niche = calculate_distribution(current_labels, 
                                                       ct_onehot, 
                                                       label_summary=label_summary, 
                                                       n_niches=len(label_summary), 
                                                       n_celltypes=n_celltypes, 
                                                       change2str=change2str,
                                                       sparse=sparse,
                                                       )
    if refine_k > 0 and cn_dist.shape[1] > refine_k:
        cn_dist = refine_dist(cn_dist, k=refine_k)
    
    if verbose:
        iterator = tqdm(range(max_iters))
    else:
        iterator = range(max_iters)
    
    for i in iterator:

    # for i in range(max_iters):

        # assign cell to cell niches
        dist_gap = measure_distribution_gap(cn_dist, micro_dist, metric=metric, precompute=precompute, 
                                            weight=None, parallel=parallel, eps=1e-10)
        new_indices = np.argmin(dist_gap, axis=1)
        new_labels = [label_summary[idx] for idx in new_indices]
        label_summary = sorted(set(new_labels))

        # calculate cell distribution for cell niches
        new_cn_dist, cell_count_niche = calculate_distribution(new_labels, 
                                                               ct_onehot, 
                                                               label_summary=label_summary, 
                                                               n_niches=len(label_summary), 
                                                               n_celltypes=n_celltypes, 
                                                               change2str=change2str,
                                                               sparse=sparse,
                                                               )
        if refine_k > 0 and new_cn_dist.shape[1] > refine_k:
            new_cn_dist = refine_dist(new_cn_dist, k=refine_k)

        # check convergence
        n_reassigned = np.sum(np.array(new_labels) != np.array(current_labels))
        # if verbose:
        #     print(f'Reassigned labels amount: {n_reassigned}.')

        # condition 1: distribution of cell niches converge
        if new_cn_dist.shape[0] == cn_dist.shape[0]: 

            if sp.issparse(new_cn_dist):
                new_cn_dist_copy = new_cn_dist.copy().toarray()
            else:
                new_cn_dist_copy = new_cn_dist.copy()
            if sp.issparse(cn_dist):
                cn_dist_copy = cn_dist.copy().toarray()
            else:
                cn_dist_copy = cn_dist.copy()

            if np.array_equal(new_cn_dist_copy, cn_dist_copy):
                if verbose:
                    print(f'Strictly converge at iteration {i+1}.')
                return new_labels, new_cn_dist, label_summary, cell_count_niche

            # if np.mean(np.sum(np.abs(new_cn_dist_copy - cn_dist_copy), axis=1)) < tol:
            if np.sqrt(np.mean((new_cn_dist_copy - cn_dist_copy) ** 2)) < tol:
            # if np.all(new_cn_dist_copy - cn_dist_copy) < tol:
            # if np.all(np.linalg.norm(new_cn_dist_copy - cn_dist_copy, axis=1)) < tol:
            # if np.mean(np.linalg.norm(new_cn_dist_copy - cn_dist_copy, axis=1)) < tol:
                if verbose:
                    print(f'Distribution of cell niches (centers) converge at iteration {i+1}.')
                return new_labels, new_cn_dist, label_summary, cell_count_niche
            
        # condition 2: few cells are reassigned
        if n_reassigned <= min(0.002 * n_cells, 1000):
            low_reassigned_rate_iter += 1
            if n_reassigned == 0:
                if verbose:
                    print(f'Strictly converge at iteration {i+1}.')
                return new_labels, new_cn_dist, label_summary, cell_count_niche
            elif low_reassigned_rate_iter >= MAX_HISTORY:
                if verbose:
                    print(f'Converge at iteration {i+1}.')
                return new_labels, new_cn_dist, label_summary, cell_count_niche
        else:
            low_reassigned_rate_iter = 0
        
        # condition 3: duplicate reassignment
        changed_indices = np.where(np.array(new_labels) != np.array(current_labels))[0]
        if any(set(changed_indices) == set(prev) for prev in history_changes):
            if verbose:
                print(f'Early stop because of duplicate reassignment at iteration {i+1}.')
            return new_labels, new_cn_dist, label_summary, cell_count_niche
        history_changes.append(list(changed_indices))
        if len(history_changes) > MAX_HISTORY:
            history_changes.pop(0)
            
        if (i == max_iters-1) and verbose:
            print(f'Unconverged at iteration {i+1}!')

        cn_dist = new_cn_dist
        current_labels = new_labels
    
    return new_labels, new_cn_dist, label_summary, cell_count_niche


def cell2cellniche_kmeans(cn_labels, ct_onehot, micro_dist, 
                          label_summary=None, n_celltypes=None, 
                          random_seed=1234, change2str=False, sparse=True):

    current_labels = cn_labels.copy()
    if label_summary is None:
        label_summary = sorted(set(current_labels))

    centers, cell_count_niche = calculate_distribution(current_labels, 
                                                       ct_onehot, 
                                                       label_summary=label_summary, 
                                                       n_niches=len(label_summary), 
                                                       n_celltypes=n_celltypes, 
                                                       change2str=change2str,
                                                       sparse=sparse,
                                                       )
    
    # # use the average of microenvironment distributions as each center
    # cn_onehot = label2onehot(current_labels, n_cols=len(label_summary), label_summary=label_summary, change2str=change2str, sparse=sparse)
    # centers = cn_onehot.T @ micro_dist / cn_onehot.T.sum(axis=1)

    if sp.issparse(centers):
        centers = centers.toarray()
    if sp.issparse(micro_dist):
        micro_dist_copy = micro_dist.copy().toarray()

    # assign cell to cell niches
    kmeans = KMeans(n_clusters=len(label_summary), init=centers, n_init=1, random_state=random_seed, verbose=False)
    new_labels = kmeans.fit_predict(micro_dist_copy)
    new_labels = [label_summary[idx] for idx in new_labels]
    label_summary = sorted(set(new_labels))

    # calculate cell distribution for cell niches
    new_cn_dist, cell_count_niche = calculate_distribution(new_labels, 
                                                           ct_onehot, 
                                                           label_summary=label_summary, 
                                                           n_niches=len(label_summary), 
                                                           n_celltypes=n_celltypes, 
                                                           change2str=change2str,
                                                           sparse=sparse,
                                                           )

    return new_labels, new_cn_dist, label_summary, cell_count_niche


def cell2cellniche_cond(cn_labels, ct_onehot, micro_dist, cn_dist_basic, n_basic_niche, label_summary,
                        precompute=None, n_celltypes=None, metric='jsd', max_iters=100, tol=1e-4, change2str=False, refine_k=0,
                        parallel=True, sparse=True, verbose=True):

    current_labels = cn_labels.copy()
    n_cells = len(current_labels)
    low_reassigned_rate_iter = 0
    history_changes = []
    MAX_HISTORY = 10

    # no new niche
    if n_basic_niche == len(label_summary):
        dist_gap = measure_distribution_gap(cn_dist_basic, micro_dist, metric=metric, precompute=precompute, 
                                            weight=None, parallel=parallel, eps=1e-10)

        new_indices = np.argmin(dist_gap, axis=1)
        new_labels = [label_summary[idx] for idx in new_indices]
        if verbose:
            print(f'No new cell niche, all cells assigned to basic niches.')
        return new_labels, None, [], None

    label_summary_basic = label_summary[:n_basic_niche].copy()
    label_summary_cond = label_summary[n_basic_niche:].copy()

    # filter labels
    selected_indices = np.where(np.isin(np.array(current_labels), np.array(label_summary_cond)))[0]
    filtered_labels = list(np.array(current_labels)[selected_indices])
    filtered_ct_onehot = ct_onehot.tocsr()[selected_indices, :]

    cn_dist_cond, cell_count_niche_cond = calculate_distribution(filtered_labels, 
                                                                 filtered_ct_onehot, 
                                                                 label_summary=label_summary_cond, 
                                                                 n_niches=len(label_summary_cond), 
                                                                 n_celltypes=n_celltypes, 
                                                                 change2str=change2str,
                                                                 sparse=sparse,
                                                                 )
    if refine_k > 0 and cn_dist_cond.shape[1] > refine_k:
        cn_dist_cond = refine_dist(cn_dist_cond, k=refine_k)

    if verbose:
        iterator = tqdm(range(max_iters))
    else:
        iterator = range(max_iters)
    
    for i in iterator:

    # for i in range(max_iters):

        # assign cell to cell niches
        cn_dist = sp.vstack([cn_dist_basic, cn_dist_cond])
        dist_gap = measure_distribution_gap(cn_dist, micro_dist, metric=metric, precompute=precompute, 
                                            weight=None, parallel=parallel, eps=1e-10)

        new_indices = np.argmin(dist_gap, axis=1)
        new_labels = [label_summary[idx] for idx in new_indices]

        new_selected_indices = np.where(np.isin(np.array(new_labels), np.array(label_summary_cond)))[0]
        if len(new_selected_indices) == 0:
            if verbose:
                print(f'No cell assigned to new cell niche at iteration {i+1}.')
            return new_labels, None, [], None
        new_filtered_labels = list(np.array(new_labels)[new_selected_indices])
        new_filtered_ct_onehot = ct_onehot.tocsr()[new_selected_indices, :]

        label_summary_cond = sorted(set(new_filtered_labels))
        label_summary = label_summary_basic + label_summary_cond

        # calculate cell distribution for cell niches
        new_cn_dist_cond, cell_count_niche_cond = calculate_distribution(new_filtered_labels, 
                                                                         new_filtered_ct_onehot, 
                                                                         label_summary=label_summary_cond, 
                                                                         n_niches=len(label_summary_cond), 
                                                                         n_celltypes=n_celltypes, 
                                                                         change2str=change2str,
                                                                         sparse=sparse,
                                                                         )
        if refine_k > 0 and new_cn_dist_cond.shape[1] > refine_k:
            new_cn_dist_cond = refine_dist(new_cn_dist_cond, k=refine_k)

        # check convergence
        n_reassigned = np.sum(np.array(new_labels) != np.array(current_labels))
        # if verbose:
        #     print(f'Reassigned labels amount: {n_reassigned}.')

        # condition 1: distribution of cell niches converge
        if new_cn_dist_cond.shape[0] == cn_dist_cond.shape[0]: 

            if sp.issparse(new_cn_dist_cond):
                new_cn_dist_cond_copy = new_cn_dist_cond.copy().toarray()
            else:
                new_cn_dist_cond_copy = new_cn_dist_cond.copy()
            if sp.issparse(cn_dist_cond):
                cn_dist_cond_copy = cn_dist_cond.copy().toarray()
            else:
                cn_dist_cond_copy = cn_dist_cond.copy()

            if np.array_equal(new_cn_dist_cond_copy, cn_dist_cond_copy):
                if verbose:
                    print(f'Strictly converge at iteration {i+1}.')
                return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
            
            if np.sqrt(np.mean((new_cn_dist_cond_copy - cn_dist_cond_copy) ** 2)) < tol:
            # if np.mean(np.linalg.norm(new_cn_dist_cond_copy - cn_dist_cond_copy, axis=1)) < tol:
                if verbose:
                    print(f'Distribution of cell niches (centers) converge at iteration {i+1}.')
                return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
            
        # condition 2: few cells are reassigned
        if n_reassigned <= min(0.002 * n_cells, 1000):
            low_reassigned_rate_iter += 1
            if n_reassigned == 0:
                if verbose:
                    print(f'Strictly converge at iteration {i+1}.')
                return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
            elif low_reassigned_rate_iter >= MAX_HISTORY:
                if verbose:
                    print(f'Converge at iteration {i+1}.')
                return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
        else:
            low_reassigned_rate_iter = 0
        
        # condition 3: duplicate reassignment
        changed_indices = np.where(np.array(new_labels) != np.array(current_labels))[0]
        if any(set(changed_indices) == set(prev) for prev in history_changes):
            if verbose:
                print(f'Early stop because of duplicate reassignment at iteration {i+1}.')
            return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
        history_changes.append(list(changed_indices))
        if len(history_changes) > MAX_HISTORY:
            history_changes.pop(0)
            
        if (i == max_iters-1) and verbose:
            print(f'Unconverged at iteration {i+1}!')

        cn_dist_cond = new_cn_dist_cond
        current_labels = new_labels
    
    return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond


# def cell2cellniche_cond1(cn_labels, ct_onehot, micro_dist, cn_dist_basic, n_basic_niche, label_summary,
#                         precompute=None, n_celltypes=None, metric='jsd', max_iters=100, tol=1e-4, change2str=False, 
#                         parallel=True, sparse=True, verbose=True):

#     current_labels = cn_labels.copy()
#     n_cells = len(current_labels)
#     low_reassigned_rate_iter = 0
#     history_changes = []
#     MAX_HISTORY = 10

#     # no new niche
#     if n_basic_niche == len(label_summary):
#         dist_gap = measure_distribution_gap(cn_dist_basic, micro_dist, metric=metric, precompute=precompute, 
#                                             weight=None, parallel=parallel, eps=1e-10)

#         new_indices = np.argmin(dist_gap, axis=1)
#         new_labels = [label_summary[idx] for idx in new_indices]
#         if verbose:
#             print(f'No new cell niche, all cells assigned to basic niches.')
#         return new_labels, None, [], None

#     label_summary_basic = label_summary[:n_basic_niche].copy()
#     label_summary_cond = label_summary[n_basic_niche:].copy()

#     # filter labels
#     selected_indices = np.where(np.isin(np.array(current_labels), np.array(label_summary_cond)))[0]
#     filtered_labels = list(np.array(current_labels)[selected_indices])
#     filtered_ct_onehot = ct_onehot.tocsr()[selected_indices, :]

#     cn_dist_cond, cell_count_niche_cond = calculate_distribution(filtered_labels, 
#                                                                  filtered_ct_onehot, 
#                                                                  label_summary=label_summary_cond, 
#                                                                  n_niches=len(label_summary_cond), 
#                                                                  n_celltypes=n_celltypes, 
#                                                                  change2str=change2str,
#                                                                  sparse=sparse,
#                                                                  )

#     if verbose:
#         iterator = tqdm(range(max_iters))
#     else:
#         iterator = range(max_iters)
    
#     for i in iterator:

#     # for i in range(max_iters):

#         # assign cell to cell niches
#         cn_dist = sp.vstack([cn_dist_basic, cn_dist_cond])
#         dist_gap = measure_distribution_gap(cn_dist, micro_dist, metric=metric, precompute=precompute, 
#                                             weight=None, parallel=parallel, eps=1e-10)

#         new_indices = np.argmin(dist_gap, axis=1)
#         new_labels = [label_summary[idx] for idx in new_indices]

#         new_selected_indices = np.where(np.isin(np.array(new_labels), np.array(label_summary_cond)))[0]
#         if len(new_selected_indices) == 0:
#             if verbose:
#                 print(f'No cell assigned to new cell niche at iteration {i+1}.')
#             return new_labels, None, [], None
#         new_filtered_labels = list(np.array(new_labels)[new_selected_indices])
#         new_filtered_ct_onehot = ct_onehot.tocsr()[new_selected_indices, :]

#         label_summary_cond = sorted(set(new_filtered_labels))
#         label_summary = label_summary_basic + label_summary_cond

#         # calculate cell distribution for cell niches
#         new_cn_dist_cond, cell_count_niche_cond = calculate_distribution(new_filtered_labels, 
#                                                                          new_filtered_ct_onehot, 
#                                                                          label_summary=label_summary_cond, 
#                                                                          n_niches=len(label_summary_cond), 
#                                                                          n_celltypes=n_celltypes, 
#                                                                          change2str=change2str,
#                                                                          sparse=sparse,
#                                                                          )
#         # if new_cn_dist_cond.shape[1] > 20:
#         #     new_cn_dist_cond = refine_dist(new_cn_dist_cond, k=20)

#         # check convergence
#         n_reassigned = np.sum(np.array(new_labels) != np.array(current_labels))
#         # if verbose:
#         #     print(f'Reassigned labels amount: {n_reassigned}.')

#         # condition 1: distribution of cell niches converge
#         if new_cn_dist_cond.shape[0] == cn_dist_cond.shape[0]: 

#             if sp.issparse(new_cn_dist_cond):
#                 new_cn_dist_cond_copy = new_cn_dist_cond.copy().toarray()
#             else:
#                 new_cn_dist_cond_copy = new_cn_dist_cond.copy()
#             if sp.issparse(cn_dist_cond):
#                 cn_dist_cond_copy = cn_dist_cond.copy().toarray()
#             else:
#                 cn_dist_cond_copy = cn_dist_cond.copy()

#             if np.array_equal(new_cn_dist_cond_copy, cn_dist_cond_copy):
#                 if verbose:
#                     print(f'Strictly converge at iteration {i+1}.')
#                 # new_cn_dist = sp.vstack([cn_dist_basic, new_cn_dist_cond])
#                 return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
            
#             if np.sqrt(np.mean((new_cn_dist_cond_copy - cn_dist_cond_copy) ** 2)) < tol:
#             # if np.mean(np.linalg.norm(new_cn_dist_cond_copy - cn_dist_cond_copy, axis=1)) < tol:
#                 if verbose:
#                     print(f'Distribution of cell niches (centers) converge at iteration {i+1}.')
#                 # new_cn_dist = sp.vstack([cn_dist_basic, new_cn_dist_cond])
#                 return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
            
#         # condition 2: few cells are reassigned
#         if n_reassigned <= min(0.002 * n_cells, 1000):
#             low_reassigned_rate_iter += 1
#             if n_reassigned == 0:
#                 if verbose:
#                     print(f'Strictly converge at iteration {i+1}.')
#                 # new_cn_dist = sp.vstack([cn_dist_basic, new_cn_dist_cond])
#                 return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
#             elif low_reassigned_rate_iter >= MAX_HISTORY:
#                 if verbose:
#                     print(f'Converge at iteration {i+1}.')
#                 # new_cn_dist = sp.vstack([cn_dist_basic, new_cn_dist_cond])
#                 return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
#         else:
#             low_reassigned_rate_iter = 0
        
#         # condition 3: duplicate reassignment
#         changed_indices = np.where(np.array(new_labels) != np.array(current_labels))[0]
#         if any(set(changed_indices) == set(prev) for prev in history_changes):
#             if verbose:
#                 print(f'Early stop because of duplicate reassignment at iteration {i+1}.')
#             # new_cn_dist = sp.vstack([cn_dist_basic, new_cn_dist_cond])
#             return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond
#         history_changes.append(list(changed_indices))
#         if len(history_changes) > MAX_HISTORY:
#             history_changes.pop(0)
            
#         if (i == max_iters-1) and verbose:
#             print(f'Unconverged at iteration {i+1}!')

#         cn_dist_cond = new_cn_dist_cond
#         current_labels = new_labels
    
#     # new_cn_dist = sp.vstack([cn_dist_basic, new_cn_dist_cond])
#     return new_labels, new_cn_dist_cond, label_summary_cond, cell_count_niche_cond


def fast_silhouette_score(X, labels, sample_size=1000, n_iterations=1000, min_count=50, 
                          equal_prob=False, replace=False, verbose=True):
    
    X_copy = X.copy()

    labels_copy = np.array(labels.copy())
    unique_labels, label_counts = np.unique(labels_copy, return_counts=True)
    label_proportions = label_counts / len(labels_copy)
    if equal_prob:
        probs = [1/len(unique_labels) for _ in range(len(unique_labels))]
    else:
        probs = label_proportions

    silhouette_scores = []
    
    if verbose:
        iterator = tqdm(range(n_iterations))
    else:
        iterator = range(n_iterations)
    
    for i in iterator:
        
        sampled_indices = []

        for label, proportion, count in zip(unique_labels, probs, label_counts):
            
            if replace:
                lower_bound = min_count
            else:
                lower_bound = min(min_count, count)
            n_samples_for_label = max(int(proportion * sample_size), lower_bound)
            
            label_indices = np.where(labels_copy == label)[0]

            sampled_label_indices = np.random.choice(label_indices, size=n_samples_for_label, replace=replace)
            sampled_indices.extend(sampled_label_indices)
        
        if sp.issparse(X_copy):
            X_copy = X_copy.tocsr()
            X_sample = X_copy[sampled_indices, :].toarray()
        else:
            X_sample = X_copy[sampled_indices, :]
        labels_sample = labels_copy[sampled_indices]
        
        score = silhouette_score(X_sample, labels_sample)
        silhouette_scores.append(score)
    
    return np.mean(silhouette_scores)


def plot_minjsd_score(score_list, cn_count_list, threshold, 
                      fig_size=None, plot=True, save=False, save_dir='./', file_name='score_vs_nichecount.pdf', **kwargs):

    if fig_size is None:
        fig_size = (int(max(cn_count_list)), int(max(cn_count_list)/2))

    fig, ax = plt.subplots(figsize=fig_size)

    ax.plot(
        cn_count_list, 
        score_list, 
        label='Score',
        marker=kwargs.get('marker', 'o'), 
        markersize=kwargs.get('marker_size', 5),        
        color=kwargs.get('color', 'black'),              
        linewidth=kwargs.get('linewidth', 2),
        )

    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold}', linewidth=kwargs.get('linewidth', 2))

    ax.set_xticks(cn_count_list)
    ax.set_xticklabels(cn_count_list)
    ax.tick_params(axis='x', labelsize=kwargs.get('tick_fontsize', 10))
    ax.tick_params(axis='y', labelsize=kwargs.get('tick_fontsize', 10))

    ax.invert_xaxis()

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.set_xlabel(kwargs.get('xlabel', "Niche Count"), fontsize=kwargs.get('label_fontsize', 12))
    ax.set_ylabel(kwargs.get('ylabel', "Score"), fontsize=kwargs.get('label_fontsize', 12))

    ax.legend(loc=kwargs.get('legend_loc', 'upper left'), fontsize=kwargs.get('legend_fontsize', 10))

    plt.title(kwargs.get('title', "Score vs Niche Count"), fontsize=kwargs.get('title_fontsize', 14))
    plt.grid(kwargs.get('grid', True))

    show_spines = kwargs.get('show_spines', False)
    if not show_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    plt.tight_layout()

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + file_name)
    
    if plot:
        plt.show()


def ctr_cond_merge(ctr_concat, cond_concat, niche_dist_key='niche_dist', cell_count_key='niche_cell_count'):

    if sp.issparse(ctr_concat.uns[niche_dist_key]):
        mtx1 = ctr_concat.uns[niche_dist_key].toarray() * ctr_concat.uns[cell_count_key][:, np.newaxis]
    else:
        mtx1 = ctr_concat.uns[niche_dist_key] * ctr_concat.uns[cell_count_key][:, np.newaxis]
    
    if sp.issparse(cond_concat.uns[niche_dist_key]):
        mtx2 = cond_concat.uns[niche_dist_key].toarray() * cond_concat.uns[cell_count_key][:, np.newaxis]
    else:
        mtx2 = cond_concat.uns[niche_dist_key] * cond_concat.uns[cell_count_key][:, np.newaxis]

    mtx2[0:mtx1.shape[0], :] = mtx2[0:mtx1.shape[0], :] + mtx1

    row_sums = mtx2.sum(axis=1, keepdims=True)
    mtx2 = mtx2 / row_sums

    row_sums = row_sums.flatten()
    mtx2 = sp.csr_matrix(mtx2)

    return mtx2, row_sums


def refine_dist(mat, k=20):

    if sp.issparse(mat):
        mat = mat.toarray()
    
    if k >= 1: 
        k = int(k)
        row_indices = np.arange(mat.shape[0])[:, None]
        topk_idx = np.argpartition(-mat, k, axis=1)[:, :k]
        mask = np.zeros_like(mat, dtype=bool)
        mask[row_indices, topk_idx] = True
    elif k >= 0 :
        mask = mat >= k
    else:
        raise ValueError('k should be greater than 0 !')

    mat[~mask] = 0

    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / row_sums

    return sp.csr_matrix(mat)