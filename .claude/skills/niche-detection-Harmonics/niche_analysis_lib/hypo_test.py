import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

from .utils import *


def ct_enrichment_test(niche_dist, cell_count_niche, idx2ct_dict, niche_summary,
                       method='fisher', alpha=0.05, fdr_method='fdr_by', log2fc_threshold=1, prop_threshold=0.01, 
                       verbose=True, eps=1e-10):
    
    if sp.issparse(niche_dist):
        niche_dist_copy = niche_dist.copy().toarray()
    else:
        niche_dist_copy = niche_dist.copy()
    
    if prop_threshold is None:
        prop_threshold = 0
    elif prop_threshold > 1:
        prop_threshold /= 100
        print(f'Proportion threshold: {prop_threshold}.')
    
    niche_data = niche_dist_copy * cell_count_niche[:, np.newaxis]
    
    cell_count_ct = np.sum(niche_data, axis=0)
    n_cell_total = np.sum(cell_count_niche)

    niche_indices = []
    niche_list = []
    ct_indices = []
    ct = []
    oddsratio_list = []
    chi2_list = []
    p_values = []
    log2fc = []
    prop = []

    n_niche, n_celltype = niche_data.shape

    if verbose:
        print(f'{n_niche} niches and {n_celltype} cell types in total.\n')

    for i in range(n_niche):

        for j in range(n_celltype):

            a = niche_data[i][j]
            b = cell_count_niche[i] - a
            c = cell_count_ct[j] - a
            d = n_cell_total - a - b - c
            
            observed = np.array([[a, b], [c, d]])

            if method.lower() == 'fisher':
                oddsratio, p_val = stats.fisher_exact(observed, alternative='two-sided')
            elif method.lower() == 'fisher_greater':
                oddsratio, p_val = stats.fisher_exact(observed, alternative='greater')
            elif method.lower() == 'chi2':
                chi2, p_val, _, _ = stats.chi2_contingency(observed, correction=True)
            else:
                raise ValueError(f"Unsupported method {method}! Supported methods are 'fisher', 'fisher_greater', and 'chi2'.")
            
            fc_ = observed[:, 0] / (observed[:, 0] + observed[:, 1] + eps)

            niche_indices.append(i)
            niche_list.append(niche_summary[i])
            ct_indices.append(j)
            ct.append(idx2ct_dict[str(j)])
            p_values.append(p_val)
            log2fc.append(np.log2((fc_[0] + eps)/(fc_[1] + eps)))
            prop.append(niche_dist_copy[i][j])

            if method.lower() == 'chi2':
                chi2_list.append(chi2)
            else:
                oddsratio_list.append(oddsratio)
    
    rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=fdr_method)

    rejected = [True if rejected[i] and log2fc[i] >= log2fc_threshold and prop[i] >= prop_threshold 
                else False for i in range(len(log2fc))]

    if method.lower() == 'chi2':
        df_results = pd.DataFrame({'niche_idx': niche_indices, 
                                   'niche': niche_list,
                                   'celltype_idx': ct_indices, 
                                   'celltype': ct,
                                   'chi2_stat': chi2_list, 
                                   'p-value': p_values,
                                   'q-value': pvals_corrected,
                                   'log2fc': log2fc,
                                   'prop': prop,
                                   'enrichment': rejected,
                                   })
    else:
        df_results = pd.DataFrame({'niche_idx': niche_indices,
                                   'niche': niche_list,
                                   'celltype_idx': ct_indices, 
                                   'celltype': ct,
                                   'oddsratio': oddsratio_list, 
                                   'p-value': p_values,
                                   'q-value': pvals_corrected,
                                   'log2fc': log2fc,
                                   'prop': prop,
                                   'enrichment': rejected,
                                   })
    
    return df_results


def cci_enrichment_test(adata_list, niche_key, ct_key, niche_summary=None, spatial_key='spatial', cut_percentage=99, 
                        method='fisher', alpha=0.05, fdr_method='fdr_by', log2fc_threshold=1, prop_threshold=0.01, 
                        # symmetric=True,
                        verbose=True, eps=1e-10):
    
    if not isinstance(adata_list, list):
        adata_list_copy = [adata_list]
    else:
        adata_list_copy = adata_list.copy()

    if prop_threshold is None:
        prop_threshold = 0
    elif prop_threshold > 1:
        prop_threshold /= 100
        print(f'Proportion threshold: {prop_threshold}.')

    adata_concat = ad.concat(adata_list_copy)
    if niche_summary is None:
        niche_summary = sorted(set(adata_concat.obs[niche_key]))
    ct_summary = sorted(set(adata_concat.obs[ct_key]))

    n_niche = len(niche_summary)
    n_celltype = len(ct_summary)

    ct2idx_dict = {ct_summary[idx]: idx for idx in range(n_celltype)}
    idx2ct_dict = {idx: ct_summary[idx] for idx in range(n_celltype)}

    if verbose:
        print(f'{n_niche} niches and {n_celltype} cell types in total.\n')
    
    niche_indices = []
    niche_list = []
    ct1_indices = []
    ct1 = []
    ct2_indices = []
    ct2 = []
    test_count = []
    bg_count = []
    test_prop = []
    bg_prop = []
    oddsratio_list = []
    p_values = []
    log2fc = []

    test_norm_list = []
    bg_norm_list = []

    test_edge_count_list = []
    bg_edge_count_list = []

    Delaunay_adjacency_mtx(adata_list_copy,
                           spatial_key=spatial_key, 
                           cut_percentage=cut_percentage, 
                           return_adata=False,
                           verbose=False,
                           )

    for idx, niche in enumerate(niche_summary):

        if verbose:
            print(f'Testing niche {niche}...')
        
        test_count_mtx = np.zeros((n_celltype, n_celltype))
        bg_count_mtx = np.zeros((n_celltype, n_celltype))

        test_pairs = []
        bg_pairs = []
        for i in range(len(adata_list_copy)):
            adj_mtx = adata_list_copy[i].obsp['delaunay_adj_mtx']
            ct_labels = adata_list_copy[i].obs[ct_key]
            cn_labels = adata_list_copy[i].obs[niche_key]

            rows, cols = adj_mtx.nonzero()
            mask = rows < cols
            rows = rows[mask]
            cols = cols[mask]
            
            test_pairs += [(ct_labels[j1], ct_labels[j2]) for j1, j2 in zip(rows, cols) if 
                           (cn_labels[j1] == niche and cn_labels[j2] == niche)]
            bg_pairs += [(ct_labels[j1], ct_labels[j2]) for j1, j2 in zip(rows, cols) if 
                         (cn_labels[j1] != niche and cn_labels[j2] != niche)]

        pair_counts = pd.Series(test_pairs).value_counts()
        for (c1, c2), count in pair_counts.items():
            idx1 = ct2idx_dict[c1]
            idx2 = ct2idx_dict[c2]
            test_count_mtx[max(idx1, idx2), min(idx1, idx2)] += count
            # if not symmetric and idx1 != idx2:
            #     test_count_mtx[min(idx1, idx2), max(idx1, idx2)] += count

        pair_counts = pd.Series(bg_pairs).value_counts()
        for (c1, c2), count in pair_counts.items():
            idx1 = ct2idx_dict[c1]
            idx2 = ct2idx_dict[c2]
            bg_count_mtx[max(idx1, idx2), min(idx1, idx2)] += count
            # if not symmetric and idx1 != idx2:
            #     bg_count_mtx[min(idx1, idx2), max(idx1, idx2)] += count

        # if symmetric:
        test_edge_count = np.sum(test_count_mtx)
        bg_edge_count = np.sum(bg_count_mtx)

        test_norm = test_count_mtx / test_edge_count
        bg_norm = bg_count_mtx / bg_edge_count
        # else:
        #     test_edge_count = np.sum(test_count_mtx, axis=1)
        #     bg_edge_count = np.sum(bg_count_mtx, axis=1)

        #     test_norm = test_count_mtx / (test_edge_count[:, np.newaxis] + eps)
        #     bg_norm = bg_count_mtx / (bg_edge_count[:, np.newaxis] + eps)

        test_norm_list.append(test_norm)
        bg_norm_list.append(bg_norm)

        test_edge_count_list.append(test_edge_count)
        bg_edge_count_list.append(bg_edge_count)

        for c1 in range(n_celltype):

            for c2 in range(n_celltype):

                # if symmetric and c1 < c2: continue
                if c1 < c2: continue

                # if symmetric:
                a = test_count_mtx[c1][c2]
                b = test_edge_count - a
                c = bg_count_mtx[c1][c2]
                d = bg_edge_count - c
                # else:
                #     a = test_count_mtx[c1][c2]
                #     b = test_edge_count[c1] - a
                #     c = bg_count_mtx[c1][c2]
                #     d = bg_edge_count[c1] - c
                
                observed = np.array([[a, b], [c, d]])

                if method.lower() == 'fisher':
                    oddsratio, p_val = stats.fisher_exact(observed, alternative='two-sided')
                elif method.lower() == 'fisher_greater':
                    oddsratio, p_val = stats.fisher_exact(observed, alternative='greater')
                else:
                    raise ValueError(f"Unsupported method {method}! Supported methods are 'fisher' and 'fisher_greater'.")
                
                fc_ = observed[:, 0] / (observed[:, 0] + observed[:, 1] + eps)

                niche_indices.append(idx)
                niche_list.append(niche)
                ct1_indices.append(c1)
                ct1.append(idx2ct_dict[c1])
                ct2_indices.append(c2)
                ct2.append(idx2ct_dict[c2])
                test_count.append(test_count_mtx[c1][c2])
                bg_count.append(bg_count_mtx[c1][c2])
                test_prop.append(test_norm[c1][c2])
                bg_prop.append(bg_norm[c1][c2])
                p_values.append(p_val)
                log2fc.append(np.log2((fc_[0] + eps)/(fc_[1] + eps)))

                oddsratio_list.append(oddsratio)
            
    rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=fdr_method)

    rejected = [True if rejected[i] and log2fc[i] >= log2fc_threshold and test_prop[i] >= prop_threshold 
                else False for i in range(len(log2fc))]

    if verbose:
        print('Finished!')
        
    df_results = pd.DataFrame({'niche_idx': niche_indices,  
                               'niche': niche_list,
                               'ct1_idx': ct1_indices,
                               'ct1': ct1,
                               'ct2_idx': ct2_indices,
                               'ct2': ct2,
                               'test_edge_count': test_count,
                               'bg_edge_count': bg_count,
                               'test_edge_prop': test_prop,
                               'bg_edge_prop': bg_prop,
                               'oddsratio': oddsratio_list, 
                               'p-value': p_values,
                               'q-value': pvals_corrected,
                               'log2fc': log2fc,
                               'enrichment': rejected,
                               })
    
    return df_results, test_norm_list, bg_norm_list, test_edge_count_list, bg_edge_count_list


def nnc_enrichment_test(adata_list, niche_key, niche_summary=None, spatial_key='spatial', cut_percentage=99, 
                        method='fisher', alpha=0.05, fdr_method='fdr_by', log2fc_threshold=1, prop_threshold=0.01, 
                        verbose=True, eps=1e-10):
    
    if not isinstance(adata_list, list):
        adata_list_copy = [adata_list]
    else:
        adata_list_copy = adata_list.copy()

    adata_concat = ad.concat(adata_list_copy)
    if niche_summary is None:
        niche_summary = sorted(set(adata_concat.obs[niche_key]))

    if prop_threshold is None:
        prop_threshold = 0
    elif prop_threshold > 1:
        prop_threshold /= 100
        print(f'Proportion threshold: {prop_threshold}.')
    
    n_niche = len(niche_summary)

    cn2idx_dict = {niche_summary[idx]: idx for idx in range(n_niche)}
    idx2cn_dict = {idx: niche_summary[idx] for idx in range(n_niche)}

    if verbose:
        print(f'{n_niche} niches in total.\n')

    Delaunay_adjacency_mtx(adata_list_copy,
                           spatial_key=spatial_key, 
                           cut_percentage=cut_percentage, 
                           return_adata=False,
                           verbose=False,
                           )
    
    niche1_indices = []
    niche1_list = []
    niche2_indices = []
    niche2_list = []
    edge_count = []
    edge_prop = []
    oddsratio_list = []
    chi2_list = []
    p_values = []
    log2fc = []

    n2n_mtx = np.zeros((n_niche, n_niche))

    pairs = []
    for i in range(len(adata_list_copy)):

        adj_mtx = adata_list_copy[i].obsp['delaunay_adj_mtx']
        labels = adata_list_copy[i].obs[niche_key]

        rows, cols = adj_mtx.nonzero()
        # mask = np.array([labels[j1] != labels[j2] for j1, j2 in zip(rows, cols)])
        # rows = rows[mask]
        # cols = cols[mask]

        # for r, c in zip(rows, cols):
        #     idx1 = cn2idx_dict[labels[r]]
        #     idx2 = cn2idx_dict[labels[c]]
        #     n2n_mtx[idx1, idx2] += 1

        pairs += [(labels[j1], labels[j2]) for j1, j2 in zip(rows, cols) if labels[j1] != labels[j2]]

    pair_counts = pd.Series(pairs).value_counts()
    for (n1, n2), count in pair_counts.items():
        idx1 = cn2idx_dict[n1]
        idx2 = cn2idx_dict[n2]
        n2n_mtx[idx1, idx2] += count

    n1_count = np.sum(n2n_mtx, axis=1)
    n2_count = np.sum(n2n_mtx, axis=0)
    total_count = np.sum(n2n_mtx)

    edge_prop_mtx = n2n_mtx/n1_count[:, np.newaxis]

    for n1 in range(n_niche):

        for n2 in range(n_niche):

            if n1 == n2: continue

            a = n2n_mtx[n1][n2]
            b = n1_count[n1] - a
            c = n2_count[n2] - a
            d = total_count - a - b - c
            
            observed = np.array([[a, b], [c, d]])

            if method.lower() == 'fisher':
                oddsratio, p_val = stats.fisher_exact(observed, alternative='two-sided')
            elif method.lower() == 'fisher_greater':
                oddsratio, p_val = stats.fisher_exact(observed, alternative='greater')
            elif method.lower() == 'chi2':
                chi2, p_val, _, _ = stats.chi2_contingency(observed, correction=True)
            else:
                raise ValueError(f"Unsupported method {method}! Supported methods are 'fisher', 'fisher_greater', and 'chi2'.")
            
            fc_ = observed[:, 0] / (observed[:, 0] + observed[:, 1] + eps)

            niche1_indices.append(n1)
            niche1_list.append(idx2cn_dict[n1])
            niche2_indices.append(n2)
            niche2_list.append(idx2cn_dict[n2])
            edge_count.append(n2n_mtx[n1][n2])
            edge_prop.append(edge_prop_mtx[n1][n2])
            p_values.append(p_val)
            log2fc.append(np.log2((fc_[0] + eps)/(fc_[1] + eps)))

            if method.lower() == 'chi2':
                chi2_list.append(chi2)
            else:
                oddsratio_list.append(oddsratio)

    rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=fdr_method)

    rejected = [True if rejected[i] and log2fc[i] >= log2fc_threshold and edge_prop[i] >= prop_threshold 
                else False for i in range(len(log2fc))]

    if method.lower() == 'chi2':
        df_results = pd.DataFrame({'niche1_idx': niche1_indices, 
                                   'niche1': niche1_list,
                                   'niche2_idx': niche2_indices,
                                   'niche2': niche2_list,
                                   'edge_count': edge_count,
                                   'edge_prop': edge_prop,
                                   'chi2_stat': chi2_list, 
                                   'p-value': p_values,
                                   'q-value': pvals_corrected,
                                   'log2fc': log2fc,
                                   'enrichment': rejected,
                                   })
    else:
        df_results = pd.DataFrame({'niche1_idx': niche1_indices, 
                                   'niche1': niche1_list,
                                   'niche2_idx': niche2_indices,
                                   'niche2': niche2_list,
                                   'edge_count': edge_count,
                                   'edge_prop': edge_prop,
                                   'oddsratio': oddsratio_list, 
                                   'p-value': p_values,
                                   'q-value': pvals_corrected,
                                   'log2fc': log2fc,
                                   'enrichment': rejected,
                                   })
    
    return df_results, edge_prop_mtx, n1_count

    