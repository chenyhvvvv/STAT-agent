import anndata as ad
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from .utils import *


class Harmonics_Model():

    def __init__(self,
                 adata_list, 
                 slice_name_list,
                 cond_list=None,
                 cond_name_list=None,
                 concat_label='slice_name',
                 proportion_label=None,
                 refine_k=0,
                 seed=1234,
                 parallel=True,
                 verbose=True,
                 ):
        
        # basic (normal) slices
        if not isinstance(adata_list, list):
            adata_list = [adata_list] 
        self.adata_list = adata_list
        self.adata_concat = ad.concat(adata_list, label=concat_label, keys=slice_name_list)

        self.n_slices_basic = len(adata_list)
        self.n_cells_basic = self.adata_concat.shape[0]

        self.seed = seed
        self.parallel = parallel
        self.verbose = verbose
        self.withcond = False

        self.proportion_label = proportion_label  # cell type distriution label for each spot, saved in .obsm[proportion_label]
        # if k >= 1: retain the top k cell types by proportion when computing niche distributions
        # if 0 < k < 1: drop all cell types with proportion < k when computing niche distributions
        # if k == 0: no niche refinement, use all cell types
        self.refine_k = refine_k  

        # condition slices
        if cond_list is not None:
            self.withcond = True
            if cond_name_list is None:
                raise ValueError('Please provide slice name(s) for condition slice(s)!')
            else:
                if not isinstance(cond_list, list):
                    cond_list = [cond_list] 
                self.cond_list = cond_list
                self.cond_concat = ad.concat(cond_list, label=concat_label, keys=cond_name_list)

                self.n_slices_cond = len(cond_list)
                self.n_cells_cond = self.cond_concat.shape[0]
        
        if self.withcond:
            self.n_slices = self.n_slices_basic + self.n_slices_cond
            self.n_cells = self.n_cells_basic + self.n_cells_cond
        else:
            self.n_slices = self.n_slices_basic
            self.n_cells = self.n_cells_basic

        if self.verbose:
            if self.withcond:
                print(f'Control set comprises {self.n_slices_basic} slices, {self.n_cells_basic} cells/spots in total.')
                print(f'Condition set comprises {self.n_slices_cond} slices, {self.n_cells_cond} cells/spots in total.')
            else:
                print(f'Dataset comprises {self.n_slices_basic} slices, {self.n_cells_basic} cells/spots in total.')

        # self.n_celltypes = None
    
        # self.adj_mtx_list_basic = []  # list for neighbor graph

        # self.ct_idx_list_basic = []  # list for cell type indices
        # self.ct_indices_basic = None  # concatenated cell type indices

        # self.ct_onehot_list_basic = []  # list for one-hot cell embedding
        # self.ct_onehot_basic = None  # concatenated one-hot cell embedding (N * K)

        # self.micro_dist_list_basic = []  # list for microenvironment cell type distribution
        # self.micro_dist_basic = None  # concatenated microenvironment cell type distribution (N * K)

        # self.n_neighbor_list_basic = []  # list for neighbor amount of cells from each slice 
        # self.cell_count_micro_basic = None  # concatenated neighbor amount of cells

        # self.init_cn_label_basic = None  # initial niche label for each cell
        # self.init_cn_count_basic = None  # initial niche count

        # self.cn_label_list_basic = []  # list for assigned cell niche label for each cell
        # self.cn_count_list_basic = []  # list for niche count

        # self.score_list_minjsd_basic = []  # list for scores
        # self.weighted_score_list_minjsd_basic = []

        # # selected solution
        # self.cn_label_summary_basic = None  # summary of cell niche label 
        # self.cn_label_basic = None  # cell niche labels for each cell
        # self.cn_dist_basic = None  # cell type distrbution for niches (Q * K)
        # self.cell_count_niche_basic = None  # cell counts for each niche

        # if self.withcond:

        #     self.adj_mtx_list_cond = []  # list for neighbor graph

        #     self.ct_idx_list_cond = []  # list for cell type indices
        #     self.ct_indices_cond = None  # concatenated cell type indices

        #     self.ct_onehot_list_cond = []  # list for one-hot cell embedding
        #     self.ct_onehot_cond = None  # concatenated one-hot cell embedding

        #     self.micro_dist_list_cond = []  # list for microenvironment cell type distribution
        #     self.micro_dist_cond = None  # concatenated microenvironment cell type distribution

        #     self.n_neighbor_list_cond = []  # list for neighbor amount of cells from each slice 
        #     self.cell_count_micro_cond = None  # concatenated neighbor amount of cells

        #     self.init_cn_label_cond = None  # initial niche label for each cell
        #     self.init_cn_count_cond = None  # initial new niche count

        #     self.cn_label_list_cond = []  # list for assigned cell niche label for each cell
        #     self.cn_count_list_cond = []  # list for new niche count

        #     self.score_list_minjsd_cond = []  # list for scores
        #     self.weighted_score_list_minjsd_cond = []

        #     # selected solution
        #     self.cn_label_summary_cond = None  # summary of cell niche label 
        #     self.cn_label_cond = None  # cell niche labels for each cell
        #     self.cn_dist_cond = None  # cell type distrbution for niches (R * K)
        #     self.cell_count_niche_cond = None  # cell counts for each niche
        

    def preprocess(self, 
                   ct_key='celltype',
                   spatial_key='spatial',
                   method='joint',
                   n_step=3, 
                   n_neighbors=20,
                   cut_percentage=99,
                   ):
        
        # initialize
        self.n_celltypes = None

        self.adj_mtx_list_basic = []  # list for neighbor graph

        self.ct_idx_list_basic = []  # list for cell type indices
        self.ct_indices_basic = None  # concatenated cell type indices

        self.ct_onehot_list_basic = []  # list for one-hot cell embedding
        self.ct_onehot_basic = None  # concatenated one-hot cell embedding (N * K)

        self.micro_dist_list_basic = []  # list for microenvironment cell type distribution
        self.micro_dist_basic = None  # concatenated microenvironment cell type distribution (N * K)

        self.n_neighbor_list_basic = []  # list for neighbor amount of cells from each slice 
        self.cell_count_micro_basic = None  # concatenated neighbor amount of cells

        if self.withcond:

            self.adj_mtx_list_cond = []  # list for neighbor graph

            self.ct_idx_list_cond = []  # list for cell type indices
            self.ct_indices_cond = None  # concatenated cell type indices

            self.ct_onehot_list_cond = []  # list for one-hot cell embedding
            self.ct_onehot_cond = None  # concatenated one-hot cell embedding

            self.micro_dist_list_cond = []  # list for microenvironment cell type distribution
            self.micro_dist_cond = None  # concatenated microenvironment cell type distribution

            self.n_neighbor_list_cond = []  # list for neighbor amount of cells from each slice 
            self.cell_count_micro_cond = None  # concatenated neighbor amount of cells
        

        if self.withcond:
            total_list = self.adata_list.copy() + self.cond_list.copy()
        else:
            total_list = self.adata_list.copy()
        
        # compute adjacency matrix (part 1)
        if method is not None:
            if method.lower() == 'delaunay':
                Delaunay_adjacency_mtx(total_list,
                                    spatial_key=spatial_key, 
                                    cut_percentage=cut_percentage, 
                                    return_adata=False,
                                    verbose=self.verbose,
                                    )
            elif method.lower() == 'knn':
                knn_adjacency_matrix(total_list, 
                                    spatial_key=spatial_key, 
                                    n_neighbors=n_neighbors, 
                                    return_adata=False, 
                                    verbose=self.verbose,
                                    )
            elif method.lower() == 'joint':
                joint_adjacency_matrix(total_list, 
                                    spatial_key=spatial_key, 
                                    cut_percentage=cut_percentage, 
                                    n_step=n_step, 
                                    n_neighbors=n_neighbors, 
                                    return_adata=False, 
                                    verbose=self.verbose,
                                    )
            else:
                raise ValueError(f"Unknown method {method}. Supported methods are 'delaunay', 'knn', and 'joint'.")
            
        elif self.proportion_label is None:
            raise ValueError("A method should be provided unless the proportion label is specificed, which is only recommended for low resolution data.\n"
                             "Supported methods are 'delaunay', 'knn', and 'joint'")
        
        # generate one-hot matrix
        if self.proportion_label is None:
            label2onehot_anndata(total_list, 
                                 ct_key=ct_key, 
                                 return_adata=False,
                                 sparse=True, 
                                 verbose=self.verbose,
                                 )

        for i in range(self.n_slices):

            # compute adjacency matrix (part 2)
            if method is not None:
                if method.lower() == 'delaunay':
                    adj_mtx = total_list[i].obsp[f'delaunay_adj_mtx']
                    adj_mtx = adj_mtx + sp.eye(adj_mtx.shape[0])
                    adj_mtx = sp.linalg.matrix_power(adj_mtx, n_step)
                    adj_mtx = (adj_mtx > 0).astype(int)
                elif method.lower() == 'knn':
                    adj_mtx = total_list[i].obsp[f'knn_adj_mtx_{n_neighbors}']
                    adj_mtx = adj_mtx + sp.eye(adj_mtx.shape[0])
                else:
                    adj_mtx = total_list[i].obsp[f'joint_adj_mtx_{n_neighbors}']
                    adj_mtx = adj_mtx + sp.eye(adj_mtx.shape[0])  # no use 
                    adj_mtx = (adj_mtx > 0).astype(int)  # no use 
                
                if i < self.n_slices_basic:
                    self.adj_mtx_list_basic.append(adj_mtx)
                else:
                    self.adj_mtx_list_cond.append(adj_mtx)
                
                if self.proportion_label is None:
                    ct_indices = list(total_list[i].obs['celltype_idx'])
                    onehot_mtx = total_list[i].obsm['onehot']

                    if i < self.n_slices_basic:
                        self.ct_idx_list_basic.append(ct_indices)
                        self.ct_onehot_list_basic.append(onehot_mtx)
                    else:
                        self.ct_idx_list_cond.append(ct_indices)
                        self.ct_onehot_list_cond.append(onehot_mtx)
                else:
                    onehot_mtx = total_list[i].obsm[self.proportion_label]  # not one-hot

                    if i < self.n_slices_basic:
                        self.ct_onehot_list_basic.append(sp.csr_matrix(onehot_mtx))
                    else:
                        self.ct_onehot_list_cond.append(sp.csr_matrix(onehot_mtx))
            else:
                onehot_mtx = total_list[i].obsm[self.proportion_label]  # not one-hot

                if i < self.n_slices_basic:
                    self.ct_onehot_list_basic.append(sp.csr_matrix(onehot_mtx))
                else:
                    self.ct_onehot_list_cond.append(sp.csr_matrix(onehot_mtx))

        self.ct_onehot_basic = sp.vstack(self.ct_onehot_list_basic.copy())
        
        if self.proportion_label is None:
            self.ct_indices_basic = list(np.concatenate([np.array(sublist) for sublist in self.ct_idx_list_basic]))

            self.adata_concat.obs['celltype_idx'] = self.ct_indices_basic.copy()
            self.adata_concat.obsm['onehot'] = self.ct_onehot_basic.copy()
            self.adata_concat.uns['ct2idx'] = total_list[0].uns['ct2idx'].copy()
            self.adata_concat.uns['idx2ct'] = total_list[0].uns['idx2ct'].copy()

        else:
            self.adata_concat.obsm['proportion'] = self.ct_onehot_basic.copy()

        self.n_celltypes = int(self.ct_onehot_basic.shape[1])

        if self.withcond:
            self.ct_onehot_cond = sp.vstack(self.ct_onehot_list_cond.copy())

            if self.proportion_label is None:
                self.ct_indices_cond = list(np.concatenate([np.array(sublist) for sublist in self.ct_idx_list_cond]))

                self.cond_concat.obs['celltype_idx'] = self.ct_indices_cond.copy()
                self.cond_concat.obsm['onehot'] = self.ct_onehot_cond.copy()
                self.cond_concat.uns['ct2idx'] = total_list[0].uns['ct2idx'].copy()
                self.cond_concat.uns['idx2ct'] = total_list[0].uns['idx2ct'].copy()
            else:
                self.cond_concat.obsm['proportion'] = self.ct_onehot_cond.copy()

        if self.verbose:
            print(f'Dataset comprises {self.n_celltypes} cell types.\n')
        
        if self.withcond:
            self.adata_list = [total_list[i] for i in range(self.n_slices_basic)].copy()
            self.cond_list = [total_list[i] for i in range(self.n_slices_basic, self.n_slices)].copy()
        else:
            self.adata_list = total_list.copy()
        
        # calculate cell type distribution for microenvironments
        if self.verbose:
            print('Calculating cell type distribution for microenvironments...')

        if method is not None:
            self.micro_dist_list_basic, self.n_neighbor_list_basic = update_microenvironment(self.adj_mtx_list_basic, 
                                                                                            self.ct_onehot_list_basic, 
                                                                                            n_celltypes=self.n_celltypes, 
                                                                                            n_slices=self.n_slices_basic, 
                                                                                            cut_edge=False, 
                                                                                            cn_labels=None, 
                                                                                            )
        else:
            self.micro_dist_list_basic = self.ct_onehot_list_basic.copy()
            for i in range(len(self.micro_dist_list_basic)):
                self.n_neighbor_list_basic.append(np.ones(self.micro_dist_list_basic[i].shape[0]))

        self.micro_dist_basic = sp.vstack(self.micro_dist_list_basic)
        self.cell_count_micro_basic = np.concatenate([np.array(sublist) for sublist in self.n_neighbor_list_basic])

        for i in range(self.n_slices_basic):

            self.adata_list[i].obs['n_neighbors'] = self.n_neighbor_list_basic[i].copy()
            self.adata_list[i].obsm['micro_dist'] = self.micro_dist_list_basic[i].tocsr().copy()
        
        self.adata_concat.obs['n_neighbors'] = list(self.cell_count_micro_basic.copy())
        self.adata_concat.obsm['micro_dist'] = self.micro_dist_basic.tocsr().copy()
        
        cell_count_micro = self.cell_count_micro_basic.copy()

        if self.withcond:

            if method is not None:
                self.micro_dist_list_cond, self.n_neighbor_list_cond = update_microenvironment(self.adj_mtx_list_cond, 
                                                                                            self.ct_onehot_list_cond, 
                                                                                            n_celltypes=self.n_celltypes, 
                                                                                            n_slices=self.n_slices_cond, 
                                                                                            cut_edge=False, 
                                                                                            cn_labels=None, 
                                                                                            )
            else:
                self.micro_dist_list_cond = self.ct_onehot_list_cond.copy()
                for i in range(len(self.micro_dist_list_cond)):
                    self.n_neighbor_list_cond.append(np.ones(self.micro_dist_list_cond[i].shape[0]))

            self.micro_dist_cond = sp.vstack(self.micro_dist_list_cond)
            self.cell_count_micro_cond = np.concatenate([np.array(sublist) for sublist in self.n_neighbor_list_cond])

            for i in range(self.n_slices_cond):

                self.cond_list[i].obs['n_neighbors'] = self.n_neighbor_list_cond[i].copy()
                self.cond_list[i].obsm['micro_dist'] = self.micro_dist_list_cond[i].tocsr().copy()
            
            self.cond_concat.obs['n_neighbors'] = list(self.cell_count_micro_cond.copy())
            self.cond_concat.obsm['micro_dist'] = self.micro_dist_cond.tocsr().copy()
        
            cell_count_micro = np.concatenate([cell_count_micro, self.cell_count_micro_cond.copy()])

        if self.verbose:
            print(f'Microenvironments comprise {np.mean(cell_count_micro):.2f} cells/spots on average. \n'
                  f'Minimum: {int(np.min(cell_count_micro))}, Maximum: {int(np.max(cell_count_micro))}\n')


    def initialize_clusters(self, 
                            dim_reduction=True,
                            explained_var=None,
                            n_components=None,
                            n_components_max=100,
                            standardize=True, 
                            method='kmeans', 
                            Qmax=20, 
                            ):
        
        # initialize
        self.init_cn_label_basic = None  # initial niche label for each cell
        self.init_cn_count_basic = None  # initial niche count
        
        # dimension reduction
        if method.lower() == 'random':
            labels = np.random.choice(np.arange(Qmax), size=self.n_cells_basic, replace=True)

        else:
            if dim_reduction:
                if self.verbose:
                    print('Performing dimension reduction...')

                x, _, _ = pca(self.micro_dist_basic, 
                            explained_var=explained_var, 
                            n_components=n_components, 
                            n_components_max=n_components_max,
                            standardize=standardize,
                            verbose=self.verbose,
                            )
            else:
                x = self.micro_dist_basic.copy().toarray()

            # clustering
            if self.verbose:
                print(f'Initializing niches...')

            if method.lower() == 'kmeans':
                kmeans = KMeans(n_clusters=Qmax, init="k-means++", random_state=self.seed)
                labels = kmeans.fit_predict(x)

            elif method.lower() == 'gmm':
                gmm = GaussianMixture(n_components=Qmax, random_state=self.seed)
                labels = gmm.fit_predict(x)
            
            else:
                raise ValueError(f"Unknown method {method}. Supported methods are 'kmeans', 'gmm', and 'random'.")
        
        labels = [int(label) for label in labels.tolist()]
        self.init_cn_label_basic = labels
        self.init_cn_count_basic = len(list(set(labels)))

        if self.verbose:
            print(f'{len(list(set(labels)))} initial niches defined.\n')
            
    
    def hier_dist_match(self, 
                        assign_metric='jsd',
                        weighted_merge=True,
                        max_iters=100, 
                        tol=1e-4, 
                        test_kmeans=False,
                        Qmin=2,
                        ):
        
        # initialize
        self.cn_label_list_basic = []  # list for assigned cell niche label for each cell
        self.cn_count_list_basic = []  # list for niche count

        self.score_list_minjsd_basic = []  # list for scores
        self.weighted_score_list_minjsd_basic = []

        np.random.seed(self.seed)

        weight_mtx = None

        cn_labels = self.init_cn_label_basic.copy()
        n_niches = self.init_cn_count_basic
        label_summary = sorted(set(cn_labels))

        if self.verbose:
            print(f'Starting from {n_niches} cell niches...\n')

        if assign_metric.lower() in ['jsd', 'kld', 'kld_reverse']: 
            if self.parallel:
                precompute = cal_log(self.micro_dist_basic.toarray()) 
            else:
                precompute = np.log(np.clip(self.micro_dist_basic.toarray(), 1e-10, 1))      
        elif assign_metric.lower() == 'cosine':
            if self.parallel:
                precompute = cal_norm(self.micro_dist_basic.toarray()) 
            else:
                precompute = np.linalg.norm(self.micro_dist_basic.toarray(), axis=1, keepdims=True) 
        else:
            precompute = None

        for _ in range(n_niches-Qmin+1):

            if self.verbose:
                print(f'Assigning cells to cell niche...')
                print(f'Current state: {label_summary}')

            # assign cell to cell niche and calculate the cell distribution of cell niches
            if not test_kmeans:
                cn_labels, cn_dist, label_summary, cell_count_niche = cell2cellniche(cn_labels, 
                                                                                     self.ct_onehot_basic, 
                                                                                     self.micro_dist_basic,
                                                                                     precompute=precompute,
                                                                                     label_summary=label_summary, 
                                                                                     n_celltypes=self.n_celltypes, 
                                                                                     metric=assign_metric, 
                                                                                     max_iters=max_iters, 
                                                                                     tol=tol, 
                                                                                     change2str=False,
                                                                                     refine_k=self.refine_k,
                                                                                     parallel=self.parallel,
                                                                                     sparse=True, 
                                                                                     verbose=self.verbose,
                                                                                     )
            else:
                cn_labels, cn_dist, label_summary, cell_count_niche = cell2cellniche_kmeans(cn_labels, 
                                                                                            self.ct_onehot_basic, 
                                                                                            self.micro_dist_basic,  
                                                                                            label_summary=label_summary, 
                                                                                            n_celltypes=self.n_celltypes, 
                                                                                            random_seed=self.seed, 
                                                                                            change2str=False,
                                                                                            sparse=True,
                                                                                            )
            
            current_n_niches = len(label_summary)

            name_encoder = LabelEncoder()
            renamed_cn_labels = name_encoder.fit_transform(cn_labels)
            renamed_cn_labels = [int(label) for label in renamed_cn_labels.tolist()]
            self.cn_label_list_basic.append(renamed_cn_labels)
            self.cn_count_list_basic.append(current_n_niches)

            if self.verbose:
                print(f'{current_n_niches} cell niches left.')

            if current_n_niches <= 1:
                if self.verbose:
                    print(f'Niche count no more than 1.\n')
                break

            # if self.verbose:
            #     print(f'Calculating score...')

            cn2cn_gap = measure_distribution_gap(cn_dist, 
                                                 cn_dist, 
                                                 metric='jsd',
                                                 precompute=None,
                                                 weight=None,
                                                 parallel=False,
                                                 )
            np.fill_diagonal(cn2cn_gap, np.inf)
            self.score_list_minjsd_basic.append(np.min(cn2cn_gap))
            
            # merge two niches
            weight_mtx = cell_count_niche[:, np.newaxis] / (cell_count_niche[:, np.newaxis] + cell_count_niche)
            weighted_cn2cn_gap = measure_distribution_gap(cn_dist, 
                                                          cn_dist, 
                                                          metric='jsd',
                                                          precompute=None,
                                                          weight=weight_mtx,
                                                          parallel=False,
                                                          )
            np.fill_diagonal(weighted_cn2cn_gap, np.inf)
            self.weighted_score_list_minjsd_basic.append(np.min(weighted_cn2cn_gap))

            if weighted_merge:
                min_index = np.unravel_index(np.argmin(weighted_cn2cn_gap), weighted_cn2cn_gap.shape)
            else:
                min_index = np.unravel_index(np.argmin(cn2cn_gap), cn2cn_gap.shape)

            # if self.verbose:
            #     print(f'Done!')

            if current_n_niches <= Qmin:
                if self.verbose:
                    print(f'Niche count no more than {Qmin}.\n')
                break

            if self.verbose:
                print(f'Merging cell niche {label_summary[min_index[1]]} and cell niche {label_summary[min_index[0]]}...')
            
            merged_name = label_summary[min_index[1]]
            for l in range(len(cn_labels)):
                if cn_labels[l] == label_summary[min_index[0]]:
                    cn_labels[l] = merged_name
            
            label_summary = sorted(set(cn_labels))

            if self.verbose:
                print(f'Done!\n')
            
        if self.verbose:
            print(f'Finished!\n')
        

    def select_solution(self,
                        n_niche=None, 
                        niche_key='niche_label', 
                        auto=True, 
                        metric='jsd',
                        threshold=0.1,
                        return_adata=True,
                        plot=True, 
                        save=False,
                        fig_size=None,
                        save_dir='./', 
                        file_name='score_vs_nichecount_basic.pdf',
                        **kwargs,
                        ):
        
        # initialize
        self.cn_label_summary_basic = None  # summary of cell niche label 
        self.cn_label_basic = None  # cell niche labels for each cell
        self.cn_dist_basic = None  # cell type distrbution for niches (Q * K)
        self.cell_count_niche_basic = None  # cell counts for each niche

        # select solution based on specified niche count
        if n_niche is not None:
            if self.verbose:
                print('Selecting solution based on specified niche count...')
            if n_niche in self.cn_count_list_basic:
                solution_idx = self.cn_count_list_basic.index(n_niche)
                solution = self.cn_label_list_basic[solution_idx]
            else:
                raise ValueError(f'No available solution for {n_niche} cell niches!\n'
                                 f'Available niche counts are {self.cn_count_list_basic}.')
        
        # automatically select best solution
        elif auto:
            if self.verbose:
                print('Automatically selecting best solution...')
            candidate_indices = []
            if metric.lower() == 'jsd' or metric.lower() == 'wjsd':
                if metric.lower() == 'jsd':
                    score_list = self.score_list_minjsd_basic.copy()
                else:
                    score_list = self.weighted_score_list_minjsd_basic.copy()
                for solution_idx in range(len(score_list)):
                    if solution_idx == 0:
                        if score_list[solution_idx] >= threshold:
                            candidate_indices.append(solution_idx)
                            if self.verbose:
                                print(f'Better solutions may be found if **Qmax** is set larger!')
                    else:
                        if score_list[solution_idx-1] < threshold and score_list[solution_idx] >= threshold:
                            candidate_indices.append(solution_idx)
                if len(candidate_indices) == 0:
                    raise ValueError('No proper solution has been found!\n'
                                     'Please try decreasing the threshold or manually set the number of niches.')
                else:
                    if self.verbose:
                        print(f'Recommended number of niches are {[self.cn_count_list_basic[idx] for idx in candidate_indices]}')
                        print(f'Selecting {self.cn_count_list_basic[candidate_indices[0]]} niches as the best solution.')
                    solution = self.cn_label_list_basic[candidate_indices[0]]
                
                if plot or save:
                    plot_minjsd_score(score_list, 
                                      self.cn_count_list_basic, 
                                      threshold, 
                                      fig_size=fig_size, 
                                      plot=plot,
                                      save=save, 
                                      save_dir=save_dir, 
                                      file_name=file_name,
                                      **kwargs,
                                      )
            else:
                raise ValueError(f"Unknown metric {metric}. Supported metric are 'jsd' and 'wjsd'.")

        else:
            raise ValueError('Please either set a real number for **n_niche** or set **auto** to Ture')

        self.cn_label_summary_basic = [str(label) for label in sorted(set(solution))]  # summary of cell niche label 
        self.cn_label_basic = [str(label) for label in solution]  # cell niche labels for each cell

        start = 0
        for i in range(len(self.adata_list)):
            self.adata_list[i].obs[niche_key] = solution[start:start+self.adata_list[i].shape[0]].copy()
            self.adata_list[i].obs[niche_key] = self.adata_list[i].obs[niche_key].astype(str)
            start += self.adata_list[i].shape[0]

        self.adata_concat.obs[niche_key] = solution.copy()
        self.adata_concat.obs[niche_key] = self.adata_concat.obs[niche_key].astype(str)

        cn_dist_basic, cell_count_niche_basic = calculate_distribution(solution, 
                                                                       self.ct_onehot_basic, 
                                                                       label_summary=sorted(set(solution)), 
                                                                       n_niches=len(list(set(solution))), 
                                                                       n_celltypes=self.n_celltypes,  
                                                                       change2str=False,
                                                                       sparse=True,
                                                                       )
        self.cn_dist_basic = cn_dist_basic.tocsr()  # cell type distrbution for niches
        self.cell_count_niche_basic = cell_count_niche_basic  # cell counts for each niche

        self.adata_concat.uns['niche_label_summary'] = self.cn_label_summary_basic.copy()
        self.adata_concat.uns['niche_dist'] = self.cn_dist_basic.copy()
        self.adata_concat.uns['niche_cell_count'] = self.cell_count_niche_basic.copy()

        if self.verbose:
            print(f'Done!\n')

        if return_adata:
            adata_list = self.adata_list.copy()
            adata_concat = self.adata_concat.copy()
            return adata_list, adata_concat         


    def label_transfer(self, 
                       assign_metric='jsd',
                       niche_key='niche_label', 
                       return_adata=True,
                       ):
        
        # initialize
        self.cn_label_summary_trans = None  # summary of cell niche label 
        self.cn_label_trans = None  # cell niche labels for each cell
        self.cn_dist_trans = None  # cell type distrbution for niches (R * K)
        self.cell_count_niche_trans = None  # cell counts for each niche

        # assign cells to fixed niches
        if self.verbose:
            print('Assigning cells to fixed niches...')
        
        fixed_label_summary = [int(label) for label in self.cn_label_summary_basic]

        if self.refine_k > 0 and self.cn_dist_basic.shape[1] > self.refine_k:
            cn_dist_basic_refined = refine_dist(self.cn_dist_basic, k=self.refine_k)
        else:
            cn_dist_basic_refined = self.cn_dist_basic.copy()

        dist_gap = measure_distribution_gap(cn_dist_basic_refined, 
                                            self.micro_dist_cond, 
                                            metric=assign_metric, 
                                            precompute=None,
                                            weight=None, 
                                            parallel=self.parallel,
                                            eps=1e-10,
                                            )
        
        new_indices = np.argmin(dist_gap, axis=1)
        transfered_labels = [fixed_label_summary[idx] for idx in new_indices]   

        self.cn_label_summary_trans = [str(label) for label in sorted(set(transfered_labels))]
        self.cn_label_trans = [str(label) for label in transfered_labels]  # cell niche labels for each cell

        start = 0
        for i in range(len(self.cond_list)):
            self.cond_list[i].obs[niche_key] = transfered_labels[start:start+self.cond_list[i].shape[0]].copy()
            self.cond_list[i].obs[niche_key] = self.cond_list[i].obs[niche_key].astype(str)
            start += self.cond_list[i].shape[0]

        self.cond_concat.obs[niche_key] = self.cn_label_trans.copy()
        
        cn_dist_trans, cell_count_niche_trans = calculate_distribution(transfered_labels, 
                                                                       self.ct_onehot_cond, 
                                                                       label_summary=sorted(set(transfered_labels)), 
                                                                       n_niches=len(sorted(set(transfered_labels))), 
                                                                       n_celltypes=self.n_celltypes, 
                                                                       change2str=False,
                                                                       sparse=True,
                                                                       )
        self.cn_dist_trans = cn_dist_trans.tocsr()  # cell type distrbution for cell niches 
        self.cell_count_niche_trans = cell_count_niche_trans  # cell counts for cell niches 

        self.cond_concat.uns['niche_label_summary'] = self.cn_label_summary_trans.copy()
        self.cond_concat.uns['niche_dist'] = self.cn_dist_trans.copy()
        self.cond_concat.uns['niche_cell_count'] = self.cell_count_niche_trans.copy()

        if self.verbose:
            print(f'Done!\n')

        if return_adata:
            cond_list = self.cond_list.copy()
            cond_concat = self.cond_concat.copy()
            return cond_list, cond_concat     


    def initialize_clusters_cond(self, 
                                 assign_metric='jsd',
                                 threshold=0.1,
                                 min_cell_per_niche=100,
                                 dim_reduction=True,
                                 explained_var=None,
                                 n_components=None,
                                 n_components_max=100,
                                 standardize=True, 
                                 method='kmeans', 
                                 Rmax=10, 
                                 ):
        
        # initialize
        self.init_cn_label_cond = None  # initial niche label for each cell
        self.init_cn_count_cond = None  # initial new niche count
        
        # assign cells to fixed niches
        if self.verbose:
            print('Assigning cells to fixed niches...')
        
        fixed_label_summary = [int(label) for label in self.cn_label_summary_basic]
        n_niches_basic = len(fixed_label_summary)

        if self.refine_k > 0 and self.cn_dist_basic.shape[1] > self.refine_k:
            cn_dist_basic_refined = refine_dist(self.cn_dist_basic, k=self.refine_k)
        else:
            cn_dist_basic_refined = self.cn_dist_basic.copy()

        dist_gap = measure_distribution_gap(cn_dist_basic_refined, 
                                            self.micro_dist_cond, 
                                            metric=assign_metric, 
                                            precompute=None,
                                            weight=None, 
                                            parallel=self.parallel,
                                            eps=1e-10,
                                            )
        
        new_indices = np.argmin(dist_gap, axis=1)
        rough_labels = [fixed_label_summary[idx] for idx in new_indices]

        row_min_values = np.min(dist_gap, axis=1)
        selected_rows = np.where(row_min_values > threshold)[0]

        if self.verbose:
            print(f'{int(self.n_cells_cond-len(list(selected_rows)))} out of {int(self.n_cells_cond)} cells are assigned to fixed niches.\n')

        need_assign = len(list(selected_rows))
        if int(need_assign / min_cell_per_niche) < Rmax:
            Rmax = int(need_assign / min_cell_per_niche)
            if Rmax == 0:
                raise ValueError('Could not find enough cells to define a new cell niche! Try decreasing threshold or min_cell_per_niche.')
            if self.verbose:
                print(f'Rmax is changed to {Rmax} according to the minimum average cell number per niche.\n')
        
        if Rmax == 1:
            labels = [int(n_niches_basic)] * need_assign
            
        else:
            if method.lower() == 'random':
                labels = np.random.choice(np.arange(Rmax), size=need_assign, replace=True)

            else:
                # dimension reduction
                if dim_reduction:
                    if self.verbose:
                        print('Performing dimension reduction...')

                    x = self.micro_dist_cond.copy().toarray()[selected_rows, :]
                    x, _, _ = pca(x, 
                                explained_var=explained_var, 
                                n_components=n_components, 
                                n_components_max=n_components_max,
                                standardize=standardize,
                                verbose=self.verbose,
                                )
                else:
                    x = self.micro_dist_cond.copy().toarray()[selected_rows, :]

                # clustering
                if self.verbose:
                    print(f'Initializing niches...')

                if method.lower() == 'kmeans':
                    kmeans = KMeans(n_clusters=Rmax, init="k-means++", random_state=self.seed)
                    labels = kmeans.fit_predict(x)

                elif method.lower() == 'gmm':
                    gmm = GaussianMixture(n_components=Rmax, random_state=self.seed)
                    labels = gmm.fit_predict(x)
                
                else:
                    raise ValueError(f"Unknown method {method}. Supported methods are 'kmeans', 'gmm', and 'random'.")
                
            labels = [int(label+n_niches_basic) for label in labels.tolist()]
                
        new_labels = np.array(rough_labels, dtype=int)
        new_labels[selected_rows] = np.array(labels, dtype=int)

        self.init_cn_label_cond = [int(label) for label in new_labels.tolist()]
        self.init_cn_count_cond = len(list(set(labels)))

        if self.verbose:
            print(f'{self.init_cn_count_cond} new niches defined.\n')


    def hier_dist_match_cond(self, 
                             assign_metric='jsd',
                             weighted_merge=True,
                             max_iters=100, 
                             tol=1e-4, 
                             ):

        # initialize
        self.cn_label_list_cond = []  # list for assigned cell niche label for each cell
        self.cn_count_list_cond = []  # list for new niche count

        self.score_list_minjsd_cond = []  # list for scores
        self.weighted_score_list_minjsd_cond = []
        
        np.random.seed(self.seed)

        weight_mtx = None

        cn_label_summary_basic = [int(label) for label in self.cn_label_summary_basic]

        cn_labels = self.init_cn_label_cond.copy()
        n_niches_basic = len(cn_label_summary_basic)
        n_niches_new = self.init_cn_count_cond

        selected_indices = np.where(~np.isin(np.array(cn_labels), np.array(cn_label_summary_basic)))[0]
        filtered_labels = [int(label) for label in np.array(cn_labels)[selected_indices].tolist()]
        label_summary = cn_label_summary_basic + sorted(set(filtered_labels))

        # niche refinement
        if self.refine_k > 0 and self.cn_dist_basic.shape[1] > self.refine_k:
            cn_dist_basic_refined = refine_dist(self.cn_dist_basic, k=self.refine_k)
        else:
            cn_dist_basic_refined = self.cn_dist_basic.copy()

        if self.verbose:
            print(f'Starting from {n_niches_new} new cell niches...\n')

        if assign_metric.lower() in ['jsd', 'kld', 'kld_reverse']: 
            if self.parallel:
                precompute = cal_log(self.micro_dist_cond.toarray()) 
            else:
                precompute = np.log(np.clip(self.micro_dist_cond.toarray(), 1e-10, 1))      
        elif assign_metric.lower() == 'cosine':
            if self.parallel:
                precompute = cal_norm(self.micro_dist_cond.toarray()) 
            else:
                precompute = np.linalg.norm(self.micro_dist_cond.toarray(), axis=1, keepdims=True) 
        else:
            precompute = None

        for _ in range(n_niches_new + 1):

            if self.verbose:
                print(f'Assigning cells to cell niche...')
                print(f'Current state: {label_summary}')

            # assign cell to cell niche and calculate the cell distribution of cell niches
            cn_labels, cn_dist_cond, label_summary_cond, cell_count_niche_cond = cell2cellniche_cond(cn_labels, 
                                                                                                     self.ct_onehot_cond, 
                                                                                                     self.micro_dist_cond, 
                                                                                                     cn_dist_basic_refined, 
                                                                                                     n_niches_basic, 
                                                                                                     label_summary,
                                                                                                     precompute=precompute,
                                                                                                     n_celltypes=self.n_celltypes, 
                                                                                                     metric=assign_metric, 
                                                                                                     max_iters=max_iters, 
                                                                                                     tol=tol, 
                                                                                                     change2str=False,
                                                                                                     parallel=self.parallel,
                                                                                                     refine_k=self.refine_k,
                                                                                                     sparse=True, 
                                                                                                     verbose=self.verbose,
                                                                                                     )
            
            current_n_niches_new = len(label_summary_cond)

            # calculate number of cells belong to each basic niche
            selected_indices = np.where(np.isin(np.array(cn_labels), np.array(cn_label_summary_basic)))[0]
            if len(selected_indices) == 0:
                cell_count_niche_basic = np.zeros(n_niches_basic)
            else:
                filtered_labels = list(np.array(cn_labels)[selected_indices])
                cn_onehot_basic = label2onehot(filtered_labels, 
                                               n_cols=n_niches_basic, 
                                               label_summary=cn_label_summary_basic, 
                                               change2str=False,
                                               sparse=True,
                                               )
                cell_count_niche_basic = np.array(cn_onehot_basic.T.sum(axis=1)).flatten()

            if current_n_niches_new == 0:
                cn_dist = cn_dist_basic_refined.copy()
                label_summary = cn_label_summary_basic.copy()
                cell_count_niche = cell_count_niche_basic
                self.cn_label_list_cond.append(cn_labels.copy())
            else:
                cn_dist = sp.vstack([cn_dist_basic_refined, cn_dist_cond])
                label_summary = cn_label_summary_basic + label_summary_cond
                cell_count_niche = np.concatenate([cell_count_niche_basic, cell_count_niche_cond])
            
                # filter labels
                selected_indices = np.where(np.isin(np.array(cn_labels), np.array(label_summary_cond)))[0]
                filtered_labels = list(np.array(cn_labels)[selected_indices])

                name_encoder = LabelEncoder()
                renamed_cn_labels_filtered = name_encoder.fit_transform(filtered_labels)
                renamed_cn_labels_filtered = [label+n_niches_basic for label in renamed_cn_labels_filtered]

                renamed_cn_labels = np.array(cn_labels, dtype=int)
                renamed_cn_labels[selected_indices] = np.array(renamed_cn_labels_filtered, dtype=int)
                renamed_cn_labels = [int(label) for label in renamed_cn_labels]
                
                self.cn_label_list_cond.append(renamed_cn_labels)
            self.cn_count_list_cond.append(current_n_niches_new)

            if self.verbose:
                print(f'{current_n_niches_new} new cell niches left.')

            # if self.verbose:
            #     print(f'Calculating score...')

            cn2cn_gap = measure_distribution_gap(cn_dist, 
                                                 cn_dist, 
                                                 metric='jsd',
                                                 precompute=None,
                                                 weight=None,
                                                 parallel=False,
                                                 )
            np.fill_diagonal(cn2cn_gap, np.inf)
            cn2cn_gap[:n_niches_basic, :] = np.inf
            self.score_list_minjsd_cond.append(np.min(cn2cn_gap))
            
            # merge two niches
            weight_mtx = cell_count_niche[:, np.newaxis] / (cell_count_niche[:, np.newaxis] + cell_count_niche)
            weighted_cn2cn_gap = measure_distribution_gap(cn_dist, 
                                                          cn_dist, 
                                                          metric='jsd',
                                                          precompute=None,
                                                          weight=weight_mtx,
                                                          parallel=False,
                                                          )
            np.fill_diagonal(weighted_cn2cn_gap, np.inf)
            weighted_cn2cn_gap[:n_niches_basic, :] = np.inf
            weighted_cn2cn_gap[n_niches_basic:, :n_niches_basic] = cn2cn_gap[n_niches_basic:, :n_niches_basic].copy()
            self.weighted_score_list_minjsd_cond.append(np.min(weighted_cn2cn_gap))

            # cn2cn_gap[n_niches_basic:, :n_niches_basic] = np.inf
            # weighted_cn2cn_gap[n_niches_basic:, :n_niches_basic] = np.inf
            if weighted_merge:
                min_index = np.unravel_index(np.argmin(weighted_cn2cn_gap), weighted_cn2cn_gap.shape)
            else:
                min_index = np.unravel_index(np.argmin(cn2cn_gap), cn2cn_gap.shape)

            # if self.verbose:
            #     print(f'Done!')
            
            if current_n_niches_new == 0:
                if self.verbose:
                    print(f'No new cell niche left.\n')
                break

            if min_index[1] < n_niches_basic:
                if self.verbose:
                    print(f'Merging new cell niche {label_summary[min_index[0]]} into basic cell niche {label_summary[min_index[1]]}...')
                
                for l in range(len(cn_labels)):
                    if cn_labels[l] == label_summary[min_index[0]]:
                        cn_labels[l] = label_summary[min_index[1]]
            
            else:
                if self.verbose:
                    print(f'Merging new cell niche {label_summary[min_index[1]]} and new cell niche {label_summary[min_index[0]]}...')
                
                merged_name = label_summary[min_index[1]]
                for l in range(len(cn_labels)):
                    if cn_labels[l] == label_summary[min_index[0]]:
                        cn_labels[l] = merged_name

            # if current_n_niches_new != 1:
            #     if self.verbose:
            #         print(f'Merging new cell niche {label_summary[min_index[1]]} and new cell niche {label_summary[min_index[0]]}...')
                
            #     merged_name = label_summary[min_index[1]]
            #     for l in range(len(cn_labels)):
            #         if cn_labels[l] == label_summary[min_index[0]]:
            #             cn_labels[l] = merged_name
            # else:
            #     label_summary = cn_label_summary_basic.copy()
            #     if self.verbose:
            #         print(f'Done!\n')
            #     continue
            
            selected_indices = np.where(~np.isin(np.array(cn_labels), np.array(cn_label_summary_basic)))[0]
            if len(selected_indices) == 0:
                label_summary = cn_label_summary_basic.copy()
                if self.verbose:
                    print(f'Done!\n')
                continue
            filtered_labels = [int(label) for label in np.array(cn_labels)[selected_indices].tolist()]
            label_summary = cn_label_summary_basic + sorted(set(filtered_labels))

            if self.verbose:
                print(f'Done!\n')
            
        if self.verbose:
            print(f'Finished!\n')

        
    def select_solution_cond(self,
                             n_csn=None, 
                             niche_key='niche_label',
                             csn_key='csn_label',
                             auto=True, 
                             metric='jsd',
                             threshold=0.1,
                             return_adata=True,
                             plot=True, 
                             save=False,
                             fig_size=None,
                             save_dir='./', 
                             file_name='score_vs_nichecount_cond.pdf',
                             **kwargs,
                             ):
        # initialize
        self.cn_label_summary_cond = None  # summary of cell niche label 
        self.cn_label_cond = None  # cell niche labels for each cell
        self.cn_dist_cond = None  # cell type distrbution for niches (R * K)
        self.cell_count_niche_cond = None  # cell counts for each niche

        # select solution based on CSN count
        if n_csn is not None:
            if self.verbose:
                print('Selecting solution based on specified condition specific niche count...')
            if n_csn in self.cn_count_list_cond:
                solution_idx = self.cn_count_list_cond.index(n_csn)
                solution = self.cn_label_list_cond[solution_idx]
            else:
                raise ValueError(f'No available solution for {n_csn} condition specific niches!\n'
                                 f'Available condition specific niche counts are {self.cn_count_list_cond}.')
        
        # automatically select best solution
        elif auto:
            if self.verbose:
                print('Automatically selecting best solution...')
            candidate_indices = []
            if metric.lower() == 'jsd' or metric.lower() == 'wjsd':
                if metric.lower() == 'jsd':
                    score_list = self.score_list_minjsd_cond.copy()
                else:
                    score_list = self.weighted_score_list_minjsd_cond.copy()
                for solution_idx in range(len(score_list)-1):
                    if solution_idx == 0:
                        if score_list[solution_idx] >= threshold:
                            candidate_indices.append(solution_idx)
                            if self.verbose:
                                print(f'Better solutions may be found if **Rmax** is set larger!')
                    else:
                        if score_list[solution_idx-1] < threshold and score_list[solution_idx] >= threshold:
                            candidate_indices.append(solution_idx)
                if len(candidate_indices) == 0:
                    print('No proper solution with condition specific niche has been found!\n'
                          'Assigning all cells to basic niches!\n'
                          'Please try decreasing the threshold or manually set the number of niches.')
                    solution = self.cn_label_list_cond[self.cn_count_list_cond.index(0)]
                else:
                    if self.verbose:
                        print(f'Recommended number of condition specific niches are {[self.cn_count_list_cond[idx] for idx in candidate_indices]}')
                        print(f'Selecting {self.cn_count_list_cond[candidate_indices[0]]} new niches as the best solution.')
                    solution = self.cn_label_list_cond[candidate_indices[0]]
                
                if plot or save:
                    plot_minjsd_score(score_list[:-1], 
                                      self.cn_count_list_cond[:-1], 
                                      threshold, 
                                      fig_size=fig_size, 
                                      plot=plot,
                                      save=save, 
                                      save_dir=save_dir, 
                                      file_name=file_name,
                                      **kwargs,
                                      )
            else:
                raise ValueError(f"Unknown metric {metric}. Supported metric are 'jsd' and 'wjsd'.")

        else:
            raise ValueError('Please either set a real number for **n_niche** or set **auto** to Ture')
        
        n_niches_basic = len(self.cn_label_summary_basic)
        
        self.cn_label_summary_cond = [str(label) for label in sorted(set(solution))]  # summary of cell niche label
        self.cn_label_cond = [str(label) for label in solution]  # cell niche labels for each cell
        csn_solution = [f'R{int(label-n_niches_basic)}' if label >= n_niches_basic else 'basic' for label in solution]

        start = 0
        for i in range(len(self.cond_list)):
            self.cond_list[i].obs[niche_key] = solution[start:start+self.cond_list[i].shape[0]].copy()
            self.cond_list[i].obs[niche_key] = self.cond_list[i].obs[niche_key].astype(str)
            self.cond_list[i].obs[csn_key] = csn_solution[start:start+self.cond_list[i].shape[0]].copy()
            start += self.cond_list[i].shape[0]

        self.cond_concat.obs[niche_key] = [str(label) for label in solution]
        self.cond_concat.obs[csn_key] = csn_solution.copy()

        cn_dist_cond, cell_count_niche_cond = calculate_distribution(solution, 
                                                                     self.ct_onehot_cond, 
                                                                     label_summary=sorted(set(solution)), 
                                                                     n_niches=len(sorted(set(solution))), 
                                                                     n_celltypes=self.n_celltypes, 
                                                                     change2str=False,
                                                                     sparse=True,
                                                                     )
        self.cn_dist_cond = cn_dist_cond.tocsr()  # cell type distrbution for cell niches 
        self.cell_count_niche_cond = cell_count_niche_cond  # cell counts for cell niches 

        self.cond_concat.uns['niche_label_summary'] = self.cn_label_summary_cond.copy()
        self.cond_concat.uns['niche_dist'] = self.cn_dist_cond.copy()
        self.cond_concat.uns['niche_cell_count'] = self.cell_count_niche_cond.copy()

        if self.verbose:
            print(f'Done!\n')

        if return_adata:
            cond_list = self.cond_list.copy()
            cond_concat = self.cond_concat.copy()
            return cond_list, cond_concat     


