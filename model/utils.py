import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from .preprocess import pca
import matplotlib.pyplot as plt
import torch

#os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'    

'''

adata       AnnData object of scanpy
num_cluster number of clusters
modelNames  covariance model name of mclust (3 letters, each stating whether
            volume / shape / orientation are constrained to be equal; possible values
            include EEE, VVV, EEV, EVE, VVE, VEV, EVE)


EII  VII EEI "VEI" "EVI" "VVI" "EEE" "EEV" "VEV" "VEE" "EVE" "VVV" "EVV" "VVE"

used_obsm   which key of adata.obsm to use as the input features
random_seed random seed
  EVE
 What does modelNames='EEE' mean?
The 3 letters mean:
    letter 1: Equal Volume / Variable volume
    letter 2: Equal Shape / Variable shape
    letter 3: Equal Orientation / Variable orientation
    'EEE' = ellipsoidal clusters with equal volume / shape / orientation
            (the most restrictive and simplest model)
   
        EVE: 
                ari:  0.3478715559149661
                nmi:  0.42593450551164164
                ami:  0.4235860798177266
                v_measure:  0.42593450551164175
                homogeneity:  0.4169236794396089
'''

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    # fix the random seed
    np.random.seed(random_seed)
    # rpy2: the bridge between Python and R
    import rpy2.robjects as robjects
    # equivalent to library(mclust) in R
    robjects.r.library("mclust")
    
    import rpy2.robjects.numpy2ri
    # activate() turns on automatic conversion: numpy arrays passed to R functions are
    # converted into R matrices/vectors from now on
    rpy2.robjects.numpy2ri.activate()
    # fetch set.seed from the R namespace
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)  
    # fetch Mclust from the R namespace; it can then be called like a Python function
    rmclust = robjects.r['Mclust']
    

    # run the Mclust clustering
    # adata.obsm[used_obsm] reads the value of that key, e.g. adata.obsm['spaFDG'],
    # of shape (n_cells, 64)
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    # the mclust res object has about 14 slots; the classification labels are accessed
    # by name here
    # mclust_res = np.array(res[-2])
    mclust_res = np.array(res.rx2('classification'))
    # store mclust_res into adata.obs['mclust']
    adata.obs['mclust'] = mclust_res
    # cast adata.obs['mclust'] to integers
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    # cast adata.obs['mclust'] to a categorical variable
    # adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


# key        input key of the embedding in adata.obsm
# add_key    output key where the labels are stored in adata.obs
# method     clustering method: mclust / leiden / louvain
# start      start of the resolution search for leiden / louvain
# end        end of the resolution search for leiden / louvain
# increment  step size of the resolution search for leiden / louvain
# use_pca    whether to reduce dimensions with PCA first
# n_comps    number of PCA components
def clustering(adata, n_clusters=7, key='emb', add_key='spaLLM', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20,modelNames='EEE'):
    """\
    Spatial clustering based the latent representation.
    
    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    # if use_pca=True: call pca() to reduce adata.obsm[key] to n_comps dimensions
    # and store the result in adata.obsm[key + '_pca'], e.g. 'emb_pca', 'spaFDG_pca'
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters,modelNames=modelNames)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters,modelNames=modelNames)
       # copy adata.obs['mclust'] into adata.obs[add_key], e.g. 'spaLLM'
       adata.obs[add_key] = adata.obs['mclust']

    elif method == 'leiden':
       if use_pca: 
         # call search_res to find a resolution that makes leiden produce n_clusters
         # clusters, then run leiden once with that resolution
         # 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       # # run leiden once with the resolution found above
       sc.tl.leiden(adata, random_state=0, resolution=res)
       # copy adata.obs['leiden'] into adata.obs[add_key], e.g. 'spaLLM'
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']

# start      start of the resolution search for leiden / louvain
# end        end of the resolution search for leiden / louvain
# increment  step size of the resolution search for leiden / louvain
'''

leiden always yields too many clusters   decrease resolution   start=0.01, end=1.0, increment=0.005
leiden always yields too few clusters    increase resolution   start=0.5, end=5.0, increment=0.01
no exact match found                     use a finer step    increment=0.001
mclust too slow                          use leiden          method='leiden'
embedding dimension too high              PCA first           use_pca=True, n_comps=30

'''
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values
  
  """  
  import pandas as pd  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  plt.show()


def add_noise_by_zeroing(matrix, zero_prob):
    """
    Zero out a fraction of the entries of the given matrix as noise.

    Parameters:
    - matrix (torch.Tensor): input matrix.
    - zero_prob (float): probability of zeroing each entry, in [0, 1].

    Returns:
    - the noised matrix.
    """
    # build a mask with the same shape as the input, where each entry survives with
    # probability 1 - zero_prob
    noise_mask = torch.bernoulli((1 - zero_prob) * torch.ones_like(matrix)).to(matrix.device)
    # apply the mask, zeroing the sampled entries
    noisy_matrix = matrix * noise_mask
    return noisy_matrix


def add_noise_by_zeroing_columns(matrix, zero_prob=0.1):
    """
    Randomly select a fraction of the columns of the given matrix and zero out the
    whole column as noise.

    Parameters:
    - matrix (torch.Tensor): input matrix.
    - zero_prob (float): probability of zeroing a column, in [0, 1].

    Returns:
    - the noised matrix.
    """
    # get the shape of the matrix
    num_columns = matrix.size(1)  # number of columns

    # draw random column indices deciding which columns to zero out
    zero_mask = torch.bernoulli((1 - zero_prob) * torch.ones(num_columns)).to(matrix.device)

    # build a mask with the same shape as the matrix
    column_mask = zero_mask.unsqueeze(0).expand_as(matrix)  # broadcast the column mask to the matrix shape

    # zero out the selected columns
    noisy_matrix = matrix * column_mask
    return noisy_matrix


def add_gaussian_noise(matrix, mean=0.0, std=0.001):
    """
    Add element-wise Gaussian noise to the given matrix.

    Parameters:
    - matrix (torch.Tensor): input matrix.
    - mean (float): mean of the Gaussian noise.
    - std (float): standard deviation of the Gaussian noise.

    Returns:
    - the noised matrix.
    """
    # draw noise with the same shape as the input
    noise = torch.normal(mean=mean, std=std, size=matrix.size()).to(matrix.device)
    # add the noise to the matrix
    noisy_matrix = matrix + noise
    return noisy_matrix
