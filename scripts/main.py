import os
import numpy as np
from scanpy.plotting import pca_loadings
from sklearn.decomposition import PCA
import torch
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import normalize,StandardScaler

import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)



from model.preprocess import fix_seed
from model.preprocess import clr_normalize_each_cell, pca
from model.preprocess import construct_neighbor_graph
from model.spaCFCL_util import Train_spaCFCL
from model.utils import clustering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import kneighbors_graph

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, mutual_info_score,v_measure_score
from sklearn.metrics import homogeneity_score
from scipy.optimize import linear_sum_assignment


def map_clusters_to_names(pred, true_labels_str, true_names):
    pred_unique = np.unique(pred)
    K = len(pred_unique); G = len(true_names)
    cnt = np.zeros((K, G), dtype=np.int64)
    for i, p in enumerate(pred):
        pi = int(np.where(pred_unique == p)[0][0])
        gi = int(np.where(true_names == true_labels_str[i])[0][0])
        cnt[pi, gi] += 1
    row_ind, col_ind = linear_sum_assignment(-cnt)
    name_for_cluster = {}
    name_counter = {}
    for r, c in zip(row_ind, col_ind):
        name_for_cluster[int(pred_unique[r])] = str(true_names[c])
        name_counter[str(true_names[c])] = 1
    leftovers = [i for i in range(K) if int(pred_unique[i]) not in name_for_cluster]
    for r in leftovers:
        best_gt = int(np.argmax(cnt[r]))
        base = str(true_names[best_gt])
        name_counter[base] = name_counter.get(base, 0) + 1
        name_for_cluster[int(pred_unique[r])] = f'{base}-{chr(96+name_counter[base])}'
    return name_for_cluster




# compute ARI, NMI, AMI, v_measure, Homogeneity
# Adjusted Rand Index: similarity between two cluster assignments, in [-1, 1], closer to 1 is better
def cal_ARI(label: np.ndarray, pred: np.ndarray) -> float:
    valid_indices = label != -1
    return adjusted_rand_score(label[valid_indices], pred[valid_indices])

# Normalized Mutual Information: similarity between two cluster assignments, in [0, 1], closer to 1 is better
def cal_NMI(label: np.ndarray, pred: np.ndarray) -> float:
    valid_indices = label != -1
    return normalized_mutual_info_score(label[valid_indices], pred[valid_indices])

# Adjusted Mutual Information: similarity between two cluster assignments, in [-1, 1], closer to 1 is better
def cal_AMI(label: np.ndarray, pred: np.ndarray) -> float:
    valid_indices = label != -1
    return adjusted_mutual_info_score(label[valid_indices], pred[valid_indices])

# V-measure: similarity between two cluster assignments, in [0, 1], closer to 1 is better
def cal_v_measure(label: np.ndarray, pred: np.ndarray) -> float:
    valid_indices = label != -1
    return v_measure_score(label[valid_indices], pred[valid_indices])

# Homogeneity: similarity between two cluster assignments, in [0, 1], closer to 1 is better
def cal_Homogeneity(label: np.ndarray, pred: np.ndarray) -> float:
    valid_indices = label != -1
    return homogeneity_score(label[valid_indices], pred[valid_indices])

def cal_mutual_info_score(label: np.ndarray, pred: np.ndarray) -> float:
    valid_indices = label != -1
    return mutual_info_score(label[valid_indices], pred[valid_indices])

device = torch.device('cuda')
os.environ['R_HOME'] = 'R path '

file_fold = 'data path'
  
# image output path
attention_path = os.path.join(file_fold, 'attention')
if not os.path.exists(attention_path):
    os.makedirs(attention_path)


adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')

adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')



emb_genef = np.load(file_fold + 'Geneformer_embedding_human_D1.npy').astype(np.float32)  

emb_scgpt = np.load(file_fold + 'ScGPT_embedding_cls_API_human_D1.npy').astype(np.float32) 




scaler_genef = StandardScaler()
scaler_scgpt = StandardScaler()
emb_genef = scaler_genef.fit_transform(emb_genef)
emb_scgpt = scaler_scgpt.fit_transform(emb_scgpt)


# make gene / protein names unique to avoid downstream errors.
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

# datatype
datatype = '10x'
# random seed
random_seed = 2024
fix_seed(random_seed)


annotation = pd.read_csv(file_fold + 'annotation.csv')


# read the manually annotated cell types and convert to integer codes.
label = annotation['manual-anno'].astype('category').cat.codes.to_numpy()
print('label11:',label)
print(label.dtype)
sc.pp.filter_genes(adata_omics1, min_cells=10) 
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)

sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)
adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']] # 
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=200)

adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1) 

k1=6  # 3 4 5 6 7 8 9 
data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=datatype,n_neighbors=k1)

print('data456:',data)

k =17 # 14 15 16 17 18 19 20 
data['adj_emb_genef'] = kneighbors_graph(emb_genef, k , mode="connectivity",
                                         metric="correlation", include_self=False)
data['adj_emb_scgpt'] = kneighbors_graph(emb_scgpt,k , mode="connectivity",
                                         metric="correlation", include_self=False)

# cl_temperature 
# temp_annealing  =0.98
temp_annealing_step =50 
weight_factors =[2.22, 1.88, 1.02, 5.08, 8.56,5.70, 7.37, 6.08, 8.25]
# ========== scheme 3: training (dual LLM + CL) ==========
model = Train_spaCFCL(data, datatype=datatype, device=device,
                     embedding_genef=emb_genef,
                     embedding_scgpt=emb_scgpt,
                     random_seed=2024, learning_rate=0.0001,
                     weight_decay=0.00,
                     dim_output=64,
                     weight_factors = weight_factors,
                     cl_temperature=0.1,
                     neg_sample_ratio=50,
                     temp_annealing=0.99,
                     temp_annealing_step=40 # 40
                      )
output = model.train(epochs=200)

# copy the RNA AnnData object as the output container
adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['spaFDG'] = output['spaFDG'].copy()
adata.obsm['alpha'] = output['alpha']
adata.obsm['alpha_omics1'] = output['alpha_omics1']
adata.obsm['alpha_omics2'] = output['alpha_omics2']
adata.obsm['alpha_att1']= output['alpha_att1']
adata.obsm['alpha_att2']= output['alpha_att2']
adata.obsm['alpha_llm_spa'] = output['alpha_llm_spa']   # Geneformer vs scGPT spa branch weights
adata.obsm['alpha_llm_fea'] = output['alpha_llm_fea']   # Geneformer vs scGPT fea branch weights
print(f"Final CL temperature: {output['final_temperature']:.4f}")

#   mclust   leiden  louvain
tool = 'mclust' 
clustering(adata, key='spaFDG', add_key='spaFDG', n_clusters=6, method=tool, use_pca=True,start=0.1, end=3.0, increment=0.01,n_comps=15,modelNames='EEI')

 
# copy the AnnData object as adata_plt (plt = plot, dedicated to plotting)
adata_plt = adata.copy()
# scanpy plotting functions (sc.pl.spatial, sc.pl.umap, ...) read keys of adata.obs as colour
# convert adata.obs['spaFDG'] to a categorical variable
adata_plt.obs['spaFDG'] = adata.obs['spaFDG'].astype('category')

pred = adata.obs['spaFDG'].to_numpy()
pred = np.ravel(pred)


# flatten the label array to 1-D
# ============================================================
# verify the barcode correspondence between label and pred
# ============================================================
print("\n[check] Barcode order consistency check ...")
assert len(annotation) == adata.n_obs, \
    f"ERROR: row count mismatch! annotation.csv={len(annotation)} rows, adata={adata.n_obs} cells"
# check whether the barcode order matches exactly
barcode_match = np.all(annotation['Barcode'].values == adata.obs_names.values)
if barcode_match:
    print("OK: barcode order verified - annotation_D1.csv and adata_RNA.h5ad rows are aligned")
else:
    # locate the mismatching barcodes
    mismatch_idx = np.where(annotation['Barcode'].values != adata.obs_names.values)[0]
    print(f"ERROR: {len(mismatch_idx)} barcode order mismatches found!")
    print(f"  first 5 mismatching positions: {mismatch_idx[:5].tolist()}")
    print(f"  first 3 annotation: {annotation['Barcode'].values[:3].tolist()}")
    print(f"  first 3 adata:      {adata.obs_names.values[:3].tolist()}")
    print("\nMake sure annotation.csv and adata_RNA.h5ad are ordered by the same barcodes!")
    raise AssertionError("barcode order mismatch")




ari_score = cal_ARI(label, pred)
print("ari: ", ari_score)
nmi_score = cal_NMI(label, pred)
print("nmi: ", nmi_score)
ami_score = cal_AMI(label, pred)
print("ami: ", ami_score)
v_score = cal_v_measure(label, pred)
print("v_measure: ", v_score)
homogeneity = cal_Homogeneity(label, pred)
print("homogeneity: ", homogeneity)
mi_score = cal_mutual_info_score(label, pred)
print("mutual_info: ", mi_score)

#exit()

# switch: whether to run the Gate Collapse diagnosis  True False 
DO_CHECK = False 

# switch: whether to visualize the attention weights
DO_VISUALIZATION_ATTENTION = False  

# switch: whether to write the training results (spaFDG_name + id2name) back into
# adata_plt and save the .h5ad
#   True  = attach the names in memory right after training and write to disk (the plotting
#           script picture_.py can read it directly)
#   False = skip naming / writing, the training results stay only in files / terminal logs
DO_PERSIST = False




if DO_CHECK:
    # ======================================================================
    # [Gate Collapse diagnosis] mean/std/min/max of every attention weight
    # ======================================================================
    print("=" * 70)
    print("[Gate Collapse diagnosis] mean/std/min/max of every attention weight")
    print("=" * 70)

    attention_keys = [
        # note: in example_Lymph_A1.py lines 131-134 the variable names
        # emb_genef/emb_scgpt are swapped with respect to the file contents
        # so [:,0] is actually scGPT and [:,1] is actually Geneformer
        ('alpha_llm_spa', 'Geneformer-spa vs scGPT-spa   [0]=scGPT [1]=Geneformer'),
        ('alpha_llm_fea', 'Geneformer-fea vs scGPT-fea   [0]=scGPT [1]=Geneformer'),
        ('alpha_att1',    'LLM-spa vs RNA-spa            [0]=LLM   [1]=RNA'),
        ('alpha_att2',    'LLM-fea vs RNA-fea            [0]=LLM   [1]=RNA'),
        ('alpha',         'RNA vs Protein                [0]=RNA   [1]=Protein'),
    ]

    diag_records = []
    for att_key, att_desc in attention_keys:
        if att_key not in adata.obsm:
            print(f"\n>> {att_key} ({att_desc})  NOT FOUND in adata.obsm")
            continue

        arr = np.array(adata.obsm[att_key], dtype=np.float64)
        col0 = arr[:, 0]
        col1 = arr[:, 1]
        dev = float(np.max(np.abs(col0 + col1 - 1.0))) if arr.shape[1] == 2 else np.nan

        print(f"\n>> {att_key} ({att_desc})")
        print(f"   [:,0]  mean={col0.mean():.4f}  std={col0.std():.4f}  "
            f"min={col0.min():.4f}  max={col0.max():.4f}")
        print(f"   [:,1]  mean={col1.mean():.4f}  std={col1.std():.4f}  "
            f"min={col1.min():.4f}  max={col1.max():.4f}")
        print(f"   sum-to-1 deviation: {dev:.2e}  "
            f"({'~0 means softmax gate' if dev < 1e-4 else 'NOT sum-to-1'})")

        mean_max = max(col0.mean(), col1.mean())
        if mean_max > 0.90:
            dominant = 0 if col0.mean() > col1.mean() else 1
            print(f"   [WARNING] Gate Collapse detected! [{dominant}] mean={mean_max:.4f} > 0.90")
        elif mean_max > 0.80:
            dominant = 0 if col0.mean() > col1.mean() else 1
            print(f"   [NOTE] Slight bias: [{dominant}] mean={mean_max:.4f} (acceptable but skewed)")
        else:
            print(f"   [OK] Gate healthy: max-mean={mean_max:.4f}, balanced")

        diag_records.append({
            'name': att_key,
            'desc': att_desc,
            'col0_mean': col0.mean(), 'col0_std': col0.std(),
            'col0_min':  col0.min(),  'col0_max':  col0.max(),
            'col1_mean': col1.mean(), 'col1_std': col1.std(),
            'col1_min':  col1.min(),  'col1_max':  col1.max(),
        })



    # ======================================================================
    # [histogram diagnosis] weight distribution of every gate
    # ======================================================================

    n_plots = len(diag_records)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, rec in zip(axes, diag_records):
        arr = np.array(adata.obsm[rec['name']])
        ax.hist(arr[:, 0], bins=30, alpha=0.6, color='steelblue',
                    label=f"col-0 (emb1)  mean={rec['col0_mean']:.3f}")
        ax.hist(arr[:, 1], bins=30, alpha=0.6, color='darkorange',
                    label=f"col-1 (emb2)  mean={rec['col1_mean']:.3f}")
        ax.set_title(rec['desc'], fontsize=10)
        ax.set_xlabel('Attention weight')
        ax.set_ylabel('Count')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(attention_path + '/alpha_histogram_diag.jpeg',
                    format='jpeg', bbox_inches='tight', dpi=200)
    plt.show()
    print(f"\n[INFO] diagnostic histogram saved to: {attention_path + '/alpha_histogram_diag.jpeg'}")


# visualization
# ====== global font setting: Arial ======

if DO_VISUALIZATION_ATTENTION:
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # # visualisation: (fusion of the LLM spatial result and the RNA spatial result)
    # alpha_att1[:, 0]: weight of the LLM spatial fusion result
    # alpha_att1[:, 1]: weight of the RNA spatial fusion result
    plt.rcParams['figure.figsize'] = (8, 5)
    df = pd.DataFrame(columns=['LLM ', 'RNA', 'label'])
    alpha_att1_arr = np.array(adata.obsm['alpha_att1'], dtype=np.float64)
    df['RNA'], df['LLM '] = alpha_att1_arr[:, 1], alpha_att1_arr[:, 0]
    df['label'] = adata.obs['spaFDG'].values
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_spaFDG', 'Modality', 'Weight value']
    df['Weight value'] = df['Weight value'].astype(float)
    ax = sns.violinplot(data=df, x='label_spaFDG', y='Weight value', hue='Modality', hue_order=['RNA', 'LLM '], split=True, inner='quart', linewidth=1, show=False)
    ax.set_title(' Spatial graph fusion weight')
    ax.set_xlabel('Cluster')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')

    plt.tight_layout(w_pad=0.05)

    plt.savefig(attention_path + '/alpha_att1_spatial.jpeg', format='jpeg', bbox_inches='tight', dpi=600)
    plt.show()


    # visualisation: (fusion of the LLM feature result and the RNA feature result)
    # alpha_att2[:, 0]: weight of the LLM feature fusion result
    # alpha_att2[:, 1]: weight of the RNA feature fusion result
    plt.rcParams['figure.figsize'] = (8, 5)
    df = pd.DataFrame(columns=['LLM ', 'RNA', 'label'])
    alpha_att2_arr = np.array(adata.obsm['alpha_att2'], dtype=np.float64)
    df['RNA'], df['LLM '] = alpha_att2_arr[:, 1], alpha_att2_arr[:, 0]
    df['label'] = adata.obs['spaFDG'].values
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_spaFDG', 'Modality', 'Weight value']
    df['Weight value'] = df['Weight value'].astype(float)
    ax = sns.violinplot(data=df, x='label_spaFDG', y='Weight value', hue='Modality', hue_order=['RNA', 'LLM '], split=True, inner='quart', linewidth=1, show=False)
    ax.set_title('Feature graph fusion weight')
    ax.set_xlabel('Cluster')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')

    plt.tight_layout(w_pad=0.05)
    plt.savefig(attention_path + '/alpha_att2_feature.jpeg', format='jpeg', bbox_inches='tight', dpi=600)
    plt.show()

    # visualisation: (fusion of the Geneformer spatial graph and the scGPT spatial graph)
    # note: in example_Lymph_A1.py lines 131-134 the variable names emb_genef / emb_scgpt
    #       are swapped with respect to the file contents
    #       emb_genef actually holds adata2_RNA_embedding.npy (i.e. scGPT)
    #       emb_scgpt actually holds Geneformer_embedding.npy (i.e. Geneformer)
    # hence alpha_llm_spa[:, 0] = scGPT weight, alpha_llm_spa[:, 1] = Geneformer weight
    plt.rcParams['figure.figsize'] = (8, 5)
    df = pd.DataFrame(columns=['Geneformer', 'scGPT', 'label'])
    alpha_llm_spa_arr = np.array(adata.obsm['alpha_llm_spa'], dtype=np.float64)
    df['scGPT'], df['Geneformer'] = alpha_llm_spa_arr[:, 0], alpha_llm_spa_arr[:, 1]
    df['label'] = adata.obs['spaFDG'].values
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_spaFDG', 'Modality', 'Weight value']
    df['Weight value'] = df['Weight value'].astype(float)
    ax = sns.violinplot(data=df, x='label_spaFDG', y='Weight value', hue='Modality', hue_order=['Geneformer', 'scGPT'], split=True, inner='quart', linewidth=1, show=False)
    ax.set_title('LLM spatial graph fusion weight')
    ax.set_xlabel('Cluster')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')
    plt.tight_layout(w_pad=0.05)
    plt.savefig(attention_path + '/alpha_llm_spa.jpeg', format='jpeg', bbox_inches='tight', dpi=600)
    plt.show()

    # visualisation: (fusion of the Geneformer feature graph and the scGPT feature graph)
    # input order as above: alpha_llm_fea[:, 0] = scGPT weight, alpha_llm_fea[:, 1] = Geneformer weight
    plt.rcParams['figure.figsize'] = (8, 5)
    df = pd.DataFrame(columns=['Geneformer', 'scGPT', 'label'])
    alpha_llm_fea_arr = np.array(adata.obsm['alpha_llm_fea'], dtype=np.float64)
    df['scGPT'], df['Geneformer'] = alpha_llm_fea_arr[:, 0], alpha_llm_fea_arr[:, 1]
    df['label'] = adata.obs['spaFDG'].values
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_spaFDG', 'Modality', 'Weight value']
    df['Weight value'] = df['Weight value'].astype(float)
    ax = sns.violinplot(data=df, x='label_spaFDG', y='Weight value', hue='Modality', hue_order=['Geneformer', 'scGPT'], split=True, inner='quart', linewidth=1, show=False)
    ax.set_title('LLM feature graph fusion weight')
    ax.set_xlabel('Cluster')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')
    plt.tight_layout(w_pad=0.05)
    plt.savefig(attention_path + '/alpha_llm_fea.jpeg', format='jpeg', bbox_inches='tight', dpi=600)
    plt.show()

    # visualisation: (fusion of RNA and protein)
    # alpha[:, 0]: weight of the RNA fusion result
    # alpha[:, 1]: weight of the protein fusion result
    plt.rcParams['figure.figsize'] = (8, 5)
    df = pd.DataFrame(columns=['RNA', 'Protein', 'label'])
    alpha_arr = np.array(adata.obsm['alpha'], dtype=np.float64)
    df['RNA'], df['Protein'] = alpha_arr[:, 0], alpha_arr[:, 1]
    df['label'] = adata.obs['spaFDG'].values
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
    df['Weight value'] = df['Weight value'].astype(float)
    ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue='Modality', split=True, inner='quart', linewidth=1, show=False)
    ax.set_title('Modality fusion weight')
    ax.set_xlabel('Cluster')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')
    plt.tight_layout(w_pad=0.05)
    plt.savefig(attention_path + '/alpha_modality.jpeg', format='jpeg', bbox_inches='tight', dpi=600)
    plt.show()








# ============================================================
# persisting the training results - only a single adata_with_results.h5ad is stored
#   adata.obs['spaFDG_name']       : anatomical name of every spot (capsule/cortex/...)
#   adata.uns['spaFDG_id2name']    : cluster_id (int) -> name (str) mapping
#   adata.uns['spaFDG_name_order'] : fixed name order (ascending cluster_id)
# ============================================================


if DO_PERSIST:
    save_dir = os.path.join(file_fold, 'res')
    os.makedirs(save_dir, exist_ok=True)
    print(f'\n[SAVE] persisting training results -> {save_dir}')

    # build anatomical names (read annotation.csv, then Hungarian 1-to-1 mapping)
    pred_int = adata.obs['spaFDG'].astype(int).to_numpy()
    name_for_cluster = {}
    anno_path = os.path.join(file_fold, 'annotation.csv')
    if os.path.exists(anno_path):
        anno = pd.read_csv(anno_path)
        if 'manual-anno' in anno.columns:
            gt_str = anno['manual-anno'].astype(str).to_numpy()
            gt_names = anno['manual-anno'].astype('category').cat.categories.to_numpy()
            name_for_cluster = map_clusters_to_names(pred_int, gt_str, gt_names)
            print(f'  cluster -> name mapping: {name_for_cluster}')
        else:
            print(f'  [WARN] annotation.csv has no manual-anno column, skipping mapping')
    else:
        print(f'  [WARN] {anno_path} not found, skipping mapping')

    # write back into adata_plt (downstream plots use spaFDG_name directly)
    if name_for_cluster:
        name_arr = np.array([name_for_cluster.get(int(c), f'c{c}') for c in pred_int])
        name_cats = [name_for_cluster[int(c)] for c in sorted(name_for_cluster.keys())]
        adata_plt.obs['spaFDG_name'] = pd.Categorical(name_arr, categories=name_cats)
        adata_plt.uns['spaFDG_id2name'] = {f'cluster_{k}': v for k, v in name_for_cluster.items()}
        adata_plt.uns['spaFDG_name_order'] = name_cats
    else:
        adata_plt.obs['spaFDG_name'] = adata_plt.obs['spaFDG'].astype(str).astype('category')

    # store a single h5ad
    adata_plt.write(os.path.join(save_dir, 'adata_with_results.h5ad'))
    print(f'  adata_with_results.h5ad saved (including the spaFDG_name column)')
else:
    print(f'  [SKIP] DO_PERSIST=False, skipping naming / writing (adata_plt keeps its end-of-training state)')


