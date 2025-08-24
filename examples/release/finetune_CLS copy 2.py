# %%
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

from scgpt_config import *
from rich.console import Console
from rich.table import Table
from gene2ncbi import initialize_genept_embeddings,get_embs_to_include,initialize_go_embeddings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from NCBI.tokenizer.gene_tokenizer import GeneVocab
import NCBI as scg
from NCBI.model import TransformerModel, AdversarialDiscriminator
from NCBI.tokenizer import tokenize_and_pad_batch, random_mask_value
from NCBI.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)

from NCBI.preprocess import Preprocessor
from NCBI import SubsetsBatchSampler
from NCBI.utils import set_seed, category_str2int, eval_scib_metrics
import argparse

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
# os.environ["WANDB_MODE"] = "offline"


console = Console()
# device = torch.cuda.device(2)
os.environ['WANDB_MODE'] = 'disabled'
os.environ["WANDB_DISABLED"] = "true"
# sc.set_figure_params(figsize=(6, 6))
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# os.environ["WANDB_MODE"] = "offline"

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="PBMC_10K",
    model_type ="scgenept_ncbi_gpt",
    do_train=True,
    load_model="/root/autodl-tmp/geneptfile/pretrained/scgpt/",
    pretrained_model_dir='/root/autodl-tmp/geneptfile/',
    mask_ratio=0.4,
    epochs=50,
    n_bins=51,
    GEPC=True,  # Masked value prediction for cell embedding
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,
    lr=1e-4,
    batch_size=16,
    layer_size=128,
    nlayers=4,
    nhead=4,
    # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    log_interval=10,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
)
run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)

# table = Table(title="训练参数")
# table.add_column("参数", justify="right", style="cyan", no_wrap=True)
# table.add_column("值", style="magenta")
# for key, value in hyperparameter_defaults.items():
#     table.add_row(key, str(value))
# console.print(table)

config = wandb.config
print(config)

set_seed(config.seed)

# %%
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 1200  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = True
DSBN = True  # Domain-spec batchnorm
explicit_zero_prob = True  # whether explicit bernoulli for zeros

# settings for train

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
# MVC = config.MVC  # Masked value prediction for cell embedding
# ECS = config.ecs_thres > 0  # Elastic cell similarity objective
# DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
# INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
# input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
# cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
# adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
# adv_D_delay_epochs = 0
# mvc_decoder_style = "inner product"
# ecs_threshold = config.ecs_thres
# dab_weight = config.dab_weight

# %%
dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
# save the whole script to the dir
os.system(f"cp {__file__} {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


# %% [markdown]
# ## Loading and preparing data
if dataset_name == "PBMC_10K":
    adata = scvi.data.pbmc_dataset(save_path="/root/autodl-tmp/HiCelleraaai/data/pbmc", remove_extracted_data= False)  # 11990 × 3346
    ori_batch_col = "batch"
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
    adata.var = adata.var.set_index("gene_symbols")
    data_is_raw = True

print("obs维度:", adata.obs.shape)
print("obs列名:", adata.obs.columns.tolist())
print("前几行:")
print(adata.obs.head())
print(" 输出: (细胞数, 基因数):",adata.shape)
# 输出: (细胞数, 基因数)
print("细胞数 (n_obs):", adata.n_obs)
print("基因数 (n_vars):", adata.n_vars)
print(adata)
print(adata.layers.keys())
print("==== AnnData object 基本信息 ====")
print(adata)
print("\n==== 数据维度 ====")
print("adata.shape =", adata.shape)
print("n_obs (细胞数):", adata.n_obs)
print("n_vars (基因数):", adata.n_vars)

print("\n==== obs（细胞注释）表头 ====")
print(adata.obs.head())
print("\nobs 列名:", adata.obs.columns.tolist())
print("obs.shape =", adata.obs.shape)

print("\n==== var（基因注释）表头 ====")
print(adata.var.head())
print("\nvar 列名:", adata.var.columns.tolist())
print("var.shape =", adata.var.shape)

print("\n==== 可用 layers ====")
print(list(adata.layers.keys()))

print("\n==== uns（无结构注释） ====")
print(list(adata.uns.keys()))

print("\n==== 主表达矩阵前 5×5 ====")
try:
    print(adata.X[:5, :5])
except Exception as e:
    print("主矩阵太大或为稀疏格式，未直接输出（", e, "）")
    
import matplotlib.pyplot as plt
#细胞类型分布条形图
adata.obs['celltype'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Cell Type Distribution')
plt.ylabel('Number of Cells')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures/pbmc_celltype_bar.png', dpi=300)
#
# 
# 2. 批次分布和标签分布条形图
# 批次
adata.obs['batch'].value_counts().sort_index().plot(kind='bar', color='orange')
plt.title('Batch Distribution')
plt.xlabel('Batch')
plt.ylabel('Number of Cells')
plt.tight_layout()
plt.savefig('figures/pbmc_batch_bar.png', dpi=300)

# 标签
adata.obs['str_labels'].value_counts().plot(kind='bar', color='green')
plt.title('Label Distribution')
plt.ylabel('Number of Cells')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures/pbmc_label_bar.png', dpi=300)
#
import scanpy as sc
sc.pp.normalize_total(adata)#. UMAP（或 t-SNE）降维可视化，按细胞类型、批次、标签上色
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
# 按细胞类型
sc.pl.umap(adata, color='celltype', save='_pbmc_umap_celltype.png', wspace=0.5, legend_loc='on data', frameon=False, title='UMAP by Cell Type', show=False)
# 按批次
sc.pl.umap(adata, color='batch', save='_pbmc_umap_batch.png', wspace=0.5, legend_loc='right margin', frameon=False, title='UMAP by Batch', show=False)
# 按str_labels
sc.pl.umap(adata, color='str_labels', save='_pbmc_umap_labels.png', wspace=0.5, legend_loc='right margin', frameon=False, title='UMAP by Label', show=False)
# 指定基因在 UMAP 或小提琴图中的表达
# (1) UMAP上的表达
sc.pl.umap(adata, color=['ISG15', 'NOC2L'], save='_pbmc_umap_genes.png', cmap='viridis', show=False)
# (2) 小提琴图，按细胞类型分组
sc.pl.violin(adata, keys=['ISG15', 'NOC2L'], groupby='celltype', save='_pbmc_violin_genes.png', show=False)

# 选取前10个高变基因   ##热图：不同细胞类型的高表达基因
sc.pp.highly_variable_genes(adata, n_top_genes=10)
top_genes = adata.var[adata.var['highly_variable']].index.tolist()
sc.pl.heatmap(adata, top_genes, groupby='celltype', save='_pbmc_heatmap_topgenes.png', show=False)

# 各细胞类型的文库大小(n_counts)分布
import seaborn as sns
import pandas as pd
plt.figure(figsize=(10,5))
sns.boxplot(x='celltype', y='n_counts', data=adata.obs, palette='Set3')
plt.xticks(rotation=45, ha='right')
plt.title('Library Size (n_counts) by Cell Type')
plt.tight_layout()
plt.savefig('figures/pbmc_librarysize_by_celltype.png', dpi=300)


# 每个细胞类型的基因数分布（基因数可用表达基因数或总基因数）
# 假设每个细胞表达的基因数为非零表达的基因数
gene_counts = (adata.X > 0).sum(axis=1)
adata.obs['n_genes_by_counts'] = gene_counts
sns.boxplot(x='celltype', y='n_genes_by_counts', data=adata.obs)
plt.xticks(rotation=45, ha='right')
plt.title('Detected Genes per Cell by Cell Type')
plt.tight_layout()
plt.savefig('figures/pbmc_ngenes_by_celltype.png', dpi=300)




# %%
# make the batch category column
adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels

adata.var["gene_name"] = adata.var.index.tolist()

if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    embsize = config.layer_size 
    nhead = config.nhead
    nlayers = config.nlayers  
    d_hid = config.layer_size


# %%
# set up the preprocessor, use the args to config the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)

# %%
if per_seq_batch_sample:
    # sort the adata by batch_id in advance
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

# %% [markdown]
# ## Tokenize input

# %%
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)####11990,1200
genes = adata.var["gene_name"].tolist()


#
# 假设类别文件中包含了对应 celltype 信息
# 在这里，我们假设类别信息也存储在 `.obs` 中的某个列中，比如 'celltype'

data_is_raw = False
filter_gene_by_counts = False
# 将 celltype 转化为数值编码
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
# adata.obs["celltype"]
# 把每个不同的 celltype（细胞类型）分配一个唯一的整数编号（label），但编号顺序依赖于 celltype 字符串的排序（通常是按字母顺序，或最先出现的顺序）。
# 取出AnnData对象 .obs 表中 "celltype" 这一列，每个细胞的类型名称，如：“B cell”、“T cell”等。
# .astype("category")

# 把这一列的数据类型转换为“分类类型”（category）。
# 分类类型是 Pandas 里一种专门用于处理有限个类别数据的类型（比如离散的细胞类型）。
# .cat.codes

# 对分类类型的每个类别分配一个唯一的整数编码（从0开始）。
# 例如，假如有3种类型：["B cell", "Monocyte", "T cell"]，会自动分别编码成 [0, 1, 2]（顺序由pandas自动决定，通常是按字母顺序）。
# .values

# 得到上一步编码后的整数标签，返回一个NumPy数组。
adata.obs["celltype_id"] = celltype_id_labels
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
print(id2type)
type2id = {v: k for k, v in id2type.items()}
#.keys() 只拿到所有key
# .values() 只拿到所有value
# .items() 同时拿到成对的 key 和 value#
###############################################
#"{0: 'B cells', 1: 'CD14+ Monocytes', 2: 'CD4 T cells', 3: 'CD8 T cells',
# 4: 'Dendritic Cells', 5: 'FCGR3A+ Monocytes', 6: 'Megakaryocytes', 7: 'NK cells', 8: 'Other'}"
adata.obs["celltype_id"] = adata.obs["celltype"].apply(lambda x: type2id.get(x, 62))
celltype_id_labels = adata.obs["celltype_id"].values
celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
# num_types = len(set(celltypes_labels))
celltypes_labels = np.array(celltypes_labels)
print(celltypes_labels)


batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_celltype_id_labels,
    valid_celltype_id_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels,celltype_id_labels, batch_ids, test_size=0.1, shuffle=True
)
print(train_celltype_labels)
print(train_celltype_id_labels)
print(len(train_celltype_labels))
print(len(train_celltype_id_labels))

# train_class_labels_tensor = torch.tensor(train_celltype_id_labels, dtype=torch.long)
# valid_class_labels_tensor = torch.tensor(valid_celltype_id_labels, dtype=torch.long)

# %%
if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)


# Get the embedding types to include in the model training 
embs_to_include = get_embs_to_include(config.model_type)
# Get GenePT embeddings to include
genept_embs, genept_emb_type, genept_emb_dim, found_genes_genept = initialize_genept_embeddings(embs_to_include, genes, vocab, config.model_type, config.pretrained_model_dir)

go_embs_to_include, go_emb_type, go_emb_dim, found_genes_go = initialize_go_embeddings(embs_to_include, genes, vocab, config.model_type, config.pretrained_model_dir)

# %%
# %%
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=True,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)


# data_loader# %%
def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
    
    tensor_labels_train = torch.tensor(train_celltype_id_labels, dtype=torch.long)
    tensor_labels_valid = torch.tensor(valid_celltype_id_labels, dtype=torch.long)

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_labels_train = tensor_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_labels_valid = tensor_labels_valid[valid_sort_ids]

    # train_data_pt = {
    #     "gene_ids": input_gene_ids_train,
    #     "values": input_values_train,
    #     "target_values": target_values_train,
    #     "batch_labels": tensor_batch_labels_train,
    # }
    # valid_data_pt = {
    #     "gene_ids": input_gene_ids_valid,
    #     "values": input_values_valid,
    #     "target_values": target_values_valid,
    #     "batch_labels": tensor_batch_labels_valid,
    # }
    
    train_data_pt = {
    "gene_ids": input_gene_ids_train,
    "values": input_values_train,
    "target_values": target_values_train,
    "batch_labels": tensor_batch_labels_train,
    "class_labels": tensor_labels_train,  # 新增分类标签
    }
    
    valid_data_pt = {
    "gene_ids": input_gene_ids_valid,
    "values": input_values_valid,
    "target_values": target_values_valid,
    "batch_labels": tensor_batch_labels_valid,
    "class_labels": tensor_labels_valid,
    }

    # plot_value_distributions(train_data_pt, valid_data_pt, mask_value, pad_value)
    # plot_boxplot_by_batch(train_data_pt, mask_value, pad_value, 'train')
    # plot_boxplot_by_batch(valid_data_pt, mask_value, pad_value, 'valid')

    # plot_violin_by_class(train_data_pt, mask_value, pad_value, 'train')
    # plot_violin_by_class(valid_data_pt, mask_value, pad_value, 'valid')
    # plot_input_vs_target(train_data_pt, mask_value, pad_value, 'train')
    # plot_input_vs_target(valid_data_pt, mask_value, pad_value, 'valid')
    # plot_label_counts(train_data_pt, 'train')
    # plot_label_counts(valid_data_pt, 'valid')

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,

) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


import matplotlib.pyplot as plt


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_id = 0 # 这是第3张显卡
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,    
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC,
    do_dab=True,
    use_batch_labels=True,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
    n_input_bins=n_input_bins,
    # cell_emb_style=cell_emb_style,   ###先不急研究这个
    # mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
    ##genept
    embs_to_include = embs_to_include,
    genept_embs = genept_embs, 
    genept_emb_type = genept_emb_type, 
    genept_emb_size = genept_emb_dim,
    go_embs_to_include = go_embs_to_include,
    go_emb_type = go_emb_type,
    go_emb_size = go_emb_dim,
)


if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)
wandb.watch(model)

criterion_cls = nn.CrossEntropyLoss()
criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        type_labels = batch_data["class_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                CLS=CLS,
                CCE=False,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                # do_sample=do_sample_in_train,
            )
            #               
            cls_output = output_dict["cls_output"]
            cls_loss = criterion_cls(cls_output, type_labels)               
            # print("one batch off 44444444444444444444444444444444")
            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics_to_log = {"train/mse": loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if config.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            if config.GEPC and explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )
            if config.ecs_thres > 0:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + config.dab_weight * loss_dab + cls_loss
            metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        # for batch_data in loader:
        #     input_gene_ids = batch_data["gene_ids"].to(device)
        #     input_values = batch_data["values"].to(device)
        #     target_values = batch_data["target_values"].to(device)
        #     batch_labels = batch_data["batch_labels"].to(device)
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            type_labels = batch_data["class_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                # output_dict = model(
                #     input_gene_ids,
                #     input_values,
                #     src_key_padding_mask=src_key_padding_mask,
                #     batch_labels=batch_labels if DSBN else None,
                # )
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                    CLS=CLS,
                    CCE=False,
                    MVC=config.GEPC,
                    ECS=config.ecs_thres > 0,
                    # do_sample=do_sample_in_train,
                )
                cls_output = output_dict["cls_output"]
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)
#################CLA
            accuracy = (cls_output.argmax(1) == type_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            # total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = cls_output.argmax(1).cpu().numpy()
            predictions.append(preds)

    if return_raw:
        predictions= np.concatenate(predictions, axis=0)
    wandb.log(
        {
        "valid/mse": total_loss / total_num,
        "valid/mre": total_error / total_num,
        "valid/dab": total_dab / total_num,
        "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab)
        / total_num,
        "epoch": epoch,
        },
    )

    return total_loss / total_num, total_error / total_num , predictions


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[input_layer_key].A
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )

    celltypes_labels = adata_t.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)
    print(celltypes_labels)



    # celltypes = adata_test.obs["celltype"].unique()
    # num_types = len(np.unique(celltype_id_labels))
    # id2type = dict(enumerate(adata_test.obs["celltype"].astype("category").cat.categories))

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["str_batch"],
            title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["batch_umap"] = fig

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["celltype_umap"] = fig

    if len(include_types) == 1:
        return results


best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
        # label_id =train_class_labels_tensor,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
        # label_id =valid_class_labels_tensor,
    )

    if config.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_mre, predictions = evaluate(
        model,
        loader=valid_loader,
        return_raw=True,
    )
    
    print(predictions) #1199
    print(valid_celltype_id_labels) #1199
    valid_celltype_id_labels = np.array(valid_celltype_id_labels)
    predictions = np.array(predictions)
    ###################################################################################################################################

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    cm = confusion_matrix(valid_celltype_id_labels, predictions, labels=list(id2type.keys()))
    plt.figure(figsize=(20, 16))
    confusion_matrix_image = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=id2type.values(),
                                         yticklabels=id2type.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show(block=True)
    plt.savefig(save_dir / "confusion_matrix.png", dpi=600)
    import sklearn
    acc = sklearn.metrics.accuracy_score(valid_celltype_id_labels, predictions)
    from sklearn.metrics import classification_report
    unique_labels = np.unique(valid_celltype_id_labels)
    report = classification_report(valid_celltype_id_labels, predictions, target_names=list(id2type.values()),
                                   labels=list(id2type.keys()))
    class_report = classification_report(valid_celltype_id_labels, predictions, target_names=list(id2type.values()),
                                         labels=list(id2type.keys()), output_dict=True)
    # 将class_report转为excel
    class_report = pd.DataFrame(class_report).transpose()
    class_report.to_excel(save_dir / "/classification_report.xlsx")
    print(acc)
    print(report)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(valid_celltype_id_labels, predictions)
    precision = precision_score(valid_celltype_id_labels, predictions, average="macro")
    recall = recall_score(valid_celltype_id_labels, predictions, average="macro")
    macro_f1 = f1_score(valid_celltype_id_labels, predictions, average="macro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, "
        f"Precision: {precision:.3f}, "
        # f"Recall: {recall:.3f}, "
        # f"Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        # "test/recall": recall,
        # "test/macro_f1": macro_f1,
    }

    # return predictions, valid_celltype_id_labels, results

    
    
    
    
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        logger.info(f"Saving model to {save_dir}")
        torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

        # eval on testdata
        results = eval_testdata(
            best_model,
            adata_t=adata_sorted if per_seq_batch_sample else adata,
            include_types=["cls"],
        )
        results["batch_umap"].savefig(
            save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
        )

        results["celltype_umap"].savefig(
            save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
        )
        metrics_to_log = {"test/" + k: v for k, v in results.items()}
        metrics_to_log["test/batch_umap"] = wandb.Image(
            str(save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
            caption=f"celltype avg_bio epoch {best_model_epoch}",
        )

        metrics_to_log["test/celltype_umap"] = wandb.Image(
            str(save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
            caption=f"celltype avg_bio epoch {best_model_epoch}",
        )
        metrics_to_log["test/best_model_epoch"] = best_model_epoch
        wandb.log(metrics_to_log)
        wandb.log({"avg_bio": results.get("avg_bio", 0.0)})
        
    torch.save(best_model.state_dict(), save_dir / "last_model.pt")

    scheduler.step()


torch.save(best_model.state_dict(), save_dir / "best_model.pt")


artifact = wandb.Artifact(f"best_model", type="model")
glob_str = os.path.join(save_dir, "best_model.pt")
artifact.add_file(glob_str)
run.log_artifact(artifact)

run.finish()
wandb.finish()
gc.collect()

# %%

