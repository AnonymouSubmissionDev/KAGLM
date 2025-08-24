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
device = torch.cuda.device(2)
os.environ['WANDB_MODE'] = 'disabled'
os.environ["WANDB_DISABLED"] = "true"
# sc.set_figure_params(figsize=(6, 6))
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# os.environ["WANDB_MODE"] = "offline"

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="AnnDataManager",
    do_train=True,
    load_model="/mnt/12T/home/liuym/codes/xxgxxxgpt/data/models/pretrained/scgpt/",
    pretrained_model_dir='/mnt/12T/home/liuym/codes/xxgxxxgpt/data/models/',
    mask_ratio=0.4,
    epochs=2,
    n_bins=51,
    GEPC=True,  # Masked value prediction for cell embedding
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,
    lr=1e-4,
    batch_size=64,
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

#################这个是虚拟数据，用不了
# %% [markdown]
# ## Loading and preparing data
if dataset_name == "AnnDataManager":
    adata = scvi.data.annotation_simulation("1","/root/autodl-tmp/hzdata/scdata/scVI-data-master/simulation/")  # 11990 × 3346

print(adata.obs["labels"])
print(adata.obs[['labels', 'batch']].head())

print(adata.var_names)


