import os
from typing import List
import scanpy
import anndata
import numpy as np
import pandas as pd
import logging
import os
import shutil
import tarfile
import warnings
from typing import Tuple
import numpy as np
logger = logging.getLogger(__name__)


def load_pbmc_dataset(
    save_path: str = "data/",
    remove_extracted_data: bool = True,
) -> anndata.AnnData:
    # urls = [
    #     "https://github.com/YosefLab/scVI-data/raw/master/gene_info.csv",
    #     "https://github.com/YosefLab/scVI-data/raw/master/pbmc_metadata.pickle",
    # ]
    # save_fns = ["gene_info_pbmc.csv", "pbmc_metadata.pickle"]

    # for i in range(len(urls)):
        # _download(urls[i], save_path, save_fns[i])

    de_metadata = pd.read_csv(os.path.join(save_path, "gene_info_pbmc.csv"), sep=",")
    pbmc_metadata = pd.read_pickle(os.path.join(save_path, "pbmc_metadata.pickle"))
    pbmc8k = load_dataset_10x(
        "pbmc8k",
        save_path=save_path,
        var_names="gene_ids",
        remove_extracted_data=remove_extracted_data,
    )
    pbmc4k = load_dataset_10x(
        "pbmc4k",
        save_path=save_path,
        var_names="gene_ids",
        remove_extracted_data=remove_extracted_data,
    )
    barcodes = np.concatenate((pbmc8k.obs_names, pbmc4k.obs_names))

    adata = pbmc8k.concatenate(pbmc4k)
    adata.obs_names = barcodes

    dict_barcodes = dict(zip(barcodes, np.arange(len(barcodes))))
    subset_cells = []
    barcodes_metadata = pbmc_metadata["barcodes"].index.values.ravel().astype(np.str)
    for barcode in barcodes_metadata:
        if (
            barcode in dict_barcodes
        ):  # barcodes with end -11 filtered on 10X website (49 cells)
            subset_cells += [dict_barcodes[barcode]]
    adata = adata[np.asarray(subset_cells), :].copy()
    idx_metadata = np.asarray(
        [not barcode.endswith("11") for barcode in barcodes_metadata], dtype=np.bool
    )
    genes_to_keep = list(
        de_metadata["ENSG"].values
    )  # only keep the genes for which we have de data
    difference = list(
        set(genes_to_keep).difference(set(adata.var_names))
    )  # Non empty only for unit tests
    for gene in difference:
        genes_to_keep.remove(gene)

    adata = adata[:, genes_to_keep].copy()
    design = pbmc_metadata["design"][idx_metadata]
    raw_qc = pbmc_metadata["raw_qc"][idx_metadata]
    normalized_qc = pbmc_metadata["normalized_qc"][idx_metadata]

    design.index = adata.obs_names
    raw_qc.index = adata.obs_names
    normalized_qc.index = adata.obs_names
    adata.obs["batch"] = adata.obs["batch"].astype(np.int64)
    adata.obsm["design"] = design
    adata.obsm["raw_qc"] = raw_qc
    adata.obsm["normalized_qc"] = normalized_qc

    adata.obsm["qc_pc"] = pbmc_metadata["qc_pc"][idx_metadata]
    labels = pbmc_metadata["clusters"][idx_metadata]
    cell_types = pbmc_metadata["list_clusters"]
    adata.obs["labels"] = labels
    adata.uns["cell_types"] = cell_types
    adata.obs["str_labels"] = [cell_types[i] for i in labels]

    adata.var["n_counts"] = np.squeeze(np.asarray(np.sum(adata.X, axis=0)))

    return adata

available_datasets = {
    "1.1.0": [
        "frozen_pbmc_donor_a",
        "frozen_pbmc_donor_b",
        "frozen_pbmc_donor_c",
        "fresh_68k_pbmc_donor_a",
        "cd14_monocytes",
        "b_cells",
        "cd34",
        "cd56_nk",
        "cd4_t_helper",
        "regulatory_t",
        "naive_t",
        "memory_t",
        "cytotoxic_t",
        "naive_cytotoxic",
    ],
    "1.3.0": ["1M_neurons"],
    "2.1.0": ["pbmc8k", "pbmc4k", "t_3k", "t_4k", "neuron_9k"],
    "3.0.0": [
        "pbmc_1k_protein_v3",
        "pbmc_10k_protein_v3",
        "malt_10k_protein_v3",
        "pbmc_1k_v2",
        "pbmc_1k_v3",
        "pbmc_10k_v3",
        "hgmm_1k_v2",
        "hgmm_1k_v3",
        "hgmm_5k_v3",
        "hgmm_10k_v3",
        "neuron_1k_v2",
        "neuron_1k_v3",
        "neuron_10k_v3",
        "heart_1k_v2",
        "heart_1k_v3",
        "heart_10k_v3",
    ],
    "3.1.0": ["5k_pbmc_protein_v3", "5k_pbmc_protein_v3_nextgem"],
}


def load_dataset_10x(
    dataset_name: str = None,
    filename: str = None,
    save_path: str = "data/10X",
    url: str = None,
    return_filtered: bool = True,
    remove_extracted_data: bool = False,
    **scanpy_read_10x_kwargs,
):
   

    adata = scanpy.read_10x_mtx(path_to_data_folder, **scanpy_read_10x_kwargs)
    adata.var_names_make_unique()
    scanpy.pp.filter_cells(adata, min_counts=1)
    scanpy.pp.filter_genes(adata, min_counts=1)

    return adata

dataset_to_group = dict(
    [
        (dataset_name, group)
        for group, list_datasets in available_datasets.items()
        for dataset_name in list_datasets
    ]
)

group_to_url_skeleton = {
    "1.1.0": "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_gene_bc_matrices.tar.gz",
    "1.3.0": "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_gene_bc_matrices_h5.h5",
    "2.1.0": "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_gene_bc_matrices.tar.gz",
    "3.0.0": "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_feature_bc_matrix.h5",
    "3.1.0": "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_feature_bc_matrix.h5",
}

group_to_filename_skeleton = {
    "1.1.0": "{}_gene_bc_matrices.tar.gz",
    "1.3.0": "{}_gene_bc_matrices_h5.h5",
    "2.1.0": "{}_gene_bc_matrices.tar.gz",
    "3.0.0": "{}_feature_bc_matrix.h5",
    "3.1.0": "{}_feature_bc_matrix.h5",
}



def _find_path_to_mtx(save_path: str) -> Tuple[str, str]:
    """
    Returns exact path for the data in the archive.

    This is required because 10X doesn't have a consistent way of storing their data.
    Additionally, the function returns whether the data is stored in compressed format.

    Returns
    -------
    path in which files are contains and their suffix if compressed.

    """
    for root, subdirs, files in os.walk(save_path):
        # do not consider hidden files
        files = [f for f in files if not f[0] == "."]
        contains_mat = [
            filename == "matrix.mtx" or filename == "matrix.mtx.gz"
            for filename in files
        ]
        contains_mat = np.asarray(contains_mat).any()
        if contains_mat:
            is_tar = files[0][-3:] == ".gz"
            suffix = ".gz" if is_tar else ""
            return root, suffix
    raise FileNotFoundError("No matrix.mtx(.gz) found in path (%s)." % save_path)



def _download(url, save_path: str, filename: str):
    """Writes data from url to file."""
    if os.path.exists(os.path.join(save_path, filename)):
        logger.info(f"File {os.path.join(save_path, filename)} already downloaded")
        return
    # elif url is None:
    #     logger.info(
    #         f"No backup URL provided for missing file {os.path.join(save_path, filename)}"
    #     )
    #     return
    # req = urllib.request.Request(url, headers={"User-Agent": "Magic Browser"})
    # try:
    #     r = urllib.request.urlopen(req)
    #     if r.getheader("Content-Length") is None:
    #         raise FileNotFoundError(
    #             f"Found file with no content at {url}. "
    #             "This is possibly a directory rather than a file path."
    #         )
    # except urllib.error.HTTPError as exc:
    #     if exc.code == "404":
    #         raise FileNotFoundError(f"Could not find file at {url}") from exc
    #     raise exc
    # logger.info("Downloading file at %s" % os.path.join(save_path, filename))

    # def read_iter(file, block_size=1000):
    #     """
    #     Iterates through file.

    #     Given a file 'file', returns an iterator that returns bytes of
    #     size 'blocksize' from the file, using read().
    #     """
    #     while True:
    #         block = file.read(block_size)
    #         if not block:
    #             break
    #         yield block

    # # Create the path to save the data
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # block_size = 1000

    # filesize = int(r.getheader("Content-Length"))
    # filesize = np.rint(filesize / block_size)
    # with open(os.path.join(save_path, filename), "wb") as f:
    #     iterator = read_iter(r, block_size=block_size)
    #     for data in track(
    #         iterator, style="tqdm", total=filesize, description="Downloading..."
    #     ):
    #         f.write(data)

adata = load_pbmc_dataset("/mnt/12T/home/liuym/codes/HiCeller-main/data") 