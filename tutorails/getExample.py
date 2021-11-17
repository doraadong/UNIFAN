import os
import argparse

import scanpy as sc

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, type=str,
                        default='./facs.h5ad', help="string, path to the downloaded data, "
                                                         "default './facs.h5ad'")
    parser.add_argument('-i', '--folder', required=True, type=str,
                        default='../example/input', help="string, path to the folder to save the data, "
                                                         "default '../example/input'")
    parser.add_argument('-t', '--tissue', required=True, type=str,
                        default=None, help="string, specify the output tissue; if using the default None, then all "
                                           "tissues will be outputted and saved separately in the folder; default None")
    parser.add_argument('-k', '--topk', required=False, default=2000, type=int,
                        help="integer, optional, number of most variable genes, default 2000")

    args = parser.parse_args()
    print(args)

    parent_folder = args.folder
    filepath = args.path
    tissue = args.tissue
    topk = args.topk

    adata = sc.read(filepath, dtype='float64')

    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)
    sc.pp.filter_genes(adata, min_cells=5)

    # get unnormalized version to infer highly variable genes
    full_data_unnorm = adata[adata.obs["age"] == "3m", :].copy()  # equivalent to Tabula Muris data (MARS)

    # not include Brain_Myeloid and Marrow (MARS)
    full_data_unnorm = full_data_unnorm[full_data_unnorm.obs["tissue"] != "Marrow", :].copy()
    full_data_unnorm = full_data_unnorm[full_data_unnorm.obs["tissue"] != "Brain_Myeloid", :].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)

    sc.pp.log1p(adata)

    sc.pp.scale(adata, max_value=10)

    full_data = adata[adata.obs["age"] == "3m", :]  # equivalent to Tabula Muris data (MARS)

    # not include Brain_Myeloid and Marrow (MARS)
    full_data = full_data[full_data.obs["tissue"] != "Marrow", :]
    full_data = full_data[full_data.obs["tissue"] != "Brain_Myeloid", :]

    if tissue is None:
        # save each tissue separately
        for tissue in set(full_data.obs['tissue'].values):
            print(f"Saving output for {tissue}")

            # get most variable genes using unnormalized
            subset_unnorm = full_data_unnorm[full_data_unnorm.obs['tissue'] == tissue].copy()

            # selecting genes using Seurat v3 method
            sc.pp.highly_variable_genes(subset_unnorm, n_top_genes=topk, flavor='seurat_v3')

            # keep only the current tissue
            subset = full_data[full_data.obs['tissue'] == tissue].copy()
            subset.var["highly_variable"] = subset_unnorm.var["highly_variable"]

            # write full data
            subset.write_h5ad(os.path.join(parent_folder, f"{tissue}_facts_processed_3m.h5ad"))
    else:
        # get most variable genes using unnormalized
        subset_unnorm = full_data_unnorm[full_data_unnorm.obs['tissue'] == tissue].copy()

        # selecting genes using Seurat v3 method
        sc.pp.highly_variable_genes(subset_unnorm, n_top_genes=topk, flavor='seurat_v3')

        # keep only the current tissue
        subset = full_data[full_data.obs['tissue'] == tissue].copy()
        subset.var["highly_variable"] = subset_unnorm.var["highly_variable"]

        # write full data
        subset.write_h5ad(os.path.join(parent_folder, f"{tissue}_facts_processed_3m.h5ad"))



if __name__ == '__main__':
    main()
