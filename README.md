# UNIFAN
Unsupervised cell functional annotation for single-cell RNA-Seq

```shell
usage: main.py [-h] -i INPUT -o OUTPUT -p PROJECT -t TISSUE [-l LABEL]
               [-v VARIABLE] [-r PRIOR] [-f {gene_sets,gene,gene_gene_sets}]
               [-a ALPHA] [-b BETA] [-g GAMMA] [-u TAU] [-d DIM] [-s BATCH]
               [-na NANNO] [-ns NSCORE] [-nu NAUTO] [-nc NCLUSTER]
               [-nze NZENCO] [-nzd NZDECO] [-dze DIMZENCO] [-dzd DIMZDECO]
               [-nre NRENCO] [-dre DIMRENCO] [-drd DIMRDECO]
               [-n {sigmoid,non-negative,gaussian}] [-m RANDOM] [-c CUDA]
               [-w NWORKERS]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        string, path to the input expression data, default
                        '../input/data.h5ad'
  -o OUTPUT, --output OUTPUT
                        string, path to the output folder, default
                        '../output/'
  -p PROJECT, --project PROJECT
                        string, identifier for the project, e.g., tabula_muris
  -t TISSUE, --tissue TISSUE
                        string, tissue where the input data is sampled from
  -l LABEL, --label LABEL
                        string, optional, the column / field name of the
                        ground truth label, if available; used for evaluation
                        only; default 'None'
  -v VARIABLE, --variable VARIABLE
                        string, optional, the column / field name of the
                        highly variable genes; default 'highly_variable'
  -r PRIOR, --prior PRIOR
                        string, optional, gene set file names used to learn
                        the gene set activity scores, use '+' to separate
                        multiple gene set names, default
                        c5.go.bp.v7.4.symbols.gmt+c2.cp.v7.4.symbols.gmt+TF-
                        DNA
  -f {gene_sets,gene,gene_gene_sets}, --features {gene_sets,gene,gene_gene_sets}
                        string, optional, features used for the annotator, any
                        of 'gene_sets', 'gene' or 'gene_gene_sets', default
                        'gene_gene_sets'
  -a ALPHA, --alpha ALPHA
                        float, optional, hyperparameter for the L1 term in the
                        set cover loss, default 1e-2
  -b BETA, --beta BETA  float, optional, hyperparameter for the set cover term
                        in the set cover loss, default 1e-5
  -g GAMMA, --gamma GAMMA
                        float, optional, hyperparameter for the exclusive L1
                        term, default 1e-3
  -u TAU, --tau TAU     float, optional, hyperparameter for the annotator
                        loss, default 10
  -d DIM, --dim DIM     integer, optional, dimension for the low-dimensional
                        representation, default 32
  -s BATCH, --batch BATCH
                        integer, optional, batch size for training except for
                        pretraining annotator (fixed at 32), default 128
  -na NANNO, --nanno NANNO
                        integer, optional, number of epochs to pretrain the
                        annotator, default 50
  -ns NSCORE, --nscore NSCORE
                        integer, optional, number of epochs to train the gene
                        set activity model, default 70
  -nu NAUTO, --nauto NAUTO
                        integer, optional, number of epochs to pretrain the
                        annocluster model, default 50
  -nc NCLUSTER, --ncluster NCLUSTER
                        integer, optional, number of epochs to train the
                        annocluster model, default 25
  -nze NZENCO, --nzenco NZENCO
                        float, optional, number of hidden layers for encoder
                        of annocluster, default 3
  -nzd NZDECO, --nzdeco NZDECO
                        float, optional, number of hidden layers for decoder
                        of annocluster, default 2
  -dze DIMZENCO, --dimzenco DIMZENCO
                        integer, optional, number of nodes for hidden layers
                        for encoder of annocluster, default 128
  -dzd DIMZDECO, --dimzdeco DIMZDECO
                        integer, optional, number of nodes for hidden layers
                        for decoder of annocluster, default 128
  -nre NRENCO, --nrenco NRENCO
                        integer, optional, number of hidden layers for the
                        encoder of gene set activity scores model, default 5
  -dre DIMRENCO, --dimrenco DIMRENCO
                        integer, optional, number of nodes for hidden layers
                        for encoder of gene set activity scores model, default
                        128
  -drd DIMRDECO, --dimrdeco DIMRDECO
                        integer, optional, number of nodes for hidden layers
                        for decoder of gene set activity scores model, default
                        128
  -n {sigmoid,non-negative,gaussian}, --network {sigmoid,non-negative,gaussian}
                        string, optional, the encoder for the gene set
                        activity model, any of 'sigmoid', 'non-negative' or
                        'gaussian', default 'non-negative'
  -m RANDOM, --random RANDOM
                        integer, optional, random seed for the initialization,
                        default 0
  -c CUDA, --cuda CUDA  boolean, optional, if use GPU for neural network
                        training, default False
  -w NWORKERS, --nworkers NWORKERS
                        integer, optional, number of works for dataloader,
                        default 8
```
