import os 
import pickle
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, StrVector, ListVector, IntVector, FloatVector

pandas2ri.activate()  

def index_to_array(df):
    # Split the index on 'x' and convert each part to float
    arr = np.array([list(map(float, idx.split('x'))) for idx in df.index])
    return arr

def CT_expression_mean_var(sc_adata):
    
    X   = sc_adata.X.tocsr()                 # ensure CSR (fast row slicing)
    ctype = sc_adata.obs["cellType"]         # or whatever your column is called
    cell_types = ctype.unique()

    # helper: pre-allocate results
    n_genes = X.shape[1]
    mean_mat = np.empty((len(cell_types), n_genes), dtype=np.float32)
    var_mat  = np.empty_like(mean_mat)


    for k, ct in enumerate(cell_types):
        idx = np.where(ctype == ct)[0]    # the row indices for this type
        X_sub = X[idx]                    # still sparse

        # mean  (E[X])
        mu = X_sub.mean(axis=0).A1        # .A1 -> 1-D np.array
        mean_mat[k] = mu

        # variance  (E[X²] − (E[X])²)
        mu2 = X_sub.power(2).mean(axis=0).A1
        var_mat[k] = mu2 - mu**2

    mean_df = pd.DataFrame(mean_mat.T, index=sc_adata.var_names, columns=cell_types)
    var_df  = pd.DataFrame(var_mat.T , index=sc_adata.var_names, columns=cell_types)
    
    return mean_df.T, var_df.T

def adata_to_dgC_fast(adata, Matrix):
    X = adata.X
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)


    row_idx, col_idx = X.nonzero()          # works for CSR or CSC
    rows = IntVector(row_idx + 1)           # R is 1-based
    cols = IntVector(col_idx + 1)
    # values
    vals = FloatVector(X.data)

    dgC = Matrix.sparseMatrix(
        i = rows, j = cols, x = vals,
        dims = IntVector(X.shape),
        dimnames = ListVector({
            'rownames': StrVector(list(adata.obs_names)),
            'colnames': StrVector(list(adata.var_names))
        }),
        repr = "C"                          # “C” = column-compressed
    )
    return dgC

def FUSION_Init(adata_list, sc_adata, domain_size, R_HOME, R_USER, min_gene: int = 100, min_spot: int = 5):

    domain_size =int(1.8*domain_size)
    
    folder_path = "preprocess_ref"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
            
    Matrix = importr('Matrix')
    base = importr('base')

    sc_count_r = adata_to_dgC_fast(sc_adata.T, Matrix)
    sc_meta_r = pandas2ri.py2rpy(sc_adata.obs)

    if not os.path.exists('mean_basis.csv'):

        mean_basis, var_basis=CT_expression_mean_var(sc_adata)

        mean_basis.to_csv(folder_path+'/mean_basis.csv')

        var_basis.to_csv(folder_path+'/var_basis.csv')

    else:

        pass

    for idx in range(len(adata_list)):

        sp_count_list = ro.ListVector({str(i): R_df for i, R_df in enumerate([adata_to_dgC_fast(df, Matrix) for df in adata_list[idx]])})

        r_code = f"""

        library(IRIS)

        create_coord_df <- function(df) {{
            # Extract column names
            
            col_names <- colnames(df)

            split_names <- strsplit(col_names, "x")
            
            # Create a new DataFrame with columns 'x' and 'y'
            coord_df <- data.frame(
            x = sapply(split_names, function(pair) as.numeric(pair[1])),
            y = sapply(split_names, function(pair) as.numeric(pair[2]))
            )
            
            # Set the row names of the new DataFrame to be the same as the original DataFrame
            rownames(coord_df) <- col_names
            
            return(coord_df)
        }}

        initial_value <- function(h5_files, sc_count, sc_meta, numCluster) {{
            h5_files          <- lapply(h5_files, t)
            renamed_pos_files <- lapply(h5_files, create_coord_df)

            capture.output(
                IRIS_object <- createIRISObject(
                    spatial_countMat_list = h5_files,
                    spatial_location_list = renamed_pos_files,
                    sc_count   = sc_count,
                    sc_meta    = sc_meta,
                    ct.varname = 'cellType',
                    sample.varname = 'sampleID',
                    minCountGene = {min_gene},
                    minCountSpot = {min_spot}
                )
            )

            result <- capture.output(
                IRIS_object <- IRIS_spatial(IRIS_object, numCluster = numCluster)
            )
            return(IRIS_object@IRIS_Prop)
        }}
        """

        ro.r(r_code)
        
        # Run the R function on the list of dgCMatrix objects
        initial_result = ro.r['initial_value'](sp_count_list, sc_count_r, sc_meta_r, domain_size)

        pd_dt=ro.conversion.rpy2py(initial_result)
        
        grouped_dfs = {group: group_df for group, group_df in pd_dt.groupby('Slice')}
        
        # Print each group as a separate DataFrame
        initial_prop=[]
        merged_pos=[]
        
        for group_df in grouped_dfs.items():

            initial_prop.append(group_df[1].iloc[:,4:])
            
            merged_pos.append(group_df[1][['x','y']].values)

        with open(folder_path+'/init_CT_s{}.pkl'.format(idx), 'wb') as f:
            pickle.dump(initial_prop, f)
            
        with open(folder_path+'/init_pos_s{}.pkl'.format(idx), 'wb') as f:
            pickle.dump(merged_pos, f)

        # with open(folder_path+'/initial_domain_section{}.pkl'.format(idx), 'wb') as f:
        #     pickle.dump(initial_domain, f)

        print('Section{} initialization complete'.format(idx))
            









