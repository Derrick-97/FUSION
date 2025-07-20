import numpy as np 
import pandas as pd
import torch
import os
import pickle
import re
from typing import List

def pre_processing(df_list, mean_basis, var_basis, log_fc_cut):
    
    def sc_gene_sel(
        m_basis: pd.DataFrame,          # genes × cell-types  (mean expr.)
        var_basis: pd.DataFrame,        # genes × cell-types  (variance)
        log_fc_cut: float = 1.5,        # ln-fold-change  (exp≈4.48×)
        vmean_cut: float = 100,         # var / mean upper bound
        min_genes: int = 500            # guarantee at least this many genes
    ) -> List[str]:

        # ---- safety: align the two matrices ----
        m_basis = m_basis.loc[var_basis.index, var_basis.columns]

        n_types = m_basis.shape[1]

        # ------------------------------------------------------------
        # Step 1  (vectorised enrichment)
        # ------------------------------------------------------------
        select_gene=[]
        for c in m_basis.columns:
            mean_scores = m_basis.drop(columns=[c]).mean(axis=1)
            select_gene.extend(m_basis[m_basis[c]/mean_scores > np.exp(log_fc_cut) ].index)
            
        genes_fc=list(set(select_gene))

        # ------------------------------------------------------------
        # Step 2  (variance / mean)
        # ------------------------------------------------------------
        var_ratio = var_basis.loc[genes_fc] / m_basis.loc[genes_fc]
        var_ratio=var_ratio.dropna(axis=0, how='any').mean(axis=1)
        
        keep_vm   = var_ratio < vmean_cut
        if keep_vm.sum() < min_genes:
            try:
                kth = np.nanpercentile(var_ratio, 100*min_genes / len(var_ratio.T))
                keep_vm |= var_ratio <= kth
            except:
                
                raise Exception('Too few genes remained!')

        selected = var_ratio.index[keep_vm]
        return selected.tolist()


    def df_clip(df_list, sc_count):

        # Find the common columns
        common_columns = set(df_list[0].columns).intersection(*[df.columns for df in df_list[1:]])
        
        if sc_count is not None:
            
            common_columns=common_columns.intersection(sc_count.columns.values)
    
        dfs_with_origin = []
        for i, df in enumerate(df_list):
            df_common = df[list(common_columns)].copy()  # Select common columns
            df_common['origin'] = f'batch{i+1}'  # Add the identifier column
            dfs_with_origin.append(df_common)

        # Step 3: Concatenate the DataFrames
        merged_df = pd.concat(dfs_with_origin, ignore_index=True)
            
        return merged_df
    
    def process_sp(df, bc_parition):
    
        sp_count=df

        genes=list(sp_count.iloc[:,:-1].columns.values)

        genes.sort()

        sp_count=sp_count[genes]
        
        # print(sp_count)

        sp_count = torch.from_numpy(sp_count.values)
        
        return sp_count, genes
    
    gene_list=sc_gene_sel(mean_basis.T, var_basis.T, log_fc_cut)
    
    merged_df=df_clip(df_list, mean_basis.loc[:,gene_list])
    
    print(merged_df.index)
    
    print('Batch merging complete')
    parition=merged_df.groupby('origin').apply(lambda x: pd.Series({
    'start_idx': merged_df.index.get_loc(x.index[0]),  # Convert start rowname to integer index
    'end_idx': merged_df.index.get_loc(x.index[-1])+1    # Convert end rowname to integer index
    }))
    print('parition complete')
    print(parition)
    sp_count, sel_genes= process_sp(merged_df, parition)
    
    if mean_basis is not None:
        
        mean_basis=mean_basis[list(sel_genes)].copy()
        
    return sp_count, mean_basis, parition

def slice_matching(folder_path):
    
    # Initialize an empty list to store all the ndarrays
    pos_list = []
    
    # Function to extract the index from the filename (e.g., 'pos_4.pkl' -> 4)
    def extract_index(filename):
        match = re.search(r'init_pos_s(\d+)\.pkl', filename)
        return int(match.group(1)) if match else None
    
    # Collect all filenames that match the pattern 'pos_<index>.pkl'
    files_with_index = [f for f in os.listdir(folder_path) if f.startswith('init_pos') and f.endswith('.pkl')]
    
    # Sort files based on their index
    sorted_files = sorted(files_with_index, key=extract_index)
    
    # Load the files in order and extend the arrays into 'all_arrays'
    for filename in sorted_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as file:
            arrays_list = pickle.load(file)
            pos_list.append(arrays_list)
    
    # Now 'all_arrays' contains all ndarrays from the loaded pickle files in sorted order
    print(f"Total slice loaded: {len(pos_list)}")

    return pos_list


