import os
import tqdm
import random
import torch
import shutil
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Dirichlet, Multinomial
import numpy as np
import pandas as pd
from init_ref import initial_ref, initial_cluster
from dataprocess import pre_processing, slice_matching
from ref_ import VI_LDA, q_ttow, training_MBLDA, M_step

def merged_array_2_df(list1, list2, list3, celltype):

    df_list = []
    
    # Iterate over both lists and merge arrays
    for k in range(len(list1)):
        # Get the N*D array from list1 and N*2 array from list2
        array1 = list1[k]
        array2 = list2[k].reshape(-1,1)
        array3= list3[k]
        
        # Concatenate along the columns (axis=1)
        merged_array = np.concatenate([array1, array2, array3], axis=1)
        
        # Create a DataFrame from the merged array
        df = pd.DataFrame(merged_array)
        
        # Add the section index as a new column
        df['section'] = k
        
        # Append the DataFrame to the list
        df_list.append(df)
    
    # Concatenate all DataFrames into a big DataFrame
    final_df = pd.concat(df_list, ignore_index=True)
    
    final_df.columns = ['x', 'y', 'domain'] + celltype + ['slide']
    
    return final_df
    

def initial_prop(sp_slice_list, sc_tensor):

    #dataprocess/initial_prop
    
    #return initial_prop
    return

def format_index(index, decimals):
    # Split the index by 'x', convert to float, and round to the specified number of decimals
    rounded_values = [np.round(float(i), decimals) for i in index.split('x')]
    # Join the rounded values with '*' without specifying digits in f-string
    return 'x'.join(map(str, rounded_values))


def FUSION_preprocess(adata_list, log_fc_cut, seed):
    
    folder_path = "preprocess_ref"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    m_basis=pd.read_csv(folder_path+"/mean_basis.csv", index_col=0)
    
    var_basis=pd.read_csv(folder_path+"/var_basis.csv", index_col=0)   
    
    # Create the folder

    all_pos_list = slice_matching(folder_path)

    for idx, pos_list in enumerate(all_pos_list):

        adata_df=[]
        
        for i, merged_pos in enumerate(pos_list):

            coords_all = adata_list[idx][i].obsm["spatial"]        # 形状 (M, 2)

            target_coords = merged_pos     
            
            decimals = 3
            all_key     = np.round(coords_all,  decimals).astype(str)  # M×2 → 字符串
            target_key  = np.round(target_coords, decimals).astype(str)

            # 2) 把每行拼成 (x, y) 元组后放进集合，做 O(1) 查询
            target_set = {tuple(row) for row in target_key}

            mask = np.array([tuple(row) in target_set for row in all_key])
            adata_subset = adata_list[idx][i][mask].copy()

            adata_df.append(adata_subset.to_df())
        
        sp_count, mean_basis, parition=pre_processing(adata_df,\
                                                      m_basis, var_basis, log_fc_cut)
    
        torch.save(sp_count, folder_path+'/sp_count_s{}.pt'.format(idx))
    
        parition.to_csv(folder_path+'/parition_s{}.csv'.format(idx))
        
    
    print('preprocess complete')


  
def FUSION_main(sp_slice_list, topic_size, domain_size, spatial_penalty, remove_tmp_files, device, seed):

    folder_path = "preprocess_ref"
    
    if len(sp_slice_list) != len(spatial_penalty):
        
        raise ValueError("The number penalty does not equal to slice sections")

    if not os.path.exists('preprocess_ref'):

        raise ValueError("No processed spatial data found")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

    out=[]

    embeddings=[]

    for section_idx in range(len(sp_slice_list)):

        sp_count=torch.load(folder_path+'/sp_count_s{}.pt'.format(section_idx)).to(device)
        parition=pd.read_csv(folder_path+'/parition_s{}.csv'.format(section_idx), index_col=0)
        
        sc_count=pd.read_csv(folder_path+'/mean_basis.csv', index_col=0)
        
        with open(folder_path+'/init_pos_s{}.pkl'.format(section_idx), 'rb') as f:
            pos_list = pickle.load(f)
        
        with open(folder_path+'/init_CT_s{}.pkl'.format(section_idx), 'rb') as f:
            initial_prop_list = pickle.load(f)

        initial_prop=pd.concat(initial_prop_list, ignore_index=True).values

        CellType=list(initial_prop_list[0].columns)
            
        initial_topic, initial_prop, sc_tensor, exp_log_ttow, exp_log_dtot, kernel_list, alpha_1, alpha_2\
        =initial_ref(sp_count, initial_prop, sc_count.values, pos_list, topic_size, device, seed)

        LDA = VI_LDA(sp_count, topic_size, initial_topic).to(device)

        optimizer = optim.Adam(LDA.parameters(), lr=0.5)
        
        losses = []
        
        eps=1e-2
        
        # Training Loop
        LDA.train()
        
        for epoch in tqdm.tqdm(range(50)):
                
            # print(training.post_topic)
            optimizer.zero_grad()

            loss = LDA(exp_log_ttow, exp_log_dtot, sc_tensor,kernel_list,parition,sp_count,\
                              initial_prop,spatial_penalty[section_idx][0],spatial_penalty[section_idx][1])
        
            losses.append(loss.item())
    
            loss.backward()
            
            optimizer.step()
            
            # if epoch %1 ==0:
            exp_log_ttow, exp_log_dtot, alpha_ttow, alpha_dtot\
            = q_ttow(sp_count, F.softmax(LDA.post_topic,dim=-1).detach().clone(), alpha_1,alpha_2)
                     
        LDA.eval()

        initial_topic, initial_domain, exp_log_dtot, pi_prior, kernel_list, alpha_1,alpha_2 = initial_cluster(LDA, sp_count, pos_list, parition, topic_size, domain_size, device, seed)

        domain_assign, losses= training_MBLDA(sp_count, initial_topic, initial_domain, exp_log_dtot, pi_prior, kernel_list, parition, alpha_1, alpha_2, device)

        CT_proportion=[]
        
        domain_list=[]

        i_domina_list=[]

        domain_num=[]

        for i, pos in enumerate(pos_list):

            do_assign=domain_assign[parition.values[i][0]:parition.values[i][1]]

            i_domina_list.append(np.argmax(initial_domain.cpu().detach().numpy(),axis=1)[parition.values[i][0]:parition.values[i][1]])
        
            CT_assign=LDA.decon(sc_tensor.shape[0]).detach().cpu().clone()[parition.values[i][0]:parition.values[i][1]]

            embeddings.append(LDA.decon(topic_size).detach().cpu().clone()[parition.values[i][0]:parition.values[i][1]])
        
            CT_proportion.append(CT_assign)
            
            domain_list.append(do_assign)

            domain_num.append(len(set(domain_list[-1])))
        
        print('Detected Domain Size across Slices:{}'.format(domain_num))

        if np.median(domain_num)<0.5*domain_size:

            print('Domains Collapse')

            domain_list = i_domina_list

        #print(CT_assign.shape)
        out.append(merged_array_2_df(pos_list, domain_list, CT_proportion, CellType))

    if remove_tmp_files:
        shutil.rmtree('preprocess_ref')

    return out, embeddings

        

        

        #return LDA.post_topic.detach().clone()




        


        



    

    

