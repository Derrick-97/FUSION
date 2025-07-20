import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ref_ import sparsify_kernel_matrix,compute_rbf_kernel_matrix
import random
import numpy as np

def kmeans(tensor, K):
        
    from sklearn.cluster import KMeans

    # Convert the tensor to a NumPy array
    tensor_np = tensor.clone().detach().numpy()

    # Define the number of clusters
    num_clusters = K

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters,random_state=999)
    kmeans.fit(tensor_np)

    # Get the cluster labels
    return torch.from_numpy(kmeans.labels_).long(), torch.from_numpy(kmeans.cluster_centers_)
    
    
def initial_ref(sp_count, initial_prop, sc_ref, pos_list, topic_size, device, seed):
    
    kernel_list=[]

    for pos in pos_list:
    
        kernel_list.append(sparsify_kernel_matrix(compute_rbf_kernel_matrix(torch.from_numpy(pos)), top_k=10,norm=True).to(device))
    
    #IRIS_prop=torch.from_numpy(IRIS.iloc[:,4:].values).to(device)
    
    initial_prop=torch.from_numpy(initial_prop).to(device)
    
    sc_tensor=torch.from_numpy(sc_ref).to(device)
    
    doc_size=sp_count.shape[0]
    
    vocal_size=sp_count.shape[1]
    
    alpha_1=torch.ones(topic_size,vocal_size).to(device)
    
    alpha_2=torch.ones(doc_size, topic_size).to(device)
    
    initial_topic= torch.ones(doc_size, vocal_size, topic_size).to(device)
    
    alpha_ttow= torch.ones(topic_size,vocal_size).to(device)
    
    alpha_dtot=torch.ones(doc_size, topic_size).to(device)
    
    exp_log_ttow=torch.digamma(alpha_ttow)-torch.digamma(alpha_ttow.sum(-1,keepdims=True))
        
    exp_log_dtot=torch.digamma(alpha_dtot)-torch.digamma(alpha_dtot.sum(-1,keepdims=True))
    
    return initial_topic, initial_prop, sc_tensor, exp_log_ttow, exp_log_dtot, kernel_list, alpha_1, alpha_2



def initial_cluster(LDA, sp_count, pos_list, parition, topic_size, domain_size, device, seed):

    doc_size=sp_count.shape[0]

    vocal_size=sp_count.shape[1]

    pi_prior=torch.ones(doc_size, domain_size).to(device)

    kernel_list=[]

    for pos in pos_list:
    
        kernel_list.append(sparsify_kernel_matrix(compute_rbf_kernel_matrix(torch.from_numpy(pos)), top_k=30,norm=True).to(device))
    
    initial_topic_MBLDA=LDA.post_topic.detach().clone()
    
    initial_topic_MBLDA.require_grad=False
    
    alpha_1=torch.ones(topic_size,vocal_size).to(device)
    
    alpha_2=torch.ones(domain_size, topic_size).to(device)
    
    alpha_ttow= torch.ones(topic_size,vocal_size).to(device)
    
    exp_log_ttow=torch.digamma(alpha_ttow)-torch.digamma(alpha_ttow.sum(-1,keepdims=True))
    
    alpha_dtot= torch.ones(domain_size, topic_size).to(device)
    
    exp_log_dtot=torch.digamma(alpha_dtot)-torch.digamma(alpha_dtot.sum(-1,keepdims=True))
    
    post_domain_tmp, domain_center = kmeans(LDA.decon(topic_size).detach().cpu().clone(), domain_size)
    
    post_domain=[]
    
    for i, pos in enumerate(pos_list):
    
        do_assign=post_domain_tmp[parition.values[i][0]:parition.values[i][1]]
    
        post_domain.append(do_assign)

    initial_domain=F.one_hot(torch.cat(post_domain), num_classes=domain_size).to(device)
    
    exp_log_dtot= (LDA.decon(topic_size).unsqueeze(1)*initial_domain.unsqueeze(-1)).sum(0)/initial_domain.unsqueeze(-1).sum(0).view(-1,1)
    
    return initial_topic_MBLDA, initial_domain, exp_log_dtot, pi_prior, kernel_list, alpha_1,alpha_2
