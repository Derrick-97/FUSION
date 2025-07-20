import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class VI_LDA(nn.Module):
    def __init__(self, sp_count, topic_size, initial_topic):
        super(VI_LDA, self).__init__()
        
        self.doc_size, self.vocal_size = sp_count.shape[0], sp_count.shape[1]
        
        self.topic_size=topic_size
        
        self.post_topic=nn.Parameter(initial_topic)
        
        self.sp_count=sp_count
    
    def exp_log_like_f1(self,sp_count, exp_log_ttowt):
        
        #sp_count:D*P
        #exp_log_ttow: T*W
        
        # print(exp_log_ttowt)
        posterior_prob=F.softmax(self.post_topic,dim=-1)
        
        return (sp_count*torch.sum(posterior_prob*torch.transpose(exp_log_ttowt, -2, -1),-1)).sum(1).mean()
    
    def exp_log_like_f2(self,sp_count, exp_log_dtot):
        
        #sp_count:D*P
        #exp_log_dtot: D*T
        posterior_prob=F.softmax(self.post_topic,dim=-1)
        
        return (sp_count*torch.sum(posterior_prob*exp_log_dtot.view(self.doc_size,1,self.topic_size),-1)).sum(1).mean()

    def exp_log_like_f3(self,kernel_mat, parition,B):
        
        domain_prob=self.decon(B)
        
        loss=0
        
        for i, kernel in enumerate(kernel_mat):
                        
            c_domain=domain_prob[parition.values[i][0]:parition.values[i][1],:].float()
            
            adjacency_matrix = (kernel >0).int().float()
            
            L=torch.diag(adjacency_matrix.sum(0))-adjacency_matrix
            
            loss+=torch.trace(torch.transpose(c_domain, -2, -1)@L@c_domain)
            #self.post_topic.shape[-1]**2
            
        return loss/domain_prob.shape[0]
    
    def loss_ref(self, sp_count, reference):
        
        #reference: K by P tensor

        CT_prop=self.decon(reference.shape[0])

        ref_nomalized= reference/reference.sum(1,keepdims=True)
        sp_count_nomalized=sp_count/sp_count.sum(1,keepdims=True)
        
        return torch.norm(sp_count_nomalized - CT_prop @ ref_nomalized, p='fro')

    def regularize(self, IRIS_prop):
        
        #reference: K by P tensor

        CT_prop=self.decon(IRIS_prop.shape[1])
        
        return torch.norm(IRIS_prop - CT_prop, p='fro')
    
    
    def decon(self, B):
        
        unorm_prob= (F.softmax(self.post_topic,dim=-1)*self.sp_count.view(self.doc_size,self.vocal_size,1)).sum(1)

        CT_prop=unorm_prob/(unorm_prob.sum(-1,keepdims=True)+1e-20)
        
        D=CT_prop.shape[1]
        
        D_by_B = D // B

        # Initialize an empty list to hold the sums of the groups
        result = []
        
        # Sum the groups of columns
        for i in range(B - 1):
            start_col = i * D_by_B
            end_col = (i + 1) * D_by_B
            result.append(CT_prop[:, start_col:end_col].sum(dim=1))
        
        # Handle the remaining columns for the last group
        result.append(CT_prop[:, (B - 1) * D_by_B:].sum(dim=1))
        
        # Stack the results to form the final tensor
        result_tensor = torch.stack(result, dim=1)

        return result_tensor
    
    
    def forward(self,exp_log_ttowt, exp_log_dtot,ref,kernel_mat, parition, sp_count_1,IRIS_prop,weight,weight2):
        
        entropy=(-F.softmax(self.post_topic,dim=-1)*torch.log(F.softmax(self.post_topic,dim=-1)+1e-20)).sum(-1)

        return  weight2*self.exp_log_like_f3(kernel_mat, parition,ref.shape[0])+100*self.regularize(IRIS_prop)\
        -weight*(self.exp_log_like_f1(self.sp_count, exp_log_ttowt)+self.exp_log_like_f2(self.sp_count, exp_log_dtot)\
                       + (entropy*self.sp_count).sum(1).mean())
        
        
        
         
def q_ttow(sp_count, post_topic, alpha_1,alpha_2):
    
    #alpha1: T*W
    #alpha2: D*T
    
    alpha_ttow=torch.transpose(torch.sum(sp_count.unsqueeze(-1)* post_topic,0), -2,-1)+alpha_1
    
    alpha_dtot=torch.sum(sp_count.unsqueeze(-1)* post_topic,1)+alpha_2
    
    exp_log_ttow=torch.digamma(alpha_ttow)-torch.digamma(alpha_ttow.sum(-1,keepdims=True))
    
    exp_log_dtot=torch.digamma(alpha_dtot)-torch.digamma(alpha_dtot.sum(-1,keepdims=True))
    
    return exp_log_ttow, exp_log_dtot, alpha_ttow, alpha_dtot

class VI_domain(nn.Module):
    def __init__(self, sp_count, post_topic, initial_domain):
        super(VI_domain, self).__init__()
        
        self.doc_size, self.vocal_size = sp_count.shape[0], sp_count.shape[1]
        
        self.post_topic=post_topic
        
        self.sp_count=sp_count
        
        self.post_domain=nn.Parameter(initial_domain)

    def exp_log_like_f0(self, exp_log_ttowt):
        
        #sp_count:D*P
        #exp_log_ttow: T*W
        
        # print(exp_log_ttowt)
        posterior_prob=F.softmax(self.post_topic,dim=-1)
        
        return (self.sp_count*torch.sum(posterior_prob*torch.transpose(exp_log_ttowt, -2, -1),-1)).sum(1).mean()
    
    def exp_log_like_f1(self,sp_count, exp_log_dtot):
        
        #sp_count:D*P
        #exp_log_dtot: O*T
        #posterior_domain: D*O
        posterior_prob=F.softmax(self.post_topic,dim=-1)
        
        domain_prob=F.softmax(self.post_domain,dim=-1)
        
        exp_tmp1=(sp_count.unsqueeze(-1)*posterior_prob).sum(1)

        return (domain_prob.unsqueeze(-1)*exp_log_dtot*exp_tmp1.unsqueeze(1)).sum([1,2]).mean()

    def Gaussian_log_like_f1(self,sp_count, exp_log_dtot):
        
        #sp_count:D*P
        #exp_log_dtot: O*T
        #posterior_domain: D*O
        
        domain_prob=F.softmax(self.post_domain,dim=-1)
        
        exp_tmp1=self.decon(exp_log_dtot.shape[1])

        return (-domain_prob*((exp_tmp1.unsqueeze(1)-exp_log_dtot)**2).sum(-1)).sum(1).mean()

        #return (domain_prob.unsqueeze(-1)*exp_log_dtot*exp_tmp1.unsqueeze(1)).sum([1,2]).mean()
    
    def exp_log_like_f2(self,sp_count, prior_pi, kernel_mat, parition):
        
        domain_prob=F.softmax(self.post_domain,dim=-1)
        
        return (domain_prob*torch.log(batch_product(kernel_mat, prior_pi, parition)+1e-20)).sum(1).mean()
    
    def exp_log_like_f3(self,kernel_mat, parition):
        
        domain_prob=F.softmax(self.post_domain,dim=-1)
        
        loss=0
        
        for i, kernel in enumerate(kernel_mat):
            
            c_domain=domain_prob[parition.values[i][0]:parition.values[i][1],:].float()
            
            adjacency_matrix = (kernel >0).int().float()
            
            L=torch.diag(adjacency_matrix.sum(0))-adjacency_matrix
            
            loss+=torch.trace(torch.transpose(c_domain, -2, -1)@L@c_domain)
            #self.post_topic.shape[-1]**2
            
        return loss/domain_prob.shape[0]

    def decon(self, B):
        
        unorm_prob= (F.softmax(self.post_topic,dim=-1)*self.sp_count.view(self.doc_size,self.vocal_size,1)).sum(1)

        CT_prop=unorm_prob/(unorm_prob.sum(-1,keepdims=True)+1e-20)
        
        D=CT_prop.shape[1]
        
        D_by_B = D // B
        if D % B != 0:
            pad_size = B - (D % B)  # Columns needed to make D divisible by B
        else:
            pad_size = 0
        
        # Pad the tensor with zeros along the last dimension
        padded_tensor = torch.nn.functional.pad(CT_prop, (0, pad_size))
        
        # Reshape and sum over the appropriate dimension
        D_new = D + pad_size
        
        result_tensor = padded_tensor.view(CT_prop.shape[0], B, D_new // B).sum(dim=2)

        return result_tensor

    def loss_ref(self, sp_count, reference):
        
        #reference: K by P tensor

        CT_prop=self.decon(reference.shape[0])

        ref_nomalized= reference/reference.sum(1,keepdims=True)
        sp_count_nomalized=sp_count/sp_count.sum(1,keepdims=True)
        
        return torch.norm(sp_count_nomalized - CT_prop @ ref_nomalized, p='fro')
    
    def forward(self, exp_log_dtot, prior_pi, kernel_list, parition):
        entropy=(-F.softmax(self.post_domain,dim=-1)*F.log_softmax(self.post_domain,dim=-1)).sum(-1)

        return  0.2*self.exp_log_like_f3( kernel_list, parition)-2000*self.Gaussian_log_like_f1(self.sp_count, exp_log_dtot)\
        -0.2*(self.exp_log_like_f2(self.sp_count, prior_pi, kernel_list, parition)+ entropy.mean()) 
        

def batch_product(kernel_list, tensor, parition):
    
    result=[]

    for i, kernel in enumerate(kernel_list):
        
        result.append(F.softmax(kernel,dim=1)@tensor[parition.values[i][0]:parition.values[i][1],:])
        
    return torch.concat(result)

def M_step(sp_count, post_topic,post_domain, alpah_1,alpha_2, kernel_mat,parition, device):
    
    #alpha1: T*W
    #alpha2: O*T
    #alpha3: B*O

    one_hot_probability = np.zeros_like(post_domain.detach().cpu().numpy())
    
    one_hot_probability[np.arange(post_domain.shape[0]), np.argmax(post_domain.detach().cpu().numpy(), axis=1)] = 1

    one_hot_probability=torch.from_numpy(one_hot_probability).to(device)

    exp_log_dtot= (post_topic.unsqueeze(1)*one_hot_probability.unsqueeze(-1)).sum(0)/one_hot_probability.unsqueeze(-1).sum(0).view(-1,1)
    
    pi_prior= batch_product(kernel_mat, post_domain, parition)
    
    pi_prior=pi_prior/pi_prior.sum(1,keepdims=True)

    attention=[]

    for i, kernel in enumerate(kernel_mat):
        
        attention.append(kernel*(post_domain[parition.values[i][0]:parition.values[i][1],:]@post_domain[parition.values[i][0]:parition.values[i][1],:].T))

    return pi_prior, exp_log_dtot, attention

def training_MBLDA(sp_count, initial_topic, initial_domain, exp_log_dtot, pi_prior, kernel_list, parition, alpha_1, alpha_2, device):
    
    training = VI_domain(sp_count, initial_topic, initial_domain.float()).to(device)

    optimizer = optim.Adam(training.parameters(), lr=0.5)

    losses = []

    attention=kernel_list

    # Training Loop
    training.train()

    for epoch in tqdm.tqdm(range(50)):

            # print(training.post_topic)
            optimizer.zero_grad()
            # optimizer2.zero_grad()

            loss = training( exp_log_dtot, pi_prior, kernel_list, parition)
        
            losses.append(loss.item())

            loss.backward(retain_graph=True)

            optimizer.step()

            # if epoch %1 ==0:
            pi_prior,  exp_log_dtot, attention\
            = M_step(sp_count, training.decon(initial_topic.shape[-1]), F.softmax(training.post_domain.detach().float(), dim=-1), alpha_1, alpha_2, kernel_list,parition, device)

            # print(F.softmax(training.post_domain.float(), dim=-1))
    training.eval()

    domain_assign = np.argmax(F.softmax(training.post_domain.float(),dim=1).detach().cpu().numpy(), axis=1)
    
    return domain_assign, losses

def sparsify_kernel_matrix(kernel_matrix, top_k=3, norm=False):
    # Create a mask for the top_k closest neighbors
    values, indices = torch.topk(kernel_matrix, top_k, dim=-1, largest=True)
    
    # Create a mask of the same shape as the kernel matrix
    mask = torch.zeros_like(kernel_matrix, dtype=torch.bool)
    
    # Use advanced indexing to set the top_k indices to True
    mask.scatter_(1, indices, True)
    
    # Apply the mask to the kernel matrix
    sparse_matrix = torch.where(mask, kernel_matrix, torch.tensor(0.0, device=kernel_matrix.device))
    
    if norm:
        sparse_matrix=sparse_matrix/sparse_matrix.sum(1,keepdims=True)
    
    return sparse_matrix

def compute_rbf_kernel_matrix(coords, sigma=10.0):
    # Compute the pairwise Euclidean distance matrix
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    distance_matrix = torch.sum(diff ** 2, -1)

    # Apply the RBF kernel
    kernel_matrix = torch.exp(-distance_matrix / (2 * sigma ** 2))
    
    return kernel_matrix.float()