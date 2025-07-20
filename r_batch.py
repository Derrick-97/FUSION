import torch
import pickle
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
    
def WGAN(tensor_pair, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    class EmbeddingTransformer(nn.Module):
        def __init__(self, embedding_dim):
            super(EmbeddingTransformer, self).__init__()
            
            # Create a 10-layer network as before
            layers = []
            input_dim = embedding_dim
            
            for _ in range(5):
                layers.append(nn.Linear(input_dim, 256))
                layers.append(nn.ReLU())
                input_dim = 256
            
            layers.append(nn.Linear(256, embedding_dim))  # Final layer to project back to embedding dimension
            self.model = nn.Sequential(*layers)
            self.model.apply(self.init_weights_zero)
            
        def forward(self, x):
            return self.model(x)
    
        @staticmethod
        def init_weights_zero(m):
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        
    
    class Discriminator(nn.Module):
        def __init__(self, embedding_dim):
            super(Discriminator, self).__init__()
            
            # Define layers
            layers = []
            input_dim = embedding_dim
            
            # Create 10-layer structure with residual connections
            for i in range(5):
                fc = nn.Linear(input_dim, 256)
                layers.append(fc)
                layers.append(nn.LeakyReLU(0.2))
                input_dim = 256
            
            # Final layer for binary classification
            self.layers = nn.Sequential(*layers)
            self.output_layer = nn.Linear(256, 1)
            
        def forward(self, x):
            out = x
            for layer in self.layers:
                out = layer(out)
            
            # Output layer for discriminator score
            return self.output_layer(out)
    
    # Define WGAN Loss (with Gradient Penalty)
    def compute_gradient_penalty(D, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1).to(real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(d_interpolates.size()).to(real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    D = tensor_pair[0].shape[1]

    batch_size=64
    n_epochs=50
    
    # Sample data (replace these with actual embeddings)
    label0_embeddings = tensor_pair[1].float().to("cuda")  # N1 * D embeddings (label 0)
    label1_embeddings = tensor_pair[0].float().to("cuda")  # N2 * D embeddings (label 1)
    
    dataset_label0 = TensorDataset(label0_embeddings)
    dataset_label1 = TensorDataset(label1_embeddings)
    
    dataloader = DataLoader(dataset_label0, batch_size=batch_size, shuffle=True)
    #dataloader_label1 = DataLoader(dataset_label1, batch_size=batch_size, shuffle=True)
    
    transformer = EmbeddingTransformer(D).to("cuda")
    discriminator = Discriminator(D).to("cuda")
    
    # Optimizers
    lr = 5e-4
    transformer_optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.5, 0.9))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    lambda_gp = 10  # Gradient penalty weight
    
    for epoch in range(n_epochs):
        #for batch_label0, batch_label1 in zip(dataloader_label0, dataloader_label1):
    
        for idx, batch_label0 in enumerate(dataloader):
            #print(batch_label0[0])
    
            distances = torch.cdist(batch_label0[0], label1_embeddings)  # N1 x N2 tensor of distances
    
            # Find the index of the minimum distance in tensor2 for each row in tensor1
            closest_indices = distances.argmin(dim=1)
            
            # Replace each row in tensor1 with the closest row in tensor2
            real_embeddings = label1_embeddings[closest_indices].to('cuda')-batch_label0[0].to("cuda")
            
            fake_embeddings = transformer(batch_label0[0].to("cuda"))
             # Fake embeddings generated from label 0
                
            #fake_embeddings = fake_embeddings/fake_embeddings.sum(axis=1, keepdims=True)
            
            if real_embeddings.shape[0] != fake_embeddings.shape[0]:
    
                continue
                
            # Train Discriminator
            discriminator_optimizer.zero_grad()
            
            # Detach the fake embeddings so generator doesn’t get updated here
            fake_embeddings_detached = fake_embeddings.detach()
            
            # Compute discriminator loss
            d_real = discriminator(real_embeddings)
            d_fake = discriminator(fake_embeddings_detached)
            
            gradient_penalty = compute_gradient_penalty(discriminator, real_embeddings, fake_embeddings_detached)
            d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gradient_penalty
    
            # Update discriminator
            d_loss.backward()
            discriminator_optimizer.step()
        
            # Train Generator
            transformer_optimizer.zero_grad()
            
            g_loss = -discriminator(fake_embeddings).mean()
            
            # Update transformer without retain_graph
            g_loss.backward()
            transformer_optimizer.step()

    transformed_embeddings_eval = transformer(label0_embeddings).detach()
    
    return transformed_embeddings_eval


def embedding_matching(tensor1, tensor2, seed):
    
    torch.manual_seed(seed)
    # Compute pairwise distances
    tensor1+=torch.randn_like(tensor1)*0.1
    
    distances = torch.cdist(tensor1, tensor2)  # N1 x N2 tensor of distances
    
    # Find the index of the minimum distance in tensor2 for each row in tensor1
    closest_indices = distances.argmin(dim=1)
    
    # Replace each row in tensor1 with the closest row in tensor2
    new_tensor1 = tensor2[closest_indices]

    return new_tensor1+torch.randn_like(tensor1)*0.02, tensor2



def FUSION_correction(adata_list, embeddings, seed):
    
    section_num=0

    embedding_merged=[]

    for idx in range(len(adata_list)):

        section_num+=len(adata_list[idx])

        embeddings_list=[embeddings[section_num-len(adata_list[idx])]]
        
        for i in range(section_num-len(adata_list[idx]), section_num-1):

            transformed_embedding_res=WGAN([embeddings_list[0],embeddings[i+1]], seed)

            output_embed=embeddings[i+1]+transformed_embedding_res.cpu()
            
            output_embed, harbor_embeddings=embedding_matching(output_embed, embeddings_list[0], seed)
            
            embeddings_list.append(output_embed)

        embedding_merged.append(torch.concat(embeddings_list).detach().cpu().numpy())
        
    return embedding_merged
