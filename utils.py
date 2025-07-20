import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import NearestNeighbors
    
def section_alignment(out, method='Distance'):
    
    if method =='Distance':
        
        def domian_order(slide_, habor_do):

            #slide_= out[0][(out[0]['slide']==0)]

            domain_idx=np.array(list(set(slide_['domain'].values)))

            harbor_idx=np.where(domain_idx==habor_do)[0]

            clusters = [
            slide_[slide_['domain']==i][['x','y']].values for i in domain_idx
            ]

            # Function to compute the minimum distance between two clusters
            def compute_min_distance(cluster1, cluster2):
                distances = np.linalg.norm(cluster1[:, np.newaxis, :] - cluster2[np.newaxis, :, :], axis=2)
                return np.min(distances)

            # Function to compute the distance matrix for multiple clusters
            def compute_distance_matrix(clusters):
                num_clusters = len(clusters)
                distance_matrix = np.zeros((num_clusters, num_clusters))

                for i in range(num_clusters):
                    for j in range(i + 1, num_clusters):  # Only compute upper triangular part
                        min_distance = compute_min_distance(clusters[i], clusters[j])
                        distance_matrix[i, j] = min_distance
                        distance_matrix[j, i] = min_distance  # Symmetric matrix

                return distance_matrix

            # Compute the distance matrix
            distance_matrix = compute_distance_matrix(clusters)

            distance_dict=dict(zip(domain_idx,distance_matrix[harbor_idx[0]]))

            output_dict=dict(zip(distance_dict.keys(), np.argsort(np.argsort(list(distance_dict.values()))))) 

            return output_dict


        def harbor_domain(out, d_idx):

            n_do=list(set(out[d_idx]['domain']))

            res={}
            for d in n_do:

                corr_list=out[d_idx][(out[d_idx]['domain']==d)&(out[d_idx]['slide']==0)][['x','y']]

                from scipy.spatial.distance import euclidean
                from scipy.optimize import minimize

                def geometric_median(points):
                    def objective(x):
                        return sum(euclidean(x, p) for p in points)
                    initial_guess = np.mean(points, axis=0)
                    result = minimize(objective, initial_guess, method='COBYLA')
                    return result.x

                res[d]=np.median(corr_list.values, axis=0)

            geomedian=np.nan_to_num(np.vstack(list(res.values())), nan=0)
            res=np.argmax(geomedian,axis=0)

            return n_do[res[0]]

        def slice_matching(out):

            for idx, section in enumerate(out):

                slide_=section[(section['slide']==0)]

                harbor_idx= harbor_domain(out, idx)

                label_mapping=domian_order(slide_, int(harbor_idx))

                section['domain'] = section['domain'].replace(label_mapping)

            return out
        
    if method == 'CT':
    
        def domian_order(slide_, int_slide):

            import numpy as np

            def find_top_columns(matrix, k=10, metric='mean'):

                # 计算每列的评估指标
                if metric == 'mean':
                    col_values = np.mean(matrix, axis=0)
                elif metric == 'sum':
                    col_values = np.sum(matrix, axis=0)
                elif metric == 'max':
                    col_values = np.max(matrix, axis=0)
                elif metric == 'min':
                    col_values = np.min(matrix, axis=0)
                else:
                    raise ValueError("metric必须是'mean','sum','max'或'min'")

                # 按指标值降序排列，获取列索引
                sorted_indices = np.argsort(col_values)[::-1]  # 从大到小排序

                # 返回前k个列的索引及排序后的指标值
                return sorted_indices[:k], col_values[sorted_indices]

            #slide_= out[0][(out[0]['slide']==0)]
            i_domian_idx=np.array(list(set(int_slide['domain'].values)))

            cutoff=find_top_columns(int_slide.iloc[:,3:-1].values)[0] 


            i_clusters = [
            int_slide[int_slide['domain']==i].iloc[:,cutoff+3].values for idx, i in enumerate(i_domian_idx)
            ]

            domain_idx=np.array(list(set(slide_['domain'].values))) 

            clusters = [
            slide_[slide_['domain']==i].iloc[:,cutoff+3].values for idx, i in enumerate(domain_idx)
            ]


            def compute_min_distance(X1, X2):
                mu1 = np.median(X1, axis=0)  # (D,)
                mu2 = np.median(X2, axis=0)  # (D,)
                return np.linalg.norm(mu1 - mu2)  # 标量值

            # Function to compute the distance matrix for multiple clusters
            def compute_distance_matrix(i_clusters, clusters):
                num_row = len(i_clusters)

                num_col = len(clusters)

                distance_matrix = np.zeros((num_row, num_col))

                for i in range(num_row):
                    for j in range(num_col):  # Only compute upper triangular part
                        min_distance = compute_min_distance(i_clusters[i], clusters[j])
                        distance_matrix[i, j] = min_distance

                return distance_matrix


            def allocate_min_indices(arr):

                n_rows, n_cols = arr.shape
                allocated = set()              # 记录已占用的行索引
                result = np.zeros(n_cols, dtype=int)

                # 生成每列的排序索引（值从小到大）
                sorted_indices = np.argsort(arr, axis=0, kind='stable')  # 稳定排序确保同值时按原始顺序

                for col in range(n_cols):
                    # 遍历该列的候选行索引（从最小值开始）
                    for row in sorted_indices[:, col]:
                        if row not in allocated:
                            result[col] = row
                            allocated.add(row)
                            break
                    else:
                        raise ValueError(f"列 {col} 无可用行")
                return result

            # Compute the distance matrix
            distance_matrix = compute_distance_matrix(i_clusters, clusters)

            refined_order=allocate_min_indices(distance_matrix)

            return dict(zip(list(domain_idx), refined_order))

        def slice_matching(out):

            int_slide=out[0][out[0]['slide']==0]

            for idx, section in enumerate(out[1:]):

                slide_=section[(section['slide']==0)]

                label_mapping=domian_order(slide_, int_slide)

                section['domain'] = section['domain'].replace(label_mapping)

            return out
        
    return slice_matching(out)



def domain_plot(selected_pixel_class, pos, color_dict, size, figsize):
    

    coordinates = pos
    # [(pixel_class==9)|(pixel_class==8)|(pixel_class==7)|(pixel_class==5)|(pixel_class==1)|(pixel_class==2)|(pixel_class==3)]

    classes = selected_pixel_class
    # [(pixel_class==9)|(pixel_class==8)|(pixel_class==7)|(pixel_class==5)|(pixel_class==1)|(pixel_class==2)|(pixel_class==3)]

    all_class=list(set(selected_pixel_class))
    all_class.sort()
    
    
    color_mapping=dict(zip(all_class, [i for i in all_class]))


    mapped_class=[]

    for i in classes:

        mapped_class.append(int(color_mapping[i]))

    # Create a DataFrame
    data = {'X': coordinates[:, 0], 'Y': coordinates[:, 1], 'Class': mapped_class}
    df = pd.DataFrame(data)
   
    
    color_order=[i for i in list(set(classes))]

    colors = [color_dict[i] for i in color_order]
    
    cmap = ['#%02x%02x%02x' % rgb for rgb in colors]

    fig,ax=plt.subplots(figsize=figsize)
    ax.axis('off')
    # Set scatter style and edge properties
    scatter_kws = {'s': size, 'marker': 'o', 'edgecolor': None, 'linewidths': 0}

    # Plot using Seaborn

    scatterplot=sns.scatterplot(data=df, x='X', y='Y', hue='Class',palette=cmap,**scatter_kws)
    #else:
     #   scatterplot=sns.scatterplot(data=df, x='X', y='Y', hue='Class',**scatter_kws)

    legend = scatterplot.legend(frameon=False)
    legend.set_title('Topic')

    
    plt.legend(bbox_to_anchor=(1.15,1), loc='upper right')
        

def compute_lisi(embedding, labels, k=30):
    """
    Compute Local Inverse Simpson's Index (LISI) for given labels.
    
    Parameters:
        embedding (np.ndarray): The low-dimensional embedding of the data (n_samples, n_features).
        labels (np.ndarray): The labels to evaluate (e.g., dataset labels for iLISI or cluster labels for cLISI).
        k (int): Number of neighbors for kNN.
    
    Returns:
        np.ndarray: LISI score for each sample.
    """
    # Fit kNN on the embedding
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    
    # Compute LISI for each point
    lisi_scores = []
    for i, neighbors in enumerate(indices):
        neighbor_labels = labels[neighbors]
        label_counts = np.unique(neighbor_labels, return_counts=True)[1]
        probabilities = label_counts / label_counts.sum()
        simpson_index = np.sum(probabilities ** 2)
        inverse_simpson = 1 / simpson_index
        lisi_scores.append(inverse_simpson)
    
    return np.array(lisi_scores)
