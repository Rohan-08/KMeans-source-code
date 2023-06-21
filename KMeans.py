# -*- coding: utf-8 -*-
"""
@author: Rohan Chhabra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import warnings
warnings.filterwarnings("ignore")

num_cols= ["f"+str(i) for i in range(1,301)]
df = pd.read_csv('dataset', sep=" ", header=None, names=["class"]+num_cols)
df.head()

X= df.drop(['class'], axis=1)
X=X.values

# Setting the random seed
np.random.seed(100)

# Check if the data is already standardized, should be as sklearn is not allowed

def k_means(X, num_clusters):
    
    # defining the centroids randomly
    centroids= np.random.rand(num_clusters, X.shape[1])
    
    # variable to input the distance of each vector with each centroid
    distance_calc= np.zeros((X.shape[0], num_clusters))
    
    # Variable for getting the nearest centroid for each point
    best_centroid=np.zeros(X.shape[0])
    
    # Getting new mean values for centroids. Initially assigning it same as centroids
    centroids_upd=centroids
    
    # Getting the error term. distance between the previous centroid position and current.
    # Initialising it as random in the begining. Cant set to 0 as while loop will not execute then
    error=np.random.rand(num_clusters, X.shape[1])
    
    # Taking mean of error terms. When error becomes 0, the mean also becomes 0
    while np.mean(error)>0:
        
        centroids= centroids_upd
        
        # Calculating the distance of each vector with centroid and appending it in the nd-array created above
        for i in range(num_clusters):
             distance_calc[:, i] = np.linalg.norm(X - centroids[i], axis=1)
    
        # Getting the centroid with minimum distance
        for i in range(len(X)):
            best_centroid[i] = np.argmin(distance_calc[i])
        
        # Merging the data points with same centroid and calculating the mean value
        # If any centroid gets un-allocated, keep the same position as before
        for k in range(num_clusters):
            if X[best_centroid.astype(int)== k].shape[0]==0:
                centroids_upd[k]=centroids[k]
            else:
                centroids_upd[k] = np.mean(X[best_centroid.astype(int)== k], axis = 0)   
            
        error= np.linalg.norm(centroids_upd- centroids, axis=1)
        
    return X, best_centroid, centroids_upd

    
    
def k_means_pp(X, num_clusters):
    
    # Setting random seed
    # np.random.seed(103)
    
    # Creating centroids location array
    centroids=np.zeros((num_clusters, X.shape[1]))
    
    # Assigning any one data point to be the centroid
    init_centroid= X[np.random.randint(len(X))]
    
    for k in range(num_clusters):
        
        centroids[k]= init_centroid
        
        # Calculating the distance of that point to all the other points
        distance_init= np.zeros((num_clusters, len(X)))
        
        # Variable to get the minimum distance
        min_dist=np.zeros((1, len(X)))

        for a in range(k+1):
            for j in range(len(X)):
                distance_init[a, j]= np.square(np.linalg.norm(centroids[a] - X[j]))

        min_dist= np.amin(distance_init[:k+1], axis=0)
        
        prob_init= np.zeros((num_clusters, len(X)))
        # Calculating probability of each point to become a cluster 
        
        for m in range(len(X)):
            prob_init[k,m] = (min_dist[m])/ np.sum(min_dist)
        
        init_centroid = X[np.argmax(prob_init[k])]
        
        if k==num_clusters-1:
            centroids[k]=init_centroid
    
    # variable to input the distance of each vector with each centroid
    distance_calc= np.zeros((X.shape[0], num_clusters))
    
    # Variable for getting the nearest centroid for each point
    best_centroid=np.zeros(X.shape[0])
    
    # Getting new mean values for centroids. Initially assigning it same as centroids
    centroids_upd=centroids
    
    # Getting the error term. distance between the previous centroid position and current.
    # Initialising it as random in the begining. Cant set to 0 as while loop will not execute then
    error=np.random.rand(num_clusters, X.shape[1])
    
    # Taking mean of error terms. When error becomes 0, the mean also becomes 0
    while np.mean(error)>0:
        
        centroids= centroids_upd
        
        # Calculating the distance of each vector with centroid and appending it in the nd-array created above
        for i in range(num_clusters):
             distance_calc[:, i] = np.linalg.norm(X - centroids[i], axis=1)
    
        # Getting the centroid with minimum distance
        for i in range(len(X)):
            best_centroid[i] = np.argmin(distance_calc[i])
        
        # Merging the data points with same centroid and calculating the mean value
        # If any centroid gets un-allocated, keep the same position as before
        for k in range(num_clusters):
            if X[best_centroid.astype(int)== k].shape[0]==0:
                centroids_upd[k,:]=centroids[k]
            else:
                centroids_upd[k,:] = np.mean(X[best_centroid.astype(int)== k], axis = 0)   
            
        error= np.linalg.norm(centroids_upd- centroids, axis=1)
        
    return X, best_centroid, centroids_upd


def bisecting_kmeans(X,num_clusters):
    
    # Creating another data variable. df will keep on getting updated
    df=X
    
    # Assigning centroids 
    centroids = np.zeros((num_clusters, df.shape[1]))
    
    # Variable to save the distance of data points to its centroid
    centroids_dist = np.zeros((num_clusters+1, df.shape[0]))
    
    # Creating a variable to store the data points of small clusters
    final_dist_cluster=[]
    
    # Creating a variable to store the centroid locations of small clusters
    final_dist_centroid=[]
    
    
    if num_clusters==1:
        final_df, best_centroid, centroids_upd = k_means(df, 1)    
    
    else:
        while len(final_dist_cluster)<num_clusters:
            
            # Running k means for the current datapoint
            df, best_centroid, centroids_upd = k_means(df, 2)
            
            # We will store the data points and the centroid location for both clusters in a list to compare with all others
            
            row_1 = [row.tolist() for row in df[best_centroid.astype(int)==0]]
            row_2 = [row.tolist() for row in df[best_centroid.astype(int)==1]]
            
            final_dist_cluster.append(row_1)
            final_dist_cluster.append(row_2)
            
            # Storing both the clusters and centroids in same order as we will resuse them to calculate the distances
            # centroid_1 = [row.tolist() for row in centroids_upd[0]]
            # centroid_2 = [row.tolist() for row in centroids_upd[1]]
            
            final_dist_centroid.append(centroids_upd[0].tolist())
            final_dist_centroid.append(centroids_upd[1].tolist())
            
            # Checking so that we don't overshoot
            if len(final_dist_cluster)<num_clusters:
                # Calculating the distance for all the clusters and centroids 
                for i in range(len(final_dist_cluster)):
                    n=len(final_dist_cluster[i])
                    for j in range(n):
                        # centroids_dist[i,j] = np.linalg.norm(df[best_centroid.astype(int)==i][j] - centroids_upd[i])
                        centroids_dist[i,j] = np.linalg.norm( np.array(final_dist_cluster[i][j]) - np.array(final_dist_centroid[i]))
                
                # Taking sum of the distances
                sum_dist= np.sum(np.square(centroids_dist), axis=1)
                
                # Finding the cluster with maximum distance
                max_dist_cluster= np.argmax(sum_dist)
                
                # Making the biggest sum distance as the new dataframe to perform clustering
                df= pd.DataFrame(final_dist_cluster[max_dist_cluster]).values
                
                # Removing the largest distance data cluster from the lists
                final_dist_cluster.pop(max_dist_cluster)
                final_dist_centroid.pop(max_dist_cluster)

    
    if num_clusters>1:
        # Creating the final_dist_cluster back to a nd-array to be used in silhouette function
        final_df=np.zeros((X.shape[0], X.shape[1]))
        
        m=0
        for i in range(len(final_dist_cluster)):
            for j in range(len(final_dist_cluster[i])):
                final_df[m] = np.array(final_dist_cluster[i][j])
                m+=1
                
        # Creating "centroids_upd" from "final_dist_centroid" into a ndarray
        centroids_upd=np.zeros((num_clusters, X.shape[1]))
        
        for i in range(len(final_dist_centroid)):
            if len(final_dist_centroid[i])==0:
                continue
            else:
                centroids_upd[i] = np.array(final_dist_centroid[i])
        
        # Creating "best_centroid" ndarray which will have the centroid names
        best_centroid=np.zeros(X.shape[0])
    
        a=0
        for i in range(len(final_dist_cluster)):
            for j in range(len(final_dist_cluster[i])):
               best_centroid[a]=i
               a+=1

    return final_df, best_centroid, centroids_upd


def silhouette(df, num_clusters, best_centroid, centroids_upd):
    
    # df=final_check
    # centroids_upd= centroids_upd_check
    # best_centroid= best_centroid_check

    internal_distance= np.zeros((df.shape[0], df.shape[0]))
    # Calculating the distance of each point with all the other points. 
    # will filter out according to the clusters afterwards
    for i in range(len(df)):
        for j in range(len(df)):
            if i<j:
                internal_distance[i,j]= np.linalg.norm(df[i]-df[j])
                internal_distance[j,i]=internal_distance[i,j]
    
    # print("internal_distance.shape", internal_distance.shape)
    
    # Getting the closest cluster for each set of cluster
    cluster_distance= np.zeros((num_clusters, num_clusters))
    
    for i in range(len(centroids_upd)):
        for j in range(len(centroids_upd)):
            # setting a huge value so that i==j will be 0 and it doesnt get picekd up in argmin
            if i==j:
                cluster_distance[i,j]=10000000
            else:
                cluster_distance[i,j]= np.linalg.norm(centroids_upd[i]-centroids_upd[j])        
    
    nearest_cluster=np.zeros(num_clusters)
    
    for i in range(num_clusters):
        nearest_cluster[i] = np.argmin(cluster_distance[i])    
    
    # Variable to store the mean silhouette index
    mean_si = np.zeros(num_clusters)
    
    for k in range(num_clusters):
        n=len(internal_distance[best_centroid==k])
        
        if n==0:
            mean_si[k]=0
            continue
        else:
            s_i=np.zeros(n)
            for i in range(n):
                a_i = np.mean(internal_distance[best_centroid==k][i][best_centroid==k])
                
                b_i = np.mean(internal_distance[best_centroid==k][i][best_centroid==nearest_cluster[k].astype(int)])
                
                if np.isnan(b_i)==False:
                    s_i[i] = (b_i - a_i)/ np.max([a_i, b_i])
                else:
                    s_i[i]=0
                    
            mean_si[k]= np.mean(s_i)
        
    return np.mean(mean_si)


# K-Means
km=[0.0 for _ in range(9)]

for i in range(1,10):
    
    final_df_km, best_centroid_km, centroids_upd_km= k_means(X, i)
    s_i_km = silhouette(final_df_km, i, best_centroid_km, centroids_upd_km)
    km[i-1]=s_i_km
    print(f"The silhouette co-efficient for k={i} is {s_i_km}")

idx_km=[]
val_km=[]
for idx_, val_ in enumerate(km):
    idx_km.append(idx_+1)
    val_km.append(val_)

plt.plot(idx_km, val_km)
plt.title('KMeans - Silhouette Coefficient with number of clusters')
plt.ylabel("Silhoutte Coefficient")
plt.xlabel("Number of clusters")
plt.show()

# K-Means++
kmp=[0.0 for _ in range(9)]

for i in range(1,10):
    
    final_df_kmp, best_centroid_kmp, centroids_upd_kmp= k_means_pp(X, i)
    s_i_kmp = silhouette(final_df_kmp, i, best_centroid_kmp, centroids_upd_kmp)
    kmp[i-1]=s_i_kmp
    print(f"The silhouette co-efficient for k={i} is {s_i_kmp}")

idx_kmp=[]
val_kmp=[]
for idx_, val_ in enumerate(kmp):
    idx_kmp.append(idx_+1)
    val_kmp.append(val_)

plt.plot(idx_kmp, val_kmp)
plt.title('KMeans++ - Silhouette Coefficient with number of clusters')
plt.ylabel("Silhoutte Coefficient")
plt.xlabel("Number of clusters")
plt.show()


# Bisecting k-means
kmb=[0.0 for _ in range(9)]

for i in range(1,10):
    
    final_df_kmb, best_centroid_kmb, centroids_upd_kmb= bisecting_kmeans(X, i)
    s_i_kmb = silhouette(final_df_kmb, i, best_centroid_kmb, centroids_upd_kmb)
    kmb[i-1]=s_i_kmb
    print(f"The silhouette co-efficient for k={i} is {s_i_kmb}")

idx_kmb=[]
val_kmb=[]
for idx_, val_ in enumerate(kmb):
    idx_kmb.append(idx_+1)
    val_kmb.append(val_)

plt.plot(idx_kmb, val_kmb)
plt.title('Bisecting KMeans - Silhouette Coefficient with number of clusters')
plt.ylabel("Silhoutte Coefficient")
plt.xlabel("Number of clusters")
plt.show()

