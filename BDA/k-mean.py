from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

#Extracting each value at specific Index from Array of Array
def Extract(lst,index):
    return [item[index] for item in lst]


#load Iris Dataset
data=load_iris()

x=data.data
y=data.target

print(data.target_names)
#Plot Current Dataset
first = plt.figure(1)
plt.xlabel("Sepal Width")
plt.ylabel("Sepal Length")
plt.scatter(x=Extract(x,0),y=Extract(x,1),c=y,cmap="gist_rainbow")
first.show()

#Check K mean for various clusters
wcss=[]
for i  in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#Elbow Chart 
second = plt.figure(2)   
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
second.show()


#From Elbow chart we choose 3 as cluster value bcoz it significe elbow
kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(x)


'''Plot chart for prdicted values and as there are 3 cluster so
we assign each clsuter a color
0,1,2 indicate types of iris flowers ['setosa' 'versicolor' 'virginica'] and red, blue, green colors respectively
'''
third = plt.figure(3)

plt.scatter(x[y_kmeans==0, 0], x[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(x[y_kmeans==1, 0], x[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(x[y_kmeans==2, 0], x[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')

#To Specify centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of Iris')
plt.xlabel("Sepal Width")
plt.ylabel("Sepal Length")
third.show()



