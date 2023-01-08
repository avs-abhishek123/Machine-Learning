# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based unsupervised learning algorithm. It computes nearest neighbor graphs to find arbitrary-shaped clusters and outliers. Whereas the K-means clustering generates spherical-shaped clusters.

DBSCAN does not require K clusters initially. Instead, it requires two parameters: eps and minPts.

- eps: it is the radius of specific neighborhoods. If the distance between two points is less than or equal to esp, it will be considered its neighbors.
- minPts: minimum number of data points in a given neighborhood to form the clusters.

DBSCAN uses these two parameters to define a core point, border point, or outlier.
![dbscan](https://www.kdnuggets.com/wp-content/uploads/awan_implementing_dbscan_python_2.jpg)