# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based unsupervised learning algorithm. It computes nearest neighbor graphs to find arbitrary-shaped clusters and outliers. Whereas the K-means clustering generates spherical-shaped clusters.

DBSCAN does not require K clusters initially. Instead, it requires two parameters: eps and minPts.

- eps: it is the radius of specific neighborhoods. If the distance between two points is less than or equal to esp, it will be considered its neighbors.
- minPts: minimum number of data points in a given neighborhood to form the clusters.

DBSCAN uses these two parameters to define a core point, border point, or outlier.<br>

![dbscan](https://www.kdnuggets.com/wp-content/uploads/awan_implementing_dbscan_python_2.jpg)

How does the DBSCAN clustering algorithm work?

1. Randomly selecting any point p. It is also called core point if there are more data points than  minPts in a neighborhood.
2. It will use eps and minPts to identify all density reachable points.
3. It will create a cluster using eps and minPts if p is a core point.
4. It will move to the next data point if p is a border point. A data point is called a border point if it has fewer points than minPts in the neighborhood.
5. The algorithm will continue until all points are visited.

## Output
![DBSCAN-output](https://github.com/avs-abhishek123/Machine-Learning/blob/main/images/dbscan.png)
## Conclusion

DBSCAN is one of the many algorithms that is used for customer segmentation. You can use K-means or Hierarchical clustering to get even better results. The clustering algorithms are generally used for recommendation engines, market and customer segmentation, social network Analysis, and document analysis.

In this blog, we have learned the basics of the density-based algorithm DBCAN and how we can use it to create customer segmentation using scikit-learn. You can improve the algorithm by finding optimal eps and min_samples using silhouette score and heatmap.