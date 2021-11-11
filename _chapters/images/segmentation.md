---
title: Segmentation
keywords: (segmentation, clustering, gestalt)
order: 11 # Lecture number 13 for 2021
---
<h1>
CS131 Lecture 13 
</h1>
Brad Nikkel, Constance Horng, Chris Tan, Yutian Cai, Jialin Zhuo 

# 1 Introduction to Segmentation and Clustering
**1.1 Overview of Segmentation**

In computer vision and image processing, segmentation is the process of partitioning an image into groups of pixels that are similar and belong together. As human, we intuitively group and segment pixels together to locate meaningful objects when we look at an image. Segmentation allows a computer to accomplish the same thing by identifying and labeling pixels that go together. Figure 1 shows an example of image segmentation. In this example, pixels of the image are grouped into either tiger, grass, sand or river.

<figure>
  <center><img src="https://drive.google.com/uc?id=1FNvC7xvcnntOmCYJbs1mzr89Qu4O9Mk2" width="800"></center>
  <figcaption align="center">Figure 1: Example of image segmentation (src: Lecture 13.1)</figcaption>
</figure>

**1.2 Goal of Segmentation**
- Separate image into coherent objects
>This is the most common usage of image segmentation. Segmentation allows us to identify pixels and group those that are alike. The result gives a set of segments each representing a coherent object. This is useful in a number of applications, like removing the background from an image, or with some further image processing, detecting objects in the image. Figure 2 shows how human intuitively use segmentation to identify objects in an image.

<figure>
  <center><img src="https://drive.google.com/uc?id=1Q9F3o-3zdxj8AfPg-mc1PE9_O9n0P-vR" width="500"></center>
  <figcaption align="center">Figure 2: Human segmentation to detect objects (src: Lecture 13.1)</figcaption>
</figure>

- Use "superpixels" to achieve higher efficiency for further processing
>Another goal of segmentation is to group neighboring pixels into patches, or "superpixels", so that for further processing, we don't need to go over each pixel one by one; instead, we can treat all pixels in a patch the same. This technique is a popular area of research and can substantially speed up subsequent computations. Figure 3 gives an example of grouping pixels into "superpixels".

<figure>
  <center><img src="https://drive.google.com/uc?id=1yJWfDm4-BNZv-DX3jb6idJcXIvHCbadk" width="500"></center>
  <figcaption align="center">Figure 3: "Superpixels" for more efficient processing (src: Lecture 13.1)</figcaption>
</figure>

**1.3 Types of Segmentation**

<figure>
  <center><img src="https://drive.google.com/uc?id=17aLjFo7nOMKIt3BZ1PB3ayg63IU-Gjgw" width="700"></center>
  <figcaption align="center">Figure 4: Different types of segmentation (src: Lecture 13.1)</figcaption>
</figure>

Depending on the specific applications, different types of segmentation are used to get different desired results. Figure 4 shows an example of oversegmentation and undersegmentation. For oversegmentation, the number of patches is large and the number of pixels in a patch is small. The result renders a very fine-grained representation of the origional image. For undersegmentation, we define very few patches and the result will have much less details. It might be important to seek a balance between the two types so as to recognize more objects, and that's why a careful selection of parameters in a segmentation algorithm is essential. It is also worth mentioning that researchers have tried combining different segmentation results together to extract the most from an image.

**1.4 Clustering**

Clustering is one solution to the segmentation problem. The goal of clustering is to group together similar data points and represent them with a single token. In the case of segmentation, the data points are image pixels. There are two types of clustering:

- Bottom up clustering
>Starts from the smallest entity and iteratively join similar data points together to form bigger clusters. In bottom up clustering, tokens belong together because they are locally coherent.

- Top down clustering
>Starts from the top and keeps on splitting the collection. In top down clustering, tokens belong together because they lie on the same visual entity.

In order to perform clustering, we need to first tackle the following two issues:

1. How to determine if two points/patches are similar?

2. How to compute an overall grouping from pairwise similarities?

To answer the first question, there is not one metric that gives you the best estimation of similar pixels. Some metrics perform better for certain objects and some perform better for others. Regardless, some of the common metrics include proximity, pixel values, symmetry and common fate. In Figure 5, the books are stacked close to each other, so proximity can be a good metric here to group similar pixels; symmetry can be used to identify symmetrical objects like vases; common fate works well with objects that move together; pixel value is probably the most intuitive metric, and it works best when objects are described by vastly different RGB values.

<figure>
  <center><img src="https://drive.google.com/uc?id=1Tnuw3CGzjS-UWFUgYDhpZF1iBm0mK65L" width="400"></center>
  <figcaption align="center">Figure 5: Different metrics used in clustering (src: Lecture 13.1)</figcaption>
</figure>

For the second question, different clustering algorithms have different ways of grouping similar pixels or patches. Those algorithms include Agglomerative Clustering, K-Means Clustering and Mean-Shift Clustering, which we will discuss in detail in subsequent sections.

# 2 Gestalt Theory
Gestalt Theory comes from the field of psychology, in which psychologists identified a series of factors that predispose people to seeing certain patterns and groupings in objects. These groupings are central to visual perception, as we learn to recognize objects by these factors.

At the core of Gestalt Theory is the belief that "the whole is greater than the sum of its parts". Rather than examining the individual components of an object, gestalt theory attempts to look at the object as a whole. This whole object may have properties resulting from the relationships among the elements that comprise it.
<figure>
  <center><img src="https://drive.google.com/uc?id=1F-pgIvPceZ_0R_3EWlBOS5ZTcsNVuoBH" width="400"></center>
  <figcaption align="center">Figure 6: Examples of objects that create visual perception of groups (src: Lecture 13.2)</figcaption>
</figure>

**2.1 Gestalt Factors**
- Not grouped
- Proximity: objects that are close together are seen as a group
- Similarity: objects are perceived to be grouped if they share certain properties
- Common fate: objects that move in the same direction are perceived as being in the same group
- Common region: we perceive objects enclosed within the same region to be grouped together
- Parallelism: objects that are parallel are perceived to be part of the same group
- Symmetry: objects that are symmetrical are perceived to be part of the same whole
- Continuity: we perceive aligned objects to be a whole
- Closure: we perceive objects as whole even if there are gaps in them

<figure>
  <center><img src="https://drive.google.com/uc?id=1DJ9QMu1-aAfy2jO08iry2TQ5v0t_Vu3-" width="400"></center>
  <figcaption align="center">Figure 7: Examples of Gestalt factors creating perception of groups (src: Lecture 13.2)</figcaption>
</figure>

However, while intuitive, these factors can be hard to translate into algorithms.

<figure>
  <center>
  <img src="https://drive.google.com/uc?id=1AbAKMRkNOENVtgU_G3BfsJRMCW3Nozcd" width="200">
  <img src="https://drive.google.com/uc?id=1nG48P3GOmKSWKg0xhPiHr2e275tBBztl" width="200">
  </center>
  <figcaption align="center">Figure 8: Example of continuity by occluded stripes (src: Lecture 13.2)</figcaption>
</figure>


# 3 Agglomerative Clustering
Clustering is an unsupervised learning method where we want to group items $x_1, ..., x_n \in \mathbb{R}^D$ into clusters. Thus, we need a pairwise distance/similarity function to measure the distance between two tokens.

**3.1 Distance Measure**

There are two main measures of distance when using feature vectors to represent data: euclidean distance and cosine similarity. We define $x$ and $x'$ as two objects from the universe of possible objects. Then, the distance (similarity) between $x$ and $x'$ can be denoted as $sim(x, x')$.

For euclidean distance, we have:

$$dist(x, x') = \sqrt{\Sigma(x_i - x'_i)^2}$$  

For cosine similarity, we have:

<center>
$sim(x, x') $  

$= \cos(\theta)$

$= \frac{x^Tx'}{||x|| \cdot ||x'||}$

$= \frac{x^Tx'}{\sqrt{x^Tx}\sqrt{x'^Tx'}}$ 
</center>

**3.2 Desirable Properties of Clustering Algorithms**

When building a clustering algorithm, it is important to think about what features we want to have and what we want to achieve. Below are some desirable properties of clustering algorithms:

1. Scalability (in terms of both time and space)
2. Ability to deal with different data types
3. Minimal requirements for domain knowledge to determine input parameters
4. Optional: interpretability and usability, as well as incorporation of user-specified constraints

**3.3 Algorithm**

Here is the general algorithm for agglomerative clustering:
1. Define each item $x_1,...,x_n$ as its own cluster $C_1,...,C_n$.
2. Find the most similar pair of clusters $C_i$ and $C_j$. Merge $C_i$ and $C_j$ into a parent cluster.
4. Repeat steps 2 and 3 until we only have 1 cluster left.

**3.4 Questions**
>How do we define cluster similarity?

There are several ways we can compute the distance between a cluster and a data point, or between two clusters. We can compute the average distance between points, choose the maximum or mimimum distance, or compute the distance between means or medoids. 

>How many clusters do we want?

Since agglomerative clustering eventually creates a dendogram (a tree), we can use a threshold based on the maximum number of clusters or the distance between merges to decide the number of clusters we want. Alternatively, we can keep on clustering until we have a certain number of clusters, at which point we stop.


**3.5 Different measures of nearest clusters**

There are three main ways to measure distances between clusters: single link, complete link, and average link.
- Single link (single-linkage)

  $$d(C_i, C_j) = min_{x \in C_i, x' \in C_j} d(x, x')$$

  This method is equivalent to the minimum spanning tree algorithm. Once the distance between clusters is above the threshold that we set, we can stop clustering. The single link algorithm tends to produce long, skinny clusters.
<figure>
  <center><img src="https://drive.google.com/uc?id=1RaSf12j4HNhost8g0EYvAlKyaBEW2-CD" width="400"></center>
  <figcaption align="center">Figure 9: Single link measurement of nearest clusters (src: Lecture 13.3)</figcaption>
</figure>

- Complete link (complete-linkage)

  $$d(C_i, C_j) = max_{x \in C_i, x' \in C_j} d(x, x')$$

  Unlike single link, complete link utilizes the maximum distance between points in two clusters. Clusters resulting from this algorithm tend to be compact and roughly equal in diameter.
<figure>
  <center><img src="https://drive.google.com/uc?id=15mOHYDvWXe3vDtIDnAJAuxR5CZ6OzZv-" width="400"></center>
  <figcaption align="center">Figure 10: Complete link measurement of nearest clusters (src: Lecture 13.3)</figcaption>
</figure>

- Average link 

  $$d(C_i, C_j) = \frac{\Sigma_{x \in C_i, x' \in C_j}d(x, x')}{|C_i|\cdot|C_j|}$$

  Average link utilizes the average distance between items, resulting in an algorithm that is somewhere between single link and complete link. Average link is more robust against noise because for single and complete link, outliers can disturb the computed distances between clusters.
<figure>
  <center><img src="https://drive.google.com/uc?id=1I_ImgW2NII7JjvuSCsScOcqVTAYMKJxi" width="400"></center>
  <figcaption align="center">Figure 11: Average link measurement of nearest clusters (src: Lecture 13.3)</figcaption>
</figure>

**3.6 Conclusions**

Benefits to agglomerative clustering:
1. Simple to implement
2. Has many applications
3. Clusters can have adaptive shapes
4. Provides a hierarchy of clusters (which are closer than others)
5. No need to specify the number of clusters in advance

Drawbacks of agglomerative clustering:
1. May result in imbalanced clusters
2. Need to specify the threshold or number of clusters
3. Does not scale well, runtime is $O(n^3)$
4. Can get stuck at a local minima


# 4 Graph-based Segmentation
**4.1 Overview**
<figure>
  <center><img src="https://drive.google.com/uc?id=1oDd1RziSN2E1tpe-r0ckZgJ72l9flXSE" width="400"></center>
  <figcaption align="center">Figure 12: Example of original photo corresponding to segmented photo below (src: Lecture 13.4)</figcaption>
</figure>

<figure>
  <center><img src="https://drive.google.com/uc?id=1uZL0rKXcYuDJZJqMGkP-wYOP0QVxQWUX" width="400"></center>
  <figcaption align="center">Figure 13: Segmented image of original photo above (src: Lecture 13.4)</figcaption>
</figure>

Graph-based segmentaton is similar to aglomerative clustering (part 3). It is an image segmentation method introduced in 2004 (prior to many deep learning algorithms) and has remained popular due to its simplicity and ease of implementation.

Graph-based clustering for image clustering was first introduced by Felzenszwalb and Huttenlocher in their paper "Efficient Graph-Based Image Segmentation."

**4.2 Formulating Images as Graphs - Features and Weights**

<figure>
  <center><img src="https://drive.google.com/uc?id=1ZIAg_GBXTInOspZEK7-2Upmi4Ny0UHGH" width="400"></center>
  <figcaption align="center">Figure 14: Example of eight neighboring pixels modeled as a graph (src: Lecture 13.4)</figcaption>
</figure>

The basic concept behind graph-based clustering is that we can model images as graphs by:
1. Treating a pixel in an image as connected to its eight neighboring pixels, where weights represent differing intensities.
2. Set weights (which represent intensities) as the L2 (Euclidean) distance in the feature space.
3. Project every pixel into the feature space defined by vertical and horizontal position and color values (x, y, r, g, b).

**4.3 Problem Formulation**

<figure>
  <center><img src="https://drive.google.com/uc?id=1spAqSwkcNmivCkaRe3J-TxYr1q9Lrm9x" width="400"></center>
  <figcaption align="center">Figure 15: Example of three clusters identified in image graph (src: Lecture 13.4)</figcaption>
</figure>

1. We have Graph $G = (V, E)$ such that $V$ is set of nodes (on for each pixel) and $E$ is the set of undirected edges between all pairs of pixels.
2. We have a set $W$ where $w(v_i, v_j)$ represents the intensity weight/distance of the edge connecting nodes (pixels) $v_i$ and $v_j$. 
3. Thet set $S$ is a segmentation of graph $G$ where $G' = (V, E')$ and $E' \subset E$.  
4. Segmentation set $S$ divides graph $G$ into graph $G'$ such that graph $G'$ contains the set of distinct clusters $C$.

**4.4 Predicate for Segmentation**

<figure>
  <center><img src="https://drive.google.com/uc?id=1X8Q2Qy5i0G6xStxRI3OIbMKOpRyJJX-P" width="400"></center>
  <figcaption align="center">Figure 16: Example difference and internal difference (src: Lecture 13.4)</figcaption>
</figure>

- First we define a predicate $D$, which determines if a boundary for segmentation exists. This decides whether two candidate clusters are actually seperate or the same (and thus whether or not we should merge the two candidates). We use the below piecewise function to determine whether or not to merge two candidate clusters:

$$ Merge(C_1, C_2)=   \left\{
\begin{array}{ll}
      True & dif(C_1, C_2) < in(C_1, C_2) \\
      False & otherwise \\
\end{array} 
\right.  $$

- Above, $dif(C_1, C_2)$ is the difference between any two clusters $C_1$ and $C_2$ (i.e. how different is cluster $C_1$ from $C_2$). On the other hand, $in(C_1, C_2)$ is the difference between pixels within a candidate cluster $C_1$ and the pixels within a candidate cluster $C_2$. 

$$dif(C_1,C_2) = \min_{v_i\in C_1, v_j \in C_2, (C_1, C_2) \in E} w(v_i, v_j)$$

- Above, we see that the minimum weight edge connnecting a node $v_i$ in cluster $C_1$ with node $v_j$ in cluster $C_2$ is the difference between components $C_1$ and $C_2$.

$$in(C_1, C_2)= \min_{C \in \{C_1, C-2\}} \left[ \max_{v_i,v_j \in C} \left[ w(v_i,v_j) + \frac{k}{|C|} \right] \right]$$ 

- Above, we see that the internal difference, $in(C_1, C_2)$, takes the maximum weight edge distance connecting any two iternal points within each cluster $C_1$ and $C_2$. This corresponds to the most different pair of points within each respective cluster candidate $C_1$ and $C_2$. We then we take the minimum of these two internal maximums.

- The intuition behind this is that if the external difference between clusters $C_1$ and $C_2$ turns out to be smaller than the internal distance of a single cluster, then we should merge clusters $C_1$ and $C_2$.

- The fraction $\frac{k}{\|C\|}$ is the threshold difference between components and the internal nodes in a component. If we set $k$ to be large, our threshold will favor large (and fewer) objects. A smaller $k$ wil make the threshold favor smaller (and more) clusters.

<figure>
  <center><img src="https://drive.google.com/uc?id=1dfvGtPwwxfTffbrMFKmP6IpgYVpuEAke" width="400"></center>
  <figcaption align="center">Figure 17: Example of effects small and large k (src: Lecture 13.4)</figcaption>
</figure>

**4.5 Performance**
1. In practice, edges are chosen are limited to the top 10 nearest neighbors in feature space. This then allows us use our graph model with a greedy algorithm to achieve time complexity of $O(n \log n)$ where $n$ is the number of pixels in the image.

<figure>
  <center><img src="https://drive.google.com/uc?id=1PkTcHWKplO7hywnrj4A_4onpxv3d6JaV" width="400"></center>
  <figcaption align="center">Figure 18: Examples of graph-clustered-images (src: Lecture 13.4)</figcaption>
</figure>









