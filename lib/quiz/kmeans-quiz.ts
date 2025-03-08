import type { QuizQuestion } from "../types"

export const beginner: QuizQuestion[] = [
  {
    question: "What type of learning does K-means clustering represent?",
    options: [
      { id: "a", text: "Supervised learning" },
      { id: "b", text: "Unsupervised learning" },
      { id: "c", text: "Reinforcement learning" },
      { id: "d", text: "Semi-supervised learning" },
    ],
    correctAnswer: "b",
    explanation:
      "K-means is an unsupervised learning algorithm because it doesn't require labeled data. It finds patterns and structures in the data without prior knowledge of the correct outputs.",
  },
  {
    question: "What does the 'K' in K-means represent?",
    options: [
      { id: "a", text: "The number of iterations" },
      { id: "b", text: "The number of clusters" },
      { id: "c", text: "The number of features" },
      { id: "d", text: "The kernel function" },
    ],
    correctAnswer: "b",
    explanation:
      "The 'K' in K-means represents the number of clusters that the algorithm will attempt to find in the data. This is a hyperparameter that must be specified before running the algorithm.",
  },
  {
    question: "How are initial centroids typically chosen in the K-means algorithm?",
    options: [
      { id: "a", text: "Always at the center of the dataset" },
      { id: "b", text: "Randomly from the data points" },
      { id: "c", text: "At equal distances from each other" },
      { id: "d", text: "Based on the density of data points" },
    ],
    correctAnswer: "b",
    explanation:
      "In the standard K-means algorithm, initial centroids are typically chosen randomly from the data points. However, there are more sophisticated initialization methods like K-means++.",
  },
  {
    question: "What metric does K-means typically use to assign points to clusters?",
    options: [
      { id: "a", text: "Manhattan distance" },
      { id: "b", text: "Euclidean distance" },
      { id: "c", text: "Cosine similarity" },
      { id: "d", text: "Jaccard index" },
    ],
    correctAnswer: "b",
    explanation:
      "K-means typically uses Euclidean distance (straight-line distance) to measure the similarity between data points and cluster centroids.",
  },
  {
    question: "What is the objective function that K-means tries to minimize?",
    options: [
      { id: "a", text: "The number of iterations" },
      { id: "b", text: "The number of clusters" },
      { id: "c", text: "The sum of squared distances between points and their assigned centroids" },
      { id: "d", text: "The maximum distance between any two points in a cluster" },
    ],
    correctAnswer: "c",
    explanation:
      "K-means aims to minimize the sum of squared distances (inertia) between each data point and its assigned cluster centroid, which is also known as the within-cluster sum of squares.",
  },
]

export const intermediate: QuizQuestion[] = [
  {
    question: "What is the 'elbow method' used for in K-means clustering?",
    options: [
      { id: "a", text: "To initialize the centroids" },
      { id: "b", text: "To determine the optimal number of clusters" },
      { id: "c", text: "To measure the quality of clustering" },
      { id: "d", text: "To speed up the convergence of the algorithm" },
    ],
    correctAnswer: "b",
    explanation:
      "The elbow method is used to determine the optimal number of clusters (K) by plotting the sum of squared distances against different values of K and looking for the 'elbow' point where the rate of decrease sharply changes.",
  },
  {
    question: "What is K-means++?",
    options: [
      { id: "a", text: "A faster version of K-means" },
      { id: "b", text: "A method for initializing centroids that improves convergence" },
      { id: "c", text: "A variant of K-means that can handle categorical data" },
      { id: "d", text: "A parallel implementation of K-means" },
    ],
    correctAnswer: "b",
    explanation:
      "K-means++ is an initialization method that selects initial centroids in a way that they are distant from each other, which often leads to better convergence and results compared to random initialization.",
  },
  {
    question: "What is the silhouette coefficient used for in clustering?",
    options: [
      { id: "a", text: "To initialize centroids" },
      { id: "b", text: "To determine the optimal number of clusters" },
      { id: "c", text: "To measure how similar an object is to its own cluster compared to other clusters" },
      { id: "d", text: "To speed up the K-means algorithm" },
    ],
    correctAnswer: "c",
    explanation:
      "The silhouette coefficient measures how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1, where a high value indicates the object is well matched to its cluster and poorly matched to neighboring clusters.",
  },
  {
    question: "What is a limitation of the K-means algorithm?",
    options: [
      { id: "a", text: "It can only handle numerical data" },
      { id: "b", text: "It assumes clusters are spherical and equally sized" },
      { id: "c", text: "It requires labeled data" },
      { id: "d", text: "It can only find a maximum of 10 clusters" },
    ],
    correctAnswer: "b",
    explanation:
      "K-means assumes that clusters are spherical, equally sized, and have similar densities. It may perform poorly when these assumptions are violated, such as with clusters of different shapes, sizes, or densities.",
  },
  {
    question:
      "What is the time complexity of the K-means algorithm with respect to the number of samples n, features d, clusters k, and iterations i?",
    options: [
      { id: "a", text: "O(n)" },
      { id: "b", text: "O(n*k*d*i)" },
      { id: "c", text: "O(n²)" },
      { id: "d", text: "O(2ⁿ)" },
    ],
    correctAnswer: "b",
    explanation:
      "The time complexity of K-means is O(n*k*d*i), where n is the number of samples, k is the number of clusters, d is the number of features, and i is the number of iterations. This makes it relatively efficient for large datasets.",
  },
]

export const advanced: QuizQuestion[] = [
  {
    question:
      "Which of the following clustering algorithms is better suited for finding non-spherical clusters compared to K-means?",
    options: [
      { id: "a", text: "K-medoids" },
      { id: "b", text: "DBSCAN" },
      { id: "c", text: "Hierarchical clustering" },
      { id: "d", text: "Mini-batch K-means" },
    ],
    correctAnswer: "b",
    explanation:
      "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) can find arbitrarily shaped clusters based on density, making it better suited for non-spherical clusters compared to K-means, which assumes spherical clusters.",
  },
  {
    question: "What is the main difference between K-means and K-medoids?",
    options: [
      { id: "a", text: "K-medoids can handle categorical data directly" },
      { id: "b", text: "K-medoids uses actual data points as cluster centers" },
      { id: "c", text: "K-medoids requires fewer iterations to converge" },
      { id: "d", text: "K-medoids doesn't require specifying the number of clusters" },
    ],
    correctAnswer: "b",
    explanation:
      "In K-medoids, the cluster centers (medoids) are actual data points from the dataset, whereas in K-means, the centroids are the mean of all points in the cluster and may not correspond to any actual data point.",
  },
  {
    question: "What is the Gaussian Mixture Model (GMM) in relation to K-means?",
    options: [
      { id: "a", text: "A preprocessing step for K-means" },
      { id: "b", text: "A method to initialize K-means centroids" },
      { id: "c", text: "A probabilistic generalization of K-means" },
      { id: "d", text: "A technique to visualize K-means results" },
    ],
    correctAnswer: "c",
    explanation:
      "Gaussian Mixture Model (GMM) is a probabilistic model that assumes data points are generated from a mixture of several Gaussian distributions. It can be seen as a soft probabilistic generalization of K-means, where points have probabilities of belonging to each cluster.",
  },
  {
    question: "What is the 'curse of dimensionality' in the context of K-means clustering?",
    options: [
      { id: "a", text: "The difficulty in visualizing clusters in high dimensions" },
      { id: "b", text: "The exponential increase in computational complexity with dimensions" },
      { id: "c", text: "The tendency of distance measures to become less meaningful in high dimensions" },
      { id: "d", text: "The requirement to use more clusters in high-dimensional spaces" },
    ],
    correctAnswer: "c",
    explanation:
      "In high-dimensional spaces, the 'curse of dimensionality' refers to the phenomenon where distance measures (like Euclidean distance used in K-means) become less meaningful as the number of dimensions increases, making it harder to distinguish between points.",
  },
  {
    question: "What is spectral clustering and how does it relate to K-means?",
    options: [
      { id: "a", text: "It's a preprocessing step that transforms data before applying K-means" },
      { id: "b", text: "It's a method to visualize K-means results using spectral decomposition" },
      { id: "c", text: "It's a technique that applies K-means in the eigenspace of a similarity matrix" },
      { id: "d", text: "It's an alternative name for K-means when applied to image data" },
    ],
    correctAnswer: "c",
    explanation:
      "Spectral clustering transforms the data using the eigendecomposition of a similarity matrix (like the graph Laplacian), and then applies K-means in this transformed space. This allows it to find clusters that would be difficult to identify with standard K-means.",
  },
]
