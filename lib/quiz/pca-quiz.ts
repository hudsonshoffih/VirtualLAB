import type { QuizQuestion } from "../types"

export const beginner: QuizQuestion[] = [
  {
    question: "What is the primary goal of Principal Component Analysis (PCA)?",
    options: [
      { id: "a", text: "Classification" },
      { id: "b", text: "Clustering" },
      { id: "c", text: "Dimensionality reduction" },
      { id: "d", text: "Regression" }
    ],
    correctAnswer: "c",
    explanation: "PCA is primarily used for dimensionality reduction by transforming data into fewer uncorrelated variables called principal components."
  },
  {
    question: "In PCA, what are the principal components?",
    options: [
      { id: "a", text: "Features of the dataset" },
      { id: "b", text: "Eigenvalues of the covariance matrix" },
      { id: "c", text: "Eigenvectors of the covariance matrix" },
      { id: "d", text: "Data points in the dataset" }
    ],
    correctAnswer: "c",
    explanation: "Principal components are the eigenvectors of the data's covariance matrix, representing new axes in the transformed space."
  },
  {
    question: "Which statistical technique is often used to standardize data before applying PCA?",
    options: [
      { id: "a", text: "Z-score normalization" },
      { id: "b", text: "Min-Max scaling" },
      { id: "c", text: "Log transformation" },
      { id: "d", text: "None of the above" }
    ],
    correctAnswer: "a",
    explanation: "Z-score normalization ensures that each feature has zero mean and unit variance, which is essential for PCA to perform effectively."
  },
  {
    question: "What is the purpose of eigenvalues in PCA?",
    options: [
      { id: "a", text: "They represent the variance explained by each principal component." },
      { id: "b", text: "They determine the number of principal components to retain." },
      { id: "c", text: "They are used for data visualization." },
      { id: "d", text: "They measure the similarity between data points." }
    ],
    correctAnswer: "a",
    explanation: "Eigenvalues in PCA indicate how much variance each principal component explains from the original dataset."
  },
  {
    question: "What is the typical range of values for eigenvalues in PCA when analyzing a dataset?",
    options: [
      { id: "a", text: "Any real value" },
      { id: "b", text: "Always between 0 and 1" },
      { id: "c", text: "Always positive" },
      { id: "d", text: "Always integers" }
    ],
    correctAnswer: "a",
    explanation: "Eigenvalues in PCA can be any real number, though for valid principal components, they are typically non-negative real values."
  },
  {
    question: "How is the total variance of the data distributed among the principal components?",
    options: [
      { id: "a", text: "Equally among all principal components" },
      { id: "b", text: "Proportional to their eigenvalues" },
      { id: "c", text: "Proportional to the number of data points" },
      { id: "d", text: "Proportional to the number of features" }
    ],
    correctAnswer: "b",
    explanation: "The variance captured by each principal component is directly proportional to its corresponding eigenvalue."
  },
  {
    question: "When should you use PCA as a preprocessing step in a machine learning pipeline?",
    options: [
      { id: "a", text: "When you want to increase the number of features" },
      { id: "b", text: "When you suspect multicollinearity among features" },
      { id: "c", text: "When you want to remove outliers" },
      { id: "d", text: "PCA should never be used as a preprocessing step." }
    ],
    correctAnswer: "b",
    explanation: "PCA is helpful when multicollinearity exists among features, as it transforms correlated variables into uncorrelated components."
  },
  {
    question: "What does the scree plot in PCA help determine?",
    options: [
      { id: "a", text: "The optimal number of clusters in the data" },
      { id: "b", text: "The correlation between features" },
      { id: "c", text: "The explained variance by each principal component" },
      { id: "d", text: "The significance of outliers in the dataset" }
    ],
    correctAnswer: "c",
    explanation: "A scree plot helps visualize the variance explained by each principal component to decide how many to retain."
  },
  {
    question: "In PCA, what is the relationship between the first principal component and the second principal component?",
    options: [
      { id: "a", text: "They are orthogonal (uncorrelated) to each other." },
      { id: "b", text: "They are positively correlated." },
      { id: "c", text: "They are negatively correlated." },
      { id: "d", text: "There is no defined relationship between them." }
    ],
    correctAnswer: "a",
    explanation: "Principal components are constructed to be orthogonal to each other, meaning they are statistically uncorrelated."
  },
  {
    question: "In PCA, what is the relationship between the number of principal components retained and the amount of variance explained by the retained components?",
    options: [
      { id: "a", text: "More retained components explain less variance." },
      { id: "b", text: "More retained components explain more variance." },
      { id: "c", text: "The number of retained components does not affect the explained variance." },
      { id: "d", text: "The relationship depends on the type of dataset." }
    ],
    correctAnswer: "b",
    explanation: "Retaining more components captures more of the original dataset’s variance, though with diminishing returns after a point."
  }
]
export const intermediate: QuizQuestion[] = [
  {
    question: "Why is it important to center the data before applying PCA?",
    options: [
      { id: "a", text: "It simplifies eigenvalue calculation" },
      { id: "b", text: "To ensure the mean of each feature is 1" },
      { id: "c", text: "To ensure the data is uncorrelated" },
      { id: "d", text: "To make sure PCA captures maximum variance from the origin" }
    ],
    correctAnswer: "d",
    explanation: "Centering the data ensures that the mean is at the origin, allowing PCA to correctly find directions of maximum variance."
  },
  {
    question: "What happens if we retain all principal components in PCA?",
    options: [
      { id: "a", text: "Dimensionality is reduced" },
      { id: "b", text: "Variance is lost" },
      { id: "c", text: "Original data can be perfectly reconstructed" },
      { id: "d", text: "The model becomes unstable" }
    ],
    correctAnswer: "c",
    explanation: "Retaining all components allows for perfect reconstruction of the original dataset."
  },
  {
    question: "Which of the following affects the direction of principal components?",
    options: [
      { id: "a", text: "Scaling of features" },
      { id: "b", text: "Number of clusters" },
      { id: "c", text: "Sample size" },
      { id: "d", text: "Outliers only" }
    ],
    correctAnswer: "a",
    explanation: "PCA is sensitive to the scale of the data; unscaled features can dominate the variance and distort results."
  },
  {
    question: "What does a small eigenvalue indicate in PCA?",
    options: [
      { id: "a", text: "High variance along the component" },
      { id: "b", text: "Component captures significant information" },
      { id: "c", text: "Component is highly correlated with others" },
      { id: "d", text: "Component contributes little to data variance" }
    ],
    correctAnswer: "d",
    explanation: "Small eigenvalues indicate that the component explains very little variance and can often be discarded."
  },
  {
    question: "If PCA components explain 90% variance, what does that imply?",
    options: [
      { id: "a", text: "10% of data has been lost" },
      { id: "b", text: "The model captures all important information" },
      { id: "c", text: "The data was perfectly reconstructed" },
      { id: "d", text: "Remaining components are noisy or less significant" }
    ],
    correctAnswer: "d",
    explanation: "The remaining components (10%) likely represent noise or less important information in the data."
  }
]
export const advanced: QuizQuestion[] = [
  {
    question: "Which matrix decomposition is most commonly used in PCA implementation?",
    options: [
      { id: "a", text: "QR decomposition" },
      { id: "b", text: "LU decomposition" },
      { id: "c", text: "Singular Value Decomposition (SVD)" },
      { id: "d", text: "Cholesky decomposition" }
    ],
    correctAnswer: "c",
    explanation: "SVD is commonly used in PCA as it is more numerically stable and can handle non-square matrices."
  },
  {
    question: "How is PCA different from Linear Discriminant Analysis (LDA)?",
    options: [
      { id: "a", text: "PCA uses label information; LDA doesn't" },
      { id: "b", text: "LDA reduces variance; PCA maximizes it" },
      { id: "c", text: "PCA is unsupervised; LDA is supervised" },
      { id: "d", text: "They are both the same" }
    ],
    correctAnswer: "c",
    explanation: "PCA is unsupervised and focuses on variance, while LDA is supervised and focuses on maximizing class separability."
  },
  {
    question: "What is the geometric interpretation of principal components?",
    options: [
      { id: "a", text: "Axes along which data points cluster tightly" },
      { id: "b", text: "New coordinate axes aligned with the maximum variance directions" },
      { id: "c", text: "Random projections of the dataset" },
      { id: "d", text: "Lines that connect the centroids of clusters" }
    ],
    correctAnswer: "b",
    explanation: "Principal components are new axes that represent directions of maximum variance in the data."
  },
  {
    question: "What is a limitation of PCA?",
    options: [
      { id: "a", text: "It increases the feature space" },
      { id: "b", text: "It is only applicable to classification problems" },
      { id: "c", text: "It assumes linear relationships in data" },
      { id: "d", text: "It can be used only with binary data" }
    ],
    correctAnswer: "c",
    explanation: "PCA assumes linear relationships and may fail to capture non-linear structures in the data."
  },
  {
    question: "Which component should be selected if you want to visualize high-dimensional data in 2D?",
    options: [
      { id: "a", text: "The two components with the smallest eigenvalues" },
      { id: "b", text: "The two original features with least correlation" },
      { id: "c", text: "The first two principal components" },
      { id: "d", text: "Any two randomly selected components" }
    ],
    correctAnswer: "c",
    explanation: "The first two principal components explain the most variance and are ideal for 2D visualizations."
  }
]


