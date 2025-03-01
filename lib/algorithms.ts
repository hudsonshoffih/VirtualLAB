import type { Algorithm } from "./types"

const algorithms: Algorithm[] = [
  {
    id: "eda",
    slug: "eda",
    title: "Exploratory Data Analysis",
    description: "Learn how to analyze and visualize datasets to understand patterns and relationships.",
    tutorialContent:
      "# Exploratory Data Analysis\n\nEDA is an approach to analyzing datasets to summarize their main characteristics, often with visual methods.",
  },
  {
    id: "dataset-insights",
    slug: "dataset-insights",
    title: "Dataset Insights and Statistics",
    description: "Discover statistical methods to extract meaningful insights from your data.",
    tutorialContent:
      "# Dataset Insights and Statistics\n\nLearn how to calculate and interpret key statistical measures to understand your dataset's characteristics.",
  },
  {
    id: "evaluation-metrics",
    slug: "evaluation-metrics",
    title: "Evaluation Metrics",
    description: "Master the metrics used to assess model performance and make informed decisions.",
    tutorialContent:
      "# Evaluation Metrics\n\nUnderstand how to measure and compare the performance of different machine learning models using appropriate metrics.",
  },
  {
    id: "linear-regression",
    slug: "linear-regression",
    title: "Linear Regression",
    description: "Master the fundamentals of linear regression for predictive modeling.",
    tutorialContent:
      "# Linear Regression\n\nLinear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables.",
  },
  {
    id: "logistic-regression",
    slug: "logistic-regression",
    title: "Logistic Regression",
    description: "Learn how to model probability of events using the logistic function.",
    tutorialContent:
      "# Logistic Regression\n\nLogistic regression is a statistical model that uses a logistic function to model a binary dependent variable.",
  },
  {
    id: "knn",
    slug: "knn",
    title: "K-Nearest Neighbors",
    description: "Explore this simple yet powerful classification and regression algorithm.",
    tutorialContent:
      "# K-Nearest Neighbors\n\nKNN is a non-parametric method used for classification and regression, where the output is based on the k closest training examples.",
  },
  {
    id: "random-forest",
    slug: "random-forest",
    title: "Random Forest",
    description: "Understand how ensemble learning with decision trees creates robust models.",
    tutorialContent:
      "# Random Forest\n\nRandom Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes.",
  },
  {
    id: "svm",
    slug: "svm",
    title: "Support Vector Machines",
    description: "Understand the powerful classification algorithm and its applications.",
    tutorialContent:
      "# Support Vector Machines\n\nSVM is a supervised learning model that analyzes data for classification and regression analysis.",
  },
  {
    id: "ensemble-models",
    slug: "ensemble-models",
    title: "Ensemble Models",
    description: "Learn how combining multiple models can improve prediction accuracy and robustness.",
    tutorialContent:
      "# Ensemble Models\n\nEnsemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.",
  },
  {
    id: "kmeans",
    slug: "kmeans",
    title: "K-Means Clustering",
    description: "Discover how to group similar data points into clusters using this popular algorithm.",
    tutorialContent:
      "# K-Means Clustering\n\nK-means clustering is a method of vector quantization that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean.",
  },
  {
    id: "pca",
    slug: "pca",
    title: "Principal Component Analysis",
    description: "Master dimensionality reduction techniques to simplify complex datasets.",
    tutorialContent:
      "# Principal Component Analysis\n\nPCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.",
  },
]

export function getAlgorithms(): Algorithm[] {
  return algorithms
}

export function getAlgorithmBySlug(slug: string): Algorithm | undefined {
  return algorithms.find((algorithm) => algorithm.slug === slug)
}

