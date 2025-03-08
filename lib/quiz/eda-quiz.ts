import type { QuizQuestion } from "../types"

export const beginner: QuizQuestion[] = [
  {
    question: "What is the primary purpose of Exploratory Data Analysis?",
    options: [
      { id: "a", text: "To build predictive models" },
      { id: "b", text: "To understand patterns and relationships in data" },
      { id: "c", text: "To clean and preprocess data" },
      { id: "d", text: "To visualize data in 3D" },
    ],
    correctAnswer: "b",
    explanation:
      "EDA is primarily used to understand the patterns, relationships, and structure of data before formal modeling.",
  },
  {
    question: "Which of the following is NOT typically part of EDA?",
    options: [
      { id: "a", text: "Checking for missing values" },
      { id: "b", text: "Identifying outliers" },
      { id: "c", text: "Deploying models to production" },
      { id: "d", text: "Visualizing distributions" },
    ],
    correctAnswer: "c",
    explanation:
      "Deploying models to production is part of the model deployment phase, not exploratory data analysis.",
  },
  {
    question: "What statistical measure is most useful for identifying the central tendency in skewed data?",
    options: [
      { id: "a", text: "Mean" },
      { id: "b", text: "Mode" },
      { id: "c", text: "Median" },
      { id: "d", text: "Standard deviation" },
    ],
    correctAnswer: "c",
    explanation:
      "The median is less affected by outliers and skewed distributions, making it a better measure of central tendency for skewed data.",
  },
  {
    question: "Which visualization is best for comparing distributions of multiple groups?",
    options: [
      { id: "a", text: "Pie chart" },
      { id: "b", text: "Line chart" },
      { id: "c", text: "Box plot" },
      { id: "d", text: "Scatter plot" },
    ],
    correctAnswer: "c",
    explanation:
      "Box plots are excellent for comparing distributions across multiple groups, showing median, quartiles, and outliers.",
  },
  {
    question: "What is a correlation coefficient used for?",
    options: [
      { id: "a", text: "To measure the size of a dataset" },
      { id: "b", text: "To measure the relationship between two variables" },
      { id: "c", text: "To count the number of categories" },
      { id: "d", text: "To calculate the median" },
    ],
    correctAnswer: "b",
    explanation:
      "A correlation coefficient measures the strength and direction of the relationship between two variables.",
  },
]


export const intermediate: QuizQuestion[] = [
  {
    question: "Which of the following correlation values indicates the strongest relationship?",
    options: [
      { id: "a", text: "0.3" },
      { id: "b", text: "-0.8" },
      { id: "c", text: "0.1" },
      { id: "d", text: "-0.4" },
    ],
    correctAnswer: "b",
    explanation:
      "The absolute value of the correlation coefficient indicates the strength of the relationship. -0.8 has the largest absolute value (0.8), indicating the strongest relationship.",
  },
  {
    question: "What is the interquartile range (IQR) used for in EDA?",
    options: [
      { id: "a", text: "To measure the central tendency" },
      { id: "b", text: "To identify the most frequent value" },
      { id: "c", text: "To measure the spread of the middle 50% of data" },
      { id: "d", text: "To calculate the mean" },
    ],
    correctAnswer: "c",
    explanation:
      "The IQR is the difference between the 75th and 25th percentiles, representing the spread of the middle 50% of the data.",
  },
  {
    question:
      "Which technique is most appropriate for visualizing the relationship between a categorical and a numerical variable?",
    options: [
      { id: "a", text: "Scatter plot" },
      { id: "b", text: "Box plot" },
      { id: "c", text: "Line chart" },
      { id: "d", text: "Histogram" },
    ],
    correctAnswer: "b",
    explanation:
      "Box plots are ideal for showing the distribution of a numerical variable across different categories.",
  },
  {
    question: "What does a Q-Q plot help determine?",
    options: [
      { id: "a", text: "The correlation between variables" },
      { id: "b", text: "Whether data follows a normal distribution" },
      { id: "c", text: "The number of outliers" },
      { id: "d", text: "The median value" },
    ],
    correctAnswer: "b",
    explanation:
      "Q-Q (quantile-quantile) plots compare the quantiles of the data against the quantiles of a theoretical distribution (often normal) to assess if the data follows that distribution.",
  },
  {
    question: "Which of the following is NOT a valid method for handling missing data?",
    options: [
      { id: "a", text: "Imputation with mean values" },
      { id: "b", text: "Removing rows with missing values" },
      { id: "c", text: "Ignoring missing values in the analysis" },
      { id: "d", text: "Imputation with regression models" },
    ],
    correctAnswer: "c",
    explanation:
      "Ignoring missing values without proper handling can lead to biased results and is not a valid approach in data analysis.",
  },
]


export const advanced: QuizQuestion[] = [
  {
    question: "Which of the following techniques is most appropriate for identifying multivariate outliers?",
    options: [
      { id: "a", text: "Z-score" },
      { id: "b", text: "Box plot" },
      { id: "c", text: "Mahalanobis distance" },
      { id: "d", text: "Histogram" },
    ],
    correctAnswer: "c",
    explanation:
      "Mahalanobis distance measures the distance between a point and a distribution, accounting for the covariance structure, making it suitable for identifying multivariate outliers.",
  },
  {
    question: "What is the purpose of a SPLOM (Scatter Plot Matrix) in EDA?",
    options: [
      { id: "a", text: "To visualize time series data" },
      { id: "b", text: "To show pairwise relationships between multiple variables" },
      { id: "c", text: "To identify clusters in data" },
      { id: "d", text: "To compare categorical variables" },
    ],
    correctAnswer: "b",
    explanation:
      "A Scatter Plot Matrix (SPLOM) shows all pairwise scatter plots of variables in a dataset, allowing for the examination of relationships between multiple variables simultaneously.",
  },
  {
    question: "Which statistical test would you use to determine if two samples come from the same distribution?",
    options: [
      { id: "a", text: "t-test" },
      { id: "b", text: "Chi-square test" },
      { id: "c", text: "Kolmogorov-Smirnov test" },
      { id: "d", text: "ANOVA" },
    ],
    correctAnswer: "c",
    explanation:
      "The Kolmogorov-Smirnov test is a non-parametric test that compares the cumulative distributions of two samples to determine if they come from the same distribution.",
  },
  {
    question: "What is the purpose of dimensionality reduction techniques like PCA in EDA?",
    options: [
      { id: "a", text: "To increase the number of features" },
      { id: "b", text: "To visualize high-dimensional data in lower dimensions" },
      { id: "c", text: "To remove outliers" },
      { id: "d", text: "To impute missing values" },
    ],
    correctAnswer: "b",
    explanation:
      "Principal Component Analysis (PCA) and other dimensionality reduction techniques help visualize high-dimensional data in lower dimensions while preserving as much variance as possible.",
  },
  {
    question: "Which of the following is a robust measure of correlation that is less sensitive to outliers?",
    options: [
      { id: "a", text: "Pearson correlation" },
      { id: "b", text: "Spearman's rank correlation" },
      { id: "c", text: "Covariance" },
      { id: "d", text: "R-squared" },
    ],
    correctAnswer: "b",
    explanation:
      "Spearman's rank correlation is based on the ranks of the data rather than the actual values, making it less sensitive to outliers and non-linear relationships.",
  },
]
