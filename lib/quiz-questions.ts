import type { QuizQuestion } from "./types"

// EDA Quiz Questions
const edaQuizQuestions: Record<string, QuizQuestion[]> = {
  beginner: [
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
  ],
  intermediate: [
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
  ],
  advanced: [
    {
        question: "What is the IQR (Interquartile Range)?",
        options: [
          { id: "a", text: "Range of the dataset" },
          { id: "b", text: "Difference between Q1 and Q3" },
          { id: "c", text: "Median of the dataset" },
          { id: "d", text: "Sum of all quartiles" },
        ],
        correctAnswer: "b",
        explanation: "IQR measures the spread of the middle 50% of data, helping to detect outliers.",
      },
      {
        question: "What can cause a Simpson’s paradox during EDA?",
        options: [
          { id: "a", text: "Outliers" },
          { id: "b", text: "Skewed distributions" },
          { id: "c", text: "Confounding variables" },
          { id: "d", text: "Multicollinearity" },
        ],
        correctAnswer: "c",
        explanation: "Confounding variables can reverse trends when data is grouped, leading to misleading results.",
      },
      {
        question: "Which test is used to check if a numerical feature follows a normal distribution?",
        options: [
          { id: "a", text: "Chi-square test" },
          { id: "b", text: "Shapiro-Wilk test" },
          { id: "c", text: "ANOVA test" },
          { id: "d", text: "T-test" },
        ],
        correctAnswer: "b",
        explanation: "Shapiro-Wilk tests whether data is normally distributed, useful for deciding statistical tests.",
      },
      {
        question: "When visualizing high-dimensional data, what technique can you use to reduce dimensions?",
        options: [
          { id: "a", text: "Logistic Regression" },
          { id: "b", text: "Random Forest" },
          { id: "c", text: "PCA (Principal Component Analysis)" },
          { id: "d", text: "Naive Bayes" },
        ],
        correctAnswer: "c",
        explanation: "PCA reduces high-dimensional data to fewer dimensions while preserving variance.",
      },
      {
        question: "What is an appropriate way to handle highly imbalanced classes during EDA?",
        options: [
          { id: "a", text: "Undersampling the majority class" },
          { id: "b", text: "Oversampling the minority class" },
          { id: "c", text: "Using SMOTE (Synthetic Minority Oversampling Technique)" },
          { id: "d", text: "All of the above" },
        ],
        correctAnswer: "d",
        explanation: "Handling imbalanced classes often involves techniques like undersampling, oversampling, or using SMOTE.",
      },
      {
        question: "Which visualization helps detect hierarchical relationships in data?",
        options: [
          { id: "a", text: "Dendrogram" },
          { id: "b", text: "Histogram" },
          { id: "c", text: "Heatmap" },
          { id: "d", text: "Strip plot" },
        ],
        correctAnswer: "a",
        explanation: "Dendrograms are tree-like diagrams that show hierarchical clustering relationships.",
      },      
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
  ],
}

// Linear Regression Quiz Questions
const linearRegressionQuizQuestions: Record<string, QuizQuestion[]> = {
  beginner: [
    {
      question: "What is the primary goal of linear regression?",
      options: [
        { id: "a", text: "To classify data into categories" },
        { id: "b", text: "To predict a continuous target variable based on one or more predictors" },
        { id: "c", text: "To identify clusters in data" },
        { id: "d", text: "To reduce the dimensionality of data" },
      ],
      correctAnswer: "b",
      explanation:
        "Linear regression aims to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the data.",
    },
    {
      question: "In a simple linear regression equation y = mx + b, what does 'b' represent?",
      options: [
        { id: "a", text: "The slope of the line" },
        { id: "b", text: "The y-intercept" },
        { id: "c", text: "The correlation coefficient" },
        { id: "d", text: "The error term" },
      ],
      correctAnswer: "b",
      explanation: "In the equation y = mx + b, 'b' represents the y-intercept, which is the value of y when x = 0.",
    },
    {
      question: "Which of the following metrics is commonly used to evaluate a linear regression model?",
      options: [
        { id: "a", text: "Accuracy" },
        { id: "b", text: "Precision" },
        { id: "c", text: "R-squared" },
        { id: "d", text: "F1 score" },
      ],
      correctAnswer: "c",
      explanation:
        "R-squared (coefficient of determination) measures the proportion of variance in the dependent variable that is predictable from the independent variables.",
    },
    {
      question: "What does a positive coefficient for a predictor variable indicate in linear regression?",
      options: [
        { id: "a", text: "The target variable decreases as the predictor increases" },
        { id: "b", text: "The target variable increases as the predictor increases" },
        { id: "c", text: "There is no relationship between the variables" },
        { id: "d", text: "The predictor variable is not significant" },
      ],
      correctAnswer: "b",
      explanation:
        "A positive coefficient indicates that as the predictor variable increases, the target variable also increases, showing a direct relationship.",
    },
    {
      question: "What assumption of linear regression is violated when the residuals are not normally distributed?",
      options: [
        { id: "a", text: "Linearity" },
        { id: "b", text: "Homoscedasticity" },
        { id: "c", text: "Normality of errors" },
        { id: "d", text: "Independence of observations" },
      ],
      correctAnswer: "c",
      explanation:
        "The normality assumption states that the residuals (errors) should be normally distributed. This is important for valid hypothesis testing and confidence intervals.",
    },
  ],
  intermediate: [
    {
      question: "What is multicollinearity in the context of linear regression?",
      options: [
        { id: "a", text: "When the target variable has multiple peaks in its distribution" },
        { id: "b", text: "When predictor variables are highly correlated with each other" },
        { id: "c", text: "When the relationship between variables is non-linear" },
        { id: "d", text: "When the residuals are not normally distributed" },
      ],
      correctAnswer: "b",
      explanation:
        "Multicollinearity occurs when two or more predictor variables are highly correlated, making it difficult to determine the individual effect of each predictor on the target variable.",
    },
    {
      question: "Which technique can help address the issue of multicollinearity?",
      options: [
        { id: "a", text: "Increasing the sample size" },
        { id: "b", text: "Ridge regression" },
        { id: "c", text: "Using more categorical variables" },
        { id: "d", text: "Removing outliers" },
      ],
      correctAnswer: "b",
      explanation:
        "Ridge regression adds a penalty term to the linear regression objective function, which helps reduce the impact of multicollinearity by shrinking the coefficients.",
    },
    {
      question: "What does heteroscedasticity mean in linear regression?",
      options: [
        { id: "a", text: "The residuals have constant variance across all levels of predictors" },
        { id: "b", text: "The residuals are normally distributed" },
        { id: "c", text: "The residuals have non-constant variance across levels of predictors" },
        { id: "d", text: "The predictors are highly correlated" },
      ],
      correctAnswer: "c",
      explanation:
        "Heteroscedasticity occurs when the variance of the residuals is not constant across all levels of the predictor variables, violating the homoscedasticity assumption.",
    },
    {
      question: "What is the purpose of the train-test split in linear regression modeling?",
      options: [
        { id: "a", text: "To increase the model's complexity" },
        { id: "b", text: "To evaluate the model's performance on unseen data" },
        { id: "c", text: "To remove outliers from the dataset" },
        { id: "d", text: "To normalize the predictor variables" },
      ],
      correctAnswer: "b",
      explanation:
        "The train-test split divides the data into training and testing sets, allowing us to train the model on one subset and evaluate its performance on unseen data to assess generalization.",
    },
    {
      question: "What is the difference between R-squared and adjusted R-squared?",
      options: [
        { id: "a", text: "R-squared is always higher than adjusted R-squared" },
        { id: "b", text: "Adjusted R-squared accounts for the number of predictors in the model" },
        { id: "c", text: "R-squared is used for multiple regression, adjusted R-squared for simple regression" },
        { id: "d", text: "They measure different aspects of model fit" },
      ],
      correctAnswer: "b",
      explanation:
        "Adjusted R-squared modifies the R-squared by taking into account the number of predictors in the model, penalizing the addition of variables that don't improve the model significantly.",
    },
  ],
  advanced: [
    {
      question:
        "Which of the following regularization techniques adds a penalty term proportional to the sum of the absolute values of the coefficients?",
      options: [
        { id: "a", text: "Ridge regression" },
        { id: "b", text: "Lasso regression" },
        { id: "c", text: "Elastic Net" },
        { id: "d", text: "Ordinary least squares" },
      ],
      correctAnswer: "b",
      explanation:
        "Lasso (Least Absolute Shrinkage and Selection Operator) regression adds a penalty term proportional to the sum of the absolute values of the coefficients (L1 regularization), which can lead to sparse models by forcing some coefficients to zero.",
    },
    {
      question: "What is the primary advantage of polynomial regression over simple linear regression?",
      options: [
        { id: "a", text: "It always results in a better fit regardless of the data" },
        { id: "b", text: "It can capture non-linear relationships between variables" },
        { id: "c", text: "It requires fewer observations to train" },
        { id: "d", text: "It is less prone to overfitting" },
      ],
      correctAnswer: "b",
      explanation:
        "Polynomial regression extends linear regression by adding polynomial terms (squared, cubed, etc.) of the predictors, allowing it to capture non-linear relationships between variables.",
    },
    {
      question: "What is the Cook's distance used for in regression diagnostics?",
      options: [
        { id: "a", text: "To measure the goodness of fit" },
        { id: "b", text: "To identify influential observations" },
        { id: "c", text: "To test for multicollinearity" },
        { id: "d", text: "To check for normality of residuals" },
      ],
      correctAnswer: "b",
      explanation:
        "Cook's distance measures the influence of each observation on the regression results. High values indicate observations that significantly affect the model's coefficients if removed.",
    },
    {
      question: "In the context of linear regression, what does the Variance Inflation Factor (VIF) measure?",
      options: [
        { id: "a", text: "The variance of the residuals" },
        { id: "b", text: "The severity of multicollinearity" },
        { id: "c", text: "The goodness of fit of the model" },
        { id: "d", text: "The significance of individual predictors" },
      ],
      correctAnswer: "b",
      explanation:
        "The Variance Inflation Factor (VIF) quantifies the severity of multicollinearity by measuring how much the variance of an estimated regression coefficient is increased due to collinearity with other predictors.",
    },
    {
      question: "What is the purpose of quantile regression compared to ordinary least squares regression?",
      options: [
        { id: "a", text: "It is less computationally intensive" },
        { id: "b", text: "It provides a more complete picture of the relationship between variables" },
        { id: "c", text: "It always results in better predictions" },
        { id: "d", text: "It requires fewer assumptions about the data" },
      ],
      correctAnswer: "b",
      explanation:
        "Quantile regression models the relationship between predictors and specific quantiles of the response variable, providing a more complete picture of the relationship across the entire distribution, not just the mean.",
    },
  ],
}

// SVM Quiz Questions
const svmQuizQuestions: Record<string, QuizQuestion[]> = {
  beginner: [
    {
      question: "What does SVM stand for?",
      options: [
        { id: "a", text: "Statistical Variable Model" },
        { id: "b", text: "Support Vector Machine" },
        { id: "c", text: "System Variance Method" },
        { id: "d", text: "Structured Variable Mechanism" },
      ],
      correctAnswer: "b",
      explanation:
        "SVM stands for Support Vector Machine, which is a supervised learning algorithm used for classification and regression tasks.",
    },
    {
      question: "What is the main objective of an SVM classifier?",
      options: [
        { id: "a", text: "To minimize the number of support vectors" },
        { id: "b", text: "To find the hyperplane with the largest margin between classes" },
        { id: "c", text: "To maximize the number of misclassifications" },
        { id: "d", text: "To create as many decision boundaries as possible" },
      ],
      correctAnswer: "b",
      explanation:
        "The main objective of SVM is to find the optimal hyperplane that maximizes the margin between different classes, which helps in better generalization.",
    },
    {
      question: "What are support vectors in SVM?",
      options: [
        { id: "a", text: "All data points in the dataset" },
        { id: "b", text: "The data points closest to the decision boundary" },
        { id: "c", text: "The center points of each class" },
        { id: "d", text: "The outliers that are removed during preprocessing" },
      ],
      correctAnswer: "b",
      explanation:
        "Support vectors are the data points that lie closest to the decision boundary (hyperplane). They are critical in defining the margin and are the most difficult to classify.",
    },
    {
      question: "Which of the following is NOT a common kernel used in SVM?",
      options: [
        { id: "a", text: "Linear kernel" },
        { id: "b", text: "Polynomial kernel" },
        { id: "c", text: "Gaussian kernel" },
        { id: "d", text: "Logarithmic kernel" },
      ],
      correctAnswer: "d",
      explanation:
        "Common kernels in SVM include linear, polynomial, radial basis function (RBF or Gaussian), and sigmoid. Logarithmic kernel is not a standard kernel type used in SVM.",
    },
    {
      question: "What is the purpose of the kernel trick in SVM?",
      options: [
        { id: "a", text: "To reduce the number of support vectors" },
        { id: "b", text: "To speed up the training process" },
        { id: "c", text: "To handle linearly inseparable data by transforming it into higher dimensions" },
        { id: "d", text: "To reduce the dimensionality of the data" },
      ],
      correctAnswer: "c",
      explanation:
        "The kernel trick allows SVM to operate in a higher-dimensional space without explicitly computing the coordinates of the data in that space, making it possible to handle linearly inseparable data.",
    },
  ],
  intermediate: [
    {
      question: "What is the C parameter in SVM used for?",
      options: [
        { id: "a", text: "It controls the kernel function selection" },
        { id: "b", text: "It determines the number of support vectors" },
        { id: "c", text: "It controls the trade-off between margin maximization and classification error" },
        { id: "d", text: "It sets the learning rate for the algorithm" },
      ],
      correctAnswer: "c",
      explanation:
        "The C parameter is a regularization parameter that controls the trade-off between having a wide margin and correctly classifying training points. A smaller C allows for a wider margin but more misclassifications.",
    },
    {
      question: "What does the gamma parameter control in an RBF kernel?",
      options: [
        { id: "a", text: "The maximum number of iterations" },
        { id: "b", text: "The influence of a single training example" },
        { id: "c", text: "The width of the margin" },
        { id: "d", text: "The number of support vectors" },
      ],
      correctAnswer: "b",
      explanation:
        "In the RBF kernel, gamma defines how far the influence of a single training example reaches. Low gamma means a far reach (smoother decision boundary), while high gamma means a close reach (more complex boundary).",
    },
    {
      question: "Which of the following statements about SVM is true?",
      options: [
        { id: "a", text: "SVM always performs better than neural networks" },
        { id: "b", text: "SVM is not affected by the curse of dimensionality" },
        { id: "c", text: "SVM can be used for both classification and regression tasks" },
        { id: "d", text: "SVM cannot handle non-linear decision boundaries" },
      ],
      correctAnswer: "c",
      explanation:
        "SVM can be used for both classification (SVC) and regression (SVR) tasks. For regression, it tries to find a function that deviates from the observed target values by a value no greater than a specified margin.",
    },
    {
      question: "What is the effect of increasing the C parameter in SVM?",
      options: [
        { id: "a", text: "It increases the margin width" },
        { id: "b", text: "It decreases the penalty for misclassification" },
        { id: "c", text: "It increases the penalty for misclassification" },
        { id: "d", text: "It has no effect on the model" },
      ],
      correctAnswer: "c",
      explanation:
        "Increasing the C parameter increases the penalty for misclassification, leading to a narrower margin but potentially better classification of training points. This can lead to overfitting if C is too large.",
    },
    {
      question: "What is the time complexity of training an SVM with respect to the number of samples n?",
      options: [
        { id: "a", text: "O(n)" },
        { id: "b", text: "O(n log n)" },
        { id: "c", text: "O(n²) to O(n³)" },
        { id: "d", text: "O(2ⁿ)" },
      ],
      correctAnswer: "c",
      explanation:
        "The time complexity of training an SVM is between O(n²) and O(n³), depending on the implementation and the data. This is one of the limitations of SVM for very large datasets.",
    },
  ],
  advanced: [
    {
      question: "What is the dual formulation of SVM used for?",
      options: [
        { id: "a", text: "To reduce the computational complexity" },
        { id: "b", text: "To apply the kernel trick without explicitly computing transformations" },
        { id: "c", text: "To handle multi-class classification" },
        { id: "d", text: "To perform feature selection" },
      ],
      correctAnswer: "b",
      explanation:
        "The dual formulation of SVM allows for the application of the kernel trick, enabling the algorithm to operate in higher-dimensional spaces without explicitly computing the transformations of the input data.",
    },
    {
      question: "Which of the following is a limitation of SVMs?",
      options: [
        { id: "a", text: "They cannot handle non-linear boundaries" },
        { id: "b", text: "They do not provide probability estimates directly" },
        { id: "c", text: "They cannot be used for regression problems" },
        { id: "d", text: "They require fewer parameters than other algorithms" },
      ],
      correctAnswer: "b",
      explanation:
        "Traditional SVMs do not provide probability estimates directly. Techniques like Platt scaling need to be applied to convert SVM outputs to probabilities.",
    },
    {
      question: "What is the purpose of the ν (nu) parameter in ν-SVM?",
      options: [
        { id: "a", text: "It controls the number of support vectors" },
        { id: "b", text: "It sets an upper bound on the fraction of margin errors" },
        { id: "c", text: "It determines the kernel function to use" },
        { id: "d", text: "It specifies the learning rate" },
      ],
      correctAnswer: "b",
      explanation:
        "In ν-SVM, the ν parameter sets an upper bound on the fraction of margin errors and a lower bound on the fraction of support vectors. It provides more intuitive control over the model's behavior than the C parameter.",
    },
    {
      question: "What is the One-Class SVM used for?",
      options: [
        { id: "a", text: "Binary classification problems" },
        { id: "b", text: "Multi-class classification problems" },
        { id: "c", text: "Anomaly detection or novelty detection" },
        { id: "d", text: "Regression problems" },
      ],
      correctAnswer: "c",
      explanation:
        "One-Class SVM is used for anomaly detection or novelty detection. It learns a decision boundary that encompasses the normal data points and can identify outliers or new data points that differ significantly from the training data.",
    },
    {
      question: "Which of the following techniques can be used to handle multi-class classification with SVM?",
      options: [
        { id: "a", text: "One-vs-Rest (OvR)" },
        { id: "b", text: "One-vs-One (OvO)" },
        { id: "c", text: "Error-Correcting Output Codes (ECOC)" },
        { id: "d", text: "All of the above" },
      ],
      correctAnswer: "d",
      explanation:
        "All three techniques—One-vs-Rest, One-vs-One, and Error-Correcting Output Codes—can be used to extend binary SVMs to handle multi-class classification problems.",
    },
  ],
}

// K-means Quiz Questions
const kmeansQuizQuestions: Record<string, QuizQuestion[]> = {
  beginner: [
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
  ],
  intermediate: [
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
  ],
  advanced: [
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
  ],
}

// Add more algorithm quiz questions here...

// Map to store all quiz questions by algorithm
const quizQuestionsMap: Record<string, Record<string, QuizQuestion[]>> = {
  eda: edaQuizQuestions,
  "linear-regression": linearRegressionQuizQuestions,
  svm: svmQuizQuestions,
  kmeans: kmeansQuizQuestions,
  // Add more algorithms here
}

// Function to get quiz questions for a specific algorithm and difficulty level
export function getQuizQuestions(algorithmSlug: string, difficultyLevel: string): QuizQuestion[] {
  // If the algorithm exists in our map
  if (quizQuestionsMap[algorithmSlug]) {
    // If the difficulty level exists for this algorithm
    if (quizQuestionsMap[algorithmSlug][difficultyLevel]) {
      return quizQuestionsMap[algorithmSlug][difficultyLevel]
    }

    // If difficulty level doesn't exist, return beginner questions as fallback
    return quizQuestionsMap[algorithmSlug]["beginner"] || []
  }

  // If algorithm doesn't exist, return empty array
  return []
}

