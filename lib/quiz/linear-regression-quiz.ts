import type { QuizQuestion } from "../types"
export const beginner: QuizQuestion[] = [
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
]

export const intermediate: QuizQuestion[] = [
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
]

export const advanced: QuizQuestion[] = [
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
]
