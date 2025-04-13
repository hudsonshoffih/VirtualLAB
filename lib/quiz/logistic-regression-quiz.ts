import type { QuizQuestion } from "../types"

export const beginner: QuizQuestion[] = [
{
        question: "What is the main goal of logistic regression?",
        options: [
          { id: "a", text: "To predict a continuous variable" },
          { id: "b", text: "To classify data into categories" },
          { id: "c", text: "To cluster similar data points" },
          { id: "d", text: "To calculate the correlation between features" }
        ],
        correctAnswer: "b",
        explanation: "Logistic regression is used for classification problems, predicting categorical outcomes."
      },
      {
        question: "What type of output does logistic regression produce?",
        options: [
          { id: "a", text: "Discrete class label" },
          { id: "b", text: "Probability value" },
          { id: "c", text: "Cluster ID" },
          { id: "d", text: "Numerical score" }
        ],
        correctAnswer: "b",
        explanation: "It outputs a probability score, which is then used to assign a class."
      },
      {
        question: "What type of output does logistic regression produce?",
        options: [
          { id: "a", text: "Discrete class label" },
          { id: "b", text: "Probability value" },
          { id: "c", text: "Cluster ID" },
          { id: "d", text: "Numerical score" }
        ],
        correctAnswer: "b",
        explanation: "It outputs a probability score, which is then used to assign a class."
      },
      {
        question: "Which loss function is commonly used in binary logistic regression?",
        options: [
          { id: "a", text: "Mean Squared Error" },
          { id: "b", text: "Cross-Entropy Loss" },
          { id: "c", text: "Hinge Loss" },
          { id: "d", text: "Absolute Error" }
        ],
        correctAnswer: "b",
        explanation: "Cross-entropy loss is used to quantify the difference between predicted and actual probabilities."
      },
      {
        question: "Which of the following is an assumption of logistic regression?",
        options: [
          { id: "a", text: "Linearity between independent and dependent variable" },
          { id: "b", text: "Normal distribution of dependent variable" },
          { id: "c", text: "No multicollinearity among predictors" },
          { id: "d", text: "All features must be binary" }
        ],
        correctAnswer: "c",
        explanation: "Logistic regression assumes that predictor variables are not highly correlated with each other."
      },
    ]
    export const intermediate: QuizQuestion[] = [
        {
            question: "How does logistic regression deal with multicollinearity?",
            options: [
              { id: "a", text: "It ignores it" },
              { id: "b", text: "It combines collinear variables" },
              { id: "c", text: "It suffers and may produce unstable results" },
              { id: "d", text: "It automatically drops variables" }
            ],
            correctAnswer: "c",
            explanation: "Multicollinearity can lead to high variance in coefficient estimates in logistic regression."
          },
          {
            question: "What does an odds ratio greater than 1 indicate in logistic regression?",
            options: [
              { id: "a", text: "Negative effect of predictor" },
              { id: "b", text: "No effect of predictor" },
              { id: "c", text: "Positive effect of predictor" },
              { id: "d", text: "Inconclusive result" }
            ],
            correctAnswer: "c",
            explanation: "An odds ratio > 1 implies that the predictor increases the odds of the target event."
          },
          {
            question: "What technique can be used to prevent overfitting in logistic regression?",
            options: [
              { id: "a", text: "Standardization" },
              { id: "b", text: "Cross-validation" },
              { id: "c", text: "Regularization" },
              { id: "d", text: "Clustering" }
            ],
            correctAnswer: "c",
            explanation: "Regularization (L1 or L2) penalizes large coefficients to prevent overfitting."
          },
          {
            question: "Which of these is a good metric to evaluate logistic regression performance?",
            options: [
              { id: "a", text: "RMSE" },
              { id: "b", text: "Accuracy" },
              { id: "c", text: "Mean Absolute Error" },
              { id: "d", text: "R² Score" }
            ],
            correctAnswer: "b",
            explanation: "Accuracy is commonly used for evaluating classification models like logistic regression."
          },
          {
            question: "What does the decision boundary represent in logistic regression?",
            options: [
              { id: "a", text: "The maximum likelihood" },
              { id: "b", text: "The threshold to separate classes" },
              { id: "c", text: "The number of clusters" },
              { id: "d", text: "The gradient descent path" }
            ],
            correctAnswer: "b",
            explanation: "The decision boundary separates predicted class labels based on probability threshold (often 0.5)."
          },                                                  
    ]
    export const advanced: QuizQuestion[] = [
        {
            question: "What is the effect of using L1 regularization in logistic regression?",
            options: [
              { id: "a", text: "Shrinks all coefficients equally" },
              { id: "b", text: "Performs feature selection by forcing some coefficients to zero" },
              { id: "c", text: "Increases the model complexity" },
              { id: "d", text: "None of the above" }
            ],
            correctAnswer: "b",
            explanation: "L1 regularization (Lasso) tends to produce sparse models by eliminating less important features."
          },
          {
            question: "In logistic regression, how is the log-odds (logit) function defined?",
            options: [
              { id: "a", text: "ln(p)" },
              { id: "b", text: "ln(1 - p)" },
              { id: "c", text: "ln(p / (1 - p))" },
              { id: "d", text: "p²" }
            ],
            correctAnswer: "c",
            explanation: "The logit function is the log of the odds: ln(p / (1 - p)), where p is the probability of the positive class."
          },
          {
            question: "Why is logistic regression considered a linear model?",
            options: [
              { id: "a", text: "Because it fits a linear line to the data" },
              { id: "b", text: "Because it models the target variable linearly" },
              { id: "c", text: "Because it models the log-odds linearly with predictors" },
              { id: "d", text: "Because the sigmoid function is linear" }
            ],
            correctAnswer: "c",
            explanation: "Logistic regression is linear in the parameters with respect to the log-odds of the response."
          },
          {
            question: "Which technique helps in dealing with imbalanced datasets in logistic regression?",
            options: [
              { id: "a", text: "Drop columns with missing values" },
              { id: "b", text: "Oversampling the minority class" },
              { id: "c", text: "Increasing the learning rate" },
              { id: "d", text: "Using PCA" }
            ],
            correctAnswer: "b",
            explanation: "Oversampling the minority class or undersampling the majority helps handle class imbalance."
          },
          {
            question: "How is the optimal set of coefficients found in logistic regression?",
            options: [
              { id: "a", text: "By solving a linear system of equations" },
              { id: "b", text: "By minimizing the squared error" },
              { id: "c", text: "By maximizing the likelihood function using optimization" },
              { id: "d", text: "By random search" }
            ],
            correctAnswer: "c",
            explanation: "Logistic regression uses maximum likelihood estimation (MLE) to find the best-fitting parameters."
          },                                        
    ]