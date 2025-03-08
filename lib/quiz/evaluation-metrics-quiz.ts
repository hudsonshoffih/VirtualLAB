import type { QuizQuestion } from "../types"

export const beginner: QuizQuestion[] = [
  {
    question: "What is the purpose of evaluation metrics in machine learning?",
    options: [
      { id: "a", text: "To determine how long a model takes to train" },
      { id: "b", text: "To assess how well a model performs on a given task" },
      { id: "c", text: "To calculate the computational resources needed" },
      { id: "d", text: "To visualize the model architecture" },
    ],
    correctAnswer: "b",
    explanation:
      "Evaluation metrics provide quantitative measures to assess how well a machine learning model performs on a specific task, helping to compare different models and tune hyperparameters.",
  },
  {
    question: "Which metric is commonly used to evaluate regression models?",
    options: [
      { id: "a", text: "Accuracy" },
      { id: "b", text: "Mean Squared Error (MSE)" },
      { id: "c", text: "F1 Score" },
      { id: "d", text: "Precision" },
    ],
    correctAnswer: "b",
    explanation:
      "Mean Squared Error (MSE) is a common evaluation metric for regression models. It calculates the average of the squared differences between predicted and actual values, penalizing larger errors more heavily.",
  },
  {
    question: "What does accuracy measure in classification problems?",
    options: [
      { id: "a", text: "The percentage of correct predictions out of all predictions" },
      { id: "b", text: "The percentage of correctly identified positive instances" },
      { id: "c", text: "The average error in predictions" },
      { id: "d", text: "The computational efficiency of the model" },
    ],
    correctAnswer: "a",
    explanation:
      "Accuracy measures the percentage of correct predictions (both true positives and true negatives) out of all predictions made by the model. It is calculated as (TP + TN) / (TP + TN + FP + FN).",
  },
  {
    question: "What is a confusion matrix used for?",
    options: [
      { id: "a", text: "To display model architecture" },
      { id: "b", text: "To show the distribution of training data" },
      { id: "c", text: "To summarize the performance of a classification model" },
      { id: "d", text: "To measure computational complexity" },
    ],
    correctAnswer: "c",
    explanation:
      "A confusion matrix is a table that summarizes the performance of a classification model by showing the counts of true positives, false positives, true negatives, and false negatives, allowing for the calculation of various evaluation metrics.",
  },
  {
    question: "When is R-squared (coefficient of determination) typically used?",
    options: [
      { id: "a", text: "In clustering algorithms" },
      { id: "b", text: "In regression models" },
      { id: "c", text: "In classification models" },
      { id: "d", text: "In reinforcement learning" },
    ],
    correctAnswer: "b",
    explanation:
      "R-squared (coefficient of determination) is typically used to evaluate regression models. It represents the proportion of variance in the dependent variable that is predictable from the independent variables.",
  },
]

export const intermediate: QuizQuestion[] = [
  {
    question: "When would precision be a more important metric than recall?",
    options: [
      { id: "a", text: "When the cost of false positives is higher than false negatives" },
      { id: "b", text: "When the cost of false negatives is higher than false positives" },
      { id: "c", text: "When the dataset is perfectly balanced" },
      { id: "d", text: "When computational efficiency is the primary concern" },
    ],
    correctAnswer: "a",
    explanation:
      "Precision is more important when the cost of false positives is higher than the cost of false negatives. For example, in spam detection, classifying legitimate emails as spam (false positives) can be more problematic than letting some spam through (false negatives).",
  },
  {
    question: "What does the ROC curve illustrate?",
    options: [
      { id: "a", text: "The relationship between training time and model accuracy" },
      { id: "b", text: "The trade-off between true positive rate and false positive rate at different threshold settings" },
      { id: "c", text: "The learning rate over training epochs" },
      { id: "d", text: "The relationship between model complexity and error" },
    ],
    correctAnswer: "b",
    explanation:
      "The Receiver Operating Characteristic (ROC) curve illustrates the trade-off between the true positive rate (sensitivity) and false positive rate (1-specificity) at various threshold settings, providing a graphical representation of a classifier's performance across different decision thresholds.",
  },
  {
    question: "What does AUC-ROC stand for and what does it measure?",
    options: [
      { id: "a", text: "Area Under Curve - Rate of Change; measures how quickly a model learns" },
      { id: "b", text: "Area Under Curve - Receiver Operating Characteristic; measures the ability of a classifier to distinguish between classes" },
      { id: "c", text: "Average Use Case - Return On Cost; measures the efficiency of model training" },
      { id: "d", text: "Area Under Curve - Recovery of Components; measures feature importance" },
    ],
    correctAnswer: "b",
    explanation:
      "AUC-ROC stands for Area Under the Receiver Operating Characteristic curve. It measures the ability of a classification model to distinguish between classes. A higher AUC indicates better model performance, with 1.0 being perfect classification and 0.5 being no better than random guessing.",
  },
  {
    question: "What is the F1 score?",
    options: [
      { id: "a", text: "The arithmetic mean of precision and recall" },
      { id: "b", text: "The harmonic mean of precision and recall" },
      { id: "c", text: "The geometric mean of precision and recall" },
      { id: "d", text: "The larger of precision or recall" },
    ],
    correctAnswer: "b",
    explanation:
      "The F1 score is the harmonic mean of precision and recall, calculated as 2 * (precision * recall) / (precision + recall). It provides a balance between precision and recall, especially useful when class distribution is imbalanced.",
  },
  {
    question: "What is the purpose of k-fold cross-validation?",
    options: [
      { id: "a", text: "To speed up model training by using fewer data points" },
      { id: "b", text: "To provide a more reliable measure of model performance by using multiple test sets" },
      { id: "c", text: "To reduce model complexity" },
      { id: "d", text: "To increase the size of the training dataset" },
    ],
    correctAnswer: "b",
    explanation:
      "k-fold cross-validation splits the data into k subsets (folds), then trains and evaluates the model k times, each time using a different fold as the test set and the remaining folds as the training set. This provides a more reliable estimate of model performance by reducing the variance of the evaluation metric.",
  },
]

export const advanced: QuizQuestion[] = [
  {
    question: "What is the Matthews Correlation Coefficient (MCC) and when is it particularly useful?",
    options: [
      { id: "a", text: "It measures the speed of model convergence; useful for comparing training efficiency" },
      { id: "b", text: "It's a measure of correlation between features; useful for feature selection" },
      { id: "c", text: "It's a balanced measure for binary classification; useful for imbalanced datasets" },
      { id: "d", text: "It measures the complexity of a model; useful for avoiding overfitting" },
    ],
    correctAnswer: "c",
    explanation:
      "The Matthews Correlation Coefficient (MCC) is a balanced measure for binary classification that takes into account all four values in the confusion matrix (TP, TN, FP, FN). It returns a value between -1 and +1, where +1 represents perfect prediction, 0 represents random prediction, and -1 represents complete disagreement. It's particularly useful for imbalanced datasets where accuracy can be misleading.",
  },
  {
    question: "What is the Kappa statistic (Cohen's Kappa) used for?",
    options: [
      { id: "a", text: "To measure agreement between model predictions and random predictions" },
      { id: "b", text: "To compare two different models" },
      { id: "c", text: "To evaluate clustering algorithms" },
      { id: "d", text: "To measure the computational efficiency of a model" },
    ],
    correctAnswer: "a",
    explanation:
      "Cohen's Kappa measures the agreement between model predictions and actual values, corrected for the agreement expected by chance alone. It provides a more robust measure than simple accuracy, especially for imbalanced datasets, as it takes into account the possibility of agreement occurring by chance. A Kappa of 1 indicates perfect agreement, 0 indicates agreement equivalent to chance, and negative values indicate worse than chance agreement."
  },
  {
    question: "What is the Brier score and when would you use it?",
    options: [
      { id: "a", text: "A measure of calibration for classification problems; used to assess probabilistic predictions" },
      { id: "b", text: "A measure of clustering quality; used in unsupervised learning" },
      { id: "c", text: "A metric for reinforcement learning; used to evaluate policy performance" },
      { id: "d", text: "A measure of model complexity; used to prevent overfitting" },
    ],
    correctAnswer: "a",
    explanation:
      "The Brier score is a measure of the accuracy of probabilistic predictions in classification problems. It's calculated as the mean squared difference between predicted probabilities and the actual outcomes (0 or 1). Lower values indicate better calibrated predictions. It's particularly useful when you're interested not just in the class predictions but in the quality of the predicted probabilities.",
  },
]