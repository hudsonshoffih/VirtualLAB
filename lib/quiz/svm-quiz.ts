import type { QuizQuestion } from "../types"

export const beginner: QuizQuestion[] = [
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
]

export const intermediate: QuizQuestion[] = [
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
]

export const advanced: QuizQuestion[] = [
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
]
