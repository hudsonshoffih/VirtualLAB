import type { QuizQuestion } from "../types"
export const beginner: QuizQuestion[] = [
    {
        question: "What does the 'K' in K-Nearest Neighbors represent?",
        options: [
          { id: "a", text: "The number of features" },
          { id: "b", text: "The number of nearest neighbors to consider" },
          { id: "c", text: "The size of the dataset" },
          { id: "d", text: "The distance metric used" }
        ],
        correctAnswer: "b",
        explanation: "The 'K' in KNN represents the number of nearest neighbors considered for classification or regression."
      },
      {
        question: "What type of machine learning algorithm is KNN?",
        options: [
          { id: "a", text: "Supervised" },
          { id: "b", text: "Unsupervised" },
          { id: "c", text: "Reinforcement" },
          { id: "d", text: "Semi-supervised" }
        ],
        correctAnswer: "a",
        explanation: "KNN is a supervised learning algorithm because it uses labeled data for training."
      },
      {
        question: "Which of the following is a common distance metric used in KNN?",
        options: [
          { id: "a", text: "Euclidean distance" },
          { id: "b", text: "Manhattan distance" },
          { id: "c", text: "Cosine similarity" },
          { id: "d", text: "All of the above" }
        ],
        correctAnswer: "d",
        explanation: "KNN can use various distance metrics, including Euclidean, Manhattan, and Cosine similarity."
      },
      {
        question: "What is the main purpose of KNN?",
        options: [
          { id: "a", text: "To cluster data points" },
          { id: "b", text: "To classify or predict based on nearest neighbors" },
          { id: "c", text: "To reduce dimensionality" },
          { id: "d", text: "To optimize hyperparameters" }
        ],
        correctAnswer: "b",
        explanation: "KNN is used for classification and regression tasks by considering the nearest neighbors."
      },
      {
        question: "Does KNN require a training phase?",
        options: [
          { id: "a", text: "Yes, it trains a model" },
          { id: "b", text: "No, it is a lazy learning algorithm" },
          { id: "c", text: "Yes, but only for regression tasks" },
          { id: "d", text: "No, it uses reinforcement learning" }
        ],
        correctAnswer: "b",
        explanation: "KNN is a lazy learning algorithm, meaning it does not explicitly train a model but uses the entire dataset during prediction."
      },
]
export const intermediate: QuizQuestion[] = [
    {
        question: "How does the choice of 'K' affect the performance of KNN?",
        options: [
          { id: "a", text: "A small K may lead to overfitting" },
          { id: "b", text: "A large K may lead to underfitting" },
          { id: "c", text: "Both a and b" },
          { id: "d", text: "It has no effect" }
        ],
        correctAnswer: "c",
        explanation: "A small K can make the model sensitive to noise, while a large K can smooth out predictions too much."
      },
      {
        question: "What is the curse of dimensionality in KNN?",
        options: [
          { id: "a", text: "It refers to the difficulty of visualizing high-dimensional data" },
          { id: "b", text: "It refers to the exponential increase in computational cost with dimensions" },
          { id: "c", text: "It refers to the sparsity of data in high-dimensional spaces" },
          { id: "d", text: "All of the above" }
        ],
        correctAnswer: "d",
        explanation: "The curse of dimensionality affects KNN as distances become less meaningful in high-dimensional spaces."
      },
      {
        question: "How can ties be resolved in KNN classification?",
        options: [
          { id: "a", text: "By choosing the class with the smallest index" },
          { id: "b", text: "By increasing the value of K" },
          { id: "c", text: "By using weighted voting based on distance" },
          { id: "d", text: "By randomly selecting a class" }
        ],
        correctAnswer: "c",
        explanation: "Weighted voting gives more importance to closer neighbors, helping resolve ties."
      },
      {
        question: "What preprocessing step is crucial for KNN?",
        options: [
          { id: "a", text: "Feature scaling" },
          { id: "b", text: "One-hot encoding" },
          { id: "c", text: "Dimensionality reduction" },
          { id: "d", text: "Removing outliers" }
        ],
        correctAnswer: "a",
        explanation: "Feature scaling ensures that all features contribute equally to the distance calculation."
      },
      {
        question: "How can KNN be used for regression tasks?",
        options: [
          { id: "a", text: "By averaging the target values of the neighbors" },
          { id: "b", text: "By taking the mode of the target values" },
          { id: "c", text: "By clustering the neighbors" },
          { id: "d", text: "By using a decision tree" }
        ],
        correctAnswer: "a",
        explanation: "In regression, KNN predicts the target value by averaging the values of the K nearest neighbors."
      },
]
export const advanced: QuizQuestion[] = [
    {
        question: "What is weighted KNN?",
        options: [
          { id: "a", text: "A version of KNN that assigns weights to features" },
          { id: "b", text: "A version of KNN that assigns weights to neighbors based on distance" },
          { id: "c", text: "A version of KNN that uses weighted averages for regression" },
          { id: "d", text: "A version of KNN that uses weighted voting for classification" }
        ],
        correctAnswer: "b",
        explanation: "Weighted KNN assigns higher weights to closer neighbors, making them more influential in predictions."
      },
      {
        question: "How can KNN handle large datasets efficiently?",
        options: [
          { id: "a", text: "By using dimensionality reduction techniques" },
          { id: "b", text: "By using approximate nearest neighbor search" },
          { id: "c", text: "By using data structures like KD-Trees" },
          { id: "d", text: "All of the above" }
        ],
        correctAnswer: "d",
        explanation: "Techniques like dimensionality reduction, approximate search, and KD-Trees can improve KNN's efficiency."
      },
      {
        question: "What is the impact of feature scaling on KNN?",
        options: [
          { id: "a", text: "It has no impact" },
          { id: "b", text: "It ensures that all features contribute equally to distance calculations" },
          { id: "c", text: "It reduces the number of neighbors considered" },
          { id: "d", text: "It increases the computational cost" }
        ],
        correctAnswer: "b",
        explanation: "Feature scaling ensures that features with larger ranges do not dominate the distance metric."
      },
      {
        question: "How can cross-validation be used in KNN?",
        options: [
          { id: "a", text: "To select the optimal value of K" },
          { id: "b", text: "To evaluate the model's performance" },
          { id: "c", text: "Both a and b" },
          { id: "d", text: "It cannot be used in KNN" }
        ],
        correctAnswer: "c",
        explanation: "Cross-validation helps in selecting the best K and evaluating the model's generalization performance."
      },
      {
        question: "What is the primary limitation of KNN?",
        options: [
          { id: "a", text: "It cannot handle categorical data" },
          { id: "b", text: "It is computationally expensive for large datasets" },
          { id: "c", text: "It cannot be used for regression tasks" },
          { id: "d", text: "It requires a large number of features" }
        ],
        correctAnswer: "b",
        explanation: "KNN requires storing and searching through the entire dataset, making it computationally expensive for large datasets."
      },
]