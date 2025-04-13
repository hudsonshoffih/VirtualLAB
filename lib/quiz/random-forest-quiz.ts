import type { QuizQuestion } from "../types"

export const beginner: QuizQuestion[] = [
    {
        question: "What is a key advantage of using Random Forest over a single Decision Tree?",
        options: [
          { id: "a", text: "Random Forest runs faster than Decision Trees" },
          { id: "b", text: "Random Forest is less likely to overfit" },
          { id: "c", text: "Random Forest does not require any data preprocessing" },
          { id: "d", text: "Random Forest needs fewer training examples" }
        ],
        correctAnswer: "b",
        explanation: "By combining multiple trees, Random Forest reduces the risk of overfitting and increases accuracy."
      },
      {
        question: "What type of algorithm is Random Forest?",
        options: [
          { id: "a", text: "Linear model" },
          { id: "b", text: "Unsupervised learning" },
          { id: "c", text: "Ensemble learning" },
          { id: "d", text: "Reinforcement learning" }
        ],
        correctAnswer: "c",
        explanation: "Random Forest is an ensemble method that aggregates predictions from multiple decision trees."
      },
      {
        question: "Which of the following is true about Random Forest?",
        options: [
          { id: "a", text: "It always builds the same trees" },
          { id: "b", text: "It randomly selects features and samples for each tree" },
          { id: "c", text: "It uses gradient descent" },
          { id: "d", text: "It needs labeled data and a scoring function" }
        ],
        correctAnswer: "b",
        explanation: "Random Forest builds diverse trees by randomly selecting features and samples (bootstrapping)."
      },
      {
        question: "Which problem is Random Forest NOT typically used for?",
        options: [
          { id: "a", text: "Classification" },
          { id: "b", text: "Regression" },
          { id: "c", text: "Clustering" },
          { id: "d", text: "Feature selection" }
        ],
        correctAnswer: "c",
        explanation: "Random Forest is primarily used for supervised tasks like classification and regression, not clustering."
      },
      {
        question: "What is the output of Random Forest in a classification task?",
        options: [
          { id: "a", text: "The average of all predictions" },
          { id: "b", text: "The sum of all predictions" },
          { id: "c", text: "The most frequent prediction among trees" },
          { id: "d", text: "The prediction from the last tree" }
        ],
        correctAnswer: "c",
        explanation: "Random Forest uses majority voting among decision trees for classification tasks."
      },                        
]
export const intermediate: QuizQuestion[] = [
    {
        question: "How does Random Forest reduce variance compared to a single decision tree?",
        options: [
          { id: "a", text: "By increasing the depth of trees" },
          { id: "b", text: "By pruning trees more aggressively" },
          { id: "c", text: "By averaging predictions from multiple trees" },
          { id: "d", text: "By using gradient descent optimization" }
        ],
        correctAnswer: "c",
        explanation: "Random Forest averages the predictions from several trees, reducing the variance."
      },
      {
        question: "In Random Forest, what does the term 'bootstrap sampling' refer to?",
        options: [
          { id: "a", text: "Using all features in each tree" },
          { id: "b", text: "Using a different target variable" },
          { id: "c", text: "Using random samples with replacement from the training set" },
          { id: "d", text: "Creating only shallow trees" }
        ],
        correctAnswer: "c",
        explanation: "Bootstrap sampling involves creating datasets by sampling with replacement from the original data."
      },
      {
        question: "Which parameter in Random Forest controls the number of trees?",
        options: [
          { id: "a", text: "max_features" },
          { id: "b", text: "n_estimators" },
          { id: "c", text: "max_depth" },
          { id: "d", text: "criterion" }
        ],
        correctAnswer: "b",
        explanation: "`n_estimators` defines how many trees the Random Forest builds."
      },
      {
        question: "Why is feature randomness important in Random Forest?",
        options: [
          { id: "a", text: "To reduce training time" },
          { id: "b", text: "To make each tree identical" },
          { id: "c", text: "To encourage tree diversity and prevent overfitting" },
          { id: "d", text: "To ensure perfect accuracy" }
        ],
        correctAnswer: "c",
        explanation: "Randomly choosing features ensures trees are diverse, improving generalization."
      },
      {
        question: "What is the effect of increasing the number of trees in Random Forest?",
        options: [
          { id: "a", text: "Increases overfitting" },
          { id: "b", text: "Reduces model bias significantly" },
          { id: "c", text: "Leads to more complex single trees" },
          { id: "d", text: "Improves performance up to a point, then saturates" }
        ],
        correctAnswer: "d",
        explanation: "After a certain number of trees, model performance plateaus as additional trees add less value."
      },                              
]
export const advanced: QuizQuestion[] = [
    {
        question: "How can Random Forest be used for feature importance analysis?",
        options: [
          { id: "a", text: "By counting the number of splits each feature makes" },
          { id: "b", text: "By calculating Gini impurity or information gain reduction from splits" },
          { id: "c", text: "By ignoring unused features" },
          { id: "d", text: "By increasing tree depth" }
        ],
        correctAnswer: "b",
        explanation: "Random Forest estimates feature importance by evaluating the decrease in impurity when splitting."
      },
      {
        question: "Which of the following can lead to Random Forest overfitting?",
        options: [
          { id: "a", text: "Too few trees" },
          { id: "b", text: "High correlation between trees" },
          { id: "c", text: "Using pruning" },
          { id: "d", text: "Shallow trees" }
        ],
        correctAnswer: "b",
        explanation: "If trees are highly correlated, the ensemble loses its diversity and may overfit."
      },
      {
        question: "How does Random Forest handle missing values in datasets?",
        options: [
          { id: "a", text: "It drops them automatically" },
          { id: "b", text: "It predicts them using regression" },
          { id: "c", text: "It uses surrogate splits or ignores them" },
          { id: "d", text: "It replaces them with zeros" }
        ],
        correctAnswer: "c",
        explanation: "Some implementations use surrogate splits or ignore missing values during training."
      },
      {
        question: "You want to use Random Forest for imbalanced classification. What strategy helps improve performance?",
        options: [
          { id: "a", text: "Increase the number of trees" },
          { id: "b", text: "Use class weights or sampling techniques" },
          { id: "c", text: "Add more features" },
          { id: "d", text: "Lower max_depth" }
        ],
        correctAnswer: "b",
        explanation: "Class weights or techniques like SMOTE help handle class imbalance more effectively."
      },
      {
        question: "How does Random Forest reduce the risk of overfitting compared to a single Decision Tree?",
        options: [
          { id: "a", text: "By using regularization" },
          { id: "b", text: "By learning from reinforcement feedback" },
          { id: "c", text: "By combining results from diverse trees built on different samples and features" },
          { id: "d", text: "By increasing model complexity" }
        ],
        correctAnswer: "c",
        explanation: "Random Forest lowers overfitting by aggregating uncorrelated trees trained on different parts of the data."
      },                              
]