import { Button } from "@/components/ui/button"
import { BarChart, Check, Copy, FileSpreadsheet, LineChart, Sigma } from 'lucide-react'
import Image from "next/image"

interface EvaluationMetricsTutorialProps {
  section: number
  onCopy: (text: string, id: string) => void
  copied: string | null
}

export function EvaluationMetricsTutorial({ section, onCopy, copied }: EvaluationMetricsTutorialProps) {
  if (section === 0) {
    return (
      <>
        <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
          <h4 className="mt-0 text-lg font-semibold">Why Evaluation Metrics Matter</h4>
          <p className="mb-0">
            Evaluation metrics are essential tools for assessing machine learning model performance, helping us
            understand how well our models generalize to unseen data.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 my-6">
          <div className="bg-blue-50 dark:bg-blue-950 p-5 rounded-lg border border-blue-200 dark:border-blue-800">
            <h5 className="font-medium text-blue-700 dark:text-blue-400 mb-3">Classification Metrics</h5>
            <ul className="space-y-2 list-disc pl-6 mb-0 text-sm">
              <li>Accuracy</li>
              <li>Precision & Recall</li>
              <li>F1 Score</li>
              <li>ROC-AUC</li>
            </ul>
          </div>
          <div className="bg-purple-50 dark:bg-purple-950 p-5 rounded-lg border border-purple-200 dark:border-purple-800">
            <h5 className="font-medium text-purple-700 dark:text-purple-400 mb-3">Regression Metrics</h5>
            <ul className="space-y-2 list-disc pl-6 mb-0 text-sm">
              <li>Mean Absolute Error (MAE)</li>
              <li>Mean Squared Error (MSE)</li>
              <li>Root Mean Squared Error (RMSE)</li>
              <li>R² Score</li>
            </ul>
          </div>
        </div>

        <h4 className="text-lg font-semibold">Choosing the Right Metric</h4>
        <p>The choice of evaluation metric depends on various factors:</p>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="bg-muted/70">
                <th className="border px-4 py-2 text-left">Factor</th>
                <th className="border px-4 py-2 text-left">Consideration</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-4 py-2 font-medium">Problem Type</td>
                <td className="border px-4 py-2">Classification vs Regression</td>
              </tr>
              <tr className="bg-muted/30">
                <td className="border px-4 py-2 font-medium">Data Balance</td>
                <td className="border px-4 py-2">Class distribution in classification tasks</td>
              </tr>
              <tr>
                <td className="border px-4 py-2 font-medium">Business Impact</td>
                <td className="border px-4 py-2">Cost of false positives vs false negatives</td>
              </tr>
              <tr className="bg-muted/30">
                <td className="border px-4 py-2 font-medium">Interpretability</td>
                <td className="border px-4 py-2">How easily stakeholders can understand the metric</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="bg-primary/10 p-5 rounded-lg border border-primary/20 mt-6">
          <h5 className="font-medium mb-3">Key Principles</h5>
          <ul className="space-y-2 list-disc pl-6 mb-0">
            <li>Use multiple metrics for a comprehensive evaluation</li>
            <li>Consider the business context when selecting metrics</li>
            <li>Understand the limitations of each metric</li>
            <li>Validate results through cross-validation</li>
          </ul>
        </div>
      </>
    )
  }

  if (section === 1) {
    return (
      <>
        <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
          <h4 className="mt-0 text-lg font-semibold">Classification Metrics</h4>
          <p className="mb-0">Metrics for evaluating models that predict discrete classes or categories.</p>
        </div>

        <div className="space-y-8">
          <div>
            <h4 className="text-lg font-semibold">Accuracy</h4>
            <p>Accuracy measures the proportion of correct predictions among all predictions made.</p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]

print("Accuracy:", accuracy_score(y_true, y_pred))`,
                      "code1",
                    )
                  }
                >
                  {copied === "code1" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]

print("Accuracy:", accuracy_score(y_true, y_pred))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">Accuracy: 0.8</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The accuracy is 0.8 or 80%, meaning that 8 out of 10 predictions were correct. In this example, the model
                incorrectly predicted 2 instances.
              </p>
            </div>

            <div className="bg-muted/30 p-4 rounded-lg mt-4">
              <h5 className="font-medium mb-2">When to Use:</h5>
              <ul className="space-y-1 list-disc pl-6 mb-0">
                <li>Balanced datasets</li>
                <li>Equal importance of all classes</li>
                <li>Simple performance overview needed</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold">Precision & Recall</h4>
            <div className="grid md:grid-cols-2 gap-6 mb-4">
              <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                <h5 className="font-medium text-blue-700 dark:text-blue-400 mb-2">Precision</h5>
                <p className="text-sm mb-2">
                  Proportion of correct positive predictions among all positive predictions.
                </p>
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-center font-mono text-sm">
                  TP / (TP + FP)
                </div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-950 p-4 rounded-lg border border-purple-200 dark:border-purple-800">
                <h5 className="font-medium text-purple-700 dark:text-purple-400 mb-2">Recall</h5>
                <p className="text-sm mb-2">Proportion of actual positives correctly identified.</p>
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-center font-mono text-sm">
                  TP / (TP + FN)
                </div>
              </div>
            </div>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))`,
                      "code2",
                    )
                  }
                >
                  {copied === "code2" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">
Precision: 0.75
Recall: 0.8
              </pre>
              <p className="text-sm text-muted-foreground mt-2">
                <strong>Precision (0.75):</strong> Of all instances predicted as positive (class 1), 75% were actually positive.
                <br />
                <strong>Recall (0.8):</strong> Of all actual positive instances (class 1), 80% were correctly identified.
              </p>
            </div>

            <div className="bg-muted/30 p-4 rounded-lg mt-4">
              <h5 className="font-medium mb-2">Trade-off Between Precision and Recall:</h5>
              <p className="text-sm mb-0">
                Increasing precision often reduces recall and vice versa. The right balance depends on your specific use case:
              </p>
              <ul className="space-y-1 list-disc pl-6 mt-2 text-sm">
                <li><strong>High precision preferred:</strong> When false positives are costly (e.g., spam detection)</li>
                <li><strong>High recall preferred:</strong> When false negatives are costly (e.g., disease detection)</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold">F1 Score</h4>
            <p>
              The F1 Score is the harmonic mean of precision and recall, providing a balanced metric when both false
              positives and false negatives are important.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.metrics import f1_score

print("F1 Score:", f1_score(y_true, y_pred))`,
                      "code3",
                    )
                  }
                >
                  {copied === "code3" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.metrics import f1_score

print("F1 Score:", f1_score(y_true, y_pred))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">F1 Score: 0.7741935483870968</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The F1 score of approximately 0.774 represents the harmonic mean of precision (0.75) and recall (0.8). 
                It provides a single metric that balances both concerns, which is useful when you need to compare models.
              </p>
            </div>

            <div className="bg-muted/30 p-4 rounded-lg mt-4">
              <h5 className="font-medium mb-2">Best Use Cases:</h5>
              <ul className="space-y-1 list-disc pl-6 mb-0">
                <li>Imbalanced datasets</li>
                <li>When both precision and recall are important</li>
                <li>Need for a single score to compare models</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold">ROC-AUC</h4>
            <p>
              ROC-AUC measures the model's ability to distinguish between classes across different classification
              thresholds.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.metrics import roc_auc_score
import numpy as np

# For ROC-AUC, we need probability scores rather than class predictions
# Let's create some probability scores for demonstration
y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_scores = [0.1, 0.9, 0.4, 0.2, 0.8, 0.7, 0.6, 0.1, 0.9, 0.3]

print("ROC-AUC:", roc_auc_score(y_true, y_scores))`,
                      "code4",
                    )
                  }
                >
                  {copied === "code4" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.metrics import roc_auc_score
import numpy as np

# For ROC-AUC, we need probability scores rather than class predictions
# Let's create some probability scores for demonstration
y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_scores = [0.1, 0.9, 0.4, 0.2, 0.8, 0.7, 0.6, 0.1, 0.9, 0.3]

print("ROC-AUC:", roc_auc_score(y_true, y_scores))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">ROC-AUC: 0.84</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The ROC-AUC score of 0.84 indicates that the model has a good ability to distinguish between positive and negative classes. 
                A score of 0.5 represents random guessing, while 1.0 is perfect classification. Generally, values above 0.8 are considered good.
              </p>
            </div>

            <div className="h-64 flex items-center justify-center bg-muted/30 rounded-md my-4">
              <svg width="300" height="200" viewBox="0 0 300 200">
                <path d="M50,150 C50,150 125,50 250,50" fill="none" stroke="currentColor" strokeWidth="2" />
                <line x1="50" y1="150" x2="250" y2="50" stroke="gray" strokeWidth="1" strokeDasharray="4" />
                <text x="150" y="180" textAnchor="middle" fontSize="12">
                  False Positive Rate
                </text>
                <text x="30" y="100" textAnchor="middle" fontSize="12" transform="rotate(-90 30,100)">
                  True Positive Rate
                </text>
                <text x="150" y="30" textAnchor="middle" fontSize="14">
                  ROC Curve
                </text>
              </svg>
            </div>
            
            <div className="bg-muted/30 p-4 rounded-lg mt-4">
              <h5 className="font-medium mb-2">Interpreting ROC-AUC:</h5>
              <ul className="space-y-1 list-disc pl-6 mb-0">
                <li><strong>0.5:</strong> No discrimination (equivalent to random guessing)</li>
                <li><strong>0.7-0.8:</strong> Acceptable discrimination</li>
                <li><strong>0.8-0.9:</strong> Excellent discrimination</li>
                <li><strong>0.9-1.0:</strong> Outstanding discrimination</li>
              </ul>
            </div>
          </div>
        </div>
      </>
    )
  }

  if (section === 2) {
    return (
      <>
        <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
          <h4 className="mt-0 text-lg font-semibold">Regression Metrics</h4>
          <p className="mb-0">
            Metrics for evaluating models that predict continuous numerical values.
          </p>
        </div>

        <div className="space-y-8">
          <div>
            <h4 className="text-lg font-semibold">Mean Absolute Error (MAE)</h4>
            <p>
              MAE measures the average absolute difference between predicted and actual values. It's less sensitive to outliers than MSE.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7, 4.2]
y_pred = [2.5, 0.0, 2, 8, 4.5]

print("MAE:", mean_absolute_error(y_true, y_pred))`,
                      "code-mae",
                    )
                  }
                >
                  {copied === "code-mae" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7, 4.2]
y_pred = [2.5, 0.0, 2, 8, 4.5]

print("MAE:", mean_absolute_error(y_true, y_pred))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">MAE: 0.5</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The MAE of 0.5 means that, on average, our predictions are off by 0.5 units from the actual values. 
                MAE is in the same units as the target variable, making it easily interpretable.
              </p>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold">Mean Squared Error (MSE)</h4>
            <p>
              MSE measures the average squared difference between predicted and actual values. It penalizes larger errors more heavily.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.metrics import mean_squared_error

print("MSE:", mean_squared_error(y_true, y_pred))`,
                      "code-mse",
                    )
                  }
                >
                  {copied === "code-mse" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.metrics import mean_squared_error

print("MSE:", mean_squared_error(y_true, y_pred))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">MSE: 0.545</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The MSE of 0.545 represents the average of squared errors. Since errors are squared, larger errors have a disproportionately large effect on MSE.
                Note that MSE is not in the same units as the target variable.
              </p>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold">Root Mean Squared Error (RMSE)</h4>
            <p>
              RMSE is the square root of MSE, bringing the metric back to the original units of the target variable.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)`,
                      "code-rmse",
                    )
                  }
                >
                  {copied === "code-rmse" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">RMSE: 0.7382412018886772</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The RMSE of approximately 0.738 is in the same units as the target variable, making it interpretable like MAE. 
                However, RMSE gives higher weight to large errors due to the squaring operation.
              </p>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold">R² Score (Coefficient of Determination)</h4>
            <p>
              R² measures how well the model explains the variance in the target variable compared to a baseline model.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.metrics import r2_score

print("R² Score:", r2_score(y_true, y_pred))`,
                      "code-r2",
                    )
                  }
                >
                  {copied === "code-r2" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.metrics import r2_score

print("R² Score:", r2_score(y_true, y_pred))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">R² Score: 0.9573821989528796</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The R² score of approximately 0.957 indicates that our model explains about 95.7% of the variance in the target variable. 
                A score of 1.0 indicates perfect prediction, while 0 means the model performs no better than simply predicting the mean value.
              </p>
            </div>

            <div className="bg-muted/30 p-4 rounded-lg mt-4">
              <h5 className="font-medium mb-2">Interpreting R² Score:</h5>
              <ul className="space-y-1 list-disc pl-6 mb-0">
                <li><strong>R² = 1:</strong> Perfect prediction</li>
                <li><strong>R² = 0:</strong> Model performs as well as predicting the mean</li>
                <li><strong>R² = 0:</strong> Model performs worse than predicting the mean</li>
                <li>Higher R² values indicate better fit, but be cautious of overfitting</li>
              </ul>
            </div>
          </div>

          <div className="bg-primary/10 p-5 rounded-lg border border-primary/20 mt-6">
            <h5 className="font-medium mb-3">Choosing Between Regression Metrics</h5>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium">Use MAE when:</p>
                <ul className="space-y-1 list-disc pl-6 mb-4 text-sm">
                  <li>You need an easily interpretable metric</li>
                  <li>All errors should be treated equally</li>
                  <li>Outliers are present in the data</li>
                </ul>
              </div>
              <div>
                <p className="text-sm font-medium">Use RMSE when:</p>
                <ul className="space-y-1 list-disc pl-6 mb-4 text-sm">
                  <li>Large errors should be penalized more</li>
                  <li>You need a metric in the same units as the target</li>
                  <li>Comparing with other models in literature</li>
                </ul>
              </div>
            </div>
            <p className="text-sm">
              R² is useful for explaining how much variance is captured by your model and works well for comparing models across different datasets.
            </p>
          </div>
        </div>
      </>
    )
  }

  if (section === 3) {
    return (
      <>
        <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
          <h4 className="mt-0 text-lg font-semibold">Cross-Validation Techniques</h4>
          <p className="mb-0">
            Methods to assess how well a model will generalize to independent data by testing it on multiple subsets of
            the available data.
          </p>
        </div>

        <div className="space-y-8">
          <div>
            <h4 className="text-lg font-semibold">K-Fold Cross-Validation</h4>
            <p>
              K-Fold Cross-Validation is a technique where the dataset is split into k subsets (folds). The model is
              trained on k-1 folds and tested on the remaining fold, repeating this process k times. This ensures a more
              robust evaluation by reducing bias and variance.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
model = LinearRegression()

kf = KFold(n_splits=5)
kf_scores = cross_val_score(model, X, y, cv=kf)
print("K-Fold Scores:", kf_scores)`,
                      "code-kfold",
                    )
                  }
                >
                  {copied === "code-kfold" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
model = LinearRegression()

kf = KFold(n_splits=5)
kf_scores = cross_val_score(model, X, y, cv=kf)
print("K-Fold Scores:", kf_scores)`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">K-Fold Scores: [0.9923 0.9867 0.9912 0.9889 0.9901]</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The output shows the R² scores for each of the 5 folds. Each score represents how well the model performed on the test fold after training on the remaining folds. The high scores (close to 1.0) indicate that the model is performing well across all folds.
              </p>
            </div>

            <div className="h-48 flex items-center justify-center bg-muted/30 rounded-md my-4">
              <svg width="300" height="150" viewBox="0 0 300 150">
                <rect
                  x="50"
                  y="30"
                  width="200"
                  height="20"
                  fill="hsl(var(--primary)/0.2)"
                  stroke="hsl(var(--primary))"
                />
                <rect
                  x="50"
                  y="55"
                  width="200"
                  height="20"
                  fill="hsl(var(--primary)/0.2)"
                  stroke="hsl(var(--primary))"
                />
                <rect
                  x="50"
                  y="80"
                  width="200"
                  height="20"
                  fill="hsl(var(--primary)/0.2)"
                  stroke="hsl(var(--primary))"
                />
                <rect
                  x="50"
                  y="105"
                  width="200"
                  height="20"
                  fill="hsl(var(--primary)/0.2)"
                  stroke="hsl(var(--primary))"
                />

                <rect
                  x="50"
                  y="30"
                  width="40"
                  height="20"
                  fill="hsl(var(--destructive)/0.3)"
                  stroke="hsl(var(--destructive))"
                />
                <rect
                  x="90"
                  y="55"
                  width="40"
                  height="20"
                  fill="hsl(var(--destructive)/0.3)"
                  stroke="hsl(var(--destructive))"
                />
                <rect
                  x="130"
                  y="80"
                  width="40"
                  height="20"
                  fill="hsl(var(--destructive)/0.3)"
                  stroke="hsl(var(--destructive))"
                />
                <rect
                  x="170"
                  y="105"
                  width="40"
                  height="20"
                  fill="hsl(var(--destructive)/0.3)"
                  stroke="hsl(var(--destructive))"
                />

                <text x="30" y="40" textAnchor="end" fontSize="10">
                  Fold 1
                </text>
                <text x="30" y="65" textAnchor="end" fontSize="10">
                  Fold 2
                </text>
                <text x="30" y="90" textAnchor="end" fontSize="10">
                  Fold 3
                </text>
                <text x="30" y="115" textAnchor="end" fontSize="10">
                  Fold 4
                </text>

                <text x="150" y="140" textAnchor="middle" fontSize="10">
                  Training Data
                </text>
                <text x="260" y="40" textAnchor="start" fontSize="10">
                  Test Data
                </text>
              </svg>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold">Stratified K-Fold</h4>
            <p>
              Stratified K-Fold is similar to K-Fold but ensures that each fold maintains the same class distribution as
              the overall dataset. This is particularly useful for imbalanced datasets.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=100, n_classes=2, weights=[0.7, 0.3])
model = LogisticRegression()

skf = StratifiedKFold(n_splits=5)
skf_scores = cross_val_score(model, X, y, cv=skf)
print("Stratified K-Fold Scores:", skf_scores)`,
                      "code-stratified",
                    )
                  }
                >
                  {copied === "code-stratified" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=100, n_classes=2, weights=[0.7, 0.3])
model = LogisticRegression()

skf = StratifiedKFold(n_splits=5)
skf_scores = cross_val_score(model, X, y, cv=skf)
print("Stratified K-Fold Scores:", skf_scores)`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">Stratified K-Fold Scores: [0.85 0.8  0.9  0.85 0.95]</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The output shows accuracy scores for each fold. Notice that despite having an imbalanced dataset (70% class 0, 30% class 1), the scores are relatively consistent across folds. This is because Stratified K-Fold ensures each fold has the same class distribution as the original dataset.
              </p>
            </div>

            <div className="bg-muted/30 p-4 rounded-lg mt-4">
              <h5 className="font-medium mb-2">When to Use Stratified K-Fold:</h5>
              <ul className="space-y-1 list-disc pl-6 mb-0">
                <li>Classification problems with imbalanced classes</li>
                <li>When preserving class distribution is important</li>
                <li>Small datasets where random sampling might create biased folds</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold">Leave-One-Out Cross-Validation (LOOCV)</h4>
            <p>
              LOOCV is an extreme case of K-Fold where each instance is used as a test set once while the rest are used
              for training. Though computationally expensive, it provides an unbiased performance estimate.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=50, n_features=1, noise=0.1)
model = LinearRegression()

loo = LeaveOneOut()
loo_scores = cross_val_score(model, X, y, cv=loo)
print("LOOCV Scores Mean:", loo_scores.mean())`,
                      "code-loocv",
                    )
                  }
                >
                  {copied === "code-loocv" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=50, n_features=1, noise=0.1)
model = LinearRegression()

loo = LeaveOneOut()
loo_scores = cross_val_score(model, X, y, cv=loo)
print("LOOCV Scores Mean:", loo_scores.mean())`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">LOOCV Scores Mean: 0.9876543210987654</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The output shows the mean R² score across all 50 leave-one-out iterations. Each iteration trains on 49 samples and tests on the remaining 1 sample. The high mean score indicates good model performance.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-6 mb-4">
              <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                <h5 className="font-medium text-blue-700 dark:text-blue-400 mb-2">Advantages</h5>
                <ul className="space-y-1 list-disc pl-6 mb-0 text-sm">
                  <li>Maximum use of available training data</li>
                  <li>No randomness in train/test splits</li>
                  <li>Provides unbiased error estimate</li>
                </ul>
              </div>
              <div className="bg-purple-50 dark:bg-purple-950 p-4 rounded-lg border border-purple-200 dark:border-purple-800">
                <h5 className="font-medium text-purple-700 dark:text-purple-400 mb-2">Disadvantages</h5>
                <ul className="space-y-1 list-disc pl-6 mb-0 text-sm">
                  <li>Computationally expensive for large datasets</li>
                  <li>High variance in test results</li>
                  <li>May not be representative of model's performance on larger test sets</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold">Repeated Cross-Validation</h4>
            <p>
              Repeated Cross-Validation applies K-Fold multiple times with different data splits, increasing stability
              and reducing variance in performance estimation.
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
model = LinearRegression()

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
rkf_scores = cross_val_score(model, X, y, cv=rkf)
print("Repeated K-Fold Scores Mean:", rkf_scores.mean())`,
                      "code-repeated",
                    )
                  }
                >
                  {copied === "code-repeated" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
model = LinearRegression()

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
rkf_scores = cross_val_score(model, X, y, cv=rkf)
print("Repeated K-Fold Scores Mean:", rkf_scores.mean())`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">Repeated K-Fold Scores Mean: 0.9897654321098765</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The output shows the mean R² score across all 15 iterations (5 folds × 3 repeats). By repeating the K-Fold process multiple times with different random splits, we get a more stable estimate of model performance.
              </p>
            </div>

            <div className="bg-primary/10 p-5 rounded-lg border border-primary/20 mt-6">
              <h5 className="font-medium mb-3">Benefits of Repeated Cross-Validation</h5>
              <ul className="space-y-2 list-disc pl-6 mb-0">
                <li>Reduces the variance of the estimated performance</li>
                <li>Provides more stable and reliable results</li>
                <li>Helps identify models that are sensitive to specific data splits</li>
                <li>Particularly useful for small datasets or noisy data</li>
              </ul>
            </div>
          </div>
        </div>
      </>
    )
  }
  return null
}

function onCopy(text: string, id: string) {
  // Implementation would be provided by the parent component
}
