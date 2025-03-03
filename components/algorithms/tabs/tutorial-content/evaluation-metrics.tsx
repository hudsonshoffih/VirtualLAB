"use client"

import { useEffect, useState } from "react"
import {
  BookOpen,
  Target,
  LineChart,
  Repeat,
  Lightbulb,
  CheckCircle2,
  Copy,
  Check,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import { Card } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

interface EvaluationMetricsProps {
  section?: number
  onCopy?: (text: string, id: string) => void
  copied?: string | null
}

export function EvaluationMetrics({
  section: initialSection = 0,
  onCopy,
  copied: externalCopied,
}: EvaluationMetricsProps) {
  const [loading, setLoading] = useState(true)
  const [currentSection, setCurrentSection] = useState(initialSection)
  const [copied, setCopied] = useState<string | null>(externalCopied || null)

  // Define sections
  const sections = [
    { title: "Introduction to Metrics", icon: <BookOpen className="h-4 w-4" /> },
    { title: "Classification Metrics", icon: <Target className="h-4 w-4" /> },
    { title: "Regression Metrics", icon: <LineChart className="h-4 w-4" /> },
    { title: "Cross-Validation", icon: <Repeat className="h-4 w-4" /> },
    { title: "Practical Applications", icon: <Lightbulb className="h-4 w-4" /> },
    { title: "Best Practices", icon: <CheckCircle2 className="h-4 w-4" /> },
  ]

  // Simulating fetching tutorial content
  useEffect(() => {
    setTimeout(() => {
      setLoading(false)
    }, 1000)
  }, [])

  // Function to copy code to clipboard
  const copyToClipboard = (text: string, id: string) => {
    if (onCopy) {
      onCopy(text, id)
    } else {
      navigator.clipboard.writeText(text)
      setCopied(id)
      setTimeout(() => setCopied(null), 2000)
    }
  }

  // Calculate progress
  const progress = ((currentSection + 1) / sections.length) * 100

  return (
    <div className="space-y-6 py-4">
      {/* Header with progress */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold">Machine Learning Evaluation Metrics</h2>
          <Badge variant="outline" className="px-3 py-1">
            {currentSection + 1} of {sections.length}
          </Badge>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Navigation sidebar */}
        <div className="md:col-span-1">
          <Card className="p-4 sticky top-4">
            <h3 className="font-medium mb-4 text-center border-b pb-2">Learning Path</h3>
            <ul className="space-y-2">
              {sections.map((section, index) => (
                <li key={index}>
                  <Button
                    variant={currentSection === index ? "default" : "ghost"}
                    className={`w-full justify-start text-sm h-10 ${
                      index < currentSection ? "text-muted-foreground" : ""
                    }`}
                    onClick={() => setCurrentSection(index)}
                  >
                    <div className="flex items-center gap-2">
                      {section.icon}
                      <span>{section.title}</span>
                    </div>
                    {index < currentSection && <Check className="h-4 w-4 ml-auto text-green-500" />}
                  </Button>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        {/* Content area */}
        <div className="md:col-span-3">
          <Card className="p-6">
            {loading ? (
              <div className="space-y-4">
                <Skeleton className="h-8 w-3/4" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-32 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
              </div>
            ) : (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  {sections[currentSection].icon}
                  <h3 className="text-xl font-bold">{sections[currentSection].title}</h3>
                </div>

                <div className="prose dark:prose-invert max-w-none">
                  {currentSection === 0 && (
                    <>
                      <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
                        <h4 className="mt-0 text-lg font-semibold">Why Evaluation Metrics Matter</h4>
                        <p className="mb-0">
                          Evaluation metrics are essential tools for assessing machine learning model performance,
                          helping us understand how well our models generalize to unseen data.
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
                  )}

                  {currentSection === 1 && (
                    <>
                      <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
                        <h4 className="mt-0 text-lg font-semibold">Classification Metrics</h4>
                        <p className="mb-0">
                          Metrics for evaluating models that predict discrete classes or categories.
                        </p>
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
                                  copyToClipboard(
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
                                  copyToClipboard(
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
                        </div>

                        <div>
                          <h4 className="text-lg font-semibold">F1 Score</h4>
                          <p>
                            The F1 Score is the harmonic mean of precision and recall, providing a balanced metric when
                            both false positives and false negatives are important.
                          </p>

                          <div className="relative bg-black rounded-md my-4 group">
                            <div className="absolute right-2 top-2">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-gray-400 hover:text-white"
                                onClick={() =>
                                  copyToClipboard(
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
                            ROC-AUC measures the model's ability to distinguish between classes across different
                            classification thresholds.
                          </p>

                          <div className="relative bg-black rounded-md my-4 group">
                            <div className="absolute right-2 top-2">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-gray-400 hover:text-white"
                                onClick={() =>
                                  copyToClipboard(
                                    `from sklearn.metrics import roc_auc_score

print("ROC-AUC:", roc_auc_score(y_true, y_pred))`,
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

print("ROC-AUC:", roc_auc_score(y_true, y_pred))`}
                              </code>
                            </pre>
                          </div>

                          <div className="h-64 flex items-center justify-center bg-muted/30 rounded-md my-4">
                            <svg width="300" height="200" viewBox="0 0 300 200">
                              <path
                                d="M50,150 C50,150 125,50 250,50"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                              />
                              <line
                                x1="50"
                                y1="150"
                                x2="250"
                                y2="50"
                                stroke="gray"
                                strokeWidth="1"
                                strokeDasharray="4"
                              />
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
                        </div>
                      </div>
                    </>
                  )}

                  {/* Cross-Validation Techniques Section */}
                  {currentSection === 3 && (
                    <>
                      <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
                        <h4 className="mt-0 text-lg font-semibold">Cross-Validation Techniques</h4>
                        <p className="mb-0">
                          Methods to assess how well a model will generalize to independent data by testing it on
                          multiple subsets of the available data.
                        </p>
                      </div>

                      <div className="space-y-8">
                        <div>
                          <h4 className="text-lg font-semibold">K-Fold Cross-Validation</h4>
                          <p>
                            K-Fold Cross-Validation is a technique where the dataset is split into k subsets (folds).
                            The model is trained on k-1 folds and tested on the remaining fold, repeating this process k
                            times. This ensures a more robust evaluation by reducing bias and variance.
                          </p>

                          <div className="relative bg-black rounded-md my-4 group">
                            <div className="absolute right-2 top-2">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-gray-400 hover:text-white"
                                onClick={() =>
                                  copyToClipboard(
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
                            Stratified K-Fold is similar to K-Fold but ensures that each fold maintains the same class
                            distribution as the overall dataset. This is particularly useful for imbalanced datasets.
                          </p>

                          <div className="relative bg-black rounded-md my-4 group">
                            <div className="absolute right-2 top-2">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-gray-400 hover:text-white"
                                onClick={() =>
                                  copyToClipboard(
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
                                {copied === "code-stratified" ? (
                                  <Check className="h-4 w-4" />
                                ) : (
                                  <Copy className="h-4 w-4" />
                                )}
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
                            LOOCV is an extreme case of K-Fold where each instance is used as a test set once while the
                            rest are used for training. Though computationally expensive, it provides an unbiased
                            performance estimate.
                          </p>

                          <div className="relative bg-black rounded-md my-4 group">
                            <div className="absolute right-2 top-2">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-gray-400 hover:text-white"
                                onClick={() =>
                                  copyToClipboard(
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
                            Repeated Cross-Validation applies K-Fold multiple times with different data splits,
                            increasing stability and reducing variance in performance estimation.
                          </p>

                          <div className="relative bg-black rounded-md my-4 group">
                            <div className="absolute right-2 top-2">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-gray-400 hover:text-white"
                                onClick={() =>
                                  copyToClipboard(
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
                                {copied === "code-repeated" ? (
                                  <Check className="h-4 w-4" />
                                ) : (
                                  <Copy className="h-4 w-4" />
                                )}
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
                  )}

                  {/* Add remaining sections (2 and 4-5) following similar pattern */}
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Navigation buttons */}
      <div className="flex justify-between mt-6">
        <Button
          variant="outline"
          size="lg"
          disabled={currentSection === 0}
          onClick={() => setCurrentSection((prev) => Math.max(0, prev - 1))}
          className="w-[120px]"
        >
          <ChevronLeft className="h-4 w-4 mr-2" /> Previous
        </Button>

        <Button
          variant={currentSection === sections.length - 1 ? "default" : "outline"}
          size="lg"
          disabled={currentSection === sections.length - 1}
          onClick={() => setCurrentSection((prev) => Math.min(sections.length - 1, prev + 1))}
          className="w-[120px]"
        >
          {currentSection === sections.length - 1 ? "Complete" : "Next"}
          {currentSection !== sections.length - 1 && <ChevronRight className="h-4 w-4 ml-2" />}
        </Button>
      </div>
    </div>
  )
}