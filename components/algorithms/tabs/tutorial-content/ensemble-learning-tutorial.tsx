"use client"

import { useState } from "react"

// Define the copiedState state
// Remove this line as it is outside a functional component
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { BookOpen, BarChart, Lightbulb, CheckCircle, ArrowRight, ChevronLeft, ChevronRight, GitBranch, GitMerge, Layers, Zap, Award, AlertTriangle, Copy, Check } from 'lucide-react'
// No chart component imports needed

interface ensembleTutorialProps {
    section: number
    onCopy: (text: string, id: string) => void
    copied: string | null
  }
  
  export function EnsembleTutorial({ section, onCopy, copied }: ensembleTutorialProps) {
    const [activeTab, setActiveTab] = useState("explanation")


  // Code examples
  const baggingCode = `from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize base model
base_model = DecisionTreeClassifier()

# Initialize BaggingClassifier
bagging_model = BaggingClassifier(base_model, n_estimators=50, random_state=42)

# Train the model
bagging_model.fit(X_train, y_train)

# Make predictions
y_pred = bagging_model.predict(X_test)
print("Bagging Accuracy:", bagging_model.score(X_test, y_test))`

  const boostingCode = `from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize base model (weak learner)
base_model = DecisionTreeClassifier(max_depth=1)

# Initialize AdaBoostClassifier
adaboost_model = AdaBoostClassifier(base_estimator=base_model, n_estimators=50, 
                                   learning_rate=1.0, random_state=42)

# Train the model
adaboost_model.fit(X_train, y_train)

# Make predictions
y_pred = adaboost_model.predict(X_test)
print("AdaBoost Accuracy:", adaboost_model.score(X_test, y_test))`

  const stackingCode = `from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Define base models
base_models = [
    ('decision_tree', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

# Define meta-model
meta_model = LogisticRegression()

# Initialize StackingClassifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train the model
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test)
print("Stacking Accuracy:", stacking_model.score(X_test, y_test))`

  // Render content based on current section
  const [copiedState, setCopiedState] = useState<string | null>(null);
    // Section 0: Introduction
    if (section === 0) {
      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-blue-100 dark:bg-blue-900 p-2 rounded-full">
                <Layers className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300">What is Ensemble Learning?</h3>
            </div>
            <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
              Ensemble learning is a machine learning technique where multiple models (often called "weak learners") are
              combined to improve overall performance. The goal is to reduce errors by leveraging the strengths of
              different models and mitigating their individual weaknesses.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="p-5 hover:shadow-md transition-shadow border-l-4 border-l-blue-500">
              <div className="flex items-center gap-2 mb-3">
                <GitBranch className="h-5 w-5 text-blue-500" />
                <h4 className="font-medium text-lg">Bagging</h4>
              </div>
              <p className="text-muted-foreground mb-3">
                Trains multiple instances of the same model on different subsets of data.
              </p>
              <Badge variant="secondary">Reduces Variance</Badge>
            </Card>

            <Card className="p-5 hover:shadow-md transition-shadow border-l-4 border-l-purple-500">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="h-5 w-5 text-purple-500" />
                <h4 className="font-medium text-lg">Boosting</h4>
              </div>
              <p className="text-muted-foreground mb-3">
                Trains models sequentially, with each model correcting errors of previous ones.
              </p>
              <Badge variant="secondary">Reduces Bias</Badge>
            </Card>

            <Card className="p-5 hover:shadow-md transition-shadow border-l-4 border-l-green-500">
              <div className="flex items-center gap-2 mb-3">
                <GitMerge className="h-5 w-5 text-green-500" />
                <h4 className="font-medium text-lg">Stacking</h4>
              </div>
              <p className="text-muted-foreground mb-3">
                Combines predictions from multiple different models using a meta-model.
              </p>
              <Badge variant="secondary">Combines Models</Badge>
            </Card>
          </div>

          <div className="mt-6">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              Why Use Ensemble Methods?
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="bg-muted/50 p-4 rounded-md">
                <div className="flex items-start gap-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  <div>
                    <h5 className="font-medium">Improved Accuracy</h5>
                    <p className="text-sm text-muted-foreground">
                      Ensemble methods typically outperform individual models by combining their strengths.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-muted/50 p-4 rounded-md">
                <div className="flex items-start gap-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  <div>
                    <h5 className="font-medium">Reduced Overfitting</h5>
                    <p className="text-sm text-muted-foreground">
                      Ensemble methods help mitigate overfitting by averaging out individual model biases.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-muted/50 p-4 rounded-md">
                <div className="flex items-start gap-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  <div>
                    <h5 className="font-medium">Robustness</h5>
                    <p className="text-sm text-muted-foreground">
                      Ensembles are more robust to noise and outliers in the training data.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-muted/50 p-4 rounded-md">
                <div className="flex items-start gap-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  <div>
                    <h5 className="font-medium">Feature Importance</h5>
                    <p className="text-sm text-muted-foreground">
                      Many ensemble methods provide better insights into feature importance.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <BarChart className="h-5 w-5 text-primary" />
              Performance Comparison
            </h4>
            <div className="bg-white dark:bg-gray-900 p-4 rounded-lg border">
              <div className="h-64">
                <h3 className="text-sm font-medium opacity-70 mb-4 text-center">Accuracy Comparison of Ensemble Methods</h3>
                <div className="flex flex-col h-full justify-end space-y-2">
                  <div className="w-full flex items-center">
                    <span className="text-xs text-muted-foreground w-24">Decision Tree</span>
                    <div className="h-8 bg-gray-300 dark:bg-gray-700" style={{ width: '65%' }} />
                    <span className="ml-2 text-xs">65%</span>
                  </div>
                  <div className="w-full flex items-center">
                    <span className="text-xs text-muted-foreground w-24">Random Forest</span>
                    <div className="h-8 bg-blue-500 dark:bg-blue-600" style={{ width: '82%' }} />
                    <span className="ml-2 text-xs">82%</span>
                  </div>
                  <div className="w-full flex items-center">
                    <span className="text-xs text-muted-foreground w-24">AdaBoost</span>
                    <div className="h-8 bg-purple-500 dark:bg-purple-600" style={{ width: '78%' }} />
                    <span className="ml-2 text-xs">78%</span>
                  </div>
                  <div className="w-full flex items-center">
                    <span className="text-xs text-muted-foreground w-24">Gradient Boost</span>
                    <div className="h-8 bg-amber-500 dark:bg-amber-600" style={{ width: '86%' }} />
                    <span className="ml-2 text-xs">86%</span>
                  </div>
                  <div className="w-full flex items-center">
                    <span className="text-xs text-muted-foreground w-24">Stacking</span>
                    <div className="h-8 bg-green-500 dark:bg-green-600" style={{ width: '88%' }} />
                    <span className="ml-2 text-xs">88%</span>
                  </div>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-4">
                This chart shows typical accuracy improvements when using ensemble methods compared to a single decision tree model.
                Results may vary depending on the dataset and specific implementation.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-blue-100 dark:bg-blue-900 p-2 rounded-full">
                  <CheckCircle className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="text-lg font-semibold text-blue-800 dark:text-blue-300">Advantages</h3>
              </div>
              <ul className="space-y-3 text-blue-700 dark:text-blue-300">
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Higher accuracy than individual models</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>More stable and robust predictions</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Reduced risk of overfitting</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Better handling of complex relationships</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950 dark:to-orange-950 p-6 rounded-lg border border-amber-100 dark:border-amber-900">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-amber-100 dark:bg-amber-900 p-2 rounded-full">
                  <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                </div>
                <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-300">Limitations</h3>
              </div>
              <ul className="space-y-3 text-amber-700 dark:text-amber-300">
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Increased computational complexity</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Longer training times</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Reduced interpretability</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Higher memory requirements</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-muted/20 p-6 rounded-lg border mt-6">
            <h4 className="font-medium text-lg mb-4 flex items-center gap-2">
              <Layers className="h-5 w-5 text-primary" />
              Ensemble Learning Visualization
            </h4>
            <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
              <img
                src="/placeholder.svg?height=400&width=800"
                alt="Ensemble Learning Methods"
                className="max-w-full h-auto"
              />
            </div>
            <p className="mt-4 text-sm text-muted-foreground">
              The diagram illustrates how different ensemble methods combine multiple models to make predictions.
              Bagging uses parallel training with different data subsets, boosting trains models sequentially to correct
              errors, and stacking uses a meta-model to combine predictions from diverse base models.
            </p>
          </div>
        </div>
      )
    }

    // Section 1: Bagging
    if (section === 1) {
        function copyToClipboard(baggingCode: string, arg1: string): void {
            throw new Error("Function not implemented.")
        }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-blue-100 dark:bg-blue-900 p-2 rounded-full">
                <GitBranch className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300">
                Bagging (Bootstrap Aggregating)
              </h3>
            </div>
            <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
              Bagging is an ensemble technique where multiple instances of the same model are trained on different
              random subsets of the data. These subsets are created through bootstrapping (sampling with replacement).
              The final prediction is the average (for regression) or majority vote (for classification).
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="explanation">How It Works</TabsTrigger>
              <TabsTrigger value="code">Code Example</TabsTrigger>
              <TabsTrigger value="visualization">Visualization</TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4">
              <div className="bg-muted/30 p-5 rounded-lg border">
                <h4 className="font-medium text-lg mb-3">Bagging Process</h4>
                <ol className="space-y-4">
                  <li className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Bootstrap Sampling</h5>
                      <p className="text-muted-foreground text-sm">
                        Create multiple random subsets of the training data by sampling with replacement. Each subset
                        has the same size as the original dataset.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Parallel Training</h5>
                      <p className="text-muted-foreground text-sm">
                        Train the same type of model independently on each subset. This creates multiple models with
                        different perspectives on the data.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Aggregation</h5>
                      <p className="text-muted-foreground text-sm">
                        Combine predictions from all models through averaging (for regression) or majority voting (for
                        classification).
                      </p>
                    </div>
                  </li>
                </ol>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <h4 className="font-medium">Why Bagging Works</h4>
                  </div>
                  <ul className="space-y-2 text-muted-foreground text-sm">
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Reduces variance by averaging out individual model errors</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Prevents overfitting, especially in high-variance models like decision trees</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Improves stability and accuracy of predictions</span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Lightbulb className="h-5 w-5 text-yellow-500" />
                    <h4 className="font-medium">Popular Bagging Algorithms</h4>
                  </div>
                  <ul className="space-y-2 text-muted-foreground text-sm">
                    <li className="flex items-start gap-2">
                      <Badge variant="outline" className="mt-0.5">
                        RF
                      </Badge>
                      <span>
                        <span className="font-medium text-foreground">Random Forest:</span> Bagging with decision trees
                        and random feature selection
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Badge variant="outline" className="mt-0.5">
                        ET
                      </Badge>
                      <span>
                        <span className="font-medium text-foreground">Extra Trees:</span> Similar to Random Forest but
                        with random thresholds
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Badge variant="outline" className="mt-0.5">
                        BC
                      </Badge>
                      <span>
                        <span className="font-medium text-foreground">Bagging Classifier:</span> Generic implementation
                        for any base estimator
                      </span>
                    </li>
                  </ul>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="code" className="space-y-4">
              <div className="relative">
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code className="language-python">{baggingCode}</code>
                </pre>
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute top-2 right-2"
                  onClick={() => copyToClipboard(baggingCode, "bagging")}
                >
                  {copiedState === "bagging" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>

              <div className="bg-muted/30 p-5 rounded-lg border">
                <h4 className="font-medium text-lg mb-3">Code Explanation</h4>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      1
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We first import the necessary libraries and load the Iris dataset, splitting it into training
                        and testing sets.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      2
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We initialize a{" "}
                        <code className="text-xs bg-muted-foreground/20 px-1 py-0.5 rounded">
                          DecisionTreeClassifier
                        </code>{" "}
                        as our base model.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      3
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We create a{" "}
                        <code className="text-xs bg-muted-foreground/20 px-1 py-0.5 rounded">BaggingClassifier</code>{" "}
                        with 50 estimators (50 decision trees), each trained on a different bootstrap sample.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      4
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We train the model and evaluate its accuracy on the test set.
                      </p>
                    </div>
                  </li>
                </ul>
              </div>

              <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-lg border border-blue-100 dark:border-blue-900">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="h-5 w-5 text-blue-500" />
                  <h4 className="font-medium">Pro Tip</h4>
                </div>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  When using bagging, you can access the{" "}
                  <code className="text-xs bg-blue-100 dark:bg-blue-900 px-1 py-0.5 rounded">estimators_</code>{" "}
                  attribute to inspect individual models in the ensemble. You can also use{" "}
                  <code className="text-xs bg-blue-100 dark:bg-blue-900 px-1 py-0.5 rounded">oob_score=True</code> to
                  enable out-of-bag evaluation, which provides an unbiased estimate of model performance.
                </p>
              </div>
            </TabsContent>

            <TabsContent value="visualization" className="space-y-4">
              <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg border flex items-center justify-center overflow-hidden">
                <img
                  src="/placeholder.svg?height=400&width=800"
                  alt="Bagging Process Visualization"
                  className="max-w-full h-auto"
                />
              </div>
              <p className="text-sm text-muted-foreground">
                The diagram illustrates the bagging process: multiple bootstrap samples are created from the original
                dataset, individual models are trained on each sample, and their predictions are combined through voting
                or averaging.
              </p>

              <div className="mt-6">
                <h4 className="font-medium text-lg mb-3">Random Forest: A Special Case of Bagging</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="p-5">
                    <h5 className="font-medium mb-2">How Random Forest Extends Bagging</h5>
                    <p className="text-sm text-muted-foreground">
                      Random Forest adds an extra layer of randomness to bagging. In addition to building each tree on a
                      different bootstrap sample, it also selects a random subset of features at each split in the
                      decision tree. This additional randomness helps to further decorrelate the trees, making the
                      ensemble even more effective.
                    </p>
                  </Card>

                  <div className="bg-muted/30 p-5 rounded-lg border">
                    <h5 className="font-medium mb-2">Random Forest Parameters</h5>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      <li className="flex items-start gap-2">
                        <Badge variant="outline" className="mt-0.5">
                          n_estimators
                        </Badge>
                        <span>Number of trees in the forest (default: 100)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Badge variant="outline" className="mt-0.5">
                          max_features
                        </Badge>
                        <span>Number of features to consider for best split</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Badge variant="outline" className="mt-0.5">
                          bootstrap
                        </Badge>
                        <span>Whether to use bootstrap samples (default: True)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Badge variant="outline" className="mt-0.5">
                          oob_score
                        </Badge>
                        <span>Whether to use out-of-bag samples for validation</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      )
    }

    // Section 2: Boosting
    if (section === 2) {
        function copyToClipboard(boostingCode: string, arg1: string): void {
            throw new Error("Function not implemented.");
        }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-violet-50 dark:from-purple-950 dark:to-violet-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-purple-100 dark:bg-purple-900 p-2 rounded-full">
                <Zap className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300">Boosting</h3>
            </div>
            <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
              Boosting trains models sequentially, where each model tries to fix the errors made by the previous one.
              This iterative process focuses more on the misclassified data points, creating a strong learner from weak
              learners. Boosting primarily reduces bias and can achieve higher accuracy than bagging in many cases.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="explanation">How It Works</TabsTrigger>
              <TabsTrigger value="code">Code Example</TabsTrigger>
              <TabsTrigger value="visualization">Types of Boosting</TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4">
              <div className="bg-muted/30 p-5 rounded-lg border">
                <h4 className="font-medium text-lg mb-3">Boosting Process</h4>
                <ol className="space-y-4">
                  <li className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Initial Model</h5>
                      <p className="text-muted-foreground text-sm">
                        Train an initial weak learner (typically a simple model like a decision stump) on the original
                        dataset.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Focus on Errors</h5>
                      <p className="text-muted-foreground text-sm">
                        Identify misclassified instances and increase their weights or importance in the dataset.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Sequential Training</h5>
                      <p className="text-muted-foreground text-sm">
                        Train the next model with more focus on previously misclassified instances.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Weighted Combination</h5>
                      <p className="text-muted-foreground text-sm">
                        Combine all models using a weighted sum, where better-performing models get higher weights.
                      </p>
                    </div>
                  </li>
                </ol>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <h4 className="font-medium">Why Boosting Works</h4>
                  </div>
                  <ul className="space-y-2 text-muted-foreground text-sm">
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Reduces bias by focusing on difficult examples</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Creates a strong learner from multiple weak learners</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Adapts to complex patterns in the data</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Often achieves higher accuracy than bagging</span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle className="h-5 w-5 text-amber-500" />
                    <h4 className="font-medium">Potential Issues</h4>
                  </div>
                  <ul className="space-y-2 text-muted-foreground text-sm">
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span>More prone to overfitting than bagging, especially with noisy data</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span>Sequential nature makes it harder to parallelize</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span>Sensitive to outliers and noisy data points</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span>Requires careful tuning of learning rate and other parameters</span>
                    </li>
                  </ul>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="code" className="space-y-4">
              <div className="relative">
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code className="language-python">{boostingCode}</code>
                </pre>
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute top-2 right-2"
                  onClick={() => copyToClipboard(boostingCode, "boosting")}
                >
                  {copiedState === "boosting" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>

              <div className="bg-muted/30 p-5 rounded-lg border">
                <h4 className="font-medium text-lg mb-3">Code Explanation</h4>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      1
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We initialize a{" "}
                        <code className="text-xs bg-muted-foreground/20 px-1 py-0.5 rounded">
                          DecisionTreeClassifier
                        </code>{" "}
                        with a shallow depth (max_depth=1) as our weak learner.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      2
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We create an{" "}
                        <code className="text-xs bg-muted-foreground/20 px-1 py-0.5 rounded">AdaBoostClassifier</code>{" "}
                        with 50 estimators and a learning rate of 1.0.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      3
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        The learning rate controls how much each weak learner contributes to the final model. A lower
                        learning rate requires more estimators but can lead to better performance.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      4
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We train the model and evaluate its accuracy on the test set.
                      </p>
                    </div>
                  </li>
                </ul>
              </div>

              <div className="bg-purple-50 dark:bg-purple-950 p-4 rounded-lg border border-purple-100 dark:border-purple-900">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="h-5 w-5 text-purple-500" />
                  <h4 className="font-medium">Pro Tip</h4>
                </div>
                <p className="text-sm text-purple-700 dark:text-purple-300">
                  When using AdaBoost, you can access the{" "}
                  <code className="text-xs bg-purple-100 dark:bg-purple-900 px-1 py-0.5 rounded">
                    feature_importances_
                  </code>{" "}
                  attribute to understand which features are most important for classification. You can also visualize
                  how the error decreases with each additional estimator using{" "}
                  <code className="text-xs bg-purple-100 dark:bg-purple-900 px-1 py-0.5 rounded">
                    staged_score(X, y)
                  </code>
                  .
                </p>
              </div>
            </TabsContent>

            <TabsContent value="visualization" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card className="p-5 hover:shadow-md transition-shadow border-l-4 border-l-purple-500">
                  <div className="flex items-center gap-2 mb-3">
                    <Badge className="bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300 hover:bg-purple-100 dark:hover:bg-purple-900">
                      AdaBoost
                    </Badge>
                  </div>
                  <p className="text-muted-foreground text-sm mb-3">
                    Adaptive Boosting adjusts weights of misclassified instances to focus subsequent models on difficult
                    examples.
                  </p>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    <li className="flex items-start gap-1">
                      <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" />
                      <span>Assigns higher weights to misclassified points</span>
                    </li>
                    <li className="flex items-start gap-1">
                      <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" />
                      <span>Uses weighted voting for final prediction</span>
                    </li>
                    <li className="flex items-start gap-1">
                      <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" />
                      <span>Sensitive to noisy data and outliers</span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-5 hover:shadow-md transition-shadow border-l-4 border-l-green-500">
                  <div className="flex items-center gap-2 mb-3">
                    <Badge className="bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300 hover:bg-green-100 dark:hover:bg-green-900">
                      Gradient Boosting
                    </Badge>
                  </div>
                  <p className="text-muted-foreground text-sm mb-3">
                    Builds models sequentially to minimize a loss function using gradient descent optimization.
                  </p>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    <li className="flex items-start gap-1">
                      <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" />
                      <span>Fits new model to residual errors of previous model</span>
                    </li>
                    <li className="flex items-start gap-1">
                      <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" />
                      <span>Uses gradient descent to minimize loss</span>
                    </li>
                    <li className="flex items-start gap-1">
                      <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" />
                      <span>More flexible than AdaBoost with different loss functions</span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-5 hover:shadow-md transition-shadow border-l-4 border-l-amber-500">
                  <div className="flex items-center gap-2 mb-3">
                    <Badge className="bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300 hover:bg-amber-100 dark:hover:bg-amber-900">
                      XGBoost
                    </Badge>
                  </div>
                  <p className="text-muted-foreground text-sm mb-3">
                    eXtreme Gradient Boosting is an optimized implementation of gradient boosting with additional
                    regularization.
                  </p>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    <li className="flex items-start gap-1">
                      <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" />
                      <span>Includes L1 and L2 regularization</span>
                    </li>
                    <li className="flex items-start gap-1">
                      <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" />
                      <span>Handles missing values automatically</span>
                    </li>
                    <li className="flex items-start gap-1">
                      <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" />
                      <span>Highly optimized for performance</span>
                    </li>
                  </ul>
                </Card>
              </div>

              <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg border flex items-center justify-center overflow-hidden mt-4">
                <img
                  src="/placeholder.svg?height=400&width=800"
                  alt="Boosting Process Visualization"
                  className="max-w-full h-auto"
                />
              </div>
              <p className="text-sm text-muted-foreground">
                The diagram illustrates how boosting sequentially builds models that focus on correcting the errors of
                previous models. Each new model gives more attention to previously misclassified instances.
              </p>

              <div className="mt-6">
                <h4 className="font-medium text-lg mb-3">Comparing Boosting Algorithms</h4>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-muted/50">
                        <th className="border px-4 py-2 text-left">Algorithm</th>
                        <th className="border px-4 py-2 text-left">Strengths</th>
                        <th className="border px-4 py-2 text-left">Weaknesses</th>
                        <th className="border px-4 py-2 text-left">Best Use Cases</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border px-4 py-2 font-medium">AdaBoost</td>
                        <td className="border px-4 py-2 text-sm">
                          Simple to implement, less prone to overfitting than decision trees
                        </td>
                        <td className="border px-4 py-2 text-sm">Sensitive to noisy data and outliers</td>
                        <td className="border px-4 py-2 text-sm">Binary classification problems with clean data</td>
                      </tr>
                      <tr className="bg-muted/20">
                        <td className="border px-4 py-2 font-medium">Gradient Boosting</td>
                        <td className="border px-4 py-2 text-sm">
                          Flexible with different loss functions, handles various data types
                        </td>
                        <td className="border px-4 py-2 text-sm">Can overfit if not tuned properly, slower training</td>
                        <td className="border px-4 py-2 text-sm">
                          Regression and classification with complex relationships
                        </td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2 font-medium">XGBoost</td>
                        <td className="border px-4 py-2 text-sm">
                          Fast, regularized to prevent overfitting, handles missing values
                        </td>
                        <td className="border px-4 py-2 text-sm">More hyperparameters to tune, can be complex</td>
                        <td className="border px-4 py-2 text-sm">
                          Structured/tabular data, competitions, production systems
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      )
    }

    // Section 3: Stacking
    if (section === 3) {
        function copyToClipboard(stackingCode: string, arg1: string): void {
            throw new Error("Function not implemented.");
        }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950 p-6 rounded-lg border border-green-100 dark:border-green-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-green-100 dark:bg-green-900 p-2 rounded-full">
                <GitMerge className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
              <h3 className="text-xl font-semibold text-green-800 dark:text-green-300">
                Stacking (Stacked Generalization)
              </h3>
            </div>
            <p className="text-green-700 dark:text-green-300 leading-relaxed">
              Stacking involves combining predictions from multiple different models (like decision trees, SVMs, etc.)
              using a meta-model (like logistic regression) that learns how to best aggregate the predictions. This
              approach leverages the strengths of diverse algorithms to create a more powerful ensemble.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="explanation">How It Works</TabsTrigger>
              <TabsTrigger value="code">Code Example</TabsTrigger>
              <TabsTrigger value="visualization">Visualization</TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4">
              <div className="bg-muted/30 p-5 rounded-lg border">
                <h4 className="font-medium text-lg mb-3">Stacking Process</h4>
                <ol className="space-y-4">
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Train Base Models</h5>
                      <p className="text-muted-foreground text-sm">
                        Train multiple different models (e.g., decision trees, SVMs, neural networks) on the same
                        dataset.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Generate Meta-Features</h5>
                      <p className="text-muted-foreground text-sm">
                        Use cross-validation to make predictions with each base model. These predictions become the
                        features for the meta-model.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Train Meta-Model</h5>
                      <p className="text-muted-foreground text-sm">
                        Train a meta-model (often logistic regression or another simple model) on the meta-features to
                        learn how to best combine the base models' predictions.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Final Prediction</h5>
                      <p className="text-muted-foreground text-sm">
                        For new data, get predictions from all base models, then feed these predictions to the
                        meta-model for the final prediction.
                      </p>
                    </div>
                  </li>
                </ol>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <h4 className="font-medium">Why Stacking Works</h4>
                  </div>
                  <ul className="space-y-2 text-muted-foreground text-sm">
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Leverages the strengths of different algorithms</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Meta-model learns which base model to trust in different situations</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Can achieve higher accuracy than any individual model</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Works well with diverse base models that capture different aspects of the data</span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Lightbulb className="h-5 w-5 text-yellow-500" />
                    <h4 className="font-medium">Best Practices</h4>
                  </div>
                  <ul className="space-y-2 text-muted-foreground text-sm">
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>Use diverse base models that perform well individually</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>Use cross-validation to prevent leakage when creating meta-features</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>Choose a simple, interpretable model for the meta-learner</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>Consider using probability outputs rather than just class predictions</span>
                    </li>
                  </ul>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="code" className="space-y-4">
              <div className="relative">
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code className="language-python">{stackingCode}</code>
                </pre>
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute top-2 right-2"
                  onClick={() => copyToClipboard(stackingCode, "stacking")}
                >
                  {copiedState === "stacking" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>

              <div className="bg-muted/30 p-5 rounded-lg border">
                <h4 className="font-medium text-lg mb-3">Code Explanation</h4>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      1
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We define two different base models: a decision tree and an SVM. These models have different
                        strengths and will capture different patterns in the data.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      2
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We choose logistic regression as our meta-model. This simple model will learn how to best
                        combine the predictions from the base models.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      3
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We create a{" "}
                        <code className="text-xs bg-muted-foreground/20 px-1 py-0.5 rounded">StackingClassifier</code>{" "}
                        that combines the base models and meta-model. Scikit-learn handles the cross-validation process
                        automatically.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      4
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">
                        We train the stacking ensemble and evaluate its performance on the test set.
                      </p>
                    </div>
                  </li>
                </ul>
              </div>

              <div className="bg-green-50 dark:bg-green-950 p-4 rounded-lg border border-green-100 dark:border-green-900">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="h-5 w-5 text-green-500" />
                  <h4 className="font-medium">Pro Tip</h4>
                </div>
                <p className="text-sm text-green-700 dark:text-green-300">
                  You can customize the cross-validation process in{" "}
                  <code className="text-xs bg-green-100 dark:bg-green-900 px-1 py-0.5 rounded">StackingClassifier</code>{" "}
                  using the <code className="text-xs bg-green-100 dark:bg-green-900 px-1 py-0.5 rounded">cv</code>{" "}
                  parameter. Setting{" "}
                  <code className="text-xs bg-green-100 dark:bg-green-900 px-1 py-0.5 rounded">passthrough=True</code>{" "}
                  will include the original features alongside the meta-features, which can improve performance in some
                  cases.
                </p>
              </div>
            </TabsContent>

            <TabsContent value="visualization" className="space-y-4">
              <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg border flex items-center justify-center overflow-hidden">
                <img
                  src="/placeholder.svg?height=400&width=800"
                  alt="Stacking Process Visualization"
                  className="max-w-full h-auto"
                />
              </div>
              <p className="text-sm text-muted-foreground">
                The diagram illustrates the stacking process: multiple base models make predictions, which are then used
                as features for a meta-model that learns how to best combine these predictions for the final output.
              </p>

              <div className="mt-6">
                <h4 className="font-medium text-lg mb-3">Stacking vs. Other Ensemble Methods</h4>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-muted/50">
                        <th className="border px-4 py-2 text-left">Feature</th>
                        <th className="border px-4 py-2 text-left">Bagging</th>
                        <th className="border px-4 py-2 text-left">Boosting</th>
                        <th className="border px-4 py-2 text-left">Stacking</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Model Diversity</td>
                        <td className="border px-4 py-2 text-sm">Same algorithm, different data subsets</td>
                        <td className="border px-4 py-2 text-sm">Same algorithm, sequential training</td>
                        <td className="border px-4 py-2 text-sm">Different algorithms</td>
                      </tr>
                      <tr className="bg-muted/20">
                        <td className="border px-4 py-2 font-medium">Training Process</td>
                        <td className="border px-4 py-2 text-sm">Parallel</td>
                        <td className="border px-4 py-2 text-sm">Sequential</td>
                        <td className="border px-4 py-2 text-sm">Two-level (base + meta)</td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Primarily Reduces</td>
                        <td className="border px-4 py-2 text-sm">Variance</td>
                        <td className="border px-4 py-2 text-sm">Bias</td>
                        <td className="border px-4 py-2 text-sm">Both bias and variance</td>
                      </tr>
                      <tr className="bg-muted/20">
                        <td className="border px-4 py-2 font-medium">Complexity</td>
                        <td className="border px-4 py-2 text-sm">Low</td>
                        <td className="border px-4 py-2 text-sm">Medium</td>
                        <td className="border px-4 py-2 text-sm">High</td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Computational Cost</td>
                        <td className="border px-4 py-2 text-sm">Medium (parallelizable)</td>
                        <td className="border px-4 py-2 text-sm">High (sequential)</td>
                        <td className="border px-4 py-2 text-sm">Very High (multiple algorithms)</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                <Card className="p-5">
                  <h5 className="font-medium mb-2">Advanced Stacking Techniques</h5>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <Badge variant="outline" className="mt-0.5">
                        Multi-level
                      </Badge>
                      <span>Using multiple layers of stacking (stacking of stacked models)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Badge variant="outline" className="mt-0.5">
                        Blending
                      </Badge>
                      <span>Using a hold-out set instead of cross-validation for meta-features</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Badge variant="outline" className="mt-0.5">
                        Feature-weighted
                      </Badge>
                      <span>Including original features alongside model predictions</span>
                    </li>
                  </ul>
                </Card>

                <div className="bg-muted/30 p-5 rounded-lg border">
                  <h5 className="font-medium mb-2">Real-world Applications</h5>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>Kaggle competitions (often winning solutions use stacking)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>Medical diagnosis combining multiple specialist models</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>Financial forecasting with diverse economic indicators</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>Recommendation systems combining content and collaborative filtering</span>
                    </li>
                  </ul>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      )
    }

    // Section 4: Comparison & Best Practices
    if (section === 4) {
      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-950 dark:to-yellow-950 p-6 rounded-lg border border-amber-100 dark:border-amber-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-amber-100 dark:bg-amber-900 p-2 rounded-full">
                <Award className="h-6 w-6 text-amber-600 dark:text-amber-400" />
              </div>
              <h3 className="text-xl font-semibold text-amber-800 dark:text-amber-300">Comparison & Best Practices</h3>
            </div>
            <p className="text-amber-700 dark:text-amber-300 leading-relaxed">
              Each ensemble method has its strengths and ideal use cases. Understanding when to use each technique and
              how to optimize their performance is key to successful machine learning applications. Let's compare these
              methods and explore best practices for implementation.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="explanation">Comparison</TabsTrigger>
              <TabsTrigger value="code">Best Practices</TabsTrigger>
              <TabsTrigger value="visualization">Performance</TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4">
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-muted/50">
                      <th className="border px-4 py-2 text-left">Method</th>
                      <th className="border px-4 py-2 text-left">Key Characteristics</th>
                      <th className="border px-4 py-2 text-left">Strengths</th>
                      <th className="border px-4 py-2 text-left">Weaknesses</th>
                      <th className="border px-4 py-2 text-left">Best For</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border px-4 py-2 font-medium">Bagging</td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Parallel training</li>
                          <li>Same algorithm</li>
                          <li>Different data subsets</li>
                          <li>Averaging/voting</li>
                        </ul>
                      </td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Reduces variance</li>
                          <li>Prevents overfitting</li>
                          <li>Parallelizable</li>
                          <li>Stable predictions</li>
                        </ul>
                      </td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Limited by base model</li>
                          <li>May not reduce bias</li>
                          <li>Requires diverse trees</li>
                        </ul>
                      </td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>High-variance models</li>
                          <li>Decision trees</li>
                          <li>Noisy data</li>
                        </ul>
                      </td>
                    </tr>
                    <tr className="bg-muted/20">
                      <td className="border px-4 py-2 font-medium">Boosting</td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Sequential training</li>
                          <li>Focus on errors</li>
                          <li>Weighted combination</li>
                          <li>Adaptive learning</li>
                        </ul>
                      </td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Reduces bias</li>
                          <li>High accuracy</li>
                          <li>Works with weak learners</li>
                          <li>Feature importance</li>
                        </ul>
                      </td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Can overfit</li>
                          <li>Sensitive to noise</li>
                          <li>Sequential (slower)</li>
                          <li>Harder to tune</li>
                        </ul>
                      </td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Structured data</li>
                          <li>Clean datasets</li>
                          <li>When accuracy is critical</li>
                        </ul>
                      </td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2 font-medium">Stacking</td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Multiple algorithms</li>
                          <li>Two-level learning</li>
                          <li>Meta-model combination</li>
                          <li>Cross-validation</li>
                        </ul>
                      </td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Leverages diverse models</li>
                          <li>Can outperform any single model</li>
                          <li>Adaptable to data characteristics</li>
                        </ul>
                      </td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Complex implementation</li>
                          <li>Computationally expensive</li>
                          <li>Risk of overfitting</li>
                          <li>Less interpretable</li>
                        </ul>
                      </td>
                      <td className="border px-4 py-2 text-sm">
                        <ul className="list-disc pl-4 space-y-1">
                          <li>Complex problems</li>
                          <li>Competitions</li>
                          <li>When multiple algorithms work well</li>
                        </ul>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <Card className="p-5 hover:shadow-md transition-shadow border-l-4 border-l-blue-500">
                  <div className="flex items-center gap-2 mb-3">
                    <GitBranch className="h-5 w-5 text-blue-500" />
                    <h4 className="font-medium">When to Use Bagging</h4>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When your model has high variance (overfitting)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When you need stable, reliable predictions</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When you have noisy data</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When parallel processing is available</span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-5 hover:shadow-md transition-shadow border-l-4 border-l-purple-500">
                  <div className="flex items-center gap-2 mb-3">
                    <Zap className="h-5 w-5 text-purple-500" />
                    <h4 className="font-medium">When to Use Boosting</h4>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When your model has high bias (underfitting)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When you need maximum predictive power</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When you have clean, well-prepared data</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When you need feature importance insights</span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-5 hover:shadow-md transition-shadow border-l-4 border-l-green-500">
                  <div className="flex items-center gap-2 mb-3">
                    <GitMerge className="h-5 w-5 text-green-500" />
                    <h4 className="font-medium">When to Use Stacking</h4>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When you have multiple well-performing models</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When you need the absolute best performance</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>When computational resources aren't limited</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>For competitions or critical applications</span>
                    </li>
                  </ul>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="code" className="space-y-6">

{/* Bagging Best Practices */}
<div className="bg-muted/30 p-5 rounded-lg border">
  <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
    <GitBranch className="h-5 w-5 text-blue-500" />
    Bagging Best Practices
  </h4>
  <ul className="space-y-3">

    {/* Use a sufficient number of estimators */}
    <li className="flex items-start gap-2">
      <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm">
        1
      </div>
      <div>
        <h5 className="font-medium">Use a sufficient number of estimators</h5>
        <p className="text-muted-foreground text-sm">
          Start with at least 50-100 estimators and increase until performance plateaus.
        </p>
        <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
          <code>bagging_model = BaggingClassifier(base_estimator=tree, n_estimators=100)</code>
        </pre>
      </div>
    </li>

    {/* Enable out-of-bag evaluation */}
    <li className="flex items-start gap-2">
      <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm">
        2
      </div>
      <div>
        <h5 className="font-medium">Enable out-of-bag evaluation</h5>
        <p className="text-muted-foreground text-sm">
          Use out-of-bag samples to estimate model performance without a separate validation set.
        </p>
        <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
          <code>bagging_model = BaggingClassifier(base_estimator=tree, oob_score=True)</code>
        </pre>
      </div>
    </li>

    {/* Tune the max_samples parameter */}
    <li className="flex items-start gap-2">
      <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm">
        3
      </div>
      <div>
        <h5 className="font-medium">Tune the max_samples parameter</h5>
        <p className="text-muted-foreground text-sm">
          Control the size of each bootstrap sample to balance diversity and performance.
        </p>
        <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
          <code>bagging_model = BaggingClassifier(base_estimator=tree, max_samples=0.8)</code>
        </pre>
      </div>
    </li>
  </ul>
</div>

{/* Boosting Best Practices */}
<div className="bg-muted/30 p-5 rounded-lg border">
  <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
    <Zap className="h-5 w-5 text-purple-500" />
    Boosting Best Practices
  </h4>
  <ul className="space-y-3">

    {/* Tune the learning rate */}
    <li className="flex items-start gap-2">
      <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm">
        1
      </div>
      <div>
        <h5 className="font-medium">Tune the learning rate</h5>
        <p className="text-muted-foreground text-sm">
          Use a smaller learning rate with more estimators for better performance.
        </p>
        <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
          <code>boost_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)</code>
        </pre>
      </div>
    </li>

    {/* Control tree complexity */}
    <li className="flex items-start gap-2">
      <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm">
        2
      </div>
      <div>
        <h5 className="font-medium">Control tree complexity</h5>
        <p className="text-muted-foreground text-sm">
          Limit tree depth to prevent overfitting, especially with noisy data.
        </p>
        <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
          <code>boost_model = GradientBoostingClassifier(max_depth=3, min_samples_leaf=5)</code>
        </pre>
      </div>
    </li>

    {/* Use early stopping */}
    <li className="flex items-start gap-2">
      <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm">
        3
      </div>
      <div>
        <h5 className="font-medium">Use early stopping</h5>
        <p className="text-muted-foreground text-sm">
          Monitor validation performance to stop training when it no longer improves.
        </p>
        <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
          <code>
            boost_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
          </code>
        </pre>
      </div>
    </li>
  </ul>
</div>

              <div className="bg-muted/30 p-5 rounded-lg border mt-4">
                <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
                  <GitMerge className="h-5 w-5 text-green-500" />
                  Stacking Best Practices
                </h4>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Choose diverse base models</h5>
                      <p className="text-muted-foreground text-sm">
                        Select models with different strengths and learning approaches for better ensemble performance.
                      </p>
                      <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
                        <code>
                          base_models = [ ('rf', RandomForestClassifier()), ('svm', SVC(probability=True)), ('knn',
                          KNeighborsClassifier()) ]
                        </code>
                      </pre>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Use cross-validation for meta-features</h5>
                      <p className="text-muted-foreground text-sm">
                        Ensure proper cross-validation to prevent data leakage when generating meta-features.
                      </p>
                      <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
                        <code>
                          stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
                        </code>
                      </pre>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Consider including original features</h5>
                      <p className="text-muted-foreground text-sm">
                        Include original features alongside meta-features for potentially better performance.
                      </p>
                      <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
                        <code>
                          stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model,
                          passthrough=True)
                        </code>
                      </pre>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Use probability outputs</h5>
                      <p className="text-muted-foreground text-sm">
                        Use probability predictions rather than just class predictions for more information.
                      </p>
                      <pre className="bg-muted p-2 rounded-md mt-1 text-xs overflow-x-auto">
                        <code>
                          stacking_model = StackingClassifier(estimators=base_models, stack_method='predict_proba')
                        </code>
                      </pre>
                    </div>
                  </li>
                </ul>
              </div>

              <div className="bg-amber-50 dark:bg-amber-950 p-4 rounded-lg border border-amber-100 dark:border-amber-900 mt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="h-5 w-5 text-amber-500" />
                  <h4 className="font-medium">General Ensemble Best Practices</h4>
                </div>
                <ul className="space-y-2 text-sm text-amber-700 dark:text-amber-300">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>
                      <span className="font-medium">Proper validation:</span> Always use cross-validation to evaluate
                      ensemble performance and prevent overfitting.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>
                      <span className="font-medium">Feature engineering:</span> Good features are still important, even
                      with powerful ensemble methods.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>
                      <span className="font-medium">Hyperparameter tuning:</span> Use grid search or random search to
                      find optimal parameters for your ensemble.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>
                      <span className="font-medium">Balance complexity and performance:</span> Consider the trade-off
                      between model complexity, training time, and performance gains.
                    </span>
                  </li>
                </ul>
              </div>
            </TabsContent>

            <TabsContent value="visualization" className="space-y-4">
              <div className="bg-white dark:bg-gray-900 p-4 rounded-lg border">
                <div className="h-64">
                  <h3 className="text-sm font-medium opacity-70 mb-4 text-center">Performance Comparison on Different Datasets</h3>
                  <div className="flex flex-col h-full justify-end space-y-2">
                    <div className="w-full flex items-center">
                      <span className="text-xs text-muted-foreground w-24">Structured Data</span>
                      <div className="flex-1 flex">
                        <div className="h-8 bg-blue-500 dark:bg-blue-600" style={{ width: '75%' }} />
                        <div className="h-8 bg-purple-500 dark:bg-purple-600" style={{ width: '15%', marginLeft: '2px' }} />
                        <div className="h-8 bg-green-500 dark:bg-green-600" style={{ width: '13%', marginLeft: '2px' }} />
                      </div>
                    </div>
                    <div className="w-full flex items-center">
                      <span className="text-xs text-muted-foreground w-24">Noisy Data</span>
                      <div className="flex-1 flex">
                        <div className="h-8 bg-blue-500 dark:bg-blue-600" style={{ width: '82%' }} />
                        <div className="h-8 bg-purple-500 dark:bg-purple-600" style={{ width: '8%', marginLeft: '2px' }} />
                        <div className="h-8 bg-green-500 dark:bg-green-600" style={{ width: '10%', marginLeft: '2px' }} />
                      </div>
                    </div>
                    <div className="w-full flex items-center">
                      <span className="text-xs text-muted-foreground w-24">High Dimensions</span>
                      <div className="flex-1 flex">
                        <div className="h-8 bg-blue-500 dark:bg-blue-600" style={{ width: '70%' }} />
                        <div className="h-8 bg-purple-500 dark:bg-purple-600" style={{ width: '15%', marginLeft: '2px' }} />
                        <div className="h-8 bg-green-500 dark:bg-green-600" style={{ width: '18%', marginLeft: '2px' }} />
                      </div>
                    </div>
                    <div className="w-full flex items-center">
                      <span className="text-xs text-muted-foreground w-24">Imbalanced Data</span>
                      <div className="flex-1 flex">
                        <div className="h-8 bg-blue-500 dark:bg-blue-600" style={{ width: '68%' }} />
                        <div className="h-8 bg-purple-500 dark:bg-purple-600" style={{ width: '12%', marginLeft: '2px' }} />
                        <div className="h-8 bg-green-500 dark:bg-green-600" style={{ width: '14%', marginLeft: '2px' }} />
                      </div>
                    </div>
                  </div>
                </div>
                <div className="flex justify-center mt-4 space-x-6">
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-blue-500 dark:bg-blue-600 mr-2"></div>
                    <span className="text-xs">Bagging</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-purple-500 dark:bg-purple-600 mr-2"></div>
                    <span className="text-xs">Boosting</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-green-500 dark:bg-green-600 mr-2"></div>
                    <span className="text-xs">Stacking</span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Computational Complexity</h4>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm">Bagging</span>
                        <span className="text-sm text-muted-foreground">Medium</span>
                      </div>
                      <Progress value={60} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm">Boosting</span>
                        <span className="text-sm text-muted-foreground">High</span>
                      </div>
                      <Progress value={80} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm">Stacking</span>
                        <span className="text-sm text-muted-foreground">Very High</span>
                      </div>
                      <Progress value={95} className="h-2" />
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mt-3">
                    Computational complexity increases from bagging to boosting to stacking, with stacking being the
                    most resource-intensive due to multiple models and cross-validation.
                  </p>
                </Card>

                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Interpretability</h4>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm">Bagging</span>
                        <span className="text-sm text-muted-foreground">Medium</span>
                      </div>
                      <Progress value={50} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm">Boosting</span>
                        <span className="text-sm text-muted-foreground">Medium-Low</span>
                      </div>
                      <Progress value={40} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm">Stacking</span>
                        <span className="text-sm text-muted-foreground">Low</span>
                      </div>
                      <Progress value={20} className="h-2" />
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mt-3">
                    As ensemble complexity increases, interpretability generally decreases. Stacking with diverse models
                    is particularly challenging to interpret.
                  </p>
                </Card>
              </div>

              <div className="bg-muted/20 p-6 rounded-lg border mt-4">
                <h4 className="font-medium text-lg mb-4">Choosing the Right Ensemble Method</h4>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-muted/50">
                        <th className="border px-4 py-2 text-left">If You Need...</th>
                        <th className="border px-4 py-2 text-left">Consider</th>
                        <th className="border px-4 py-2 text-left">Why</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border px-4 py-2">Stability and reduced overfitting</td>
                        <td className="border px-4 py-2 font-medium">Bagging (Random Forest)</td>
                        <td className="border px-4 py-2 text-sm">
                          Reduces variance through averaging multiple models trained on different data subsets
                        </td>
                      </tr>
                      <tr className="bg-muted/20">
                        <td className="border px-4 py-2">Maximum accuracy on clean data</td>
                        <td className="border px-4 py-2 font-medium">Boosting (XGBoost, LightGBM)</td>
                        <td className="border px-4 py-2 text-sm">
                          Sequentially focuses on difficult examples to reduce bias and improve accuracy
                        </td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2">Leveraging multiple algorithms</td>
                        <td className="border px-4 py-2 font-medium">Stacking</td>
                        <td className="border px-4 py-2 text-sm">
                          Combines strengths of diverse models through a meta-learner
                        </td>
                      </tr>
                      <tr className="bg-muted/20">
                        <td className="border px-4 py-2">Fast training and parallelization</td>
                        <td className="border px-4 py-2 font-medium">Bagging</td>
                        <td className="border px-4 py-2 text-sm">Independent models can be trained in parallel</td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2">Feature importance insights</td>
                        <td className="border px-4 py-2 font-medium">Boosting (especially XGBoost)</td>
                        <td className="border px-4 py-2 text-sm">Provides detailed feature importance metrics</td>
                      </tr>
                      <tr className="bg-muted/20">
                        <td className="border px-4 py-2">Winning competitions</td>
                        <td className="border px-4 py-2 font-medium">Stacking or Multi-level Stacking</td>
                        <td className="border px-4 py-2 text-sm">
                          Squeezes out maximum performance by combining multiple strong models
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      )
    }

    // Default return if section is not found
    return (
      <div className="py-8 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <BookOpen className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-xl font-medium mb-2">Section Content Coming Soon</h3>
        <p className="text-muted-foreground max-w-md mx-auto">
          We're currently developing content for this section of the ensemble learning tutorial. Check back soon!
        </p>
      </div>
    )
  }