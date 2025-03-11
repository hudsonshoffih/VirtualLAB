"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import {
  TreeDeciduous,
  BookOpen,
  Code,
  BarChart,
  Lightbulb,
  CheckCircle,
  ArrowRight,
  GitBranch,
  Shuffle,
  Sliders,
  ChevronLeft,
  ChevronRight,
  Trees,
  LineChart,
  Settings,
  Workflow,
  Sparkles,
} from "lucide-react"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartTooltipItem,
  ChartBar,
  ChartBarItem,
  ChartLine,
  ChartLineItem,
  ChartTitle,
} from "@/components/ui/chart"

interface randomForestTutorialProps {
    section: number
    onCopy: (text: string, id: string) => void
    copied: string | null
  }
  
  export function RandomForestTutorial({ section, onCopy, copied }: randomForestTutorialProps) {
    const [activeTab, setActiveTab] = useState("explanation")

  // Render content based on current section
  const [copiedState, setCopiedState] = useState<string | null>(null);

    // Section 0: Introduction to Random Forest
    if (section === 0) {
      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950 dark:to-teal-950 p-6 rounded-lg border border-emerald-100 dark:border-emerald-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-emerald-100 dark:bg-emerald-900 p-2 rounded-full">
                <Trees className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold text-emerald-800 dark:text-emerald-300">What is Random Forest?</h3>
            </div>
            <p className="text-emerald-700 dark:text-emerald-300 leading-relaxed">
              Random Forest is a powerful ensemble learning algorithm that combines multiple decision trees to make more
              accurate and robust predictions. By leveraging the wisdom of many trees and introducing randomness in both
              data sampling and feature selection, Random Forest overcomes the limitations of individual decision trees
              while maintaining their interpretability.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="h-5 w-5 text-emerald-500" />
                <h4 className="font-medium text-lg">Key Concepts</h4>
              </div>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    1
                  </span>
                  <span>Ensemble learning (combines multiple models)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    2
                  </span>
                  <span>Bootstrapping (random sampling with replacement)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    3
                  </span>
                  <span>Feature randomness (subset of features at each split)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    4
                  </span>
                  <span>Majority voting (for classification) or averaging (for regression)</span>
                </li>
              </ul>
            </Card>

            <Card className="p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center gap-2 mb-3">
                <Lightbulb className="h-5 w-5 text-yellow-500" />
                <h4 className="font-medium text-lg">Applications</h4>
              </div>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Credit risk assessment and fraud detection</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Disease diagnosis and healthcare predictions</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Stock market analysis and price prediction</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Customer churn prediction and recommendation systems</span>
                </li>
              </ul>
            </Card>
          </div>

          <div className="mt-6">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              What You'll Learn
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {[
                { icon: <TreeDeciduous className="h-4 w-4" />, text: "Decision Tree Fundamentals" },
                { icon: <Shuffle className="h-4 w-4" />, text: "Bootstrapping & Bagging" },
                { icon: <GitBranch className="h-4 w-4" />, text: "Feature Randomness" },
                { icon: <Code className="h-4 w-4" />, text: "Implementation with Scikit-Learn" },
                { icon: <Sliders className="h-4 w-4" />, text: "Hyperparameter Tuning" },
                { icon: <BarChart className="h-4 w-4" />, text: "Feature Importance" },
                { icon: <LineChart className="h-4 w-4" />, text: "Model Evaluation" },
                { icon: <Workflow className="h-4 w-4" />, text: "Practical Applications" },
              ].map((item, index) => (
                <div key={index} className="bg-muted/50 p-3 rounded-md flex items-center gap-2">
                  {item.icon}
                  <span className="text-sm">{item.text}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-muted/30 p-5 rounded-lg border mt-6">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <h4 className="font-medium text-lg">Prerequisites</h4>
            </div>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-1 text-primary" />
                <span>Basic understanding of Python programming</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-1 text-primary" />
                <span>Familiarity with fundamental machine learning concepts</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-1 text-primary" />
                <span>Knowledge of basic statistics (mean, variance, etc.)</span>
              </li>
            </ul>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950 dark:to-teal-950 p-6 rounded-lg border border-emerald-100 dark:border-emerald-900">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-emerald-100 dark:bg-emerald-900 p-2 rounded-full">
                  <Sparkles className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                </div>
                <h3 className="text-lg font-semibold text-emerald-800 dark:text-emerald-300">
                  Advantages of Random Forest
                </h3>
              </div>
              <ul className="space-y-3 text-emerald-700 dark:text-emerald-300">
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Reduces overfitting compared to single decision trees</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Handles high-dimensional data without feature scaling</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Provides feature importance measures</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Robust to outliers and non-linear data</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Can handle missing values effectively</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950 dark:to-orange-950 p-6 rounded-lg border border-amber-100 dark:border-amber-900">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-amber-100 dark:bg-amber-900 p-2 rounded-full">
                  <Settings className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                </div>
                <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-300">Limitations to Consider</h3>
              </div>
              <ul className="space-y-3 text-amber-700 dark:text-amber-300">
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>More computationally expensive than single decision trees</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Less interpretable than a single decision tree</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Prediction time can be slower for large forests</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>May not perform well on very small datasets</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Can be biased towards features with more levels in categorical data</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-muted/20 p-6 rounded-lg border mt-6">
            <h4 className="font-medium text-lg mb-4 flex items-center gap-2">
              <Workflow className="h-5 w-5 text-primary" />
              Random Forest at a Glance
            </h4>
            <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
              <img
                src="/placeholder.svg?height=400&width=800"
                alt="Random Forest Overview"
                className="max-w-full h-auto"
                onError={(e) => {
                  e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*i0o8mjFfCn-uD79-F1Cqkw.png"
                }}
              />
            </div>
            <p className="mt-4 text-sm text-muted-foreground">
              The diagram illustrates how Random Forest works: multiple decision trees are trained on different subsets
              of the data and features. Each tree makes its own prediction, and the final result is determined by
              majority voting (for classification) or averaging (for regression).
            </p>
          </div>
        </div>
      )
    }

    // Section 1: Understanding Decision Trees
    if (section === 1) {
        function copyToClipboard(decisionTreeCode: string, arg1: string): void {
            throw new Error("Function not implemented.")
        }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950 dark:to-teal-950 p-6 rounded-lg border border-emerald-100 dark:border-emerald-900">
            <h3 className="text-xl font-semibold text-emerald-800 dark:text-emerald-300 mb-3">
              Understanding Decision Trees
            </h3>
            <p className="text-emerald-700 dark:text-emerald-300 leading-relaxed">
              Decision trees are the building blocks of Random Forest. They are hierarchical structures that make
              decisions by splitting data based on feature values. Understanding how decision trees work is essential to
              grasp the power of Random Forest.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                How They Work
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Visualization
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Decision Tree Structure</h4>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Root Node</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        The top node that represents the entire dataset and the first decision point.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Internal Nodes</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Decision points that split the data based on feature values (e.g., "Is temperature &gt; 30°C?").
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Branches</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Connections between nodes representing the outcome of a decision (e.g., "Yes" or "No").
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Leaf Nodes</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Terminal nodes that provide the final prediction or classification.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">How Decision Trees Make Splits</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  Decision trees use metrics to determine the best feature and threshold for splitting data at each
                  node:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge variant="outline">Classification</Badge>
                    </h5>
                    <ul className="text-sm space-y-2">
                      <li className="flex items-start gap-2">
                        <Badge className="mt-1" variant="secondary">
                          Gini Impurity
                        </Badge>
                        <span>Measures the probability of incorrect classification</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Badge className="mt-1" variant="secondary">
                          Entropy
                        </Badge>
                        <span>Measures the disorder or uncertainty in the data</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Badge className="mt-1" variant="secondary">
                          Information Gain
                        </Badge>
                        <span>Reduction in entropy after a split</span>
                      </li>
                    </ul>
                  </div>
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge variant="outline">Regression</Badge>
                    </h5>
                    <ul className="text-sm space-y-2">
                      <li className="flex items-start gap-2">
                        <Badge className="mt-1" variant="secondary">
                          MSE
                        </Badge>
                        <span>Mean Squared Error between actual and predicted values</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Badge className="mt-1" variant="secondary">
                          MAE
                        </Badge>
                        <span>Mean Absolute Error between actual and predicted values</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Badge className="mt-1" variant="secondary">
                          Variance Reduction
                        </Badge>
                        <span>Reduction in variance after a split</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Example Decision Process</h4>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <h5 className="font-medium mb-3">Weather Example: Should I play outside?</h5>
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <ArrowRight className="h-4 w-4 text-primary" />
                      <span>
                        <strong>Root Node:</strong> Is it raining?
                      </span>
                    </div>
                    <div className="pl-6">
                      <div className="flex items-center gap-2">
                        <span className="text-red-500">No →</span>
                        <span>
                          <strong>Internal Node:</strong> Is temperature &gt; 30°C?
                        </span>
                      </div>
                      <div className="pl-6">
                        <div className="flex items-center gap-2">
                          <span className="text-red-500">No →</span>
                          <span>
                            <strong>Leaf Node:</strong> Play outside (Prediction)
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-red-500">Yes →</span>
                          <span>
                            <strong>Leaf Node:</strong> Stay inside (Prediction)
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="pl-6">
                      <div className="flex items-center gap-2">
                        <span className="text-red-500">Yes →</span>
                        <span>
                          <strong>Leaf Node:</strong> Stay inside (Prediction)
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Limitations of Single Decision Trees</h4>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Overfitting:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Single trees can memorize the training data rather than learning general patterns
                      </span>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">High Variance:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Small changes in the data can result in very different trees
                      </span>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Bias Towards Dominant Classes:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Can be biased towards majority classes in imbalanced datasets
                      </span>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Instability:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Prone to errors if certain features are missing or contain noise
                      </span>
                    </div>
                  </li>
                </ul>
                <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-950 rounded-lg">
                  <p className="text-sm text-amber-700 dark:text-amber-300">
                    <strong>Note:</strong> Random Forest addresses these limitations by combining multiple trees and
                    introducing randomness in the training process.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Simple Decision Tree Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(decisionTreeCode, "decision-tree-code")}
                    className="text-xs"
                  >
                    {copiedState === "decision-tree-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{decisionTreeCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This simplified implementation demonstrates the core concepts of a decision tree. It includes
                    functions to calculate Gini impurity, find the best split, and recursively build the tree structure.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Decision Tree Visualization</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Decision Tree Visualization"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src =
                        "https://scikit-learn.org/stable/_images/sphx_glr_plot_tree_regression_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The visualization shows a decision tree with:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Root node at the top representing the first decision</li>
                    <li>Internal nodes showing decision criteria (feature and threshold)</li>
                    <li>Branches representing the outcomes of decisions (True/False)</li>
                    <li>Leaf nodes at the bottom showing the final predictions</li>
                    <li>Color intensity indicating the confidence or purity of predictions</li>
                  </ul>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-emerald-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Decision trees make predictions by following a series of if-then rules from root to leaf</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>
                  Trees split data based on metrics like Gini impurity or entropy to maximize information gain
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Single decision trees are prone to overfitting and high variance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Random Forest addresses these limitations by combining multiple trees</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 2: Bootstrapping & Bagging
    if (section === 2) {
        function copyToClipboard(bootstrappingCode: string, arg1: string): void {
            throw new Error("Function not implemented.")
        }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
            <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-3">Bootstrapping & Bagging</h3>
            <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
              Bootstrapping and Bagging are key techniques that make Random Forest more powerful than individual
              decision trees. These methods introduce diversity among the trees, reducing overfitting and improving
              generalization.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Concepts
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Visualization
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">What is Bootstrapping?</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  Bootstrapping is a statistical resampling technique that involves randomly sampling a dataset with
                  replacement.
                </p>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Random Sampling</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        For each tree in the forest, a random sample of the training data is selected.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">With Replacement</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Samples are drawn with replacement, meaning the same data point can be selected multiple times.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Sample Size</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Typically, each bootstrap sample has the same size as the original dataset.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Out-of-Bag Samples</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Due to sampling with replacement, about 1/3 of the original data points are left out in each
                        bootstrap sample. These are called "out-of-bag" samples and can be used for validation.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">What is Bagging?</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  Bagging (Bootstrap Aggregating) is an ensemble technique that trains multiple models on different
                  bootstrap samples and combines their predictions.
                </p>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Multiple Models</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Train multiple decision trees, each on a different bootstrap sample of the data.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Independent Training</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Each tree is trained independently, without knowledge of the other trees.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Aggregation</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Combine the predictions of all trees through voting (for classification) or averaging (for
                        regression).
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Reduced Variance</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        The ensemble reduces variance and overfitting compared to a single model.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Benefits in Random Forest</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4 bg-blue-50 dark:bg-blue-950">
                    <h5 className="font-medium mb-2">Diversity Among Trees</h5>
                    <p className="text-sm text-muted-foreground">
                      Each tree sees a different subset of the data, leading to diverse models that capture different
                      patterns.
                    </p>
                  </div>
                  <div className="border rounded-lg p-4 bg-green-50 dark:bg-green-950">
                    <h5 className="font-medium mb-2">Reduced Overfitting</h5>
                    <p className="text-sm text-muted-foreground">
                      By averaging multiple trees, the ensemble is less likely to memorize noise in the training data.
                    </p>
                  </div>
                  <div className="border rounded-lg p-4 bg-purple-50 dark:bg-purple-950">
                    <h5 className="font-medium mb-2">Improved Stability</h5>
                    <p className="text-sm text-muted-foreground">
                      The model becomes more robust to outliers and variations in the data.
                    </p>
                  </div>
                  <div className="border rounded-lg p-4 bg-amber-50 dark:bg-amber-950">
                    <h5 className="font-medium mb-2">Built-in Validation</h5>
                    <p className="text-sm text-muted-foreground">
                      Out-of-bag samples provide a way to estimate model performance without a separate validation set.
                    </p>
                  </div>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Bootstrapping & Bagging Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(bootstrappingCode, "bootstrapping-code")}
                    className="text-xs"
                  >
                    {copiedState === "bootstrapping-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{bootstrappingCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how bootstrapping and bagging work in Random Forest. It creates multiple
                    bootstrap samples, trains a decision tree on each sample, and combines their predictions through
                    voting or averaging.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Bootstrapping & Bagging Visualization</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Bootstrapping and Bagging"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*9Cz-IXc8_RoEROVCBf9qwA.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The visualization illustrates the bootstrapping and bagging process:</p>
                  <ol className="list-decimal pl-5 space-y-1">
                    <li>Original dataset is shown at the top</li>
                    <li>Multiple bootstrap samples are created by random sampling with replacement</li>
                    <li>A decision tree is trained on each bootstrap sample</li>
                    <li>Predictions from all trees are combined through voting or averaging</li>
                    <li>The final prediction is more robust than any individual tree</li>
                  </ol>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Effect of Bagging on Model Performance</h4>
                <div className="h-80">
                  <ChartContainer>
                    <ChartTitle>Error Rate vs. Number of Trees</ChartTitle>
                    <ChartLine
                      data={[
                        { x: 1, y: 0.35 },
                        { x: 5, y: 0.22 },
                        { x: 10, y: 0.18 },
                        { x: 20, y: 0.15 },
                        { x: 50, y: 0.12 },
                        { x: 100, y: 0.11 },
                        { x: 200, y: 0.105 },
                      ]}
                    >
                      {(data: any) => (
                        <>
                          <ChartLineItem
                            data={data}
                            valueAccessor={(d) => d.y}
                            categoryAccessor={(d) => d.x}
                            style={{ stroke: "#10b981", strokeWidth: 2 }}
                          />
                        </>
                      )}
                    </ChartLine>
                    <ChartTooltip>
                      {(point: { point: any }): JSX.Element => (
                        <ChartTooltipContent>
                          <ChartTooltipItem label="Trees" value={point.point.data.x} />
                          <ChartTooltipItem label="Error Rate" value={point.point.data.y.toFixed(3)} />
                        </ChartTooltipContent>
                      )}
                    </ChartTooltip>
                  </ChartContainer>
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  <p>
                    The chart shows how the error rate decreases as more trees are added to the ensemble. The
                    improvement is significant at first but eventually plateaus, demonstrating the "wisdom of crowds"
                    effect in bagging.
                  </p>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-blue-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Bootstrapping creates diverse training sets by random sampling with replacement</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Bagging combines predictions from multiple models to reduce variance and overfitting</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Out-of-bag samples provide a built-in validation mechanism</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The ensemble's performance improves with more trees, but with diminishing returns</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 3: Feature Randomness
    if (section === 3) {
        function copyToClipboard(featureRandomnessCode: string, arg1: string): void {
            throw new Error("Function not implemented.")
        }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
            <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300 mb-3">Feature Randomness</h3>
            <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
              Feature randomness is what distinguishes Random Forest from simple bagging of decision trees. By
              considering only a subset of features at each split, Random Forest introduces additional diversity among
              trees and further reduces correlation between them.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Concepts
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Visualization
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">How Feature Randomness Works</h4>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Feature Subset Selection</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        At each node split, only a random subset of features is considered instead of all available
                        features.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Subset Size</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        For classification, the default subset size is typically √p features, where p is the total
                        number of features. For regression, the default is p/3 features.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Best Split Selection</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        From the random subset of features, the algorithm selects the best feature and split point based
                        on impurity measures (like Gini or entropy).
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Different Features for Each Node</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        A new random subset of features is selected for each node in each tree, creating highly diverse
                        trees.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Why Feature Randomness Matters</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4 bg-purple-50 dark:bg-purple-950">
                    <h5 className="font-medium mb-2">Decorrelation of Trees</h5>
                    <p className="text-sm text-muted-foreground">
                      Without feature randomness, trees might be too similar if certain features are very strong
                      predictors, leading to correlated errors.
                    </p>
                  </div>
                  <div className="border rounded-lg p-4 bg-indigo-50 dark:bg-indigo-950">
                    <h5 className="font-medium mb-2">Prevents Feature Dominance</h5>
                    <p className="text-sm text-muted-foreground">
                      Prevents strong features from dominating every tree, allowing weaker but still informative
                      features to contribute.
                    </p>
                  </div>
                  <div className="border rounded-lg p-4 bg-blue-50 dark:bg-blue-950">
                    <h5 className="font-medium mb-2">Increased Diversity</h5>
                    <p className="text-sm text-muted-foreground">
                      Creates more diverse trees than bagging alone, leading to better ensemble performance.
                    </p>
                  </div>
                  <div className="border rounded-lg p-4 bg-teal-50 dark:bg-teal-950">
                    <h5 className="font-medium mb-2">Robustness to Noise</h5>
                    <p className="text-sm text-muted-foreground">
                      Makes the model more robust to noisy features and irrelevant variables.
                    </p>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Bagging vs. Random Forest</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full border-collapse">
                    <thead>
                      <tr className="bg-muted/50">
                        <th className="border px-4 py-2 text-left">Aspect</th>
                        <th className="border px-4 py-2 text-left">Bagging (e.g., Bagged Trees)</th>
                        <th className="border px-4 py-2 text-left">Random Forest</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Data Sampling</td>
                        <td className="border px-4 py-2">Bootstrap samples</td>
                        <td className="border px-4 py-2">Bootstrap samples</td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Feature Selection</td>
                        <td className="border px-4 py-2">All features considered at each split</td>
                        <td className="border px-4 py-2">Random subset of features at each split</td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Tree Correlation</td>
                        <td className="border px-4 py-2">Higher correlation between trees</td>
                        <td className="border px-4 py-2">Lower correlation between trees</td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Diversity</td>
                        <td className="border px-4 py-2">Less diverse ensemble</td>
                        <td className="border px-4 py-2">More diverse ensemble</td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Performance</td>
                        <td className="border px-4 py-2">Good, but can be limited by feature dominance</td>
                        <td className="border px-4 py-2">Better, especially with many features</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Feature Randomness Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(featureRandomnessCode, "feature-randomness-code")}
                    className="text-xs"
                  >
                    {copiedState === "feature-randomness-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{featureRandomnessCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how feature randomness is implemented in Random Forest. At each node split,
                    only a random subset of features is considered, adding another layer of randomness beyond
                    bootstrapping.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Feature Randomness Visualization</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Feature Randomness in Random Forest"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*QcS0CXZQQXgF_lPrTgFXzA.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The visualization illustrates how feature randomness works in Random Forest:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Each tree considers only a subset of features at each node</li>
                    <li>Different trees use different feature subsets, creating diversity</li>
                    <li>Some trees may not use certain features at all</li>
                    <li>The combination of trees creates a more robust model than any individual tree</li>
                  </ul>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Impact of Feature Subset Size</h4>
                <div className="h-80">
                  <ChartContainer>
                    <ChartTitle>Model Performance vs. Feature Subset Size</ChartTitle>
                    <ChartBar
                      data={[
                        { category: "sqrt(p)", value: 0.92, color: "#8b5cf6" },
                        { category: "p/3", value: 0.89, color: "#8b5cf6" },
                        { category: "p/2", value: 0.87, color: "#8b5cf6" },
                        { category: "p", value: 0.84, color: "#8b5cf6" },
                        { category: "log2(p)", value: 0.91, color: "#8b5cf6" },
                      ]}
                    >
                      {(data: any) => (
                        <>
                          <ChartBarItem
                            data={data}
                            valueAccessor={(d) => d.value}
                            categoryAccessor={(d) => d.category}
                            style={{ fill: (d: { color: string }) => d.color }}
                          />
                        </>
                      )}
                    </ChartBar>
                    <ChartTooltip>
                      {({ point }) => (
                        <ChartTooltipContent>
                          <ChartTooltipItem label="Feature Subset Size" value={point.data.category} />
                          <ChartTooltipItem label="Accuracy" value={point.data.value.toFixed(2)} />
                        </ChartTooltipContent>
                      )}
                    </ChartTooltip>
                  </ChartContainer>
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  <p>
                    The chart shows how model performance varies with different feature subset sizes, where p is the
                    total number of features. The default values (sqrt(p) for classification and p/3 for regression)
                    often provide the best balance.
                  </p>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-purple-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Feature randomness is what distinguishes Random Forest from simple bagging</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Only a random subset of features is considered at each node split</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>This creates more diverse trees and reduces correlation between them</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The optimal feature subset size depends on the problem, but defaults often work well</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 4: Implementation with Scikit-Learn
    if (section === 4) {
        function copyToClipboard(rfRegressionCode: string, arg1: string): void {
            throw new Error("Function not implemented.")
        }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-950 dark:to-teal-950 p-6 rounded-lg border border-green-100 dark:border-green-900">
            <h3 className="text-xl font-semibold text-green-800 dark:text-green-300 mb-3">
              Implementation with Scikit-Learn
            </h3>
            <p className="text-green-700 dark:text-green-300 leading-relaxed">
              While understanding the theory behind Random Forest is important, for practical applications, we can
              leverage scikit-learn's optimized implementation. Let's explore how to implement Random Forest for both
              classification and regression tasks.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Implementation
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Results
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Scikit-Learn Random Forest Classes</h4>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>RandomForestClassifier</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Used for classification tasks where the target variable is categorical.
                    </p>
                    <div className="bg-muted/50 p-3 rounded text-sm">
                      <p className="font-mono">from sklearn.ensemble import RandomForestClassifier</p>
                      <p className="font-mono mt-1">rf = RandomForestClassifier(n_estimators=100, random_state=42)</p>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>RandomForestRegressor</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Used for regression tasks where the target variable is continuous.
                    </p>
                    <div className="bg-muted/50 p-3 rounded text-sm">
                      <p className="font-mono">from sklearn.ensemble import RandomForestRegressor</p>
                      <p className="font-mono mt-1">rf = RandomForestRegressor(n_estimators=100, random_state=42)</p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Important Parameters</h4>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      n_estimators
                    </Badge>
                    <div>
                      <p className="text-sm">Number of trees in the forest (default: 100)</p>
                      <p className="text-xs text-muted-foreground">
                        Higher values generally improve performance but increase computation time
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      max_depth
                    </Badge>
                    <div>
                      <p className="text-sm">Maximum depth of each tree (default: None)</p>
                      <p className="text-xs text-muted-foreground">
                        Controls complexity; lower values reduce overfitting
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      min_samples_split
                    </Badge>
                    <div>
                      <p className="text-sm">Minimum samples required to split a node (default: 2)</p>
                      <p className="text-xs text-muted-foreground">
                        Higher values prevent creating nodes with few samples
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      min_samples_leaf
                    </Badge>
                    <div>
                      <p className="text-sm">Minimum samples required in a leaf node (default: 1)</p>
                      <p className="text-xs text-muted-foreground">Higher values create more balanced trees</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      max_features
                    </Badge>
                    <div>
                      <p className="text-sm">
                        Number of features to consider for best split (default: 'sqrt' for classification, 'auto' for
                        regression)
                      </p>
                      <p className="text-xs text-muted-foreground">Controls feature randomness</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      bootstrap
                    </Badge>
                    <div>
                      <p className="text-sm">Whether to use bootstrap samples (default: True)</p>
                      <p className="text-xs text-muted-foreground">
                        If False, the whole dataset is used to build each tree
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      random_state
                    </Badge>
                    <div>
                      <p className="text-sm">Seed for random number generation (default: None)</p>
                      <p className="text-xs text-muted-foreground">Set for reproducible results</p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Implementation Steps</h4>
                <ol className="space-y-3 list-decimal pl-5">
                  <li>Import the necessary libraries and classes</li>
                  <li>Load and prepare your dataset</li>
                  <li>Split the data into training and testing sets</li>
                  <li>Create and configure the Random Forest model</li>
                  <li>Train the model using the fit() method</li>
                  <li>Make predictions using the predict() method</li>
                  <li>Evaluate the model's performance</li>
                  <li>Analyze feature importance (if needed)</li>
                </ol>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Random Forest Classification</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(rfClassificationCode, "rf-classification-code")}
                    className="text-xs"
                  >
                    {copiedState === "rf-classification-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{rfClassificationCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This example demonstrates how to implement Random Forest for a classification task using
                    scikit-learn. It includes data loading, preprocessing, model training, prediction, and evaluation.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Random Forest Regression</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(rfRegressionCode, "rf-regression-code")}
                    className="text-xs"
                  >
                    {copiedState === "rf-regression-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{rfRegressionCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This example shows how to implement Random Forest for a regression task. The approach is similar to
                    classification, but uses RandomForestRegressor and different evaluation metrics.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Classification Results</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=400"
                      alt="Confusion Matrix"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src =
                          "https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_001.png"
                      }}
                    />
                  </div>
                  <div className="flex flex-col justify-between">
                    <div>
                      <h5 className="font-medium mb-2">Confusion Matrix</h5>
                      <p className="text-sm text-muted-foreground mb-4">
                        The confusion matrix shows how well our Random Forest model is performing for each class. The
                        diagonal elements represent correct predictions, while off-diagonal elements are
                        misclassifications.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h5 className="font-medium mb-2">Performance Metrics</h5>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Accuracy:</span>
                          <Badge variant="outline">97.4%</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Precision:</span>
                          <Badge variant="outline">97.8%</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Recall:</span>
                          <Badge variant="outline">97.3%</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">F1 Score:</span>
                          <Badge variant="outline">97.5%</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Regression Results</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=400"
                      alt="Predicted vs Actual Values"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src =
                          "https://scikit-learn.org/stable/_images/sphx_glr_plot_forest_regression_001.png"
                      }}
                    />
                  </div>
                  <div className="flex flex-col justify-between">
                    <div>
                      <h5 className="font-medium mb-2">Predicted vs Actual Values</h5>
                      <p className="text-sm text-muted-foreground mb-4">
                        The scatter plot shows the relationship between predicted and actual values. Points closer to
                        the diagonal line indicate better predictions.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h5 className="font-medium mb-2">Performance Metrics</h5>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">R² Score:</span>
                          <Badge variant="outline">0.923</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Mean Absolute Error:</span>
                          <Badge variant="outline">0.342</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Mean Squared Error:</span>
                          <Badge variant="outline">0.187</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Root Mean Squared Error:</span>
                          <Badge variant="outline">0.432</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-green-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Scikit-learn provides optimized implementations for both classification and regression</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The API is consistent with other scikit-learn models: fit(), predict(), and score()</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Key parameters like n_estimators and max_features allow fine-tuning the model</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Random Forest typically performs well even with default parameters</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 5: Hyperparameter Tuning
    if (section === 5) {
        function copyToClipboard(gridSearchCode: string, arg1: string): void {
            throw new Error("Function not implemented.")
        }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950 dark:to-orange-950 p-6 rounded-lg border border-amber-100 dark:border-amber-900">
            <h3 className="text-xl font-semibold text-amber-800 dark:text-amber-300 mb-3">Hyperparameter Tuning</h3>
            <p className="text-amber-700 dark:text-amber-300 leading-relaxed">
              While Random Forest often performs well with default parameters, tuning hyperparameters can further
              improve performance. Let's explore how to find the optimal configuration for your specific dataset.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Key Parameters
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Results
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Key Hyperparameters to Tune</h4>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>n_estimators</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">The number of trees in the forest.</p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Higher values generally improve performance but increase computation time</li>
                      <li>Diminishing returns after a certain point</li>
                      <li>Typical range: 50-500</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>max_depth</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">The maximum depth of each tree.</p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Controls complexity of the model</li>
                      <li>Lower values reduce overfitting</li>
                      <li>Typical range: 5-30 or None (unlimited)</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>min_samples_split</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      The minimum number of samples required to split an internal node.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Higher values prevent creating nodes with few samples</li>
                      <li>Helps control overfitting</li>
                      <li>Typical range: 2-20</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>min_samples_leaf</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      The minimum number of samples required to be at a leaf node.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Higher values create more balanced trees</li>
                      <li>Helps prevent overfitting</li>
                      <li>Typical range: 1-10</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>max_features</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      The number of features to consider when looking for the best split.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>'sqrt': sqrt(n_features)</li>
                      <li>'log2': log2(n_features)</li>
                      <li>int: specific number of features</li>
                      <li>float: fraction of features</li>
                    </ul>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Tuning Methods</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4 bg-amber-50 dark:bg-amber-950">
                    <h5 className="font-medium mb-2">Grid Search</h5>
                    <p className="text-sm text-muted-foreground">
                      Exhaustively searches through a specified parameter grid, evaluating all possible combinations.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5 mt-2">
                      <li>Thorough but computationally expensive</li>
                      <li>Good for small parameter spaces</li>
                      <li>Guaranteed to find the best combination in the grid</li>
                    </ul>
                  </div>
                  <div className="border rounded-lg p-4 bg-orange-50 dark:bg-orange-950">
                    <h5 className="font-medium mb-2">Random Search</h5>
                    <p className="text-sm text-muted-foreground">
                      Randomly samples parameter combinations from specified distributions.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5 mt-2">
                      <li>More efficient for large parameter spaces</li>
                      <li>Can explore more values per parameter</li>
                      <li>Often finds good solutions with fewer iterations</li>
                    </ul>
                  </div>
                </div>
                <div className="border rounded-lg p-4 mt-4 bg-red-50 dark:bg-red-950">
                  <h5 className="font-medium mb-2">Bayesian Optimization</h5>
                  <p className="text-sm text-muted-foreground">
                    Uses probabilistic models to guide the search for optimal parameters.
                  </p>
                  <ul className="text-sm space-y-1 list-disc pl-5 mt-2">
                    <li>More efficient than grid or random search</li>
                    <li>Learns from previous evaluations</li>
                    <li>Particularly useful for expensive-to-evaluate models</li>
                    <li>Implemented in libraries like scikit-optimize or Optuna</li>
                  </ul>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Cross-Validation</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  When tuning hyperparameters, it's important to use cross-validation to get reliable performance
                  estimates.
                </p>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <h5 className="font-medium mb-2">K-Fold Cross-Validation</h5>
                  <ol className="text-sm space-y-2 list-decimal pl-5">
                    <li>Split the data into K folds (typically 5 or 10)</li>
                    <li>
                      For each parameter combination:
                      <ul className="list-disc pl-5 mt-1">
                        <li>Train on K-1 folds</li>
                        <li>Validate on the remaining fold</li>
                        <li>Repeat for all K folds</li>
                        <li>Average the performance metrics</li>
                      </ul>
                    </li>
                    <li>Select the parameter combination with the best average performance</li>
                    <li>Retrain on the entire training set with the best parameters</li>
                    <li>Evaluate on the test set (which was not used in tuning)</li>
                  </ol>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Grid Search CV</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(gridSearchCode, "grid-search-code")}
                    className="text-xs"
                  >
                    {copiedState === "grid-search-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{gridSearchCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how to use GridSearchCV to find the optimal hyperparameters for a Random
                    Forest model. It exhaustively searches through all combinations of the specified parameter values.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Random Search CV</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(gridSearchCode, "random-search-code")}
                    className="text-xs"
                  >
                    {copiedState === "random-search-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{gridSearchCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code shows how to use RandomizedSearchCV, which is more efficient for large parameter spaces.
                    It randomly samples parameter combinations rather than trying all possibilities.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Parameter Importance</h4>
                <div className="h-80">
                  <ChartContainer>
                    <ChartTitle>Hyperparameter Importance</ChartTitle>
                    <ChartBar
                      data={[
                        { parameter: "n_estimators", importance: 0.35, color: "#f59e0b" },
                        { parameter: "max_depth", importance: 0.28, color: "#f59e0b" },
                        { parameter: "min_samples_split", importance: 0.18, color: "#f59e0b" },
                        { parameter: "max_features", importance: 0.12, color: "#f59e0b" },
                        { parameter: "min_samples_leaf", importance: 0.07, color: "#f59e0b" },
                      ]}
                    >
                      {(data: any) => (
                        <>
                          <ChartBarItem
                            data={data}
                            valueAccessor={(d) => d.importance}
                            categoryAccessor={(d) => d.parameter}
                            style={{ fill: (d: { color: string }) => d.color }}
                          />
                        </>
                      )}
                    </ChartBar>
                    <ChartTooltip>
                      {({ point }: { point: any }) => (
                        <ChartTooltipContent>
                          <ChartTooltipItem label="Parameter" value={point.data.parameter} />
                          <ChartTooltipItem label="Importance" value={point.data.importance.toFixed(2)} />
                        </ChartTooltipContent>
                      )}
                    </ChartTooltip>
                  </ChartContainer>
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  <p>
                    The chart shows the relative importance of different hyperparameters in determining model
                    performance. This can help focus tuning efforts on the most influential parameters.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Performance Comparison</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=300&width=500"
                      alt="Hyperparameter Tuning Results"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src =
                          "https://scikit-learn.org/stable/_images/sphx_glr_plot_randomized_search_001.png"
                      }}
                    />
                  </div>
                  <div className="flex flex-col justify-between">
                    <div>
                      <h5 className="font-medium mb-2">Tuning Results</h5>
                      <p className="text-sm text-muted-foreground mb-4">
                        The plot shows how model performance varies with different hyperparameter combinations. Each
                        point represents a specific configuration, with the best ones highlighted.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h5 className="font-medium mb-2">Best Parameters</h5>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between items-center">
                          <span>n_estimators:</span>
                          <Badge variant="outline">200</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>max_depth:</span>
                          <Badge variant="outline">15</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>min_samples_split:</span>
                          <Badge variant="outline">5</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>max_features:</span>
                          <Badge variant="outline">'sqrt'</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>Performance Improvement:</span>
                          <Badge variant="outline">+3.2%</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-amber-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Hyperparameter tuning can significantly improve Random Forest performance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Focus on key parameters like n_estimators, max_depth, and max_features</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Use cross-validation to get reliable performance estimates</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>
                  Grid Search is thorough but slow; Random Search is more efficient for large parameter spaces
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Always evaluate the final model on a separate test set not used during tuning</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 6: Visualization & Interpretation
    if (section === 6) {
        function copyToClipboard(featureImportanceCode: string, arg1: string): void {
            throw new Error("Function not implemented.")
        }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-950 dark:to-blue-950 p-6 rounded-lg border border-indigo-100 dark:border-indigo-900">
            <h3 className="text-xl font-semibold text-indigo-800 dark:text-indigo-300 mb-3">
              Visualization & Interpretation
            </h3>
            <p className="text-indigo-700 dark:text-indigo-300 leading-relaxed">
              One of the advantages of Random Forest is its interpretability. While not as transparent as a single
              decision tree, Random Forest provides several ways to understand how it makes predictions and which
              features are most important.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Interpretation
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Visualizations
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Feature Importance</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  Feature importance measures how much each feature contributes to the model's predictions. Random
                  Forest provides several ways to calculate feature importance:
                </p>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Mean Decrease in Impurity (MDI)</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground">
                      Also known as Gini importance, this measures the total reduction in node impurity (Gini impurity
                      or entropy) averaged over all trees.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5 mt-2">
                      <li>Available via the feature_importances_ attribute</li>
                      <li>Fast to compute as it's calculated during training</li>
                      <li>Can be biased towards high cardinality features</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Permutation Importance</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground">
                      Measures the decrease in model performance when a feature's values are randomly shuffled.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5 mt-2">
                      <li>More reliable than MDI, especially for correlated features</li>
                      <li>Computationally more expensive</li>
                      <li>Based on actual performance decrease, not just tree structure</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Drop-Column Importance</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground">
                      Measures the decrease in model performance when a feature is removed and the model is retrained.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5 mt-2">
                      <li>Most computationally expensive method</li>
                      <li>Accounts for the model's ability to adapt without the feature</li>
                      <li>Closest to the true impact of removing a feature</li>
                    </ul>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Partial Dependence Plots</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  Partial dependence plots show how the model's predictions change as a function of a single feature,
                  while averaging out the effects of all other features.
                </p>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">What They Show:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        The marginal effect of a feature on the predicted outcome
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Interpretation:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        A flat line means the feature has little impact; a steep curve indicates strong influence
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Limitations:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Assumes features are independent; may not be accurate for highly correlated features
                      </span>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">SHAP Values</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance based on
                  game theory.
                </p>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">What They Show:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        How much each feature contributes to the prediction for each individual instance
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Advantages:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Consistent, locally accurate, and can handle feature interactions
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Visualizations:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Summary plots, force plots, and dependence plots provide different perspectives
                      </span>
                    </div>
                  </div>
                </div>
                <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <p className="text-sm text-blue-700 dark:text-blue-300">
                    <strong>Note:</strong> SHAP values are implemented in the SHAP library, which works well with
                    scikit-learn models including Random Forest.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Feature Importance</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(featureImportanceCode, "feature-importance-code")}
                    className="text-xs"
                  >
                    {copiedState === "feature-importance-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{featureImportanceCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how to calculate and visualize different types of feature importance:
                    built-in importance, permutation importance, and drop-column importance.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">SHAP Values</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(shapValuesCode, "shap-values-code")}
                    className="text-xs"
                  >
                    {copiedState === "shap-values-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{shapValuesCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code shows how to use the SHAP library to calculate and visualize SHAP values for a Random
                    Forest model, providing detailed insights into feature contributions.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Feature Importance Visualization</h4>
                <div className="h-80">
                  <ChartContainer>
                    <ChartTitle>Feature Importance</ChartTitle>
                    <ChartBar
                      data={[
                        { feature: "Feature A", importance: 0.32, color: "#6366f1" },
                        { feature: "Feature B", importance: 0.25, color: "#6366f1" },
                        { feature: "Feature C", importance: 0.18, color: "#6366f1" },
                        { feature: "Feature D", importance: 0.12, color: "#6366f1" },
                        { feature: "Feature E", importance: 0.08, color: "#6366f1" },
                        { feature: "Feature F", importance: 0.05, color: "#6366f1" },
                      ]}
                    >
                      {(data) => (
                        <>
                          <ChartBarItem
                            data={data}
                            valueAccessor={(d) => d.importance}
                            categoryAccessor={(d) => d.feature}
                            style={{ fill: (d: { color: string }) => d.color }}
                          />
                        </>
                      )}
                    </ChartBar>
                    <ChartTooltip>
                      {({ point }) => (
                        <ChartTooltipContent>
                          <ChartTooltipItem label="Feature" value={point.data.feature} />
                          <ChartTooltipItem label="Importance" value={point.data.importance.toFixed(2)} />
                        </ChartTooltipContent>
                      )}
                    </ChartTooltip>
                  </ChartContainer>
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  <p>
                    The chart shows the relative importance of each feature in the Random Forest model. Features are
                    ranked by their contribution to the model's predictions, with higher values indicating more
                    important features.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">SHAP Summary Plot</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="SHAP Summary Plot"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://shap.readthedocs.io/en/latest/_images/shap_summary_plot.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The SHAP summary plot shows:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Features ranked by importance from top to bottom</li>
                    <li>Each point represents a SHAP value for an instance</li>
                    <li>Color indicates the feature value (red = high, blue = low)</li>
                    <li>
                      Position on the x-axis shows whether the effect increases (right) or decreases (left) the
                      prediction
                    </li>
                    <li>Clustering of points shows common patterns in the data</li>
                  </ul>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Partial Dependence Plot</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Partial Dependence Plot"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src =
                        "https://scikit-learn.org/stable/_images/sphx_glr_plot_partial_dependence_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The partial dependence plot shows:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>How the predicted outcome changes as a feature value changes</li>
                    <li>The y-axis represents the model's prediction</li>
                    <li>The x-axis represents the feature value</li>
                    <li>The curve shows the marginal effect of the feature on the prediction</li>
                    <li>Steeper curves indicate stronger influence on the prediction</li>
                  </ul>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-indigo-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Feature importance helps identify which variables have the most impact on predictions</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Permutation importance is often more reliable than the built-in feature importance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Partial dependence plots show how predictions change as a feature value changes</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>
                  SHAP values provide detailed insights into how each feature contributes to individual predictions
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>These interpretation tools help build trust and understanding in the model's decisions</span>
              </li>
            </ul>
          </Card>

          <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-950 dark:to-blue-950 p-6 rounded-lg border border-indigo-100 dark:border-indigo-900 mt-6">
            <h3 className="text-xl font-semibold text-indigo-800 dark:text-indigo-300 mb-3">Conclusion</h3>
            <p className="text-indigo-700 dark:text-indigo-300 leading-relaxed mb-4">
              Random Forest is a powerful and versatile algorithm that combines the simplicity of decision trees with
              the power of ensemble learning. By understanding its core concepts and implementation details, you can
              effectively apply it to a wide range of machine learning problems.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">What We've Learned</h4>
                <ul className="text-sm space-y-1 list-disc pl-5">
                  <li>The fundamentals of decision trees</li>
                  <li>How bootstrapping and bagging reduce overfitting</li>
                  <li>The role of feature randomness in creating diverse trees</li>
                  <li>Implementation with scikit-learn for both classification and regression</li>
                  <li>Hyperparameter tuning to optimize performance</li>
                  <li>Interpretation techniques to understand model predictions</li>
                </ul>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Next Steps</h4>
                <ul className="text-sm space-y-1 list-disc pl-5">
                  <li>Explore other ensemble methods like Gradient Boosting</li>
                  <li>Apply Random Forest to real-world datasets</li>
                  <li>Experiment with feature engineering to improve performance</li>
                  <li>Combine Random Forest with other models in a voting ensemble</li>
                  <li>Deploy your model in a production environment</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="flex justify-center mt-8">
            <Button size="lg" className="gap-2">
              <ArrowRight className="h-4 w-4" />
              Continue to Practice Exercises
            </Button>
          </div>
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
          We're currently developing content for this section of the Random Forest tutorial. Check back soon!
        </p>
      </div>
    )
  }

// Code examples as constants
const decisionTreeCode = `import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """Build the decision tree"""
        self.tree = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X):
        """Predict class for X"""
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _grow_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # Return the most common class
            leaf_value = Counter(y).most_common(1)[0][0]
            return leaf_value

        # Find the best split
        best_split = self._best_split(X, y)
        
        # If no split improves the impurity, make a leaf
        if best_split is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return leaf_value

        # Split the data
        feature_idx = best_split["feature_idx"]
        threshold = best_split["threshold"]
        
        # Create child subtrees
        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = ~left_idxs
        
        left_subtree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return {
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def _best_split(self, X, y):
        """Find the best split"""
        n_samples, n_features = X.shape
        
        # If not enough samples, don't split
        if n_samples < self.min_samples_split:
            return None
        
        # Calculate parent impurity
        parent_impurity = self._gini(y)
        
        best_gain = 0
        best_split = None
        
        # Try all features
        for feature_idx in range(n_features):
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Try all possible thresholds
            for threshold in thresholds:
                # Split the data
                left_idxs = X[:, feature_idx] <= threshold
                right_idxs = ~left_idxs
                
                # Skip if one side is empty
                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue
                
                # Calculate impurity for children
                left_impurity = self._gini(y[left_idxs])
                right_impurity = self._gini(y[right_idxs])
                
                # Calculate the weighted average impurity
                n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                
                # Calculate information gain
                info_gain = parent_impurity - weighted_impurity
                
                # Update best split if this is better
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_split = {"feature_idx": feature_idx, "threshold": threshold}
        
        return best_split

    def _gini(self, y):
        """Calculate Gini impurity"""
        # Count occurrences of each class
        counts = np.bincount(y)
        # Remove zeros
        counts = counts[counts > 0]
        # Calculate probabilities
        probabilities = counts / len(y)
        # Calculate Gini impurity
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _predict_sample(self, sample, tree):
        """Predict class for a single sample"""
        # If the tree is a leaf (not a dictionary), return the value
        if not isinstance(tree, dict):
            return tree
        
        # Get the feature and threshold from the tree
        feature_idx = tree["feature_idx"]
        threshold = tree["threshold"]
        
        # Decide which subtree to follow
        if sample[feature_idx] <= threshold:
            return self._predict_sample(sample, tree["left"])
        else:
            return self._predict_sample(sample, tree["right"])

# Example usage
if __name__ == "__main__":
    # Create a simple dataset
    X = np.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ])
    y = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    
    # Create and train the decision tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    
    # Make predictions
    predictions = tree.predict(X)
    print("Predictions:", predictions)
    print("Accuracy:", np.sum(predictions == y) / len(y))`

const bootstrappingCode = `import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load a dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simplified Decision Tree (using the one from previous example)
class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        
    def fit(self, X, y):
        # Simplified fit method for demonstration
        self.X_train = X
        self.y_train = y
        return self
        
    def predict(self, X):
        # Simplified prediction using nearest neighbor for demonstration
        predictions = []
        for sample in X:
            # Find the closest training example
            distances = np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))
            nearest_idx = np.argmin(distances)
            predictions.append(self.y_train[nearest_idx])
        return np.array(predictions)

# Bootstrapping and Bagging implementation
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=3, bootstrap_ratio=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap_ratio = bootstrap_ratio
        self.trees = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Create multiple trees with bootstrap samples
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            bootstrap_indices = np.random.choice(
                n_samples, 
                size=int(n_samples * self.bootstrap_ratio), 
                replace=True  # With replacement
            )
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Train a decision tree on the bootstrap sample
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Add the tree to our forest
            self.trees.append(tree)
            
        return self
    
    def predict(self, X):
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to get predictions per sample
        predictions = predictions.T
        
        # For each sample, take the majority vote
        final_predictions = []
        for sample_predictions in predictions:
            # Get the most common class
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(most_common)
            
        return np.array(final_predictions)

# Demonstrate bootstrapping
def demonstrate_bootstrapping():
    print("Demonstrating bootstrapping:")
    n_samples = X_train.shape[0]
    
    # Create a bootstrap sample
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_bootstrap = X_train[bootstrap_indices]
    y_bootstrap = y_train[bootstrap_indices]
    
    # Count occurrences of each sample
    unique, counts = np.unique(bootstrap_indices, return_counts=True)
    
    print(f"Original dataset size: {n_samples}")
    print(f"Bootstrap sample size: {len(bootstrap_indices)}")
    print(f"Unique samples in bootstrap: {len(unique)} ({len(unique)/n_samples:.1%} of original)")
    print(f"Samples selected multiple times: {np.sum(counts > 1)}")
    print(f"Samples not selected (out-of-bag): {n_samples - len(unique)}")
    
    # Visualize the bootstrap sample distribution
    print("\nSample selection distribution:")
    for count, frequency in Counter(counts).items():
        print(f"  Selected {count} times: {frequency} samples")

# Demonstrate bagging with our Random Forest
def demonstrate_bagging():
    print("\nDemonstrating bagging with Random Forest:")
    
    # Create and train a Random Forest
    rf = RandomForest(n_estimators=10, max_depth=3)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Random Forest accuracy: {accuracy:.4f}")
    
    # Compare with a single decision tree
    tree = SimpleDecisionTree(max_depth=3)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    tree_accuracy = np.sum(tree_pred == y_test) / len(y_test)
    print(f"Single tree accuracy: {tree_accuracy:.4f}")
    
    # Show the effect of increasing the number of trees
    print("\nEffect of increasing the number of trees:")
    n_trees_list = [1, 5, 10, 20, 50, 100]
    accuracies = []
    
    for n_trees in n_trees_list:
        rf = RandomForest(n_estimators=n_trees, max_depth=3)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        accuracies.append(accuracy)
        print(f"  Trees: {n_trees}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    demonstrate_bootstrapping()
    demonstrate_bagging()`

const featureRandomnessCode = `import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load a dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simplified Decision Tree with feature randomness
class RandomizedDecisionTree:
    def __init__(self, max_depth=3, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        
    def fit(self, X, y):
        self.n_features = X.shape[1]
        
        # If max_features is not specified, use sqrt(n_features)
        if self.max_features is None:
            self.max_features = int(np.sqrt(self.n_features))
        
        # Store training data for simplified prediction
        self.X_train = X
        self.y_train = y
        
        # Build the tree (simplified for demonstration)
        self.tree = self._grow_tree(X, y, depth=0)
        
        return self
    
    def _grow_tree(self, X, y, depth):
        # Stopping criteria
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]
        
        # Select a random subset of features
        feature_indices = np.random.choice(
            self.n_features, 
            size=self.max_features, 
            replace=False
        )
        
        # Find the best split among the random features
        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)
        
        # If no good split is found, return the majority class
        if best_feature is None:
            return Counter(y).most_common(1)[0][0]
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Grow the left and right subtrees
        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Return the decision node
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _find_best_split(self, X, y, feature_indices):
        # Simplified best split finder
        best_gini = 1.0
        best_feature = None
        best_threshold = None
        
        # Try each feature in the random subset
        for feature_idx in feature_indices:
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Try each threshold
            for threshold in thresholds:
                # Split the data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                # Skip if one side is empty
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate Gini impurity
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                
                # Weighted average of Gini impurity
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_gini = (n_left * left_gini + n_right * right_gini) / len(y)
                
                # Update best split if this is better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _gini(self, y):
        # Calculate Gini impurity
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def predict(self, X):
        # Predict for each sample
        return np.array([self._predict_sample(sample, self.tree) for sample in X])
    
    def _predict_sample(self, sample, tree):
        # If tree is a leaf (not a dictionary), return the value
        if not isinstance(tree, dict):
            return tree
        
        # Get the feature and threshold
        feature = tree['feature']
        threshold = tree['threshold']
        
        # Decide which subtree to follow
        if sample[feature] <= threshold:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])

# Random Forest with feature randomness
class RandomForestWithFeatureRandomness:
    def __init__(self, n_estimators=10, max_depth=3, max_features=None, bootstrap_ratio=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap_ratio = bootstrap_ratio
        self.trees = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Create multiple trees with bootstrap samples and feature randomness
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            bootstrap_indices = np.random.choice(
                n_samples, 
                size=int(n_samples * self.bootstrap_ratio), 
                replace=True
            )
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Train a randomized decision tree
            tree = RandomizedDecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Add the tree to our forest
            self.trees.append(tree)
            
        return self
    
    def predict(self, X):
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to get predictions per sample
        predictions = predictions.T
        
        # For each sample, take the majority vote
        final_predictions = []
        for sample_predictions in predictions:
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(most_common)
            
        return np.array(final_predictions)

# Demonstrate feature randomness
def demonstrate_feature_randomness():
    print("Demonstrating feature randomness:")
    n_features = X_train.shape[1]
    
    # Try different max_features values
    max_features_options = [1, 2, 3, 4, None]  # None will use sqrt(n_features)
    
    print(f"Total number of features: {n_features}")
    print("\nAccuracy with different max_features values:")
    
    for max_features in max_features_options:
        # Create and train a Random Forest with specified max_features
        rf = RandomForestWithFeatureRandomness(
            n_estimators=20,
            max_depth=3,
            max_features=max_features
        )
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test)
        
        # Calculate accuracy
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        
        # Display the actual max_features value used
        actual_max_features = max_features if max_features is not None else int(np.sqrt(n_features))
        print(f"  max_features={max_features} (actual: {actual_max_features}): Accuracy = {accuracy:.4f}")
    
    # Compare with a Random Forest without feature randomness (uses all features)
    print("\nComparing with and without feature randomness:")
    
    # With feature randomness (sqrt of features)
    rf_with_randomness = RandomForestWithFeatureRandomness(
        n_estimators=20,
        max_depth=3,
        max_features=None  # sqrt(n_features)
    )
    rf_with_randomness.fit(X_train, y_train)
    y_pred_with = rf_with_randomness.predict(X_test)
    accuracy_with = np.sum(y_pred_with == y_test) / len(y_test)
    
    # Without feature randomness (all features)
    rf_without_randomness = RandomForestWithFeatureRandomness(
        n_estimators=20,
        max_depth=3,
        max_features=n_features  # All features
    )
    rf_without_randomness.fit(X_train, y_train)
    y_pred_without = rf_without_randomness.predict(X_test)
    accuracy_without = np.sum(y_pred_without == y_test) / len(y_test)
    
    print(f"  With feature randomness: Accuracy = {accuracy_with:.4f}")
    print(f"  Without feature randomness: Accuracy = {accuracy_without:.4f}")

if __name__ == "__main__":
    demonstrate_feature_randomness()`

const rfClassificationCode = `import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Random Forest classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=None,    # Maximum depth of each tree
    min_samples_split=2,  # Minimum samples required to split a node
    min_samples_leaf=1,   # Minimum samples required at a leaf node
    max_features='sqrt',  # Number of features to consider for best split
    bootstrap=True,       # Use bootstrap samples
    random_state=42       # For reproducibility
)

# Train the model
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Predict probabilities for a single example
sample = X_test[0].reshape(1, -1)
probabilities = rf_clf.predict_proba(sample)
print("\\nProbabilities for a single example:")
for i, prob in enumerate(probabilities[0]):
    print(f"{target_names[i]}: {prob:.4f}")

# Feature importance
feature_importance = rf_clf.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# Visualize decision boundaries (for 2 features)
def plot_decision_boundaries(X, y, model, feature_idx=(0, 1)):
    # Extract the two features we want to visualize
    X_visual = X[:, feature_idx]
    
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_visual[:, 0].min() - 1, X_visual[:, 0].max() + 1
    y_min, y_max = X_visual[:, 1].min() - 1, X_visual[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create feature vectors for all mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For visualization, we need to create a full feature vector
    # with zeros for the features we're not visualizing
    if X.shape[1] > 2:
        full_mesh_points = np.zeros((mesh_points.shape[0], X.shape[1]))
        full_mesh_points[:, feature_idx] = mesh_points
        Z = model.predict(full_mesh_points)
    else:
        Z = model.predict(mesh_points)
    
    # Reshape the predictions
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot the training points
    scatter = plt.scatter(X_visual[:, 0], X_visual[:, 1], c=y, 
                         edgecolors='k', cmap='viridis')
    
    plt.xlabel(feature_names[feature_idx[0]])
    plt.ylabel(feature_names[feature_idx[1]])
    plt.title('Random Forest Decision Boundaries')
    plt.colorbar(scatter)
    plt.show()

# Visualize decision boundaries for sepal length and sepal width
plot_decision_boundaries(X, y, rf_clf, feature_idx=(0, 1))`

const rfRegressionCode = `import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Random Forest regressor
rf_reg = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=None,    # Maximum depth of each tree
    min_samples_split=2,  # Minimum samples required to split a node
    min_samples_leaf=1,   # Minimum samples required at a leaf node
    max_features='auto',  # Number of features to consider for best split (auto = n_features)
    bootstrap=True,       # Use bootstrap samples
    random_state=42       # For reproducibility
)

# Train the model
rf_reg.fit(X_train, y_train)

# Make predictions
y_pred = rf_reg.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression: Predicted vs Actual Values')
plt.tight_layout()
plt.show()

# Feature importance
feature_importance = rf_reg.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()

# Distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='blue')
plt.xlabel('Residual Value')
plt.ylabel('Count')
plt.title('Distribution of Residuals')
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# Partial dependence plot for an important feature
from sklearn.inspection import partial_dependence, plot_partial_dependence

# Find the most important feature
most_important_feature_idx = np.argmax(feature_importance)

# Create partial dependence plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_partial_dependence(
    rf_reg, X_train, [most_important_feature_idx], 
    feature_names=feature_names,
    ax=ax
)
plt.tight_layout()
plt.show()

# Predict for a single example
sample = X_test[0].reshape(1, -1)
prediction = rf_reg.predict(sample)
print(f"\\nPrediction for a single example: {prediction[0]:.4f}")
print(f"Actual value: {y_test[0]:.4f}")

# Out-of-bag score (a built-in cross-validation mechanism)
rf_reg_oob = RandomForestRegressor(
    n_estimators=100,
    oob_score=True,  # Enable out-of-bag scoring
    random_state=42
)
rf_reg_oob.fit(X_train, y_train)
oob_score = rf_reg_oob.oob_score_
print(f"\\nOut-of-bag R² score: {oob_score:.4f}")`

const shapValuesCode = `# Example SHAP values code
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=cancer.feature_names)
`

const featureImportanceCode = `# Example code for feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
`;

const gridSearchCode = `import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Create a base model
rf = RandomForestClassifier(random_state=42)

# Create the grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available cores
    verbose=1,
    scoring='accuracy'
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))

# Get the best model
best_rf = grid_search.best_estimator_

# Evaluate on the test set
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy with best parameters: {:.4f}".format(accuracy))

# Print classification report
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# Compare with default Random Forest
default_rf = RandomForestClassifier(random_state=42)
default_rf.fit(X_train, y_train)
default_pred = default_rf.predict(X_test)
default_accuracy = accuracy_score(y_test, default_pred)
print("\\nDefault Random Forest accuracy: {:.4f}".format(default_accuracy))
print("Improvement: {:.2f}%".format(100 * (accuracy - default_accuracy) / default_accuracy))

# Plot the grid search results
results = grid_search.cv_results_
`

