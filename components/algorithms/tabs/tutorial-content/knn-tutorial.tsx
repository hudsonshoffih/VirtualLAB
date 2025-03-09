"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import {
  Users,
  BookOpen,
  Code,
  BarChart,
  Lightbulb,
  CheckCircle,
  ArrowRight,
  Network,
  GitBranch,
  Ruler,
  Zap,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Minimize2,
} from "lucide-react"

interface KnnTutorialProps {
  section?: number
  onCopy?: (text: string, id: string) => void
  copied?: string | null
}
export function KnnTutorial({ section, onCopy, copied }: KnnTutorialProps) {
  const [activeTab, setActiveTab] = useState("explanation")
  const [copiedState, setCopiedState] = useState<string | null>(null)
    // Section 0: Introduction
    if (section === 0) {
      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-purple-100 dark:bg-purple-900 p-2 rounded-full">
                <Users className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300">
                What is K-Nearest Neighbors?
              </h3>
            </div>
            <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
              K-Nearest Neighbors (KNN) is a simple, versatile, and powerful supervised learning algorithm used for both
              classification and regression tasks. Unlike many algorithms that build a model during training, KNN is a
              lazy learner that makes predictions based on the similarity of new data points to its training examples.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center gap-2 mb-3">
                <Network className="h-5 w-5 text-purple-500" />
                <h4 className="font-medium text-lg">Key Concepts</h4>
              </div>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    1
                  </span>
                  <span>Lazy learning (no explicit training phase)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    2
                  </span>
                  <span>Instance-based (stores training examples)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    3
                  </span>
                  <span>Non-parametric (makes no assumptions about data distribution)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    4
                  </span>
                  <span>Uses distance metrics to find similar examples</span>
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
                  <span>Image classification and recognition</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Recommendation systems</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Medical diagnosis and anomaly detection</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Credit scoring and financial analysis</span>
                </li>
              </ul>
            </Card>
          </div>

          <div className="mt-6">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              What You'll Learn
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              {[
                { icon: <Network className="h-4 w-4" />, text: "KNN Algorithm Fundamentals" },
                { icon: <Ruler className="h-4 w-4" />, text: "Distance Metrics" },
                { icon: <Code className="h-4 w-4" />, text: "Implementation with Scikit-Learn" },
                { icon: <BarChart className="h-4 w-4" />, text: "Finding Optimal K Values" },
                { icon: <Zap className="h-4 w-4" />, text: "Performance Optimization" },
                { icon: <GitBranch className="h-4 w-4" />, text: "Classification & Regression" },
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
                <span>Knowledge of basic data visualization</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900 mt-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-purple-100 dark:bg-purple-900 p-2 rounded-full">
                <Maximize2 className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="text-lg font-semibold text-purple-800 dark:text-purple-300">Advantages of KNN</h3>
            </div>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-3 text-purple-700 dark:text-purple-300">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Simple to understand and implement</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>No assumptions about data distribution</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Naturally handles multi-class problems</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Can be used for both classification and regression</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-950 dark:to-red-950 p-6 rounded-lg border border-orange-100 dark:border-orange-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-orange-100 dark:bg-orange-900 p-2 rounded-full">
                <Minimize2 className="h-5 w-5 text-orange-600 dark:text-orange-400" />
              </div>
              <h3 className="text-lg font-semibold text-orange-800 dark:text-orange-300">Limitations of KNN</h3>
            </div>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-3 text-orange-700 dark:text-orange-300">
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 text-orange-500 mt-1" />
                <span>Computationally expensive for large datasets</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 text-orange-500 mt-1" />
                <span>Sensitive to irrelevant features</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 text-orange-500 mt-1" />
                <span>Requires feature scaling</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 text-orange-500 mt-1" />
                <span>Choosing optimal K can be challenging</span>
              </li>
            </ul>
          </div>
        </div>
      )
    }

    // Section 1: Understanding the Algorithm
    if (section === 1) {
      function copyToClipboard(knnFromScratchCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
            <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300 mb-3">
              Understanding the KNN Algorithm
            </h3>
            <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
              K-Nearest Neighbors works by finding the K closest data points to a new example and making predictions
              based on those neighbors. It's like asking your closest friends for advice, assuming that similar examples
              should have similar outputs.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                How It Works
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
                <h4 className="font-medium text-lg mb-3">The KNN Algorithm Steps</h4>
                <ol className="space-y-4">
                  <li className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Choose the number of neighbors (K)</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Select an appropriate value for K, which represents how many nearby points to consider when
                        making a prediction.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Calculate distances</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Compute the distance between the new data point and all points in the training dataset using a
                        distance metric (e.g., Euclidean distance).
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Find K nearest neighbors</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Sort the distances and identify the K training examples with the smallest distances to the new
                        point.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Make a prediction</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        For classification: Assign the majority class among the K neighbors.
                        <br />
                        For regression: Calculate the average of the K neighbors' target values.
                      </p>
                    </div>
                  </li>
                </ol>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Classification vs. Regression</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4 bg-purple-50 dark:bg-purple-950">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge variant="outline">Classification</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Predicts a categorical class label by taking a majority vote of the K nearest neighbors.
                    </p>
                    <div className="bg-white dark:bg-gray-800 p-3 rounded-md text-sm">
                      <p className="font-mono">prediction = most_common(neighbor_classes)</p>
                    </div>
                  </div>
                  <div className="border rounded-lg p-4 bg-blue-50 dark:bg-blue-950">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge variant="outline">Regression</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Predicts a continuous value by averaging the values of the K nearest neighbors.
                    </p>
                    <div className="bg-white dark:bg-gray-800 p-3 rounded-md text-sm">
                      <p className="font-mono">prediction = mean(neighbor_values)</p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Example Scenario</h4>
                <p className="mb-3">Classifying a new flower based on its features:</p>
                <ul className="space-y-2">
                  <li>
                    <span className="font-medium">Dataset:</span> Iris dataset with features like petal length, petal
                    width, etc.
                  </li>
                  <li>
                    <span className="font-medium">Task:</span> Classify a new flower into one of three species (setosa,
                    versicolor, virginica)
                  </li>
                  <li>
                    <span className="font-medium">Process:</span> Measure the new flower's features, find the K most
                    similar flowers in our dataset, and assign the most common species among those neighbors
                  </li>
                </ul>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">KNN from Scratch</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(knnFromScratchCode, "knn-scratch-code")}
                    className="text-xs"
                  >
                    {copiedState === "knn-scratch-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{knnFromScratchCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This implementation shows the core logic of KNN without using any specialized libraries. It
                    calculates Euclidean distances between points and makes predictions based on the majority class of
                    the K nearest neighbors.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">KNN Decision Boundaries</h4>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="KNN Decision Boundaries"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src =
                        "https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">
                    The visualization shows how KNN creates decision boundaries between different classes:
                  </p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Each color represents a different class</li>
                    <li>The decision boundary is where the algorithm switches from predicting one class to another</li>
                    <li>
                      Notice how the boundary becomes smoother with higher K values (less complex, potentially less
                      overfitting)
                    </li>
                    <li>With K=1, the boundary follows the training examples exactly, which may lead to overfitting</li>
                  </ul>
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
                <span>
                  KNN is a lazy learning algorithm that makes predictions based on similarity to training data
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The algorithm uses distance metrics to find the K nearest neighbors to a new data point</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>
                  For classification, it predicts the majority class; for regression, it averages the target values
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The choice of K significantly impacts the model's behavior and performance</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 2: Distance Metrics
    if (section === 2) {
      function copyToClipboard(distanceMetricsCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-teal-50 dark:from-blue-950 dark:to-teal-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
            <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-3">Distance Metrics</h3>
            <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
              Distance metrics are at the heart of KNN, determining how similarity between data points is measured. The
              choice of distance metric can significantly impact the algorithm's performance and should be selected
              based on the nature of your data.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Common Metrics
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
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
                    <Badge>Euclidean</Badge> Distance
                  </h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    The straight-line distance between two points in Euclidean space. This is the most commonly used
                    distance metric.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center mb-3">
                    <p className="font-mono text-sm">d(a, b) = √(Σ(aᵢ - bᵢ)²)</p>
                  </div>
                  <ul className="text-sm space-y-2">
                    <li>
                      <span className="font-medium">Best for:</span> Continuous data where the scale is similar across
                      all features
                    </li>
                    <li>
                      <span className="font-medium">Example:</span> Physical measurements like height, weight, or
                      dimensions
                    </li>
                  </ul>
                </Card>

                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
                    <Badge>Manhattan</Badge> Distance
                  </h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    The sum of absolute differences between coordinates. Also known as city block or taxicab distance.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center mb-3">
                    <p className="font-mono text-sm">d(a, b) = Σ|aᵢ - bᵢ|</p>
                  </div>
                  <ul className="text-sm space-y-2">
                    <li>
                      <span className="font-medium">Best for:</span> Grid-like data or when movement is restricted to
                      specific directions
                    </li>
                    <li>
                      <span className="font-medium">Example:</span> City navigation where you can only move along
                      streets in a grid pattern
                    </li>
                  </ul>
                </Card>
              </div>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
                  <Badge>Minkowski</Badge> Distance
                </h4>
                <p className="text-sm text-muted-foreground mb-3">
                  A generalized metric that includes both Euclidean and Manhattan distances as special cases.
                </p>
                <div className="bg-muted/50 p-3 rounded text-center mb-3">
                  <p className="font-mono text-sm">d(a, b) = (Σ|aᵢ - bᵢ|ᵖ)^(1/p)</p>
                </div>
                <ul className="text-sm space-y-2">
                  <li>
                    <span className="font-medium">Parameter p:</span> When p=1, it's Manhattan distance; when p=2, it's
                    Euclidean distance
                  </li>
                  <li>
                    <span className="font-medium">Best for:</span> When you want to experiment with different distance
                    metrics by adjusting a single parameter
                  </li>
                  <li>
                    <span className="font-medium">Example:</span> Feature spaces where different dimensions have
                    different importance
                  </li>
                </ul>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
                    <Badge>Cosine</Badge> Similarity
                  </h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Measures the cosine of the angle between two vectors, focusing on orientation rather than magnitude.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center mb-3">
                    <p className="font-mono text-sm">cos(θ) = (a·b)/(||a||·||b||)</p>
                  </div>
                  <ul className="text-sm space-y-2">
                    <li>
                      <span className="font-medium">Best for:</span> Text analysis, document similarity, and
                      high-dimensional data
                    </li>
                    <li>
                      <span className="font-medium">Range:</span> -1 to 1 (1 means identical direction, 0 means
                      orthogonal, -1 means opposite)
                    </li>
                  </ul>
                </Card>

                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
                    <Badge>Hamming</Badge> Distance
                  </h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Counts the number of positions at which corresponding symbols differ in two strings of equal length.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center mb-3">
                    <p className="font-mono text-sm">d(a, b) = count(aᵢ ≠ bᵢ)</p>
                  </div>
                  <ul className="text-sm space-y-2">
                    <li>
                      <span className="font-medium">Best for:</span> Categorical data, binary vectors, or string
                      comparison
                    </li>
                    <li>
                      <span className="font-medium">Example:</span> DNA sequence analysis, error detection in data
                      transmission
                    </li>
                  </ul>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Distance Metrics Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(distanceMetricsCode, "distance-metrics-code")}
                    className="text-xs"
                  >
                    {copiedState === "distance-metrics-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{distanceMetricsCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how to implement different distance metrics from scratch and how to use them
                    with scikit-learn's KNN implementation.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Comparing Distance Metrics</h4>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Distance Metrics Comparison"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*yGMk1GSKAJcXLnRxP7RJzg.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">
                    The visualization shows how different distance metrics create different "shapes" of neighborhoods:
                  </p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Euclidean distance creates circular neighborhoods</li>
                    <li>Manhattan distance creates diamond-shaped neighborhoods</li>
                    <li>Minkowski distance with p&gt;2 creates star-like shapes</li>
                    <li>The choice of distance metric affects which points are considered "neighbors"</li>
                  </ul>
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
                <span>The choice of distance metric should match the characteristics of your data</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Euclidean distance works well for continuous data in low-dimensional space</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Manhattan distance is suitable for grid-like data or when features represent discrete steps</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>
                  Cosine similarity is better for high-dimensional data where direction matters more than magnitude
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Experiment with different metrics to find the best performance for your specific problem</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 3: Implementation with Scikit-Learn
    if (section === 3) {
      function copyToClipboard(scikitLearnCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-950 dark:to-teal-950 p-6 rounded-lg border border-green-100 dark:border-green-900">
            <h3 className="text-xl font-semibold text-green-800 dark:text-green-300 mb-3">
              Implementation with Scikit-Learn
            </h3>
            <p className="text-green-700 dark:text-green-300 leading-relaxed">
              Scikit-learn provides an efficient and easy-to-use implementation of KNN for both classification and
              regression tasks. Let's explore how to implement KNN using this popular machine learning library.
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
                <h4 className="font-medium text-lg mb-3">Scikit-Learn KNN Classes</h4>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>KNeighborsClassifier</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Used for classification tasks where the target variable is categorical.
                    </p>
                    <div className="bg-muted/50 p-3 rounded text-sm">
                      <p className="font-mono">from sklearn.neighbors import KNeighborsClassifier</p>
                      <p className="font-mono mt-1">knn = KNeighborsClassifier(n_neighbors=5)</p>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>KNeighborsRegressor</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Used for regression tasks where the target variable is continuous.
                    </p>
                    <div className="bg-muted/50 p-3 rounded text-sm">
                      <p className="font-mono">from sklearn.neighbors import KNeighborsRegressor</p>
                      <p className="font-mono mt-1">knn = KNeighborsRegressor(n_neighbors=5)</p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Important Parameters</h4>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      n_neighbors
                    </Badge>
                    <div>
                      <p className="text-sm">Number of neighbors to consider (default: 5)</p>
                      <p className="text-xs text-muted-foreground">This is the 'K' in KNN</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      weights
                    </Badge>
                    <div>
                      <p className="text-sm">Weight function used in prediction (default: 'uniform')</p>
                      <p className="text-xs text-muted-foreground">
                        Options: 'uniform', 'distance', or a custom function
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      algorithm
                    </Badge>
                    <div>
                      <p className="text-sm">Algorithm used to compute nearest neighbors (default: 'auto')</p>
                      <p className="text-xs text-muted-foreground">
                        Options: 'ball_tree', 'kd_tree', 'brute', or 'auto'
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      metric
                    </Badge>
                    <div>
                      <p className="text-sm">Distance metric to use (default: 'minkowski')</p>
                      <p className="text-xs text-muted-foreground">
                        Many options including 'euclidean', 'manhattan', 'cosine', etc.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      p
                    </Badge>
                    <div>
                      <p className="text-sm">Power parameter for Minkowski metric (default: 2)</p>
                      <p className="text-xs text-muted-foreground">
                        p=1 is Manhattan distance, p=2 is Euclidean distance
                      </p>
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
                  <li>Scale features (important for distance-based algorithms)</li>
                  <li>Create and configure the KNN model</li>
                  <li>Train the model using the fit() method</li>
                  <li>Make predictions using the predict() method</li>
                  <li>Evaluate the model's performance</li>
                </ol>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">KNN with Scikit-Learn</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(scikitLearnCode, "scikit-learn-code")}
                    className="text-xs"
                  >
                    {copiedState === "scikit-learn-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{scikitLearnCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This example demonstrates how to implement KNN classification using scikit-learn on the Iris
                    dataset. It includes data loading, preprocessing, model training, prediction, and evaluation.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Classification Results</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="aspect-square bg-muted rounded-lg flex items-center justify-center overflow-hidden">
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
                        The confusion matrix shows how well our KNN model is performing for each class. The diagonal
                        elements represent correct predictions, while off-diagonal elements are misclassifications.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h5 className="font-medium mb-2">Performance Metrics</h5>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Accuracy:</span>
                          <Badge variant="outline">96.7%</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Precision:</span>
                          <Badge variant="outline">97.2%</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Recall:</span>
                          <Badge variant="outline">96.7%</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">F1 Score:</span>
                          <Badge variant="outline">96.8%</Badge>
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
                <span>
                  Scikit-learn provides efficient implementations of KNN for both classification and regression
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>
                  Feature scaling is crucial for KNN to ensure all features contribute equally to distance calculations
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The API is consistent with other scikit-learn models: fit(), predict(), and score()</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Various parameters allow you to customize the algorithm to your specific needs</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 4: Finding the Best K Value
    if (section === 4) {
      function copyToClipboard(findingBestKCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-orange-50 to-yellow-50 dark:from-orange-950 dark:to-yellow-950 p-6 rounded-lg border border-orange-100 dark:border-orange-900">
            <h3 className="text-xl font-semibold text-orange-800 dark:text-orange-300 mb-3">
              Finding the Best K Value
            </h3>
            <p className="text-orange-700 dark:text-orange-300 leading-relaxed">
              The choice of K significantly impacts the performance of a KNN model. A small K can lead to overfitting
              and sensitivity to noise, while a large K may result in underfitting. Let's explore how to find the
              optimal K value for your specific dataset.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Approach
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
                <h4 className="font-medium text-lg mb-3">The Impact of K</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div className="border rounded-lg p-4 bg-red-50 dark:bg-red-950">
                    <h5 className="font-medium mb-2">Small K Values</h5>
                    <ul className="text-sm space-y-2 list-disc pl-5">
                      <li>More sensitive to local patterns</li>
                      <li>Can lead to overfitting</li>
                      <li>Higher variance, lower bias</li>
                      <li>More complex decision boundaries</li>
                      <li>More susceptible to noise</li>
                    </ul>
                  </div>
                  <div className="border rounded-lg p-4 bg-blue-50 dark:bg-blue-950">
                    <h5 className="font-medium mb-2">Large K Values</h5>
                    <ul className="text-sm space-y-2 list-disc pl-5">
                      <li>More sensitive to global patterns</li>
                      <li>Can lead to underfitting</li>
                      <li>Lower variance, higher bias</li>
                      <li>Smoother decision boundaries</li>
                      <li>More robust to noise</li>
                    </ul>
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">
                  The optimal K value balances these trade-offs and depends on the specific characteristics of your
                  dataset.
                </p>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Methods to Find Optimal K</h4>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2">1. Elbow Method</h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Plot the error rate or accuracy against different K values and look for the "elbow point" where
                      the rate of improvement significantly changes.
                    </p>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2">2. Cross-Validation</h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Use k-fold cross-validation to evaluate different K values and select the one with the best
                      average performance.
                    </p>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2">3. Grid Search</h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Systematically try different K values (and potentially other hyperparameters) to find the
                      combination that yields the best performance.
                    </p>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2">4. Rule of Thumb</h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Some common heuristics include using the square root of the number of samples or an odd number to
                      avoid ties in binary classification.
                    </p>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Best Practices</h4>
                <ul className="space-y-2">
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <span>Always use a separate validation set or cross-validation to evaluate K values</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <span>
                      Consider using odd values of K for classification problems with an even number of classes to avoid
                      ties
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <span>
                      Test a wide range of K values, typically from 1 to √n (where n is the number of samples)
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <span>
                      Consider the computational cost: larger K values require more computation for prediction
                    </span>
                  </li>
                </ul>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Finding Optimal K</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(findingBestKCode, "finding-best-k-code")}
                    className="text-xs"
                  >
                    {copiedState === "finding-best-k-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{findingBestKCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how to find the optimal K value by testing a range of values and plotting the
                    accuracy for each. It also shows how to use cross-validation for more robust evaluation.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">K Value vs. Accuracy</h4>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="K Value vs. Accuracy"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*RGJ-SEi_HhKn5Vf7kUkG-g.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The graph shows how accuracy changes with different K values:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Very small K values (e.g., K=1) often show high training accuracy but may overfit</li>
                    <li>As K increases, the model becomes smoother and less prone to noise</li>
                    <li>The optimal K value is where the accuracy peaks or stabilizes</li>
                    <li>Beyond the optimal point, accuracy may decrease as the model becomes too simple</li>
                    <li>The "elbow point" in the curve often indicates a good trade-off between bias and variance</li>
                  </ul>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-orange-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The optimal K value balances the trade-off between overfitting and underfitting</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Use cross-validation to find the best K value for your specific dataset</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The elbow method helps identify where increasing K provides diminishing returns</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Consider using odd values of K for classification problems to avoid ties</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 5: Optimizing with KD-Tree
    if (section === 5) {
      function copyToClipboard(kdTreeCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-pink-50 to-purple-50 dark:from-pink-950 dark:to-purple-950 p-6 rounded-lg border border-pink-100 dark:border-pink-900">
            <h3 className="text-xl font-semibold text-pink-800 dark:text-pink-300 mb-3">Optimizing with KD-Tree</h3>
            <p className="text-pink-700 dark:text-pink-300 leading-relaxed">
              KNN can become computationally expensive for large datasets, as it requires calculating distances between
              the query point and all training examples. KD-Trees and other spatial data structures can significantly
              improve performance by optimizing the nearest neighbor search.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                KD-Tree Explained
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Performance
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">What is a KD-Tree?</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  A KD-Tree (K-Dimensional Tree) is a space-partitioning data structure that organizes points in a
                  k-dimensional space. It enables efficient range searches and nearest neighbor searches.
                </p>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      Structure
                    </Badge>
                    <span className="text-sm">
                      A binary tree where each node represents a point in k-dimensional space
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      Partitioning
                    </Badge>
                    <span className="text-sm">
                      Each level of the tree splits the space along one dimension, cycling through dimensions as the
                      tree grows
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      Search
                    </Badge>
                    <span className="text-sm">
                      Allows for efficient nearest neighbor search by pruning large portions of the search space
                    </span>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">How KD-Trees Improve KNN</h4>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2">Reduced Computational Complexity</h5>
                    <p className="text-sm text-muted-foreground">
                      <span className="font-mono">O(log n)</span> search time on average compared to{" "}
                      <span className="font-mono">O(n)</span> for brute force, where n is the number of training
                      examples
                    </p>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2">Efficient Space Partitioning</h5>
                    <p className="text-sm text-muted-foreground">
                      Divides the feature space into regions, allowing the algorithm to quickly eliminate large portions
                      of the search space
                    </p>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2">Scalability</h5>
                    <p className="text-sm text-muted-foreground">
                      Makes KNN practical for larger datasets where brute force search would be prohibitively expensive
                    </p>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">When to Use KD-Trees</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4 bg-green-50 dark:bg-green-950">
                    <h5 className="font-medium mb-2">Good For</h5>
                    <ul className="text-sm space-y-2 list-disc pl-5">
                      <li>Low to medium dimensional data (typically d ≤ 20)</li>
                      <li>Large datasets where brute force is too slow</li>
                      <li>Static datasets that don't change frequently</li>
                      <li>When exact nearest neighbors are required</li>
                    </ul>
                  </div>
                  <div className="border rounded-lg p-4 bg-red-50 dark:bg-red-950">
                    <h5 className="font-medium mb-2">Less Effective For</h5>
                    <ul className="text-sm space-y-2 list-disc pl-5">
                      <li>High-dimensional data (suffers from "curse of dimensionality")</li>
                      <li>Frequently changing datasets requiring tree rebuilding</li>
                      <li>Very small datasets where brute force is already fast</li>
                      <li>When approximate nearest neighbors are sufficient</li>
                    </ul>
                  </div>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">KNN with KD-Tree</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(kdTreeCode, "kd-tree-code")}
                    className="text-xs"
                  >
                    {copiedState === "kd-tree-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{kdTreeCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how to use KD-Tree with scikit-learn's KNN implementation and compares its
                    performance with the brute force approach.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Performance Comparison</h4>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="KD-Tree vs Brute Force Performance"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*fgbU9xRlpXKpIXyWNNEGjA.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">
                    The graph shows the performance comparison between brute force and KD-Tree approaches:
                  </p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>As the dataset size increases, brute force search time grows linearly</li>
                    <li>KD-Tree search time grows logarithmically, providing significant speedup for large datasets</li>
                    <li>For very small datasets, the overhead of building the KD-Tree may outweigh the benefits</li>
                    <li>The performance advantage of KD-Trees diminishes as the dimensionality increases</li>
                  </ul>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-pink-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>KD-Trees can significantly improve the performance of KNN for large datasets</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>They work by partitioning the space and enabling efficient nearest neighbor searches</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>KD-Trees are most effective for low to medium dimensional data</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>
                  In scikit-learn, you can specify the algorithm parameter as 'kd_tree' to use this optimization
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>For high-dimensional data, consider using approximate nearest neighbor algorithms instead</span>
              </li>
            </ul>
          </Card>

          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900 mt-6">
            <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300 mb-3">Conclusion</h3>
            <p className="text-purple-700 dark:text-purple-300 leading-relaxed mb-4">
              K-Nearest Neighbors is a versatile and intuitive algorithm that can be used for both classification and
              regression tasks. Its simplicity makes it an excellent starting point for many machine learning problems.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Strengths</h4>
                <ul className="text-sm space-y-1 list-disc pl-5">
                  <li>Simple to understand and implement</li>
                  <li>No training phase (lazy learning)</li>
                  <li>Naturally handles multi-class problems</li>
                  <li>Can be used for both classification and regression</li>
                  <li>Makes no assumptions about data distribution</li>
                </ul>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Considerations</h4>
                <ul className="text-sm space-y-1 list-disc pl-5">
                  <li>Requires feature scaling</li>
                  <li>Computationally expensive for large datasets</li>
                  <li>Sensitive to irrelevant features</li>
                  <li>Choosing the optimal K value is important</li>
                  <li>Performance degrades in high dimensions</li>
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
}

// Code examples as constants
const knnFromScratchCode = `import numpy as np
from collections import Counter

def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance between two points
    """
    return np.sqrt(np.sum((a - b) ** 2))

def knn_classify(X_train, y_train, X_test, k=3):
    """
    K-Nearest Neighbors classifier from scratch
    
    Parameters:
    -----------
    X_train : array-like, shape (n_samples, n_features)
        Training data
    y_train : array-like, shape (n_samples,)
        Target values for training data
    X_test : array-like, shape (n_samples, n_features)
        Test data
    k : int, default=3
        Number of neighbors to use
        
    Returns:
    --------
    y_pred : array, shape (n_samples,)
        Predicted class labels for test data
    """
    predictions = []
    
    # Loop through all test points
    for test_point in X_test:
        # Calculate distances to all training points
        distances = [euclidean_distance(test_point, train_point) 
                    for train_point in X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        
        # Get the labels of k nearest neighbors
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        # Majority vote to determine the most common class
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
        
    return np.array(predictions)

# Example usage
if __name__ == "__main__":
    # Create a simple dataset
    X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y_train = np.array([0, 0, 1, 1, 0, 1])
    
    # Test points
    X_test = np.array([[1.1, 1.9], [7, 7]])
    
    # Predict
    predictions = knn_classify(X_train, y_train, X_test, k=3)
    print(f"Predictions: {predictions}")  # Expected: [0, 1]`

const distanceMetricsCode = `import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Custom distance metrics implementation
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p=3):
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Test different metrics with scikit-learn
metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
results = {}

for metric in metrics:
    # Create KNN classifier with specific metric
    if metric == 'minkowski':
        # For Minkowski, we can specify the power parameter p
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, p=3)
    else:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    
    # Train and predict
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[metric] = accuracy
    
    print(f"Metric: {metric}, Accuracy: {accuracy:.4f}")

# Using a custom metric with scikit-learn
# Note: This is just for demonstration, scikit-learn has built-in metrics
from sklearn.neighbors import DistanceMetric

# Create a custom distance metric
def custom_distance(x1, x2):
    # Example: weighted Euclidean distance
    weights = np.array([2.0, 1.0, 3.0, 1.0])  # Weights for each feature
    return np.sqrt(np.sum(weights * ((x1 - x2) ** 2)))

# Using the custom metric with KNeighborsClassifier
from sklearn.metrics import pairwise_distances

# Create distance matrix using custom metric
dist_matrix = pairwise_distances(X_train, X_test, metric=custom_distance)

# Note: For production use, it's better to use scikit-learn's built-in
# functionality for custom metrics through the 'metric' parameter`

const scikitLearnCode = `import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Scale the features (important for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
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
sample = X_test_scaled[0].reshape(1, -1)
probabilities = knn.predict_proba(sample)
print("\\nProbabilities for a single example:")
for i, prob in enumerate(probabilities[0]):
    print(f"{target_names[i]}: {prob:.4f}")

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
    plt.title('KNN Decision Boundaries')
    plt.colorbar(scatter)
    plt.show()

# Visualize decision boundaries for sepal length and sepal width
plot_decision_boundaries(X_train_scaled, y_train, knn, feature_idx=(0, 1))`

const findingBestKCode = `import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 1: Testing a range of K values
k_values = list(range(1, 31))
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Training accuracy
    train_pred = knn.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    train_scores.append(train_acc)
    
    # Testing accuracy
    test_pred = knn.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    test_scores.append(test_acc)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(k_values, train_scores, label='Training Accuracy')
plt.plot(k_values, test_scores, label='Testing Accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs K Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Add a vertical line at the best K value
best_k = k_values[np.argmax(test_scores)]
plt.axvline(x=best_k, color='r', linestyle='--', 
           label=f'Best K = {best_k}')
plt.legend()
plt.show()

print(f"Best K value based on test accuracy: {best_k}")
print(f"Best test accuracy: {max(test_scores):.4f}")

# Method 2: Using cross-validation for more robust evaluation
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot cross-validation results
plt.figure(figsize=(12, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN Cross-Validation Accuracy vs K Value')
plt.grid(True, alpha=0.3)

# Add a vertical line at the best K value
best_k_cv = k_values[np.argmax(cv_scores)]
plt.axvline(x=best_k_cv, color='r', linestyle='--', 
           label=f'Best K = {best_k_cv}')
plt.legend()
plt.show()

print(f"Best K value based on cross-validation: {best_k_cv}")
print(f"Best cross-validation accuracy: {max(cv_scores):.4f}")

# Train final model with the best K value
final_model = KNeighborsClassifier(n_neighbors=best_k_cv)
final_model.fit(X_train_scaled, y_train)
final_pred = final_model.predict(X_test_scaled)
final_acc = accuracy_score(y_test, final_pred)

print(f"Final model accuracy on test set: {final_acc:.4f}")`

const kdTreeCode = `import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset with 10,000 samples and 10 features
X, y = make_classification(n_samples=10000, n_features=10, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare performance of different algorithms
algorithms = ['brute', 'kd_tree', 'ball_tree']
times = []
accuracies = []

for algorithm in algorithms:
    # Create and train the model
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algorithm)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate time and accuracy
    end_time = time.time()
    elapsed_time = end_time - start_time
    accuracy = knn.score(X_test_scaled, y_test)
    
    times.append(elapsed_time)
    accuracies.append(accuracy)
    
    print(f"Algorithm: {algorithm}")
    print(f"Time: {elapsed_time:.5f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("-" * 30)

# Plot the results
plt.figure(figsize=(12, 5))

# Time comparison
plt.subplot(1, 2, 1)
plt.bar(algorithms, times, color=['blue', 'green', 'orange'])
plt.xlabel('Algorithm')
plt.ylabel('Time (seconds)')
plt.title('KNN Algorithm Performance: Time')

# Accuracy comparison
plt.subplot(1, 2, 2)
plt.bar(algorithms, accuracies, color=['blue', 'green', 'orange'])
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('KNN Algorithm Performance: Accuracy')
plt.ylim([0.9, 1.0])  # Adjust as needed

plt.tight_layout()
plt.show()

# Compare performance with increasing dataset size
sizes = [100, 500, 1000, 2000, 5000, 10000]
brute_times = []
kdtree_times = []

for size in sizes:
    # Generate dataset of specific size
    X_subset, y_subset = make_classification(n_samples=size, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Brute force
    start_time = time.time()
    knn_brute = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
    knn_brute.fit(X_train_scaled, y_train)
    knn_brute.predict(X_test_scaled)
    brute_time = time.time() - start_time
    brute_times.append(brute_time)
    
    # KD-Tree
    start_time = time.time()
    knn_kdtree = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
    knn_kdtree.fit(X_train_scaled, y_train)
    knn_kdtree.predict(X_test_scaled)
    kdtree_time = time.time() - start_time
    kdtree_times.append(kdtree_time)
    
    print(f"Dataset size: {size}")
    print(f"Brute force time: {brute_time:.5f} seconds")
    print(f"KD-Tree time: {kdtree_time:.5f} seconds")
    print(f"Speedup: {brute_time/kdtree_time:.2f}x")
    print("-" * 30)

# Plot time comparison with increasing dataset size
plt.figure(figsize=(10, 6))
plt.plot(sizes, brute_times, 'o-', label='Brute Force')
plt.plot(sizes, kdtree_times, 'o-', label='KD-Tree')
plt.xlabel('Dataset Size')
plt.ylabel('Time (seconds)')
plt.title('KNN Performance: Brute Force vs KD-Tree')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Visualize KD-Tree partitioning (simplified 2D example)
from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt

# Generate a simple 2D dataset
np.random.seed(42)
X_2d = np.random.rand(100, 2)

# Create KD-Tree
tree = KDTree(X_2d)

# Function to recursively plot KD-Tree partitions
def plot_kdtree_partition(ax, tree, depth=0, x_min=0, x_max=1, y_min=0, y_max=1):
    if depth >= 5:  # Limit recursion depth for visualization
        return
        
    # Alternate between x and y axis for splitting
    axis = depth % 2
    
    # Get the splitting point at this node
    if hasattr(tree, 'node_data'):
        split_point = tree.node_data[0]
    else:
        # For scikit-learn's KDTree, we'll use a simplified approach
        # This is an approximation since we don't have direct access to the tree structure
        points_in_region = X_2d[np.logical_and(
            np.logical_and(X_2d[:, 0] >= x_min, X_2d[:, 0] <= x_max),
            np.logical_and(X_2d[:, 1] >= y_min, X_2d[:, 1] <= y_max)
        )]
        if len(points_in_region) == 0:
            return
        split_point = np.median(points_in_region[:, axis])
    
    # Draw the partition line
    if axis == 0:
        ax.plot([split_point, split_point], [y_min, y_max], 'r-')
        # Recurse on left and right
        plot_kdtree_partition(ax, tree, depth+1, x_min, split_point, y_min, y_max)
        plot_kdtree_partition(ax, tree, depth+1, split_point, x_max, y_min, y_max)
    else:
        ax.plot([x_min, x_max], [split_point, split_point], 'b-')
        # Recurse on top and bottom
        plot_kdtree_partition(ax, tree, depth+1, x_min, x_max, y_min, split_point)
        plot_kdtree_partition(ax, tree, depth+1, x_min, x_max, split_point, y_max)

# Plot the data points and KD-Tree partitions
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(X_2d[:, 0], X_2d[:, 1], c='k', s=50)
plot_kdtree_partition(ax, tree)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('KD-Tree Space Partitioning (Simplified Visualization)')
plt.show()`

