import {
  BookOpen,
  Database,
  ListChecks,
  BarChart2,
  Network,
  CheckCircle2,
  BarChart,
  LineChart,
  FileText,
  PieChart,
  Ruler,
  Sigma,
  Lightbulb,
  Target,
  Repeat,
  Layers,
  SplitSquareVertical,
  GitBranch,
  Code,
  Users,
  Zap,
  Trees,
  TreeDeciduous,
  Shuffle,
  Sliders,
  Workflow,
  Settings,
  GitMerge,
} from "lucide-react";

export const getTutorialSections = (algorithmSlug: string) => {
  switch (algorithmSlug) {
    case "eda":
      return [
        { title: "Introduction to EDA", icon: BookOpen },
        { title: "Dataset Overview", icon: Database },
        { title: "Steps for EDA", icon: ListChecks },
        { title: "Visualizing Distributions", icon: BarChart2 },
        { title: "Correlation Analysis", icon: Network },
        { title: "Conclusion", icon: CheckCircle2 },
      ];
    case "dataset-insights":
      return [
        { title: "Introduction to Statistics", icon: BookOpen },
        { title: "Types of Data", icon: Database },
        { title: "Central Tendency", icon: Sigma },
        { title: "Measures of Dispersion", icon: Ruler },
        { title: "Distributions & Skewness", icon: BarChart2 },
        { title: "Statistical Inference", icon: LineChart },
      ];
    case "evaluation-metrics":
      return [
        { title: "Introduction to Metrics", icon: BookOpen },
        { title: "Classification Metrics", icon: Target },
        { title: "Regression Metrics", icon: LineChart },
        { title: "Cross-Validation", icon: Repeat },
        { title: "Practical Applications", icon: Lightbulb },
        { title: "Best Practices", icon: CheckCircle2 },
      ];
    case "linear-regression":
      return [
        { title: "Introduction", icon: BookOpen },
        { title: "Simple Linear Regression", icon: LineChart },
        { title: "Multiple Linear Regression", icon: Layers },
        { title: "Model Evaluation", icon: BarChart },
        { title: "Regularization Techniques", icon: SplitSquareVertical },
        { title: "Conclusion", icon: CheckCircle2 },
      ];
    case "logistic-regression":
      return [
        { title: "Introduction", icon: BookOpen },
        { title: "Sigmoid Function", icon: LineChart },
        { title: "Types & Assumptions", icon: GitBranch },
        { title: "Implementation", icon: Code },
        { title: "Model Evaluation", icon: BarChart2 },
        { title: "Conclusion", icon: CheckCircle2 },
      ];
    case "knn":
      return [
        { title: "Introduction to KNN", icon: Users },
        { title: "Understanding the Algorithm", icon: Network },
        { title: "Distance Metrics", icon: Ruler },
        { title: "Implementation with Scikit-Learn", icon: Code },
        { title: "Finding the Best K Value", icon: BarChart },
        { title: "Optimizing with KD-Tree", icon: Zap },
      ];
    case "random-forest":
      return [
        { title: "Introduction to Random Forest", icon: Trees },
        { title: "Understanding Decision Trees", icon: TreeDeciduous },
        { title: "Bootstrapping & Bagging", icon: Shuffle },
        { title: "Feature Randomness", icon: GitBranch },
        { title: "Implementation with Scikit-Learn", icon: Code },
        { title: "Hyperparameter Tuning", icon: Sliders },
        { title: "Visualization & Interpretation", icon: BarChart },
      ];
    case "svm":
      return [
        { title: "Introduction to SVM", icon: GitMerge },
        { title: "Mathematical Foundations", icon: Sigma },
        { title: "Linear SVM", icon: LineChart },
        { title: "Kernel Trick", icon: Workflow },
        { title: "Implementation with Scikit-Learn", icon: Code },
        { title: "Hyperparameter Tuning", icon: Settings },
        { title: "Visualization & Interpretation", icon: BarChart },
      ];
    case "ensemble-models":
      return [
        { title: "Introduction to Ensemble Learning", icon: Layers },
        { title: "Bagging (Bootstrap Aggregating)", icon: GitBranch },
        { title: "Boosting", icon: Zap },
        { title: "Stacking", icon: GitMerge },
        { title: "Comparison & Best Practices", icon: CheckCircle2 },
      ];

    default:
      return [
        { title: "Introduction", icon: FileText },
        { title: "Key Concepts", icon: BookOpen },
        { title: "Implementation", icon: ListChecks },
        { title: "Examples", icon: BarChart },
        { title: "Applications", icon: PieChart },
        { title: "Summary", icon: CheckCircle2 },
      ];
  }
};
