import {
    BookOpen,
    Database,
    ListChecks,
    BarChart2,
    Network,
    CheckCircle2,
    Calculator,
    BarChart,
    LineChart,
    ArrowDownUp,
    TrendingUp,
    FileText,
    PieChart,
    Ruler,
    Sigma,
    Lightbulb,
    Target, 
    Repeat,
  } from "lucide-react"
  
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
        ]
      case "dataset-insights":
        return [
          { title: "Introduction to Statistics", icon: BookOpen },
          { title: "Types of Data", icon: Database},
          { title: "Central Tendency", icon: Sigma },
          { title: "Measures of Dispersion", icon: Ruler },
          { title: "Distributions & Skewness", icon: BarChart2 },
          { title: "Statistical Inference", icon: LineChart}
        ]
        case "evaluation-metrics":
          return [
            { title: "Introduction to Metrics", icon: BookOpen  },
            { title: "Classification Metrics", icon: Target  },
            { title: "Regression Metrics", icon: LineChart },
            { title: "Cross-Validation", icon: Repeat },
            { title: "Practical Applications", icon: Lightbulb},
            { title: "Best Practices", icon: CheckCircle2 },
          ]
      default:
        return [
          { title: "Introduction", icon: FileText },
          { title: "Key Concepts", icon: BookOpen },
          { title: "Implementation", icon: ListChecks },
          { title: "Examples", icon: BarChart },
          { title: "Applications", icon: PieChart },
          { title: "Summary", icon: CheckCircle2 },
        ]
    }
  }
  
  