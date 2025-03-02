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
          { title: "Introduction to Statistics", icon: Calculator },
          { title: "Descriptive Statistics", icon: BarChart },
          { title: "Probability Distributions", icon: LineChart },
          { title: "Hypothesis Testing", icon: ArrowDownUp },
          { title: "Regression Analysis", icon: TrendingUp },
          { title: "Applying Statistics", icon: CheckCircle2 },
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
  
  