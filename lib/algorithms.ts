import type { Algorithm } from "./types"

const algorithms: Algorithm[] = [
  {
    id: "eda",
    slug: "eda",
    title: "Exploratory Data Analysis",
    description: "Learn how to analyze and visualize datasets to understand patterns and relationships.",
    tutorialContent:
      "# Exploratory Data Analysis\n\nEDA is an approach to analyzing datasets to summarize their main characteristics, often with visual methods.",
  },
  {
    id: "linear-regression",
    slug: "linear-regression",
    title: "Linear Regression",
    description: "Master the fundamentals of linear regression for predictive modeling.",
    tutorialContent:
      "# Linear Regression\n\nLinear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables.",
  },
  {
    id: "svm",
    slug: "svm",
    title: "Support Vector Machines",
    description: "Understand the powerful classification algorithm and its applications.",
    tutorialContent:
      "# Support Vector Machines\n\nSVM is a supervised learning model that analyzes data for classification and regression analysis.",
  },
]

export function getAlgorithms(): Algorithm[] {
  return algorithms
}

export function getAlgorithmBySlug(slug: string): Algorithm | undefined {
  return algorithms.find((algorithm) => algorithm.slug === slug)
}

