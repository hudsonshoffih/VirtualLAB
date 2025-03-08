import type { QuizQuestion } from "../types"
import * as edaQuiz from "./eda-quiz"
import * as linearRegressionQuiz from "./linear-regression-quiz"
import * as svmQuiz from "./svm-quiz"
import * as kmeansQuiz from "./kmeans-quiz"
import * as dataInsightsQuiz from "./data-insights-quiz"
import * as evaluationMetricsQuiz from "./evaluation-metrics-quiz"

const quizModules: Record<string, any> = {
  "eda": edaQuiz,
  "linear-regression": linearRegressionQuiz,
  "svm": svmQuiz,
  "kmeans": kmeansQuiz,
  "dataset-insights": dataInsightsQuiz,
  "evaluation-metrics": evaluationMetricsQuiz,
}

export function getQuizQuestions(algorithmSlug: string, difficultyLevel: string): QuizQuestion[] {
  if (quizModules[algorithmSlug]) {
    const questions = quizModules[algorithmSlug][difficultyLevel] || quizModules[algorithmSlug].beginner
    return questions || []
  }
  return []
}
