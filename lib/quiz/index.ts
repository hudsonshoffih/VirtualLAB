import type { QuizQuestion } from "../types";
import * as edaQuiz from "./eda-quiz";
import * as linearRegressionQuiz from "./linear-regression-quiz";
import * as svmQuiz from "./svm-quiz";
import * as kmeansQuiz from "./kmeans-quiz";
import * as dataInsightsQuiz from "./data-insights-quiz";
import * as evaluationMetricsQuiz from "./evaluation-metrics-quiz";
import * as logisticRegressionQuiz from "./logistic-regression-quiz";
import * as knnQuiz from "./knn-quiz";
import * as randomForestQuiz from "./random-forest-quiz";


interface QuizModule {
  beginner?: QuizQuestion[];
  intermediate?: QuizQuestion[];
  advanced?: QuizQuestion[];
}

const quizModules: Record<string, QuizModule> = {
  eda: edaQuiz,
  "linear-regression": linearRegressionQuiz,
  svm: svmQuiz,
  kmeans: kmeansQuiz,
  "dataset-insights": dataInsightsQuiz,
  "evaluation-metrics": evaluationMetricsQuiz,
  "logistic-regression": logisticRegressionQuiz,
  knn: knnQuiz,
  "random-forest": randomForestQuiz,
};

export function getQuizQuestions(
  algorithmSlug: string,
  difficultyLevel: string
): QuizQuestion[] {
  const quizModule = quizModules[algorithmSlug];
  if (quizModule) {
    const questions =
      quizModule[difficultyLevel as keyof QuizModule] || quizModule.beginner;
    return questions || [];
  }
  return [];
}
