export interface Algorithm {
  id: string
  slug: string
  title: string
  description: string
  tutorialContent?: string
}

export interface QuizQuestion {
  question: string
  options: { id: string; text: string }[]
  correctAnswer: string
  explanation?: string
}