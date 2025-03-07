export interface Algorithm {
  id: string
  slug: string
  title: string
  description: string
  tutorialContent?: string
}

export interface PracticeStep {
  title: string
  instruction: string
  starterCode: string
  expectedOutput?: string
  hints: string[]
  resources: { title: string; url: string }[]
}

interface CodeOutputProps {
  isExecuting: boolean
  result: {
    success: boolean
    message: string
    output?: string
    error?: string
    table_html?: string
  } | null
}

export interface QuizQuestion {
  question: string
  options: { id: string; text: string }[]
  correctAnswer: string
  explanation?: string
}
