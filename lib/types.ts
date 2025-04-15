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

export interface TitanicPassenger {
  survived: number
  pclass: number
  sex: string
  age: number | null
  sibsp: number
  parch: number
  fare: number
  embarked: string
}
