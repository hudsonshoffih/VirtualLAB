import { createLocalStorage } from "./local-storage"

export interface QuizProgress {
  completed: boolean
  score: number
  totalQuestions: number
  date: string
}

export interface AlgorithmQuizProgress {
  beginner?: QuizProgress
  intermediate?: QuizProgress
  advanced?: QuizProgress
}

export interface UserProgress {
  quizzes: Record<string, AlgorithmQuizProgress>
}

const { getItem, setItem } = createLocalStorage<UserProgress>("virtual_lab_progress", {
  quizzes: {},
})

export function getUserProgress(): UserProgress {
  return getItem() || { quizzes: {} }
}

export function isQuizLevelCompleted(algorithmSlug: string, level: string): boolean {
  const progress = getUserProgress()
  return Boolean(progress.quizzes[algorithmSlug]?.[level as keyof AlgorithmQuizProgress]?.completed)
}

export function isQuizLevelUnlocked(algorithmSlug: string, level: string): boolean {
  if (level === "beginner") return true
  
  const progress = getUserProgress()
  
  if (level === "intermediate") {
    return Boolean(progress.quizzes[algorithmSlug]?.beginner?.completed)
  }
  
  if (level === "advanced") {
    return Boolean(
      progress.quizzes[algorithmSlug]?.beginner?.completed && 
      progress.quizzes[algorithmSlug]?.intermediate?.completed
    )
  }
  
  return false
}

export function saveQuizResult(
  algorithmSlug: string, 
  level: string, 
  score: number, 
  totalQuestions: number
): void {
  const progress = getUserProgress()
  
  if (!progress.quizzes[algorithmSlug]) {
    progress.quizzes[algorithmSlug] = {}
  }
  
  progress.quizzes[algorithmSlug][level as keyof AlgorithmQuizProgress] = {
    completed: true,
    score,
    totalQuestions,
    date: new Date().toISOString()
  }
  
  setItem(progress)
}

export function getQuizProgress(
  algorithmSlug: string, 
  level: string
): QuizProgress | undefined {
  const progress = getUserProgress()
  return progress.quizzes[algorithmSlug]?.[level as keyof AlgorithmQuizProgress]
}

export function resetAllProgress(): void {
  setItem({ quizzes: {} })
}
