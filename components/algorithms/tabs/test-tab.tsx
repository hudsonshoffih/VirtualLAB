"use client"

import { useState, useEffect } from "react"
import type { Algorithm } from "@/lib/types"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { getQuizQuestions } from "@/lib/quiz-questions"
import {
  CheckCircle,
  AlertCircle,
  Timer,
  Trophy,
  Brain,
  ArrowRight,
  RotateCcw,
  Award,
  BookOpen,
  Lightbulb,
  Zap,
} from "lucide-react"

interface TestTabProps {
  algorithm: Algorithm
}

type DifficultyLevel = "beginner" | "intermediate" | "advanced"

export function TestTab({ algorithm }: TestTabProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, string>>({})
  const [testSubmitted, setTestSubmitted] = useState(false)
  const [timeRemaining, setTimeRemaining] = useState(600) // 10 minutes in seconds
  const [difficultyLevel, setDifficultyLevel] = useState<DifficultyLevel>("beginner")
  const [quizStarted, setQuizStarted] = useState(false)
  const [timerActive, setTimerActive] = useState(false)
  const [quizCompleted, setQuizCompleted] = useState(false)

  // Get questions for the current algorithm and difficulty level
  const questions = getQuizQuestions(algorithm.slug, difficultyLevel)

  // Timer effect
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null

    if (timerActive && timeRemaining > 0) {
      timer = setInterval(() => {
        setTimeRemaining((prev) => prev - 1)
      }, 1000)
    } else if (timeRemaining === 0) {
      handleSubmitTest()
    }

    return () => {
      if (timer) clearInterval(timer)
    }
  }, [timerActive, timeRemaining])

  const handleSelectAnswer = (answerId: string) => {
    setSelectedAnswers({
      ...selectedAnswers,
      [currentQuestion]: answerId,
    })
  }

  const handleNextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1)
    }
  }

  const handlePreviousQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1)
    }
  }

  const handleSubmitTest = () => {
    setTestSubmitted(true)
    setTimerActive(false)
    setQuizCompleted(true)
  }

  const calculateScore = () => {
    let correctCount = 0
    Object.entries(selectedAnswers).forEach(([questionIndex, answerId]) => {
      if (questions[Number(questionIndex)].correctAnswer === answerId) {
        correctCount++
      }
    })
    return {
      score: correctCount,
      total: questions.length,
      percentage: Math.round((correctCount / questions.length) * 100),
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`
  }

  const startQuiz = (level: DifficultyLevel) => {
    setDifficultyLevel(level)
    setQuizStarted(true)
    setTimerActive(true)
    setTimeRemaining(level === "beginner" ? 600 : level === "intermediate" ? 900 : 1200) // 10, 15, or 20 minutes
    setCurrentQuestion(0)
    setSelectedAnswers({})
    setTestSubmitted(false)
    setQuizCompleted(false)
  }

  const resetQuiz = () => {
    setQuizStarted(false)
    setTimerActive(false)
    setTestSubmitted(false)
    setSelectedAnswers({})
    setCurrentQuestion(0)
    setQuizCompleted(false)
  }

  const score = calculateScore()

  // If no questions are available
  if (!questions || questions.length === 0) {
    return (
      <div className="py-8 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <BookOpen className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-xl font-medium mb-2">Quiz Coming Soon</h3>
        <p className="text-muted-foreground max-w-md mx-auto">
          We're currently developing quiz questions for {algorithm.title}. Check back soon for interactive quizzes!
        </p>
      </div>
    )
  }

  // Quiz selection screen
  if (!quizStarted) {
    return (
      <div className="space-y-6 py-4">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold">{algorithm.title} Quiz</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          <Card
            className="p-6 hover:shadow-md transition-shadow cursor-pointer border-2 hover:border-primary/50"
            onClick={() => startQuiz("beginner")}
          >
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="bg-green-100 dark:bg-green-900 p-3 rounded-full">
                <BookOpen className="h-8 w-8 text-green-600 dark:text-green-400" />
              </div>
              <h3 className="text-xl font-semibold">Beginner</h3>
              <Badge variant="outline" className="px-3 py-1">
                10 Minutes
              </Badge>
              <p className="text-muted-foreground">Fundamental concepts and basic applications of {algorithm.title}.</p>
              <Button className="w-full mt-2">
                Start Quiz <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </Card>

          <Card
            className="p-6 hover:shadow-md transition-shadow cursor-pointer border-2 hover:border-primary/50"
            onClick={() => startQuiz("intermediate")}
          >
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="bg-blue-100 dark:bg-blue-900 p-3 rounded-full">
                <Lightbulb className="h-8 w-8 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold">Intermediate</h3>
              <Badge variant="outline" className="px-3 py-1">
                15 Minutes
              </Badge>
              <p className="text-muted-foreground">
                Deeper understanding and practical implementation of {algorithm.title}.
              </p>
              <Button className="w-full mt-2">
                Start Quiz <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </Card>

          <Card
            className="p-6 hover:shadow-md transition-shadow cursor-pointer border-2 hover:border-primary/50"
            onClick={() => startQuiz("advanced")}
          >
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="bg-purple-100 dark:bg-purple-900 p-3 rounded-full">
                <Zap className="h-8 w-8 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold">Advanced</h3>
              <Badge variant="outline" className="px-3 py-1">
                20 Minutes
              </Badge>
              <p className="text-muted-foreground">Complex scenarios and advanced techniques for {algorithm.title}.</p>
              <Button className="w-full mt-2">
                Start Quiz <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 py-4">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <h2 className="text-2xl font-bold">{algorithm.title} Quiz</h2>
          <Badge
            variant={
              difficultyLevel === "beginner"
                ? "success"
                : difficultyLevel === "intermediate"
                  ? "default"
                  : "destructive"
            }
            className="px-3 py-1"
          >
            {difficultyLevel === "beginner"
              ? "Beginner"
              : difficultyLevel === "intermediate"
                ? "Intermediate"
                : "Advanced"}
          </Badge>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-muted px-3 py-1 rounded-full">
            <Timer className="h-4 w-4 text-muted-foreground" />
            <span className={`text-sm font-medium ${timeRemaining < 60 ? "text-red-500 animate-pulse" : ""}`}>
              {formatTime(timeRemaining)}
            </span>
          </div>
          <Progress value={(currentQuestion / (questions.length - 1)) * 100} className="w-40" />
        </div>
      </div>

      {!testSubmitted ? (
        <Card className="p-6">
          <div className="mb-6">
            <div className="text-sm text-muted-foreground mb-2">
              Question {currentQuestion + 1} of {questions.length}
            </div>
            <h3 className="text-lg font-medium mb-4">{questions[currentQuestion].question}</h3>

            <RadioGroup
              value={selectedAnswers[currentQuestion] || ""}
              onValueChange={handleSelectAnswer}
              className="space-y-3"
            >
              {questions[currentQuestion].options.map((option) => (
                <div
                  key={option.id}
                  className="flex items-center space-x-2 border p-3 rounded-md hover:bg-muted/50 transition-colors"
                >
                  <RadioGroupItem value={option.id} id={`option-${option.id}`} />
                  <Label htmlFor={`option-${option.id}`} className="text-base flex-1 cursor-pointer">
                    {option.text}
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>

          <div className="flex items-center justify-between">
            <Button variant="outline" onClick={handlePreviousQuestion} disabled={currentQuestion === 0}>
              Previous
            </Button>

            {currentQuestion < questions.length - 1 ? (
              <Button onClick={handleNextQuestion} disabled={!selectedAnswers[currentQuestion]}>
                Next
              </Button>
            ) : (
              <Button onClick={handleSubmitTest} disabled={Object.keys(selectedAnswers).length < questions.length}>
                Submit Quiz
              </Button>
            )}
          </div>
        </Card>
      ) : (
        <div className="space-y-6">
          <Card className="p-6">
            <div className="flex flex-col items-center text-center mb-6">
              <div
                className={`p-4 rounded-full mb-4 ${
                  score.percentage >= 80
                    ? "bg-green-100 dark:bg-green-900"
                    : score.percentage >= 60
                      ? "bg-yellow-100 dark:bg-yellow-900"
                      : "bg-red-100 dark:bg-red-900"
                }`}
              >
                {score.percentage >= 80 ? (
                  <Trophy className="h-12 w-12 text-green-600 dark:text-green-400" />
                ) : score.percentage >= 60 ? (
                  <Award className="h-12 w-12 text-yellow-600 dark:text-yellow-400" />
                ) : (
                  <Brain className="h-12 w-12 text-red-600 dark:text-red-400" />
                )}
              </div>
              <h3 className="text-2xl font-bold mb-2">Quiz Completed!</h3>
              <p className="text-muted-foreground text-lg">
                You scored {score.score} out of {score.total} ({score.percentage}%)
              </p>

              <div className="w-full max-w-xs mt-6">
                <div className="h-4 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      score.percentage >= 80 ? "bg-green-500" : score.percentage >= 60 ? "bg-yellow-500" : "bg-red-500"
                    }`}
                    style={{ width: `${score.percentage}%` }}
                  ></div>
                </div>
                <div className="flex justify-between mt-2 text-sm text-muted-foreground">
                  <span>0%</span>
                  <span>50%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">Performance Summary</h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Correct Answers:</span>
                    <span className="font-medium">{score.score}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Incorrect Answers:</span>
                    <span className="font-medium">{score.total - score.score}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Completion Time:</span>
                    <span className="font-medium">
                      {formatTime(
                        (difficultyLevel === "beginner" ? 600 : difficultyLevel === "intermediate" ? 900 : 1200) -
                          timeRemaining,
                      )}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Difficulty Level:</span>
                    <span className="font-medium capitalize">{difficultyLevel}</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-3">Recommendations</h4>
                <ul className="space-y-2 text-sm">
                  {score.percentage < 60 && (
                    <li className="flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 text-red-500 mt-0.5" />
                      <span>Review the tutorial section on key concepts</span>
                    </li>
                  )}
                  {score.percentage < 80 && (
                    <li className="flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 text-yellow-500 mt-0.5" />
                      <span>Practice with more examples before retaking</span>
                    </li>
                  )}
                  {score.percentage >= 80 && (
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>
                        {difficultyLevel !== "advanced"
                          ? `You're ready to try the ${difficultyLevel === "beginner" ? "intermediate" : "advanced"} level!`
                          : "You've mastered this algorithm!"}
                      </span>
                    </li>
                  )}
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5" />
                    <span>Try the practice exercises for more hands-on experience</span>
                  </li>
                </ul>
              </div>
            </div>

            <Separator className="my-6" />

            <div className="flex justify-between">
              <Button variant="outline" onClick={() => setTestSubmitted(false)}>
                Review Answers
              </Button>
              <div className="space-x-2">
                <Button variant="outline" onClick={resetQuiz}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Try Different Level
                </Button>
                <Button
                  onClick={() => {
                    setSelectedAnswers({})
                    setCurrentQuestion(0)
                    setTestSubmitted(false)
                    setTimerActive(true)
                  }}
                >
                  Retake Quiz
                </Button>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="font-medium mb-4">Question Review</h3>
            <div className="space-y-6">
              {questions.map((question, index) => (
                <div key={index} className="pb-4 border-b last:border-b-0 last:pb-0">
                  <div className="flex items-start gap-2">
                    {selectedAnswers[index] === question.correctAnswer ? (
                      <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-red-500 mt-0.5" />
                    )}
                    <div className="flex-1">
                      <h4 className="font-medium">{question.question}</h4>
                      <div className="mt-2 space-y-1">
                        {question.options.map((option) => (
                          <div
                            key={option.id}
                            className={`text-sm p-3 rounded-md ${
                              option.id === question.correctAnswer
                                ? "bg-green-50 dark:bg-green-950 text-green-700 dark:text-green-300"
                                : option.id === selectedAnswers[index] && option.id !== question.correctAnswer
                                  ? "bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300"
                                  : "bg-muted/30"
                            }`}
                          >
                            {option.text}
                            {option.id === question.correctAnswer && (
                              <span className="ml-2 text-xs font-medium">(Correct Answer)</span>
                            )}
                          </div>
                        ))}
                      </div>
                      {question.explanation && (
                        <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300 rounded-md text-sm">
                          <strong>Explanation:</strong> {question.explanation}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

