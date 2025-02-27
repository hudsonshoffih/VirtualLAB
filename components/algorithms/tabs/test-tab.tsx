"use client"

import type { Algorithm } from "@/lib/types"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { useState } from "react"
import { CheckCircle, AlertCircle, Timer, Trophy } from "lucide-react"
import { Progress } from "@/components/ui/progress"

interface TestTabProps {
  algorithm: Algorithm
}

export function TestTab({ algorithm }: TestTabProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, string>>({})
  const [testSubmitted, setTestSubmitted] = useState(false)
  const [timeRemaining, setTimeRemaining] = useState(600) // 10 minutes in seconds

  // Simulated test questions
  const questions = [
    {
      question: "What is the primary purpose of Exploratory Data Analysis?",
      options: [
        { id: "a", text: "To build predictive models" },
        { id: "b", text: "To understand patterns and relationships in data" },
        { id: "c", text: "To clean and preprocess data" },
        { id: "d", text: "To visualize data in 3D" },
      ],
      correctAnswer: "b",
    },
    {
      question: "Which of the following is NOT typically part of EDA?",
      options: [
        { id: "a", text: "Checking for missing values" },
        { id: "b", text: "Identifying outliers" },
        { id: "c", text: "Deploying models to production" },
        { id: "d", text: "Visualizing distributions" },
      ],
      correctAnswer: "c",
    },
    {
      question: "What statistical measure is most useful for identifying the central tendency in skewed data?",
      options: [
        { id: "a", text: "Mean" },
        { id: "b", text: "Mode" },
        { id: "c", text: "Median" },
        { id: "d", text: "Standard deviation" },
      ],
      correctAnswer: "c",
    },
    {
      question: "Which visualization is best for comparing distributions of multiple groups?",
      options: [
        { id: "a", text: "Pie chart" },
        { id: "b", text: "Line chart" },
        { id: "c", text: "Box plot" },
        { id: "d", text: "Scatter plot" },
      ],
      correctAnswer: "c",
    },
    {
      question: "What is a correlation coefficient used for?",
      options: [
        { id: "a", text: "To measure the size of a dataset" },
        { id: "b", text: "To measure the relationship between two variables" },
        { id: "c", text: "To count the number of categories" },
        { id: "d", text: "To calculate the median" },
      ],
      correctAnswer: "b",
    },
  ]

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

  const score = calculateScore()

  return (
    <div className="space-y-6 py-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">{algorithm.title} Test</h2>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Timer className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">{formatTime(timeRemaining)}</span>
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
                <div key={option.id} className="flex items-center space-x-2">
                  <RadioGroupItem value={option.id} id={`option-${option.id}`} />
                  <Label htmlFor={`option-${option.id}`} className="text-base">
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
                Submit Test
              </Button>
            )}
          </div>
        </Card>
      ) : (
        <div className="space-y-6">
          <Card className="p-6">
            <div className="flex flex-col items-center text-center mb-6">
              <div className="bg-primary/10 p-3 rounded-full mb-4">
                <Trophy className="h-10 w-10 text-primary" />
              </div>
              <h3 className="text-2xl font-bold mb-2">Test Completed!</h3>
              <p className="text-muted-foreground">
                You scored {score.score} out of {score.total} ({score.percentage}%)
              </p>

              <div className="w-full max-w-xs mt-6">
                <div className="h-4 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full ${score.percentage >= 70 ? "bg-green-500" : score.percentage >= 50 ? "bg-yellow-500" : "bg-red-500"}`}
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
                    <span className="font-medium">8:23</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-3">Recommendations</h4>
                <ul className="space-y-2 text-sm">
                  {score.percentage < 70 && (
                    <li className="flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 text-yellow-500 mt-0.5" />
                      <span>Review the tutorial section on key concepts</span>
                    </li>
                  )}
                  {score.percentage < 50 && (
                    <li className="flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 text-red-500 mt-0.5" />
                      <span>Practice with more examples before retaking</span>
                    </li>
                  )}
                  {score.percentage >= 70 && (
                    <li className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <span>You're ready to move to the next algorithm!</span>
                    </li>
                  )}
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5" />
                    <span>Try the practice exercises for more hands-on experience</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t flex justify-between">
              <Button variant="outline">Review Answers</Button>
              <Button>Retake Test</Button>
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
                    <div>
                      <h4 className="font-medium">{question.question}</h4>
                      <div className="mt-2 space-y-1">
                        {question.options.map((option) => (
                          <div
                            key={option.id}
                            className={`text-sm p-2 rounded-md ${
                              option.id === question.correctAnswer
                                ? "bg-green-50 dark:bg-green-950 text-green-700 dark:text-green-300"
                                : option.id === selectedAnswers[index] && option.id !== question.correctAnswer
                                  ? "bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300"
                                  : ""
                            }`}
                          >
                            {option.text}
                            {option.id === question.correctAnswer && (
                              <span className="ml-2 text-xs">(Correct Answer)</span>
                            )}
                          </div>
                        ))}
                      </div>
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

