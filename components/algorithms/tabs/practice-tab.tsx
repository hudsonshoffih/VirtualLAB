"use client"

import type { Algorithm } from "@/lib/types"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"
import { CheckCircle, AlertCircle, ArrowRight } from "lucide-react"
import { Progress } from "@/components/ui/progress"

interface PracticeTabProps {
  algorithm: Algorithm
}

export function PracticeTab({ algorithm }: PracticeTabProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [userCode, setUserCode] = useState("")
  const [result, setResult] = useState<null | { success: boolean; message: string }>(null)

  // Simulated practice steps
  const steps = [
    {
      title: "Step 1: Data Preparation",
      instruction: "Write code to load and preprocess the dataset for analysis.",
      starterCode:
        "// Load the dataset\nfunction loadData() {\n  // Your code here\n}\n\n// Preprocess the data\nfunction preprocessData(data) {\n  // Your code here\n}",
    },
    {
      title: "Step 2: Implement the Algorithm",
      instruction: "Implement the core functionality of the algorithm.",
      starterCode: "// Implement the algorithm\nfunction runAlgorithm(data) {\n  // Your code here\n}",
    },
    {
      title: "Step 3: Evaluate Results",
      instruction: "Write code to evaluate the performance of your implementation.",
      starterCode: "// Evaluate performance\nfunction evaluateResults(predictions, actual) {\n  // Your code here\n}",
    },
  ]

  const handleSubmit = () => {
    // In a real app, you would validate the code
    setResult({
      success: true,
      message: "Great job! Your solution works correctly.",
    })
  }

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
      setUserCode(steps[currentStep + 1].starterCode)
      setResult(null)
    }
  }

  return (
    <div className="space-y-6 py-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">{algorithm.title} Practice</h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Progress:</span>
          <Progress value={(currentStep / (steps.length - 1)) * 100} className="w-40" />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1">
          <Card className="p-4">
            <h3 className="font-medium mb-3">{steps[currentStep].title}</h3>
            <p className="text-sm text-muted-foreground mb-4">{steps[currentStep].instruction}</p>

            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-medium mb-2">Hints</h4>
                <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                  <li>Consider the data structure you'll need</li>
                  <li>Remember to handle edge cases</li>
                  <li>Check for efficiency in your implementation</li>
                </ul>
              </div>

              <div>
                <h4 className="text-sm font-medium mb-2">Resources</h4>
                <ul className="text-sm space-y-1">
                  <li>
                    <Button variant="link" className="p-0 h-auto text-sm">
                      Documentation
                    </Button>
                  </li>
                  <li>
                    <Button variant="link" className="p-0 h-auto text-sm">
                      API Reference
                    </Button>
                  </li>
                  <li>
                    <Button variant="link" className="p-0 h-auto text-sm">
                      Example Code
                    </Button>
                  </li>
                </ul>
              </div>
            </div>
          </Card>
        </div>

        <div className="md:col-span-2">
          <Card className="p-6">
            <div className="mb-4">
              <Textarea
                value={userCode}
                onChange={(e) => setUserCode(e.target.value)}
                className="font-mono text-sm h-80"
                placeholder="Write your code here..."
              />
            </div>

            <div className="flex items-center justify-between">
              <Button onClick={handleSubmit}>Submit Solution</Button>

              <Button
                variant="outline"
                onClick={handleNext}
                disabled={currentStep === steps.length - 1 || !result?.success}
              >
                Next Step <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>

            {result && (
              <div
                className={`mt-4 p-3 rounded-md ${result.success ? "bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-900" : "bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-900"}`}
              >
                <div className="flex items-start gap-2">
                  {result.success ? (
                    <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-red-500 mt-0.5" />
                  )}
                  <div>
                    <p
                      className={`font-medium ${result.success ? "text-green-700 dark:text-green-300" : "text-red-700 dark:text-red-300"}`}
                    >
                      {result.success ? "Success!" : "Error"}
                    </p>
                    <p
                      className={`text-sm ${result.success ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}
                    >
                      {result.message}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}

