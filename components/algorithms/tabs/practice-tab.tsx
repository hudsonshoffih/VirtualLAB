"use client"

import type { Algorithm } from "@/lib/types"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useState } from "react"
import { ArrowRight } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { PythonCodeEditor } from "@/components/python-code-editor"

interface PracticeTabProps {
  algorithm: Algorithm
}

export function PracticeTab({ algorithm }: PracticeTabProps) {
  const [currentStep, setCurrentStep] = useState(0)

  // Simulated practice steps
  const steps = [
    {
      title: "Step 1: Load the Dataset",
      instruction: "Import necessary libraries and load the dataset for analysis.",
    },
    {
      title: "Step 2: Understand the Data",
      instruction: "Check the structure, data types, and general statistics of the dataset.",
    },
    {
      title: "Step 3: Evaluate Results",
      instruction: "Write code to evaluate the performance of your implementation.",
    },
  ]

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
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
            <PythonCodeEditor algorithm={algorithm.slug} />

            <div className="flex items-center justify-end mt-4">
              <Button variant="outline" onClick={handleNext} disabled={currentStep === steps.length - 1}>
                Next Step <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

