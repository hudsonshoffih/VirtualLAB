"use client"

import { useEffect, useState } from "react"
import type { Algorithm } from "@/lib/types"
import { Card } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ChevronLeft, ChevronRight, Check } from "lucide-react"
import { getTutorialSections } from "@/lib/tutorial-sections"
import { EdaTutorial } from "./tutorial-content/eda-tutorial"
import { DataInsightsTutorial } from "./tutorial-content/data-insights-tutorial"
import { EvaluationMetricsTutorial } from "./tutorial-content/evaluation-metrics-tutorial"
import { LinearRegressionTutorial } from "./tutorial-content/linear-regression-tutorial"
import { LogisticRegressionTutorial } from "./tutorial-content/logistic-regression-tutorial"


interface TutorialTabProps {
  algorithm: Algorithm
}

export function TutorialTab({ algorithm }: TutorialTabProps) {
  const [loading, setLoading] = useState(true)
  const [currentSection, setCurrentSection] = useState(0)
  const [copied, setCopied] = useState<string | null>(null)

  const sections = getTutorialSections(algorithm.slug)

  // Simulating fetching tutorial content
  useEffect(() => {
    setTimeout(() => {
      setLoading(false)
    }, 1000)
  }, [])

  // Function to copy code to clipboard
  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }

  // Calculate progress
  const progress = ((currentSection + 1) / sections.length) * 100

  const renderTutorialContent = () => {
    switch (algorithm.slug) {
      case "eda":
        return <EdaTutorial section={currentSection} onCopy={copyToClipboard} copied={copied} />
      case "dataset-insights":
        return <DataInsightsTutorial section={currentSection} onCopy={copyToClipboard} copied={copied} />
      case "evaluation-metrics":
        return <EvaluationMetricsTutorial section={currentSection} onCopy={copyToClipboard} copied={copied} />
      case "linear-regression":
        return <LinearRegressionTutorial section={currentSection} onCopy={copyToClipboard} copied={copied} />
      case "logistic-regression":
        return <LogisticRegressionTutorial section={currentSection} onCopy={copyToClipboard} copied={copied} />
      default:
        return (
          <div className="prose dark:prose-invert max-w-none">
            <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
              <h4 className="mt-0 text-lg font-semibold">Tutorial Content Coming Soon</h4>
              <p className="mb-0">The tutorial content for {algorithm.title} is currently being developed.</p>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="space-y-6 py-4">
      {/* Header with progress */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold">{algorithm.title} Tutorial</h2>
          <Badge variant="outline" className="px-3 py-1">
            {currentSection + 1} of {sections.length}
          </Badge>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Navigation sidebar */}
        <div className="md:col-span-1">
          <Card className="p-4 sticky top-4">
            <h3 className="font-medium mb-4 text-center border-b pb-2">Learning Path</h3>
            <ul className="space-y-2">
              {sections.map((section, index) => (
                <li key={index}>
                  <Button
                    variant={currentSection === index ? "default" : "ghost"}
                    className={`w-full justify-start text-sm h-10 ${
                      index < currentSection ? "text-muted-foreground" : ""
                    }`}
                    onClick={() => setCurrentSection(index)}
                  >
                    <div className="flex items-center gap-2">
                      {<section.icon className="h-4 w-4" />}
                      <span>{section.title}</span>
                    </div>
                    {index < currentSection && <Check className="h-4 w-4 ml-auto text-green-500" />}
                  </Button>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        {/* Content area */}
        <div className="md:col-span-3">
          <Card className="p-6">
            {loading ? (
              <div className="space-y-4">
                <Skeleton className="h-8 w-3/4" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-32 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
              </div>
            ) : (
              renderTutorialContent()
            )}
          </Card>
        </div>
      </div>

      {/* Navigation buttons */}
      <div className="flex justify-between mt-6">
        <Button
          variant="outline"
          size="lg"
          disabled={currentSection === 0}
          onClick={() => setCurrentSection((prev) => Math.max(0, prev - 1))}
          className="w-[120px]"
        >
          <ChevronLeft className="h-4 w-4 mr-2" /> Previous
        </Button>

        <Button
          variant={currentSection === sections.length - 1 ? "default" : "outline"}
          size="lg"
          disabled={currentSection === sections.length - 1}
          onClick={() => setCurrentSection((prev) => Math.min(sections.length - 1, prev + 1))}
          className="w-[120px]"
        >
          {currentSection === sections.length - 1 ? "Complete" : "Next"}
          {currentSection !== sections.length - 1 && <ChevronRight className="h-4 w-4 ml-2" />}
        </Button>
      </div>
    </div>
  )
}