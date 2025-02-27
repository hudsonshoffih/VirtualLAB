"use client"

import type { Algorithm } from "@/lib/types"
import { Card } from "@/components/ui/card"
import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight } from "lucide-react"

interface TutorialTabProps {
  algorithm: Algorithm
}

export function TutorialTab({ algorithm }: TutorialTabProps) {
  const [content, setContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [currentSection, setCurrentSection] = useState(0)

  // Simulating fetching tutorial content
  useEffect(() => {
    // In a real app, you would fetch the actual content
    setTimeout(() => {
      setContent(algorithm.tutorialContent || "# Tutorial content would be loaded here")
      setLoading(false)
    }, 1000)
  }, [algorithm])

  // Simulated tutorial sections
  const sections = ["Introduction", "Mathematical Foundation", "Implementation", "Applications", "Advanced Topics"]

  return (
    <div className="space-y-6 py-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">{algorithm.title} Tutorial</h2>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={currentSection === 0}
            onClick={() => setCurrentSection((prev) => Math.max(0, prev - 1))}
          >
            <ChevronLeft className="h-4 w-4 mr-1" /> Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            {currentSection + 1} of {sections.length}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={currentSection === sections.length - 1}
            onClick={() => setCurrentSection((prev) => Math.min(sections.length - 1, prev + 1))}
          >
            Next <ChevronRight className="h-4 w-4 ml-1" />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="md:col-span-1">
          <Card className="p-4">
            <h3 className="font-medium mb-3">Sections</h3>
            <ul className="space-y-1">
              {sections.map((section, index) => (
                <li key={index}>
                  <Button
                    variant={currentSection === index ? "secondary" : "ghost"}
                    className="w-full justify-start text-sm h-9"
                    onClick={() => setCurrentSection(index)}
                  >
                    {section}
                  </Button>
                </li>
              ))}
            </ul>
          </Card>
        </div>

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
              <div>
                <h3 className="text-xl font-bold mb-4">{sections[currentSection]}</h3>
                <div className="prose dark:prose-invert max-w-none">
                  {/* In a real app, you would render markdown content here */}
                  <p>This is where the tutorial content for {algorithm.title} would be displayed.</p>
                  <p>
                    The content would include explanations, formulas, code examples, and diagrams related to{" "}
                    {sections[currentSection]}.
                  </p>

                  {currentSection === 0 && (
                    <>
                      <h4>What is {algorithm.title}?</h4>
                      <p>An introduction to the algorithm, its history, and its importance in data science.</p>
                    </>
                  )}

                  {currentSection === 1 && (
                    <>
                      <h4>Mathematical Foundation</h4>
                      <p>The mathematical principles and formulas behind the algorithm.</p>
                      <div className="bg-muted p-4 rounded-md my-4">
                        <code>Mathematical formula would be displayed here</code>
                      </div>
                    </>
                  )}

                  {currentSection === 2 && (
                    <>
                      <h4>Implementation</h4>
                      <p>Step-by-step guide on implementing the algorithm.</p>
                      <div className="bg-muted p-4 rounded-md my-4">
                        <pre>
                          <code
                            dangerouslySetInnerHTML={{
                              __html:
                                "// Code example would be displayed here\nfunction example() {\n  // Implementation details\n  return result;\n}",
                            }}
                          />
                        </pre>
                      </div>
                    </>
                  )}

                  {currentSection === 3 && (
                    <>
                      <h4>Applications</h4>
                      <p>Real-world applications and use cases for the algorithm.</p>
                      <ul>
                        <li>Application 1</li>
                        <li>Application 2</li>
                        <li>Application 3</li>
                      </ul>
                    </>
                  )}

                  {currentSection === 4 && (
                    <>
                      <h4>Advanced Topics</h4>
                      <p>More complex aspects and extensions of the algorithm.</p>
                      <div className="bg-muted p-4 rounded-md my-4">
                        <p>Advanced concepts and techniques would be explained here.</p>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}

