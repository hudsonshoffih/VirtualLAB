"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { Search, FileText, Code, BookOpen, HelpCircle, Play } from "lucide-react"
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import { Button } from "@/components/ui/button"
import { getAlgorithms } from "@/lib/algorithms"
import { Badge } from "@/components/ui/badge"

type SearchMode = "algorithms" | "quiz" | "demo"

export function CommandSearch() {
  const [open, setOpen] = React.useState(false)
  const [searchMode, setSearchMode] = React.useState<SearchMode>("algorithms")
  const router = useRouter()
  const algorithms = getAlgorithms()

  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      // Only trigger if not typing in an input field
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      if (e.key === "a" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault()
        setSearchMode("algorithms")
        setOpen(true)
      } else if (e.key === "q" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault()
        setSearchMode("quiz")
        setOpen(true)
      } else if (e.key === "d" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault()
        setSearchMode("demo")
        setOpen(true)
      }
    }

    document.addEventListener("keydown", down)
    return () => document.removeEventListener("keydown", down)
  }, [])

  const runCommand = React.useCallback((command: () => unknown) => {
    setOpen(false)
    command()
  }, [])

  // Group algorithms by category
  const algorithmsByCategory = React.useMemo(() => {
    const categories: Record<string, typeof algorithms> = {
      "Data Analysis": [],
      "Supervised Learning": [],
      "Unsupervised Learning": [],
      Other: [],
    }

    algorithms.forEach((algorithm) => {
      if (algorithm.slug.includes("eda") || algorithm.slug.includes("insights") || algorithm.slug.includes("metrics")) {
        categories["Data Analysis"].push(algorithm)
      } else if (
        algorithm.slug.includes("regression") ||
        algorithm.slug.includes("svm") ||
        algorithm.slug.includes("knn") ||
        algorithm.slug.includes("forest") ||
        algorithm.slug.includes("ensemble")
      ) {
        categories["Supervised Learning"].push(algorithm)
      } else if (algorithm.slug.includes("kmeans") || algorithm.slug.includes("pca")) {
        categories["Unsupervised Learning"].push(algorithm)
      } else {
        categories["Other"].push(algorithm)
      }
    })

    return categories
  }, [algorithms])

  const getAlgorithmIcon = (slug: string) => {
    if (slug.includes("eda") || slug.includes("insights")) return FileText
    if (slug.includes("regression") || slug.includes("svm")) return Code
    return BookOpen
  }

  const getAlgorithmBadge = (slug: string) => {
    if (slug.includes("eda") || slug.includes("insights") || slug.includes("metrics")) return "Analysis"
    if (
      slug.includes("regression") ||
      slug.includes("svm") ||
      slug.includes("knn") ||
      slug.includes("forest") ||
      slug.includes("ensemble")
    )
      return "Supervised"
    if (slug.includes("kmeans") || slug.includes("pca")) return "Unsupervised"
    return "Other"
  }

  const getSearchPlaceholder = () => {
    switch (searchMode) {
      case "algorithms":
        return "Search algorithms and tutorials..."
      case "quiz":
        return "Search quizzes and tests..."
      case "demo":
        return "Search demos and interactive examples..."
      default:
        return "Search..."
    }
  }

  const getSearchTitle = () => {
    switch (searchMode) {
      case "algorithms":
        return "Search Algorithms"
      case "quiz":
        return "Search Quizzes"
      case "demo":
        return "Search Demos"
      default:
        return "Search"
    }
  }

  const renderAlgorithmResults = () => (
    <>
      {Object.entries(algorithmsByCategory).map(
        ([category, categoryAlgorithms]) =>
          categoryAlgorithms.length > 0 && (
            <CommandGroup key={category} heading={category}>
              {categoryAlgorithms.map((algorithm) => {
                const Icon = getAlgorithmIcon(algorithm.slug)
                const badgeText = getAlgorithmBadge(algorithm.slug)

                return (
                  <CommandItem
                    key={algorithm.slug}
                    value={`${algorithm.title} ${algorithm.description} ${badgeText}`}
                    onSelect={() => {
                      runCommand(() => router.push(`/algorithms/${algorithm.slug}`))
                    }}
                    className="flex items-center gap-3 p-3"
                  >
                    <Icon className="h-4 w-4 text-muted-foreground" />
                    <div className="flex-1 space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{algorithm.title}</span>
                        <Badge variant="secondary" className="text-xs">
                          {badgeText}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground line-clamp-1">{algorithm.description}</p>
                    </div>
                  </CommandItem>
                )
              })}
            </CommandGroup>
          ),
      )}
    </>
  )

  const renderQuizResults = () => (
    <CommandGroup heading="Available Quizzes">
      {algorithms.map((algorithm) => (
        <CommandItem
          key={`quiz-${algorithm.slug}`}
          value={`${algorithm.title} quiz test assessment`}
          onSelect={() => {
            runCommand(() => {
              // Navigate to the algorithm page with quiz tab
              router.push(`/algorithms/${algorithm.slug}?tab=test`)
              // Small delay to ensure navigation completes
              setTimeout(() => {
                window.location.reload()
              }, 100)
            })
          }}
          className="flex items-center gap-3 p-3"
        >
          <HelpCircle className="h-4 w-4 text-muted-foreground" />
          <div className="flex-1 space-y-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">{algorithm.title} Quiz</span>
              <Badge variant="outline" className="text-xs">
                Test
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">Test your knowledge of {algorithm.title.toLowerCase()}</p>
          </div>
        </CommandItem>
      ))}
    </CommandGroup>
  )

  const renderDemoResults = () => (
    <CommandGroup heading="Interactive Demos">
      {algorithms.map((algorithm) => (
        <CommandItem
          key={`demo-${algorithm.slug}`}
          value={`${algorithm.title} demo interactive example visualization`}
          onSelect={() => {
            runCommand(() => {
              // Navigate to the algorithm page with demo tab
              router.push(`/algorithms/${algorithm.slug}?tab=demo`)
              // Small delay to ensure navigation completes
              setTimeout(() => {
                window.location.reload()
              }, 100)
            })
          }}
          className="flex items-center gap-3 p-3"
        >
          <Play className="h-4 w-4 text-muted-foreground" />
          <div className="flex-1 space-y-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">{algorithm.title} Demo</span>
              <Badge variant="outline" className="text-xs">
                Interactive
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Interactive demo and visualization for {algorithm.title.toLowerCase()}
            </p>
          </div>
        </CommandItem>
      ))}
    </CommandGroup>
  )

  return (
    <>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          className="relative h-9 w-9 p-0 xl:h-10 xl:w-60 xl:justify-start xl:px-3 xl:py-2"
          onClick={() => {
            setSearchMode("algorithms")
            setOpen(true)
          }}
        >
          <Search className="h-4 w-4 xl:mr-2" />
          <span className="hidden xl:inline-flex">Search...</span>
          <div className="pointer-events-none absolute right-1.5 top-2 hidden h-6 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 xl:flex">
            <span className="text-xs">A</span>
          </div>
        </Button>

        {/* Quick access buttons */}
        <div className="hidden md:flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setSearchMode("quiz")
              setOpen(true)
            }}
            className="h-8 px-2 text-xs"
          >
            <HelpCircle className="h-3 w-3 mr-1" />
            Quiz
            <kbd className="ml-1 text-[10px] bg-muted px-1 rounded">Q</kbd>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setSearchMode("demo")
              setOpen(true)
            }}
            className="h-8 px-2 text-xs"
          >
            <Play className="h-3 w-3 mr-1" />
            Demo
            <kbd className="ml-1 text-[10px] bg-muted px-1 rounded">D</kbd>
          </Button>
        </div>
      </div>

      <CommandDialog open={open} onOpenChange={setOpen}>
        <div className="flex items-center justify-between border-b px-3 py-2">
          <h3 className="text-sm font-medium">{getSearchTitle()}</h3>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <kbd className="bg-muted px-1.5 py-0.5 rounded text-[10px]">A</kbd>
            <span>Algorithms</span>
            <kbd className="bg-muted px-1.5 py-0.5 rounded text-[10px]">Q</kbd>
            <span>Quiz</span>
            <kbd className="bg-muted px-1.5 py-0.5 rounded text-[10px]">D</kbd>
            <span>Demo</span>
          </div>
        </div>
        <CommandInput placeholder={getSearchPlaceholder()} />
        <CommandList>
          <CommandEmpty>No results found.</CommandEmpty>

          {searchMode === "algorithms" && renderAlgorithmResults()}
          {searchMode === "quiz" && renderQuizResults()}
          {searchMode === "demo" && renderDemoResults()}

          <CommandGroup heading="Quick Actions">
            <CommandItem onSelect={() => runCommand(() => router.push("/"))} className="flex items-center gap-3 p-3">
              <BookOpen className="h-4 w-4 text-muted-foreground" />
              <span>Go to Home</span>
            </CommandItem>
            <CommandItem
              onSelect={() => {
                setSearchMode("algorithms")
                // Keep dialog open for mode switch
                setTimeout(() => setOpen(true), 100)
              }}
              className="flex items-center gap-3 p-3"
            >
              <Search className="h-4 w-4 text-muted-foreground" />
              <span>Switch to Algorithms</span>
              <kbd className="ml-auto text-xs bg-muted px-1 rounded">A</kbd>
            </CommandItem>
            <CommandItem
              onSelect={() => {
                setSearchMode("quiz")
                setTimeout(() => setOpen(true), 100)
              }}
              className="flex items-center gap-3 p-3"
            >
              <HelpCircle className="h-4 w-4 text-muted-foreground" />
              <span>Switch to Quiz</span>
              <kbd className="ml-auto text-xs bg-muted px-1 rounded">Q</kbd>
            </CommandItem>
            <CommandItem
              onSelect={() => {
                setSearchMode("demo")
                setTimeout(() => setOpen(true), 100)
              }}
              className="flex items-center gap-3 p-3"
            >
              <Play className="h-4 w-4 text-muted-foreground" />
              <span>Switch to Demo</span>
              <kbd className="ml-auto text-xs bg-muted px-1 rounded">D</kbd>
            </CommandItem>
          </CommandGroup>
        </CommandList>
      </CommandDialog>
    </>
  )
}
