import { MainLayout } from "@/components/layouts/main-layout"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ChevronRight, Database, LineChart, Microscope, BarChart, Network } from "lucide-react"
import Link from "next/link"
import { getAlgorithms } from "@/lib/algorithms"

export default function Home() {
  const allAlgorithms = getAlgorithms()
  const featuredAlgorithms = [
    allAlgorithms[0],
    allAlgorithms[1],
    allAlgorithms[2],
    allAlgorithms[3],
    allAlgorithms[4],
    allAlgorithms[5],
    allAlgorithms[6], // EDA
    allAlgorithms[7], // Linear Regression
    allAlgorithms[8],
    allAlgorithms[9],
    allAlgorithms[10], // SVM
  ]

  const algorithmIcons: Record<string, any> = {
    eda: Database,
    "linear-regression": LineChart,
    svm: Network,
  }

  return (
    <MainLayout>
      <div className="container px-4 py-6 md:py-12">
        <div className="flex flex-col items-center text-center mb-12">
          <div className="inline-block p-2 bg-primary/10 rounded-full mb-4">
            <Microscope className="h-10 w-10 text-primary" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">Virtual Lab</h1>
          <p className="text-xl text-muted-foreground max-w-3xl">
            An interactive platform to learn, practice, and master data science algorithms through hands-on experience.
          </p>
          <div className="flex gap-4 mt-8">
            <Button asChild size="lg">
              <Link href="/algorithms/eda">Start Learning</Link>
            </Button>
            <Button variant="outline" size="lg">
              <Link href="/about">Learn More</Link>
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          {featuredAlgorithms.map((algorithm) => {
            const Icon = algorithmIcons[algorithm.slug] || BarChart

            return (
              <Card key={algorithm.title} className="transition-all hover:shadow-md">
                <CardHeader>
                  <div className="flex items-center gap-2 mb-2">
                    <Icon className="h-5 w-5 text-primary" />
                    <CardTitle>{algorithm.title}</CardTitle>
                  </div>
                  <CardDescription>{algorithm.description}</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button variant="ghost" asChild className="w-full justify-between">
                    <Link href={`/algorithms/${algorithm.slug}`}>
                      Explore <ChevronRight className="h-4 w-4" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            )
          })}
        </div>

        <div className="mt-20 border rounded-lg p-8 bg-muted/50">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-4">How It Works</h2>
              <p className="text-muted-foreground mb-6">
                Our virtual lab provides a structured learning experience with four key components:
              </p>
              <ul className="space-y-4">
                <li className="flex gap-3">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border bg-background">
                    <span className="text-sm font-medium">1</span>
                  </div>
                  <div>
                    <h3 className="font-medium">Tutorial</h3>
                    <p className="text-sm text-muted-foreground">
                      Learn the theoretical concepts and mathematical foundations.
                    </p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border bg-background">
                    <span className="text-sm font-medium">2</span>
                  </div>
                  <div>
                    <h3 className="font-medium">Demo</h3>
                    <p className="text-sm text-muted-foreground">
                      See the algorithm in action with interactive visualizations.
                    </p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border bg-background">
                    <span className="text-sm font-medium">3</span>
                  </div>
                  <div>
                    <h3 className="font-medium">Practice</h3>
                    <p className="text-sm text-muted-foreground">Apply what you've learned with guided exercises.</p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border bg-background">
                    <span className="text-sm font-medium">4</span>
                  </div>
                  <div>
                    <h3 className="font-medium">Test</h3>
                    <p className="text-sm text-muted-foreground">
                      Validate your understanding with challenges and quizzes.
                    </p>
                  </div>
                </li>
              </ul>
            </div>
            <div className="bg-background rounded-lg p-6 border">
              <div className="aspect-video bg-muted rounded-md flex items-center justify-center">
                <p className="text-muted-foreground">Interactive Demo Preview</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}

