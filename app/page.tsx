"use client"

import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import {
  ChevronRight,
  Database,
  LineChart,
  Microscope,
  BarChart,
  Network,
  Sparkles,
  Brain,
  Target,
  Zap,
  //Clock,
  //TrendingUp,
 // Users,
  Boxes,
  Layers,
  PieChart,
  CircleDot,
} from "lucide-react"
import Link from "next/link"
import { getAlgorithms } from "@/lib/algorithms"
import { motion, Easing } from "framer-motion"
import { FloatingDock } from "@/components/ui/floating-dock"
import { CommandSearch } from "@/components/search/command-search"
import { ModeToggle } from "@/components/mode-toggle"
import { HoverSidebar } from "@/components/hover-sidebar"
import { ElementType } from "react";
export default function Home() {
  const allAlgorithms = getAlgorithms();
  const featuredAlgorithms = [
    allAlgorithms[0],
    allAlgorithms[1],
    allAlgorithms[2],
    allAlgorithms[3],
    allAlgorithms[4],
    allAlgorithms[5],
    allAlgorithms[6],
    allAlgorithms[7],
    allAlgorithms[8],
    allAlgorithms[9],
    allAlgorithms[10],
  ];

const algorithmIcons: Record<string, ElementType> = {
  eda: Database,
  "dataset-insights": BarChart,
  "evaluation-metrics": PieChart,
  "linear-regression": LineChart,
  "logistic-regression": LineChart,
  knn: Network,
  "random-forest": Brain,
  svm: Network,
  "ensemble-models": Layers,
  kmeans: CircleDot,
  pca: Boxes,
};
  const dockItems = [
    {
      title: "Exploratory Data Analysis",
      icon: Database,
      href: "/algorithms/eda",
    },
    {
      title: "Dataset Insights",
      icon: BarChart,
      href: "/algorithms/dataset-insights",
    },
    {
      title: "Evaluation Metrics",
      icon: PieChart,
      href: "/algorithms/evaluation-metrics",
    },
    {
      title: "Linear Regression",
      icon: LineChart,
      href: "/algorithms/linear-regression",
    },
    {
      title: "Logistic Regression",
      icon: LineChart,
      href: "/algorithms/logistic-regression",
    },
    {
      title: "K-Nearest Neighbors",
      icon: Network,
      href: "/algorithms/knn",
    },
        {
      title: "Random Forest",
      icon: Brain,
      href: "/algorithms/random-forest",
    },
    {
      title: "Support Vector Machine",
      icon: Network,
      href: "/algorithms/svm",
    },
    {
      title: "Ensemble Models",
      icon: Layers,
      href: "/algorithms/ensemble-models",
    },
    {
      title: "K-Means Clustering",
      icon: BarChart,
      href: "/algorithms/kmeans",
    },
    {
      title: "Principal Component Analysis",
      icon: Boxes,
      href: "/algorithms/pca",
    },
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.2,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
    },
  }

  const heroVariants = {
    hidden: { scale: 0.8, opacity: 0 },
    visible: {
      scale: 1,
      opacity: 1,
      transition: {
        duration: 0.8,
        ease: "easeInOut" as Easing,
      },
    },
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-background via-background to-primary/5">
      {/* Hover Sidebar */}
      <HoverSidebar />

      {/* Full-screen header */}
      <header className="w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex h-16 items-center justify-end px-6 gap-4">
          <CommandSearch />
          <Button variant="ghost" size="sm" className="gap-2">
            <Target className="h-4 w-4" />
            Quiz
            <span className="text-xs bg-muted px-1.5 py-0.5 rounded">Q</span>
          </Button>
          <Button variant="ghost" size="sm" className="gap-2">
            <Zap className="h-4 w-4" />
            Practice
            <span className="text-xs bg-muted px-1.5 py-0.5 rounded">P</span>
          </Button>
          <ModeToggle />
        </div>
      </header>

      {/* Full-screen main content */}
      <main className="w-full">
        {/* Hero Section */}
        <motion.div
          className="w-full px-6 py-16 md:py-24"
          initial="hidden"
          animate="visible"
          variants={containerVariants}
        >
          <div className="max-w-6xl mx-auto">
            <div className="flex flex-col items-center text-center mb-16">
              <motion.div className="relative mb-8" variants={heroVariants}>
                <motion.div
                  className="absolute inset-0 bg-primary/20 rounded-full blur-3xl"
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.3, 0.6, 0.3],
                  }}
                  transition={{
                    duration: 4,
                    repeat: Number.POSITIVE_INFINITY,
                    ease: "easeInOut",
                  }}
                />
                <div className="relative inline-block p-6 bg-gradient-to-br from-primary/10 to-primary/20 rounded-full backdrop-blur-sm border border-primary/20">
                  <Microscope className="h-20 w-20 text-primary" />
                </div>
              </motion.div>

              <motion.div variants={itemVariants}>
                <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold tracking-tight mb-6">
                  <span className="text-foreground">Machine Learning </span>
                  <span className="bg-gradient-to-r from-primary via-blue-600 to-primary bg-clip-text text-transparent">
                    Virtual Lab
                  </span>
                </h1>
              </motion.div>

              <motion.p
                className="text-lg sm:text-xl md:text-2xl text-muted-foreground max-w-4xl mb-12 leading-relaxed"
                variants={itemVariants}
              >
                An interactive platform to learn, practice, and master data science algorithms through hands-on
                experience with cutting-edge visualizations.
              </motion.p>

              <motion.div className="flex flex-col sm:flex-row gap-4 mb-16" variants={itemVariants}>
                <Button
                  asChild
                  size="lg"
                  className="text-lg px-8 py-6 rounded-full bg-gradient-to-r from-primary to-blue-600 hover:from-primary/90 hover:to-blue-600/90 shadow-lg hover:shadow-xl transition-all duration-300"
                >
                  <Link href="/algorithms/eda" className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    Start Learning
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  className="text-lg px-8 py-6 rounded-full border-2 hover:bg-primary/5 transition-all duration-300 bg-transparent"
                >
                  <Link href="/about" className="flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    Learn More
                  </Link>
                </Button>
              </motion.div>

              {/* Floating Dock */}
              <motion.div
                initial={{ y: 100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1, duration: 0.8 }}
                className="w-full max-w-4xl"
              >
                <FloatingDock items={dockItems} />
              </motion.div>
            </div>
          </div>
        </motion.div>

        {/* Featured Cards Section */}
        <motion.div
          className="w-full px-6 py-16"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={containerVariants}
        >
          <div className="max-w-6xl mx-auto">
            <motion.div className="grid grid-cols-1 md:grid-cols-3 gap-8" variants={containerVariants}>
              {featuredAlgorithms.map((algorithm) => { // index is not needed here
                const Icon = algorithmIcons[algorithm.slug] || BarChart

                return (
                  <motion.div
                    key={algorithm.title}
                    variants={itemVariants}
                    whileHover={{
                      scale: 1.05,
                      transition: { duration: 0.2 },
                    }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Card className="h-full transition-all duration-300 hover:shadow-2xl hover:shadow-primary/10 border-0 bg-gradient-to-br from-card to-card/50 backdrop-blur-sm">
                      <CardHeader className="pb-4">
                        <div className="flex items-center gap-3 mb-3">
                          <motion.div
                            className="p-3 bg-primary/10 rounded-lg"
                            whileHover={{ rotate: 360 }}
                            transition={{ duration: 0.5 }}
                          >
                            <Icon className="h-6 w-6 text-primary" />
                          </motion.div>
                          <CardTitle className="text-xl">{algorithm.title}</CardTitle>
                        </div>
                        <CardDescription className="text-base leading-relaxed">{algorithm.description}</CardDescription>
                      </CardHeader>
                      <CardFooter>
                        <Button
                          variant="ghost"
                          asChild
                          className="w-full justify-between group hover:bg-primary/5 transition-all duration-300"
                        >
                          <Link href={`/algorithms/${algorithm.slug}`}>
                            Explore
                            <ChevronRight className="h-4 w-4 group-hover:translate-x-1 transition-transform duration-300" />
                          </Link>
                        </Button>
                      </CardFooter>
                    </Card>
                  </motion.div>
                )
              })}
            </motion.div>
          </div>
        </motion.div>

        {/* How It Works Section */}
        <motion.div
          className="w-full px-6 py-16"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={containerVariants}
        >
          <div className="max-w-6xl mx-auto">
            <motion.div
              className="border rounded-3xl p-8 md:p-12 bg-gradient-to-br from-muted/30 to-muted/10 backdrop-blur-sm border-primary/10"
              variants={itemVariants}
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                <div>
                  <motion.h2
                    className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-foreground to-primary bg-clip-text text-transparent"
                    variants={itemVariants}
                  >
                    How It Works
                  </motion.h2>
                  <motion.p className="text-muted-foreground mb-8 text-lg leading-relaxed" variants={itemVariants}>
                    Our virtual lab provides a structured learning experience with four key components designed to
                    accelerate your data science journey:
                  </motion.p>
                  <motion.ul className="space-y-6" variants={containerVariants}>
                    {[
                      {
                        icon: Brain,
                        title: "Tutorial",
                        desc: "Learn theoretical concepts and mathematical foundations with interactive examples.",
                      },
                      {
                        icon: Zap,
                        title: "Demo",
                        desc: "See algorithms in action with real-time visualizations and parameter adjustments.",
                      },
                      {
                        icon: Target,
                        title: "Practice",
                        desc: "Apply knowledge with guided exercises and hands-on coding challenges.",
                      },
                      {
                        icon: Sparkles,
                        title: "Test",
                        desc: "Validate understanding with adaptive quizzes and comprehensive assessments.",
                      },
                    ].map((item) => ( // index is not needed here
                      <motion.li
                        key={item.title}
                        className="flex gap-4 group"
                        variants={itemVariants}
                        whileHover={{ x: 10 }}
                        transition={{ duration: 0.2 }}
                      >
                        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full border-2 border-primary/20 bg-gradient-to-br from-primary/10 to-primary/5 group-hover:border-primary/40 transition-all duration-300">
                          <item.icon className="h-6 w-6 text-primary" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-lg mb-1">{item.title}</h3>
                          <p className="text-muted-foreground leading-relaxed">{item.desc}</p>
                        </div>
                      </motion.li>
                    ))}
                  </motion.ul>
                </div>

                <motion.div className="relative" variants={itemVariants}>
                  <div className="bg-gradient-to-br from-background to-primary/5 rounded-2xl p-8 border border-primary/10 shadow-2xl">
                    <div className="aspect-video bg-gradient-to-br from-muted/50 to-muted/20 rounded-xl flex items-center justify-center relative overflow-hidden">
                      <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-primary/20 via-transparent to-primary/20"
                        animate={{
                          x: [-100, 100, -100],
                        }}
                        transition={{
                          duration: 3,
                          repeat: Number.POSITIVE_INFINITY,
                          ease: "easeInOut",
                        }}
                      />
                      <div className="relative z-10 text-center">
                        <Microscope className="h-16 w-16 text-primary mx-auto mb-4" />
                        <p className="text-lg font-medium text-muted-foreground">Interactive Demo Preview</p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </main>
    </div>
  )
}
