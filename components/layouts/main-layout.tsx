"use client"

import React, { useState } from "react"
import { SidebarInset } from "@/components/ui/sidebar"
import { ModeToggle } from "@/components/mode-toggle"
import { Button } from "@/components/ui/button"
import {
  Menu,
  Microscope,
  Home,
  BookOpen,
  Lightbulb,
  GraduationCap,
  Database,
  LineChart,
  Code,
  BarChart,
  PieChart,
  GitBranch,
  Network,
  Layers,
  CircleDot,
  Boxes,
} from "lucide-react"
import { useSidebar } from "@/components/ui/sidebar"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { getAlgorithms } from "@/lib/algorithms"
import { CommandSearch } from "@/components/search/command-search"
import { motion, AnimatePresence } from "framer-motion"

export function MainLayout({ children }: { children: React.ReactNode }) {
  const { toggleSidebar } = useSidebar()
  const pathname = usePathname()
  const [isHovered, setIsHovered] = useState(false)

  return (
    <div className="flex min-h-screen relative">
      {/* Hover trigger area */}
      <motion.div
        className="fixed left-0 top-0 w-4 h-full z-50 bg-transparent"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      />

      {/* Animated Sidebar */}
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ x: -280, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -280, opacity: 0 }}
            transition={{
              type: "spring",
              stiffness: 300,
              damping: 30,
            }}
            className="fixed left-0 top-0 h-full z-40"
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
          >
            <MainSidebar pathname={pathname} />
          </motion.div>
        )}
      </AnimatePresence>

      <SidebarInset className="w-full">
        <motion.header
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="sticky top-0 z-10 flex h-16 items-center gap-4 border-b bg-background/80 backdrop-blur-sm px-6"
        >
          <Button variant="ghost" size="icon" onClick={toggleSidebar} className="md:hidden">
            <Menu className="h-5 w-5" />
            <span className="sr-only">Toggle menu</span>
          </Button>
          <div className="ml-auto flex items-center gap-2">
            <CommandSearch />
            <ModeToggle />
          </div>
        </motion.header>
        <motion.main
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="flex-1"
        >
          {children}
        </motion.main>
      </SidebarInset>
    </div>
  )
}

function MainSidebar({ pathname }: { pathname: string }) {
  const algorithms = getAlgorithms()

  // Map icons to algorithm slugs
  const algorithmIcons: Record<string, any> = {
    eda: Database,
    "dataset-insights": BarChart,
    "evaluation-metrics": PieChart,
    "linear-regression": LineChart,
    "logistic-regression": LineChart,
    knn: Network,
    "random-forest": GitBranch,
    svm: Code,
    "ensemble-models": Layers,
    kmeans: CircleDot,
    pca: Boxes,
  }

  const itemVariants = {
    hidden: { x: -20, opacity: 0 },
    visible: {
      x: 0,
      opacity: 1,
      transition: {
        type: "spring" as const,
        stiffness: 400,
        damping: 25,
      },
    },
  }

  const groupVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.03,
        delayChildren: 0.1,
      },
    },
  }

  return (
    <div className="w-64 h-full bg-background/95 backdrop-blur-md border-r shadow-2xl">
      <motion.div initial="hidden" animate="visible" variants={groupVariants} className="h-full flex flex-col">
        {/* Header */}
        <div className="border-b p-4">
          <motion.div variants={itemVariants}>
            <Link href="/" className="flex items-center gap-3 group">
              <motion.div
                whileHover={{ rotate: 360, scale: 1.1 }}
                transition={{ duration: 0.6 }}
                className="p-2 rounded-lg bg-primary/10"
              >
                <Microscope className="h-6 w-6 text-primary" />
              </motion.div>
              <motion.span
                className="text-xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent"
                whileHover={{ scale: 1.05 }}
              >
                Virtual Lab
              </motion.span>
            </Link>
          </motion.div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Home */}
          <motion.div variants={itemVariants}>
            <Link
              href="/"
              className={`flex items-center gap-3 p-3 rounded-lg transition-all group ${
                pathname === "/" ? "bg-primary/10 text-primary border border-primary/20" : "hover:bg-muted"
              }`}
            >
              <motion.div
                whileHover={{ scale: 1.2, rotate: 5 }}
                whileTap={{ scale: 0.95 }}
                transition={{ type: "spring", stiffness: 400, damping: 17 }}
              >
                <Home className="h-5 w-5" />
              </motion.div>
              <span className="font-medium">Home</span>
            </Link>
          </motion.div>

          {/* Algorithms Section */}
          <motion.div variants={groupVariants}>
            <motion.div variants={itemVariants}>
              <h3 className="text-sm font-semibold text-muted-foreground/80 uppercase tracking-wider mb-3">
                Algorithms
              </h3>
            </motion.div>
            <div className="space-y-1">
              {algorithms.map((algorithm, index) => (
                <motion.div key={algorithm.slug} variants={itemVariants} transition={{ delay: index * 0.02 }}>
                  <Link
                    href={`/algorithms/${algorithm.slug}`}
                    className={`flex items-center gap-3 p-3 rounded-lg transition-all group ${
                      pathname === `/algorithms/${algorithm.slug}`
                        ? "bg-primary/10 text-primary border border-primary/20"
                        : "hover:bg-muted"
                    }`}
                  >
                    <motion.div
                      whileHover={{
                        scale: 1.2,
                        rotate: [0, -5, 5, 0],
                        transition: { duration: 0.3 },
                      }}
                      whileTap={{ scale: 0.9 }}
                      className="flex items-center justify-center"
                    >
                      {algorithmIcons[algorithm.slug] ? (
                        React.createElement(algorithmIcons[algorithm.slug], {
                          className: "h-5 w-5",
                        })
                      ) : (
                        <Code className="h-5 w-5" />
                      )}
                    </motion.div>
                    <motion.span className="font-medium" whileHover={{ x: 2 }}>
                      {algorithm.title}
                    </motion.span>
                  </Link>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Resources Section */}
          <motion.div variants={groupVariants}>
            <motion.div variants={itemVariants}>
              <h3 className="text-sm font-semibold text-muted-foreground/80 uppercase tracking-wider mb-3">
                Resources
              </h3>
            </motion.div>
            <div className="space-y-1">
              {[
                { href: "/resources/tutorials", icon: BookOpen, label: "Tutorials" },
                { href: "/resources/examples", icon: Lightbulb, label: "Examples" },
                { href: "/resources/courses", icon: GraduationCap, label: "Courses" },
              ].map((item, index) => (
                <motion.div key={item.href} variants={itemVariants} transition={{ delay: index * 0.03 }}>
                  <Link
                    href={item.href}
                    className="flex items-center gap-3 p-3 rounded-lg transition-all group hover:bg-muted"
                  >
                    <motion.div
                      whileHover={{
                        scale: 1.2,
                        rotate: [0, -10, 10, 0],
                        transition: { duration: 0.4 },
                      }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <item.icon className="h-5 w-5" />
                    </motion.div>
                    <motion.span className="font-medium" whileHover={{ x: 2 }}>
                      {item.label}
                    </motion.span>
                  </Link>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}
