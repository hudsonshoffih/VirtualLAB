"use client"

import { motion } from "framer-motion"
import { Github, Linkedin, Twitter, Globe, MapPin, Calendar, Star } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

const contributors = [
  {
    id: 1,
    name: "Hudson Shoffi H",
    role: "Contributor",
    avatar: "/images/hudson.jpg?height=120&width=120",
    bio: "Full-stack developer with 8+ years of experience in machine learning and web development. Passionate about creating educational tools.",
    location: "Salem, Tamil Nadu",
    skills: ["React", "Python", "TypeScript", "Machine Learning", "Node.js"],
    gradient: "from-blue-500 to-purple-600",
    social: {
      github: "hudsonshoffih",
      linkedin: "hudson-h-3b6933291",
      twitter: "alexchen_dev",
      website: "alexchen.dev",
    },
  },
  {
    id: 2,
    name: "S Ganesh Vasanth",
    role: "Contributor",
    avatar: "/images/ganesh.jpg?height=120&width=120",
    bio: "Creative designer focused on user experience and accessibility. Believes in making complex concepts simple and beautiful.",
    location: "Hosur, Tamil Nadu",
    skills: ["Figma", "Design Systems", "Prototyping", "User Research", "Accessibility"],
    gradient: "from-pink-500 to-rose-600",
    social: {
      github: "Bombe-19",
      linkedin: "s-ganesh-vasanth-71a665374",
      twitter: "sarahj_design",
      website: "sarahjohnson.design",
    },
  },
  {
    id: 3,
    name: "Srivishal S",
    role: "Contributor",
    avatar: "/placeholder.svg?height=120&width=120",
    bio: "PhD in Computer Science with expertise in machine learning algorithms. Contributes to the educational content and algorithm implementations.",
    location: "Chennai, Tamil Nadu",
    skills: ["Python", "R", "Statistics", "Deep Learning", "Data Visualization"],
    gradient: "from-green-500 to-teal-600",
    social: {
      github: "root-daemon",
      linkedin: "michael-rodriguez-phd",
      twitter: "dr_rodriguez_ml",
      website: "michaelrodriguez.ai",
    },
  },
  {
    id: 4,
    name: "Emma Thompson",
    role: "Frontend Engineer",
    avatar: "/placeholder.svg?height=120&width=120",
    bio: "Frontend specialist with a passion for creating interactive and responsive user interfaces. Loves working with modern web technologies.",
    location: "London, UK",
    joinDate: "Apr 2023",
    contributions: 134,
    skills: ["React", "Next.js", "Tailwind CSS", "Framer Motion", "JavaScript"],
    gradient: "from-orange-500 to-red-600",
    social: {
      github: "emma-thompson",
      linkedin: "emma-thompson-fe",
      twitter: "emma_codes",
      website: "emmathompson.dev",
    },
  },
  {
    id: 5,
    name: "David Kim",
    role: "Backend Engineer",
    avatar: "/placeholder.svg?height=120&width=120",
    bio: "Backend engineer specializing in scalable systems and API development. Ensures the platform runs smoothly and efficiently.",
    location: "Seattle, WA",
    joinDate: "May 2023",
    contributions: 198,
    skills: ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
    gradient: "from-indigo-500 to-blue-600",
    social: {
      github: "david-kim-be",
      linkedin: "david-kim-backend",
      twitter: "david_codes",
      website: "davidkim.tech",
    },
  },
  {
    id: 6,
    name: "Lisa Wang",
    role: "DevOps Engineer",
    avatar: "/placeholder.svg?height=120&width=120",
    bio: "DevOps engineer focused on automation, deployment, and infrastructure. Keeps the development workflow smooth and efficient.",
    location: "Austin, TX",
    joinDate: "Jun 2023",
    contributions: 112,
    skills: ["Docker", "Kubernetes", "CI/CD", "AWS", "Terraform"],
    gradient: "from-purple-500 to-pink-600",
    social: {
      github: "lisa-wang-devops",
      linkedin: "lisa-wang-devops",
      twitter: "lisa_devops",
      website: "lisawang.io",
    },
  },
]

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
}

const cardVariants = {
  hidden: {
    opacity: 0,
    y: 50,
    scale: 0.9,
  },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      // Remove 'type' or use type: "spring" as AnimationGeneratorType if you import AnimationGeneratorType
      stiffness: 100,
      damping: 15,
      duration: 0.6,
    },
  },
}

const skillVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      stiffness: 200,
      damping: 20,
    },
  },
}

const skillsContainerVariants = {
  visible: {
    transition: {
      staggerChildren: 0.05,
    },
  },
}

// Helper function to generate social URLs
const getSocialUrl = (platform: string, username: string) => {
  const urls = {
    github: `https://github.com/${username}`,
    linkedin: `https://linkedin.com/in/${username}`,
    twitter: `https://twitter.com/${username}`,
    website: `https://${username}`,
  }
  return urls[platform as keyof typeof urls] || "#"
}

export default function ContributorsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto px-4 py-12">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <motion.h1
            className="text-5xl font-bold mb-6 bg-gradient-to-r from-primary via-purple-500 to-pink-500 bg-clip-text text-transparent"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            Our Contributors
          </motion.h1>
          <motion.p
            className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.6 }}
          >
            Meet the talented individuals who make Virtual Lab possible. Our diverse team of developers, designers, and
            researchers work together to create an exceptional learning experience.
          </motion.p>
        </motion.div>

        {/* Contributors Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16"
        >
          {contributors.map((contributor) => (
            <motion.div
              key={contributor.id}
              variants={cardVariants}
              whileHover={{
                y: -8,
                transition: { type: "spring", stiffness: 300, damping: 20 },
              }}
              className="group"
            >
              <Card className="h-full overflow-hidden border-0 shadow-lg hover:shadow-2xl transition-all duration-300 bg-gradient-to-br from-background to-muted/10 backdrop-blur-sm">
                <CardContent className="p-0">
                  {/* Gradient Header */}
                  <div className={`h-24 bg-gradient-to-r ${contributor.gradient} relative overflow-hidden`}>
                    <div className="absolute inset-0 bg-black/10" />
                    <motion.div className="absolute top-4 right-4 text-white/80" whileHover={{ scale: 1.1, rotate: 5 }}>
                      <Star className="h-5 w-5" />
                    </motion.div>
                  </div>

                  {/* Avatar */}
                  <div className="relative -mt-12 flex justify-center">
                    <motion.div
                      whileHover={{ scale: 1.05, rotate: 2 }}
                      transition={{ type: "spring", stiffness: 300, damping: 20 }}
                      className="relative"
                    >
                      <div className="w-24 h-24 rounded-full border-4 border-background shadow-xl overflow-hidden bg-gradient-to-br from-muted to-background">
                        <img
                          src={contributor.avatar || "/placeholder.svg"}
                          alt={contributor.name}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div
                        className={`absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-gradient-to-r ${contributor.gradient} border-2 border-background`}
                      />
                    </motion.div>
                  </div>

                  {/* Content */}
                  <div className="p-6 pt-4">
                    <motion.h3 className="text-xl font-bold text-center mb-1" whileHover={{ scale: 1.02 }}>
                      {contributor.name}
                    </motion.h3>
                    <p className="text-primary font-medium text-center mb-3">{contributor.role}</p>

                    {/* Stats */}
                    <div className="flex justify-center items-center gap-4 mb-4 text-sm text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <MapPin className="h-3 w-3" />
                        <span>{contributor.location}</span>
                      </div>
                    </div>

                    <p className="text-muted-foreground text-sm text-center mb-4 leading-relaxed">{contributor.bio}</p>

                    {/* Skills */}
                    <motion.div
                      className="mb-6"
                      variants={skillsContainerVariants}
                      initial="hidden"
                      whileInView="visible"
                      viewport={{ once: true }}
                    >
                      <div className="flex flex-wrap gap-2 justify-center">
                        {contributor.skills.map((skill, index) => (
                          <motion.div key={skill} variants={skillVariants}>
                            <Badge
                              variant="secondary"
                              className="text-xs hover:scale-105 transition-transform cursor-default bg-muted/50 hover:bg-muted"
                            >
                              {skill}
                            </Badge>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>

                    {/* Social Links */}
                    <div className="flex justify-center gap-3">
                      {[
                        { icon: Github, key: "github", color: "hover:text-gray-900 dark:hover:text-gray-100" },
                        { icon: Linkedin, key: "linkedin", color: "hover:text-blue-600" },
                        { icon: Twitter, key: "twitter", color: "hover:text-blue-400" },
                        { icon: Globe, key: "website", color: "hover:text-green-600" },
                      ].map(({ icon: Icon, key, color }) => {
                        const socialUsername = contributor.social[key as keyof typeof contributor.social]
                        if (!socialUsername) return null

                        return (
                          <motion.a
                            key={key}
                            href={getSocialUrl(key, socialUsername)}
                            target="_blank"
                            rel="noopener noreferrer"
                            whileHover={{ scale: 1.2, y: -2 }}
                            whileTap={{ scale: 0.95 }}
                            className={`p-2 rounded-full bg-muted/50 text-muted-foreground transition-all duration-200 ${color} hover:bg-muted hover:shadow-md`}
                          >
                            <Icon className="h-4 w-4" />
                          </motion.a>
                        )
                      })}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="text-center bg-gradient-to-r from-primary/5 via-purple-500/5 to-pink-500/5 rounded-2xl p-12 border border-primary/10"
        >
          <motion.h2
            className="text-3xl font-bold mb-4 bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent"
            whileHover={{ scale: 1.02 }}
          >
            Want to Contribute?
          </motion.h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto text-lg">
            Join our amazing community of contributors! Whether you're a developer, designer, or educator, there's a
            place for you in our team.
          </p>
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              size="lg"
              className="bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90 text-white shadow-lg hover:shadow-xl transition-all duration-300"
              asChild
            >
              <a href="https://github.com/hudsonshoffih/VirtualLAB.git" target="_blank" rel="noopener noreferrer">
                <Github className="mr-2 h-5 w-5" />
                Get Started on GitHub
              </a>
            </Button>
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}
