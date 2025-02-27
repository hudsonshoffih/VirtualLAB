import { MainLayout } from "@/components/layouts/main-layout"
import { AlgorithmTabs } from "@/components/algorithms/algorithm-tabs"
import { getAlgorithmBySlug } from "@/lib/algorithms"
import { notFound } from "next/navigation"

export default function AlgorithmPage({ params }: { params: { slug: string } }) {
  const algorithm = getAlgorithmBySlug(params.slug)

  if (!algorithm) {
    notFound()
  }

  return (
    <MainLayout>
      <div className="container py-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">{algorithm.title}</h1>
          <p className="text-muted-foreground">{algorithm.description}</p>
        </div>

        <AlgorithmTabs algorithm={algorithm} />
      </div>
    </MainLayout>
  )
}

