import { type NextRequest, NextResponse } from "next/server"
import { corsHeaders } from "@/lib/cors"

export async function POST(req: NextRequest) {
  // Handle CORS preflight request
  if (req.method === "OPTIONS") {
    return new NextResponse("ok", { headers: corsHeaders })
  }

  try {
    const data = await req.json()

    // Forward the request to the Flask backend
    const response = await fetch("http://localhost:5000/api/execute", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })

    const result = await response.json()

    return NextResponse.json(result, {
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json",
      },
    })
  } catch (error) {
    console.error("Error executing code:", error)

    return NextResponse.json(
      [
        {
          output: null,
          error: `Failed to execute code: ${error instanceof Error ? error.message : String(error)}`,
          table_html: null,
          plot: null,
        },
      ],
      {
        status: 500,
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      },
    )
  }
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "10mb",
    },
  },
}

