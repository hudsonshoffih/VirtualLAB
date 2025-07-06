import { type NextRequest, NextResponse } from "next/server";

const PYTHON_BACKEND_URL = "https://virtualab-backend.onrender.com/api/execute";

export async function POST(request: NextRequest) {
  try {
    const { code } = await request.json();

    const flaskRequest = {
      cells: [{ code }],
    };

    const response = await fetch(PYTHON_BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(flaskRequest),
    });

    if (!response.ok) {
      throw new Error(`Flask backend returned ${response.status}`);
    }

    const results = await response.json();
    const result = results[0];

    let visualization = null;

    if (result.table_html) {
      visualization = {
        type: "html_table",
        data: result.table_html,
      };
    }

    if (result.plot) {
      visualization = {
        type: "plot",
        data: `data:image/png;base64,${result.plot}`,
      };
    }

    return NextResponse.json({
      success: true,
      output: result.output || "",
      error: result.error,
      visualization,
    });
  } catch (error) {
    console.error("Python execution error:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Unknown error during code execution",
        output: "",
        visualization: null,
      },
      { status: 500 }
    );
  }
}
