import { type NextRequest, NextResponse } from "next/server";

// URL to your Flask backend
const PYTHON_BACKEND_URL = "http://localhost:5000/api/execute";

export async function POST(request: NextRequest) {
  try {
    const { code } = await request.json();

    // Prepare the request to the Flask backend
    const flaskRequest = {
      cells: [{ code }],
    };

    // Send the request to the Flask backend
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
    const result = results[0]; // Get the first cell result

    // Process the result
    let visualization = null;

    // Check if there's a table to display
    if (result.table_html) {
      visualization = {
        type: "html_table",
        data: result.table_html,
      };
    }

    // Check if there's a plot to display
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
