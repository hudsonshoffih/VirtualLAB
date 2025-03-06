"use client"

import { useEffect, useRef } from "react"
import Editor from "@monaco-editor/react"

interface CodeEditorProps {
  value: string
  onChange: (value: string) => void
  language?: string
  height?: string
  minHeight?: string
  onEditorDidMount?: (editor: any) => void
}

export function CodeEditor({
  value,
  onChange,
  language = "python",
  height = "300px",
  minHeight = "100px",
  onEditorDidMount,
}: CodeEditorProps) {
  const editorRef = useRef<any>(null)

  const handleEditorDidMount = (editor: any) => {
    editorRef.current = editor

    // Adjust editor height based on content
    if (height === "auto") {
      const updateHeight = () => {
        const contentHeight = Math.max(editor.getContentHeight(), Number.parseInt(minHeight))
        editor.layout({ width: editor.getLayoutInfo().width, height: contentHeight })
        editor.getDomNode()?.style.setProperty("height", `${contentHeight}px`)
      }

      // Update height initially and when content changes
      updateHeight()
      editor.onDidContentSizeChange(updateHeight)
    }

    // Call the passed onEditorDidMount if provided
    if (onEditorDidMount) {
      onEditorDidMount(editor)
    }
  }

  useEffect(() => {
    // This ensures the editor resizes properly when its container changes size
    const resizeObserver = new ResizeObserver(() => {
      if (editorRef.current) {
        editorRef.current.layout()
      }
    })

    const editorContainer = document.querySelector(".monaco-editor")
    if (editorContainer) {
      resizeObserver.observe(editorContainer)
    }

    return () => {
      if (editorContainer) {
        resizeObserver.unobserve(editorContainer)
      }
    }
  }, [])

  return (
    <div className={height === "auto" ? "min-h-[100px]" : ""}>
      <Editor
        height={height}
        language={language}
        value={value}
        onChange={(value) => onChange(value || "")}
        onMount={handleEditorDidMount}
        options={{
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          fontSize: 14,
          lineNumbers: "on",
          automaticLayout: true,
          tabSize: 4,
          insertSpaces: true,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
          scrollbar: {
            vertical: "auto",
            horizontal: "auto",
          },
          wordWrap: "on",
          suggest: {
            showKeywords: true,
            showSnippets: true,
            showClasses: true,
            showFunctions: true,
          },
          quickSuggestions: true,
        }}
        theme="vs-dark"
      />
    </div>
  )
}

