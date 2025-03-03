interface ExecuteCodeParams {
    code: string
    algorithm: string
    dataset: string
    parameters: Record<string, any>
  }
  
  interface AlgorithmParams {
    algorithm: string
    dataset: string
    parameters: Record<string, any>
  }
  
  class MLService {
    private baseUrl: string
    private ws: WebSocket | null = null
  
    constructor() {
      this.baseUrl = process.env.NEXT_PUBLIC_ML_API_URL || "http://localhost:8000"
    }
  
    private async fetchWithError(url: string, options: RequestInit = {}) {
      const response = await fetch(`${this.baseUrl}${url}`, options)
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || "An error occurred")
      }
      return response.json()
    }
  
    async executeCode(params: ExecuteCodeParams) {
      return this.fetchWithError("/api/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      })
    }
  
    async getDatasets() {
      return this.fetchWithError("/api/datasets")
    }
  
    async getAlgorithms() {
      return this.fetchWithError("/api/algorithms")
    }
  
    connectWebSocket(onMessage: (data: any) => void) {
      const wsUrl = this.baseUrl.replace("http", "ws")
      this.ws = new WebSocket(`${wsUrl}/ws`)
  
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        onMessage(data)
      }
  
      return () => {
        if (this.ws) {
          this.ws.close()
          this.ws = null
        }
      }
    }
  
    async runAlgorithm(params: AlgorithmParams) {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        throw new Error("WebSocket connection not established")
      }
      this.ws.send(JSON.stringify(params))
    }
  }
  
  export const mlService = new MLService()
  
  