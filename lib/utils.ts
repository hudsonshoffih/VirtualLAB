import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Generate a unique session ID for the notebook
export function generateSessionId() {
  return `session_${Math.random().toString(36).substring(2, 15)}_${Math.random().toString(36).substring(2, 15)}`
}

