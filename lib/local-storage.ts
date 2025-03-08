export function createLocalStorage<T>(key: string, initialValue: T) {
    const safelyParseJSON = (json: string | null): T | null => {
      if (!json) return null
      try {
        return JSON.parse(json) as T
      } catch (e) {
        console.error(`Error parsing JSON from localStorage key "${key}":`, e)
        return null
      }
    }
  
    const getItem = (): T | null => {
      if (typeof window === 'undefined') return initialValue 
      const item = localStorage.getItem(key)
      return safelyParseJSON(item)
    }
  

    const setItem = (value: T): void => {
      if (typeof window === 'undefined') return 
      localStorage.setItem(key, JSON.stringify(value))
    }
  
    const removeItem = (): void => {
      if (typeof window === 'undefined') return 
      localStorage.removeItem(key)
    }
  
    return { getItem, setItem, removeItem }
  }
  