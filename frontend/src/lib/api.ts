export interface Fruit {
  id: string
  local_name: string
  english_name: string
  botanical_name: string
  description: string
  nutrition: string[]
  season: string
}

export interface TopPrediction {
  class: string
  confidence: number
}

export interface FruitInfo {
  description?: string
  nutrition?: string[]
  season?: string
  price?: string
  uses?: string
}

export interface PredictResult {
  prediction: string
  confidence: number
  top3: TopPrediction[]
  message: string
  fruit_info?: FruitInfo
  input_width?: number
  input_height?: number
  input_hash?: string
}

export interface CropFractions {
  x: number
  y: number
  w: number
  h: number
}

export interface PredictOptions {
  crop?: CropFractions
  useFull?: boolean
}

const MAX_BYTES = 8 * 1024 * 1024
const ALLOWED_EXT = ['jpg', 'jpeg', 'png', 'webp']
const ALLOWED_MIME = ['image/jpeg', 'image/png', 'image/webp']

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

export function validateFile(file: File): string | null {
  const ext = file.name.includes('.')
    ? file.name.split('.').pop()!.toLowerCase()
    : ''
  const extOk = ALLOWED_EXT.includes(ext)
  const mimeOk = ALLOWED_MIME.includes(file.type)
  if (!extOk && !mimeOk) return 'Please upload JPG, PNG, or WEBP.'
  if (file.size > MAX_BYTES) return 'File must be 8 MB or smaller.'
  return null
}

export async function fetchFruits(): Promise<Fruit[]> {
  const res = await fetch(`${API_BASE}/api/fruits`)
  if (!res.ok) throw new Error('Failed to load fruit encyclopedia.')
  return res.json()
}

export function fruitSampleUrl(localName: string): string {
  return `${API_BASE}/fruit-sample/${encodeURIComponent(localName)}`
}

export async function predictFruit(
  file: File,
  options?: PredictOptions,
): Promise<PredictResult> {
  const formData = new FormData()
  formData.append('image', file, file.name)
  if (options?.useFull) {
    formData.append('use_full', '1')
  } else if (options?.crop) {
    formData.append('crop', JSON.stringify(options.crop))
  }
  const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Prediction failed')
  return data
}

export async function downloadPoster(
  file: File,
  prediction: PredictResult,
  options?: PredictOptions,
): Promise<void> {
  const formData = new FormData()
  formData.append('image', file, file.name)
  if (options?.useFull) {
    formData.append('use_full', '1')
  } else if (options?.crop) {
    formData.append('crop', JSON.stringify(options.crop))
  }
  formData.append(
    'metadata',
    JSON.stringify({
      prediction: prediction.prediction,
      confidence: prediction.confidence,
      fruit_info: prediction.fruit_info || {},
    }),
  )
  const res = await fetch(`${API_BASE}/generate-poster`, { method: 'POST', body: formData })
  if (!res.ok) {
    let message = 'Poster generation failed.'
    try {
      const data = await res.json()
      message = data.error || message
    } catch {
      /* not JSON */
    }
    throw new Error(message)
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const safeName = (prediction.prediction || 'fruit')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
  const a = document.createElement('a')
  a.href = url
  a.download = `${safeName || 'fruit'}-poster.png`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}
