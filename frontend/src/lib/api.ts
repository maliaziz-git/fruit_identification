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

import localFruitInfo from '../data/fruit_info.json'

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

function getLocalFruits(): Fruit[] {
  return Object.keys(localFruitInfo)
    .sort()
    .map((local_name) => {
      const entry = (localFruitInfo as Record<string, any>)[local_name] || {}
      return {
        id: local_name,
        local_name,
        english_name: entry.english_name || local_name,
        botanical_name: entry.botanical_name || '',
        description: entry.description || '',
        nutrition: Array.isArray(entry.nutrition) ? entry.nutrition : [],
        season: entry.season || '',
      }
    })
}

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
  try {
    const endpoint = API_BASE ? `${API_BASE}/api/fruits` : '/api/fruits'
    const res = await fetch(endpoint)
    if (res.ok) {
      const data = await res.json()
      if (Array.isArray(data) && data.length > 0) {
        return data
      }
    }
  } catch (err) {
    console.warn('Backend unavailable, using bundled encyclopedia metadata:', err)
  }
  return getLocalFruits()
}

export function fruitSampleUrl(localName: string): string {
  if (API_BASE) {
    return `${API_BASE}/fruit-sample/${encodeURIComponent(localName)}`
  }
  return `/fruit-samples/${encodeURIComponent(localName)}.jpg`
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
  const endpoint = API_BASE ? `${API_BASE}/predict` : '/predict'
  let res: Response
  try {
    res = await fetch(endpoint, { method: 'POST', body: formData })
  } catch {
    throw new Error(
      API_BASE
        ? `Could not connect to backend at ${API_BASE}. If using Render free tier, the instance may be waking up from sleep. Please wait ~30 seconds and try again.`
        : 'Backend API is not connected. If you deployed on Vercel, set the VITE_API_URL environment variable in your Vercel Project Settings to your deployed backend URL.'
    )
  }
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
  const endpoint = API_BASE ? `${API_BASE}/generate-poster` : '/generate-poster'
  let res: Response
  try {
    res = await fetch(endpoint, { method: 'POST', body: formData })
  } catch {
    throw new Error(
      API_BASE
        ? `Could not reach backend (${API_BASE}) to generate poster. Please retry in a few seconds.`
        : 'Backend API is not configured. Please set VITE_API_URL in your Vercel Project Settings.'
    )
  }
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
