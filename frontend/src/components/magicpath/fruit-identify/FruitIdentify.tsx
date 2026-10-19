import { useCallback, useRef, useState } from 'react'
import Cropper from 'react-cropper'
import type { ReactCropperElement } from 'react-cropper'
import type CropperJS from 'cropperjs'
import {
  downloadPoster,
  predictFruit,
  validateFile,
  type CropFractions,
  type PredictOptions,
  type PredictResult,
} from '../../../lib/api'

const MAX_BYTES = 8 * 1024 * 1024
const EXPORT_SIZE = 512
const DEFAULT_CROP_FRACTION = 0.85

function defaultCropFractions(nw: number, nh: number): CropFractions {
  const side = Math.min(nw, nh) * DEFAULT_CROP_FRACTION
  return {
    x: (nw - side) / 2 / nw,
    y: (nh - side) / 2 / nh,
    w: side / nw,
    h: side / nh,
  }
}

function loadImageSize(file: File): Promise<{ nw: number; nh: number }> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      resolve({ nw: img.naturalWidth, nh: img.naturalHeight })
      URL.revokeObjectURL(url)
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('Could not read image dimensions.'))
    }
    img.src = url
  })
}

function applyDefaultCrop(cropper: CropperJS, nw: number, nh: number) {
  const side = Math.min(nw, nh) * DEFAULT_CROP_FRACTION
  cropper.setData({
    x: (nw - side) / 2,
    y: (nh - side) / 2,
    width: side,
    height: side,
  })
}

function cropToFractions(cropper: CropperJS, nw: number, nh: number): CropFractions | null {
  if (nw < 1 || nh < 1) return null
  const data = cropper.getData()
  if (data.width < 1 || data.height < 1) return null
  return {
    x: data.x / nw,
    y: data.y / nh,
    w: data.width / nw,
    h: data.height / nh,
  }
}

function normalizeCroppedCanvas(sourceCanvas: HTMLCanvasElement | null): HTMLCanvasElement | null {
  if (!sourceCanvas) return null
  const out = document.createElement('canvas')
  out.width = EXPORT_SIZE
  out.height = EXPORT_SIZE
  const ctx = out.getContext('2d')
  if (!ctx) return null
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, EXPORT_SIZE, EXPORT_SIZE)
  const scale = Math.min(EXPORT_SIZE / sourceCanvas.width, EXPORT_SIZE / sourceCanvas.height)
  const w = sourceCanvas.width * scale
  const h = sourceCanvas.height * scale
  ctx.drawImage(sourceCanvas, (EXPORT_SIZE - w) / 2, (EXPORT_SIZE - h) / 2, w, h)
  return out
}

function confidenceClass(confidence: number): string {
  if (confidence >= 0.7) return 'bg-emerald-50 border-emerald-200 text-emerald-900'
  if (confidence >= 0.4) return 'bg-amber-50 border-amber-200 text-amber-900'
  return 'bg-red-50 border-red-200 text-red-900'
}

const BAR_COLORS = ['bg-primary', 'bg-emerald-400', 'bg-emerald-300']

export const FruitIdentify = () => {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const cameraInputRef = useRef<HTMLInputElement>(null)
  const cropperRef = useRef<ReactCropperElement>(null)

  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [originalFile, setOriginalFile] = useState<File | null>(null)
  const [predictOptions, setPredictOptions] = useState<PredictOptions>({})
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [cropSrc, setCropSrc] = useState<string | null>(null)
  const [showCrop, setShowCrop] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingMsg, setLoadingMsg] = useState('Identifying fruit…')
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [cropperReady, setCropperReady] = useState(false)

  const revokePreview = useCallback((url: string | null) => {
    if (url) URL.revokeObjectURL(url)
  }, [])

  const setPreview = useCallback(
    (file: File) => {
      setPreviewUrl((prev) => {
        revokePreview(prev)
        return URL.createObjectURL(file)
      })
    },
    [revokePreview],
  )

  const acceptFile = useCallback(
    (file: File) => {
      setSelectedFile(file)
      setPreview(file)
      setError(null)
    },
    [setPreview],
  )

  const closeCrop = useCallback(() => {
    setShowCrop(false)
    setCropperReady(false)
    setPendingFile(null)
    setCropSrc((prev) => {
      revokePreview(prev)
      return null
    })
  }, [revokePreview])

  const openCrop = useCallback(
    (file: File) => {
      setCropperReady(false)
      setPendingFile(file)
      setCropSrc((prev) => {
        revokePreview(prev)
        return URL.createObjectURL(file)
      })
      setShowCrop(true)
    },
    [revokePreview],
  )

  const handleFile = useCallback(
    (file: File) => {
      const err = validateFile(file)
      if (err) {
        setError(err)
        return
      }
      setPreview(file)
      openCrop(file)
    },
    [openCrop, setPreview],
  )

  const applyCrop = useCallback(() => {
    const cropper = cropperRef.current?.cropper
    if (!cropper || !pendingFile) return
    if (!cropperReady) {
      setError('Image is still loading. Wait a moment, then try again.')
      return
    }

    const imageData = cropper.getImageData()
    const fractions = cropToFractions(
      cropper,
      imageData.naturalWidth,
      imageData.naturalHeight,
    )
    if (!fractions) {
      setError('Invalid crop region. Adjust the crop box and try again.')
      return
    }
    setOriginalFile(pendingFile)
    setPredictOptions({ crop: fractions })

    const cropped = cropper.getCroppedCanvas({
      maxWidth: 1024,
      maxHeight: 1024,
      imageSmoothingEnabled: true,
      imageSmoothingQuality: 'high',
      fillColor: '#ffffff',
    })
    const canvas = normalizeCroppedCanvas(cropped)
    if (!canvas) {
      setError('Could not crop this image. Try a different area.')
      return
    }

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError('Crop failed. Please try again.')
          return
        }
        if (blob.size > MAX_BYTES) {
          setError('Cropped image exceeds 8 MB. Crop a smaller region.')
          return
        }
        const base = pendingFile.name.replace(/\.[^.]+$/i, '') || 'fruit'
        const croppedFile = new File([blob], `${base}_cropped.jpg`, { type: 'image/jpeg' })
        setSelectedFile(pendingFile)
        setPreview(croppedFile)
        setError(null)
        closeCrop()
      },
      'image/jpeg',
      0.92,
    )
  }, [closeCrop, cropperReady, pendingFile, setPreview])

  const useFullImage = useCallback(async () => {
    if (!pendingFile) return
    try {
      let nw: number
      let nh: number
      const cropper = cropperRef.current?.cropper
      if (cropperReady && cropper) {
        const imageData = cropper.getImageData()
        nw = imageData.naturalWidth
        nh = imageData.naturalHeight
      } else {
        const size = await loadImageSize(pendingFile)
        nw = size.nw
        nh = size.nh
      }
      if (nw < 1 || nh < 1) {
        setError('Could not read image dimensions. Try again.')
        return
      }
      const fractions = defaultCropFractions(nw, nh)
      setOriginalFile(pendingFile)
      setPredictOptions({ crop: fractions })
      acceptFile(pendingFile)
      closeCrop()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not use default crop.')
    }
  }, [acceptFile, closeCrop, cropperReady, pendingFile])

  const runPredict = async () => {
    const uploadFile = originalFile || selectedFile
    if (!uploadFile) {
      setError('Please choose and crop an image first.')
      return
    }
    if (!predictOptions.crop && !predictOptions.useFull) {
      setError('Please apply or confirm a crop first.')
      return
    }
    setLoading(true)
    setLoadingMsg('Identifying fruit…')
    setError(null)
    try {
      const data = await predictFruit(uploadFile, predictOptions)
      setResult(data)
      setShowResults(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const handlePoster = async () => {
    const uploadFile = originalFile || selectedFile
    if (!uploadFile || !result) return
    setLoading(true)
    setLoadingMsg('Creating poster…')
    setError(null)
    try {
      await downloadPoster(uploadFile, result, predictOptions)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Poster failed')
    } finally {
      setLoading(false)
      setLoadingMsg('Identifying fruit…')
    }
  }

  const reset = () => {
    setSelectedFile(null)
    setOriginalFile(null)
    setPredictOptions({})
    setResult(null)
    setShowResults(false)
    setPreviewUrl((prev) => {
      revokePreview(prev)
      return null
    })
    setError(null)
  }

  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-10 sm:px-6">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-foreground">Identify Fruit</h1>
        <p className="mt-2 text-muted-foreground">
          Upload or capture a photo, crop to one fruit, then run the model.
        </p>
      </div>

      {error && (
        <div
          className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
          role="alert"
        >
          {error}
        </div>
      )}

      {!showResults ? (
        <div className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-8">
          <div
            className={`flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 transition ${
              dragOver ? 'border-primary bg-secondary' : 'border-border bg-muted/30'
            } ${previewUrl ? 'border-solid' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragOver(false)
              const file = e.dataTransfer.files[0]
              if (file) handleFile(file)
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected fruit"
                className="max-h-64 w-full max-w-sm rounded-lg object-contain"
              />
            ) : (
              <>
                <span className="flex h-14 w-14 items-center justify-center rounded-full bg-primary text-2xl text-primary-foreground">
                  +
                </span>
                <p className="mt-4 font-medium text-foreground">Click or drag an image</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  JPG, PNG, or WEBP · max 8 MB
                </p>
              </>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
              e.target.value = ''
            }}
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
              e.target.value = ''
            }}
          />

          <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
            <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:flex-wrap">
              <button
                type="button"
                onClick={() => cameraInputRef.current?.click()}
                className="w-full rounded-xl border border-border bg-background px-5 py-2.5 text-sm font-medium transition hover:bg-muted sm:w-auto"
              >
                Take photo
              </button>
              {selectedFile && (
                <button
                  type="button"
                  onClick={() => openCrop(originalFile || selectedFile!)}
                  className="w-full rounded-xl border border-border bg-background px-5 py-2.5 text-sm font-medium transition hover:bg-muted sm:w-auto"
                >
                  Adjust crop
                </button>
              )}
            </div>
            <button
              type="button"
              disabled={!(originalFile || selectedFile) || loading}
              onClick={runPredict}
              className="w-full rounded-xl bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground transition hover:opacity-90 disabled:opacity-50 sm:ml-auto sm:w-auto"
            >
              Identify fruit
            </button>
          </div>
        </div>
      ) : (
        result && (
          <div className="space-y-6">
            <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
              <div className="flex flex-col gap-6 sm:flex-row sm:items-start">
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Analyzed fruit"
                    className="h-40 w-40 shrink-0 rounded-xl object-cover"
                  />
                )}
                <div className="flex-1">
                  <p className="text-sm font-medium text-muted-foreground">Prediction</p>
                  <h2 className="text-3xl font-bold text-foreground">{result.prediction}</h2>
                  <p className="mt-1 text-lg text-primary">
                    {(result.confidence * 100).toFixed(2)}% confidence
                  </p>
                  <div
                    className={`mt-4 rounded-xl border px-4 py-3 text-sm ${confidenceClass(result.confidence)}`}
                  >
                    {result.message}
                  </div>
                </div>
              </div>

              <div className="mt-8">
                <h3 className="mb-4 font-semibold text-foreground">Top 3 matches</h3>
                <div className="space-y-3">
                  {result.top3.map((item, i) => {
                    const pct = (item.confidence * 100).toFixed(2)
                    return (
                      <div key={item.class} className="flex items-center gap-3 text-sm">
                        <span className="w-28 shrink-0 truncate font-medium">{item.class}</span>
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                          <div
                            className={`h-full rounded-full ${BAR_COLORS[i]}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="w-14 text-right text-muted-foreground">{pct}%</span>
                      </div>
                    )
                  })}
                </div>
              </div>

              {result.fruit_info && (
                <div className="mt-8 grid gap-4 border-t border-border pt-8 sm:grid-cols-2">
                  <div>
                    <h3 className="font-semibold text-foreground">Description</h3>
                    <p className="mt-2 text-sm text-muted-foreground">
                      {result.fruit_info.description || '—'}
                    </p>
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground">Season</h3>
                    <p className="mt-2 text-sm text-muted-foreground">
                      {result.fruit_info.season || '—'}
                    </p>
                  </div>
                  {(result.fruit_info.nutrition?.length ?? 0) > 0 && (
                    <div className="sm:col-span-2">
                      <h3 className="font-semibold text-foreground">Nutrition</h3>
                      <ul className="mt-2 list-inside list-disc text-sm text-muted-foreground">
                        {result.fruit_info.nutrition!.map((n) => (
                          <li key={n}>{n}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={reset}
                className="rounded-xl border border-border bg-card px-5 py-2.5 text-sm font-medium transition hover:bg-muted"
              >
                Upload another
              </button>
              <button
                type="button"
                onClick={handlePoster}
                disabled={loading}
                className="rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition hover:opacity-90 disabled:opacity-50"
              >
                Download poster
              </button>
            </div>
          </div>
        )
      )}

      {showCrop && cropSrc && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          aria-label="Crop image"
          onKeyDown={(e) => e.key === 'Escape' && closeCrop()}
        >
          <div className="w-full max-w-2xl rounded-2xl bg-card p-4 shadow-xl">
            <p className="mb-3 truncate text-sm text-muted-foreground">
              {pendingFile?.name}
            </p>
            <div className="mx-auto mb-3 aspect-square w-full max-w-[520px] overflow-hidden rounded-xl bg-muted/30">
              <Cropper
                ref={cropperRef}
                src={cropSrc}
                style={{ height: '100%', width: '100%' }}
                viewMode={1}
                dragMode="crop"
                aspectRatio={NaN}
                initialAspectRatio={1}
                autoCropArea={0.85}
                autoCrop={false}
                responsive
                background={false}
                guides
                center
                highlight
                cropBoxMovable
                cropBoxResizable
                ready={() => {
                  const cropper = cropperRef.current?.cropper
                  if (!cropper) return
                  const { naturalWidth, naturalHeight } = cropper.getImageData()
                  applyDefaultCrop(cropper, naturalWidth, naturalHeight)
                  setCropperReady(true)
                }}
              />
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                disabled={!cropperReady}
                onClick={applyCrop}
                className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground disabled:opacity-50"
              >
                Apply crop
              </button>
              <button
                type="button"
                disabled={!cropperReady}
                onClick={useFullImage}
                className="rounded-lg border border-border px-4 py-2 text-sm font-medium disabled:opacity-50"
              >
                Use full image
              </button>
              <button
                type="button"
                onClick={closeCrop}
                className="rounded-lg px-4 py-2 text-sm text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40">
          <div className="rounded-xl bg-card px-8 py-6 shadow-xl">
            <p className="font-medium text-foreground">{loadingMsg}</p>
          </div>
        </div>
      )}
    </div>
  )
}
