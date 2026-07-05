import { useCallback, useRef, useState } from 'react'
import Cropper from 'react-cropper'
import type { ReactCropperElement } from 'react-cropper'
import {
  downloadPoster,
  predictFruit,
  validateFile,
  type PredictResult,
} from '../../../lib/api'

const MAX_BYTES = 8 * 1024 * 1024

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
    setPendingFile(null)
    setCropSrc((prev) => {
      revokePreview(prev)
      return null
    })
  }, [revokePreview])

  const openCrop = useCallback(
    (file: File) => {
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

    const canvas = cropper.getCroppedCanvas({
      maxWidth: 1024,
      maxHeight: 1024,
      imageSmoothingEnabled: true,
      imageSmoothingQuality: 'high',
      fillColor: '#ffffff',
    })
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
        const cropped = new File([blob], `${base}_cropped.jpg`, { type: 'image/jpeg' })
        acceptFile(cropped)
        closeCrop()
      },
      'image/jpeg',
      0.92,
    )
  }, [acceptFile, closeCrop, pendingFile])

  const useFullImage = useCallback(() => {
    if (!pendingFile) return
    acceptFile(pendingFile)
    closeCrop()
  }, [acceptFile, closeCrop, pendingFile])

  const runPredict = async () => {
    if (!selectedFile) {
      setError('Please choose and crop an image first.')
      return
    }
    setLoading(true)
    setLoadingMsg('Identifying fruit…')
    setError(null)
    try {
      const data = await predictFruit(selectedFile)
      setResult(data)
      setShowResults(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const handlePoster = async () => {
    if (!selectedFile || !result) return
    setLoading(true)
    setLoadingMsg('Creating poster…')
    setError(null)
    try {
      await downloadPoster(selectedFile, result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Poster failed')
    } finally {
      setLoading(false)
      setLoadingMsg('Identifying fruit…')
    }
  }

  const reset = () => {
    setSelectedFile(null)
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

          <div className="mt-6 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => cameraInputRef.current?.click()}
              className="rounded-xl border border-border bg-background px-5 py-2.5 text-sm font-medium transition hover:bg-muted"
            >
              Take photo
            </button>
            {selectedFile && (
              <button
                type="button"
                onClick={() => openCrop(selectedFile)}
                className="rounded-xl border border-border bg-background px-5 py-2.5 text-sm font-medium transition hover:bg-muted"
              >
                Adjust crop
              </button>
            )}
            <button
              type="button"
              disabled={!selectedFile || loading}
              onClick={runPredict}
              className="ml-auto rounded-xl bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground transition hover:opacity-90 disabled:opacity-50"
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
            <Cropper
              ref={cropperRef}
              src={cropSrc}
              style={{ height: 360, width: '100%' }}
              viewMode={1}
              dragMode="crop"
              aspectRatio={NaN}
              autoCropArea={0.85}
              responsive
              background={false}
              guides
              center
              highlight
              cropBoxMovable
              cropBoxResizable
            />
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={applyCrop}
                className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground"
              >
                Apply crop
              </button>
              <button
                type="button"
                onClick={useFullImage}
                className="rounded-lg border border-border px-4 py-2 text-sm font-medium"
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
