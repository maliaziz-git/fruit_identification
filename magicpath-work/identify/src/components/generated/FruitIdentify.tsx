import { useCallback, useState } from 'react'

type View = 'upload' | 'results'

const MOCK_RESULT = {
  prediction: 'Durian',
  confidence: 0.87,
  message: 'High confidence — likely a match.',
  top3: [
    { class: 'Durian', confidence: 0.87 },
    { class: 'Jackfruit', confidence: 0.08 },
    { class: 'Cempedak', confidence: 0.03 },
  ],
  fruit_info: {
    description: 'Durian is a well-known Malaysian tropical fruit with a thorny husk and strong aroma.',
    season: 'Seasonal in Malaysia.',
    nutrition: ['Calories: 147 kcal', 'Vitamin C: 19.7 mg', 'Potassium: 436 mg'],
  },
}

const BAR_COLORS = ['bg-primary', 'bg-emerald-400', 'bg-emerald-300']

function confidenceClass(confidence: number): string {
  if (confidence >= 0.7) return 'bg-emerald-50 border-emerald-200 text-emerald-900'
  if (confidence >= 0.4) return 'bg-amber-50 border-amber-200 text-amber-900'
  return 'bg-red-50 border-red-200 text-red-900'
}

export const FruitIdentify = () => {
  const [view, setView] = useState<View>('upload')
  const [dragOver, setDragOver] = useState(false)
  const [hasImage, setHasImage] = useState(false)
  const [loading, setLoading] = useState(false)

  const simulateIdentify = useCallback(() => {
    if (!hasImage) return
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
      setView('results')
    }, 800)
  }, [hasImage])

  const reset = () => {
    setView('upload')
    setHasImage(false)
  }

  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-background p-4">
      <div className="mx-auto w-full max-w-4xl px-4 py-10 sm:px-6">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-foreground">Identify Fruit</h1>
          <p className="mt-2 text-muted-foreground">
            Upload or capture a photo, crop to one fruit, then run the model.
          </p>
        </div>

        {view === 'upload' ? (
          <div className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-8">
            <div
              className={`flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 transition ${
                dragOver ? 'border-primary bg-secondary' : 'border-border bg-muted/30'
              } ${hasImage ? 'border-solid' : ''}`}
              onClick={() => setHasImage(true)}
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragOver(false)
                setHasImage(true)
              }}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && setHasImage(true)}
            >
              {hasImage ? (
                <div className="flex h-48 w-48 items-center justify-center rounded-xl bg-secondary text-6xl">
                  🍎
                </div>
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

            <div className="mt-6 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => setHasImage(true)}
                className="rounded-xl border border-border bg-background px-5 py-2.5 text-sm font-medium transition hover:bg-muted"
              >
                Take photo
              </button>
              {hasImage && (
                <button
                  type="button"
                  className="rounded-xl border border-border bg-background px-5 py-2.5 text-sm font-medium transition hover:bg-muted"
                >
                  Adjust crop
                </button>
              )}
              <button
                type="button"
                disabled={!hasImage || loading}
                onClick={simulateIdentify}
                className="ml-auto rounded-xl bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground transition hover:opacity-90 disabled:opacity-50"
              >
                Identify fruit
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
              <div className="flex flex-col gap-6 sm:flex-row sm:items-start">
                <div className="flex h-40 w-40 shrink-0 items-center justify-center rounded-xl bg-secondary text-5xl">
                  🍎
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-muted-foreground">Prediction</p>
                  <h2 className="text-3xl font-bold text-foreground">{MOCK_RESULT.prediction}</h2>
                  <p className="mt-1 text-lg text-primary">
                    {(MOCK_RESULT.confidence * 100).toFixed(2)}% confidence
                  </p>
                  <div
                    className={`mt-4 rounded-xl border px-4 py-3 text-sm ${confidenceClass(MOCK_RESULT.confidence)}`}
                  >
                    {MOCK_RESULT.message}
                  </div>
                </div>
              </div>

              <div className="mt-8">
                <h3 className="mb-4 font-semibold text-foreground">Top 3 matches</h3>
                <div className="space-y-3">
                  {MOCK_RESULT.top3.map((item, i) => {
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

              <div className="mt-8 grid gap-4 border-t border-border pt-8 sm:grid-cols-2">
                <div>
                  <h3 className="font-semibold text-foreground">Description</h3>
                  <p className="mt-2 text-sm text-muted-foreground">
                    {MOCK_RESULT.fruit_info.description}
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">Season</h3>
                  <p className="mt-2 text-sm text-muted-foreground">
                    {MOCK_RESULT.fruit_info.season}
                  </p>
                </div>
                <div className="sm:col-span-2">
                  <h3 className="font-semibold text-foreground">Nutrition</h3>
                  <ul className="mt-2 list-inside list-disc text-sm text-muted-foreground">
                    {MOCK_RESULT.fruit_info.nutrition.map((n) => (
                      <li key={n}>{n}</li>
                    ))}
                  </ul>
                </div>
              </div>
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
                className="rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition hover:opacity-90"
              >
                Download poster
              </button>
            </div>
          </div>
        )}

        {loading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
            <div className="rounded-xl bg-card px-8 py-6 shadow-xl">
              <p className="font-medium text-foreground">Identifying fruit…</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
