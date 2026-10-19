import { useEffect, useMemo, useState } from 'react'
import type { Fruit } from '../../../lib/api'
import { fetchFruits, fruitSampleUrl } from '../../../lib/api'

export const FruitEncyclopediaGrid = () => {
  const [fruits, setFruits] = useState<Fruit[]>([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)

  useEffect(() => {
    fetchFruits()
      .then(setFruits)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    const q = query.toLowerCase().trim()
    if (!q) return fruits
    return fruits.filter(
      (f) =>
        f.local_name.toLowerCase().includes(q) ||
        f.english_name.toLowerCase().includes(q) ||
        f.botanical_name.toLowerCase().includes(q),
    )
  }, [fruits, query])

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <p className="text-muted-foreground">Loading encyclopedia…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="mx-auto max-w-lg rounded-xl border border-red-200 bg-red-50 p-6 text-center text-red-800">
        {error}
      </div>
    )
  }

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground">Fruit Encyclopedia</h1>
        <p className="mt-2 text-muted-foreground">
          {fruits.length} Malaysian local fruit classes
        </p>
        <input
          type="search"
          placeholder="Search by local, English, or botanical name…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="mt-6 w-full max-w-md rounded-xl border border-border bg-card px-4 py-3 text-foreground outline-none ring-primary transition focus:ring-2"
          aria-label="Search fruits"
        />
      </div>

      {filtered.length === 0 ? (
        <p className="text-center text-muted-foreground">No fruits match your search.</p>
      ) : (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((fruit) => {
            const open = expanded === fruit.id
            return (
              <article
                key={fruit.id}
                className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm transition hover:border-primary/30 hover:shadow-md"
              >
                <div className="aspect-[4/3] bg-muted">
                  <img
                    src={fruitSampleUrl(fruit.local_name)}
                    alt={fruit.local_name}
                    className="h-full w-full object-cover"
                    onError={(e) => {
                      const fallback = `/fruit-samples/${encodeURIComponent(fruit.local_name)}.jpg`
                      if (!e.currentTarget.src.endsWith(fallback)) {
                        e.currentTarget.src = fallback
                      } else {
                        e.currentTarget.style.display = 'none'
                      }
                    }}
                  />
                </div>
                <div className="p-5">
                  <h2 className="text-xl font-semibold text-foreground">{fruit.local_name}</h2>
                  <p className="text-sm text-muted-foreground">{fruit.english_name}</p>
                  {fruit.botanical_name && (
                    <p className="mt-1 text-xs italic text-muted-foreground">
                      {fruit.botanical_name}
                    </p>
                  )}
                  <button
                    type="button"
                    onClick={() => setExpanded(open ? null : fruit.id)}
                    className="mt-4 rounded text-sm font-medium text-primary transition hover:underline focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
                    aria-expanded={open}
                  >
                    {open ? 'Show less' : 'Show details'}
                  </button>
                  {open && (
                    <div className="mt-4 space-y-3 border-t border-border pt-4 text-sm">
                      <p className="text-muted-foreground">{fruit.description || '—'}</p>
                      {fruit.nutrition.length > 0 && (
                        <div>
                          <p className="font-medium text-foreground">Nutrition</p>
                          <ul className="mt-1 list-inside list-disc text-muted-foreground">
                            {fruit.nutrition.map((n) => (
                              <li key={n}>{n}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {fruit.season && (
                        <p>
                          <span className="font-medium text-foreground">Season: </span>
                          <span className="text-muted-foreground">{fruit.season}</span>
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </article>
            )
          })}
        </div>
      )}
    </div>
  )
}
