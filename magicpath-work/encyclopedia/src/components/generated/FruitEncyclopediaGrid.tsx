import { useMemo, useState } from 'react'

type Fruit = {
  id: string
  local_name: string
  english_name: string
  botanical_name: string
  description: string
  nutrition: string[]
  season: string
  emoji: string
}

const MOCK_FRUITS: Fruit[] = [
  {
    id: 'durian',
    local_name: 'Durian',
    english_name: 'Durian',
    botanical_name: 'Durio zibethinus',
    description: 'Known as the king of fruits, prized for its creamy flesh and distinctive aroma.',
    nutrition: ['Rich in vitamin C', 'Good source of potassium', 'Contains healthy fats'],
    season: 'June – August',
    emoji: '🌰',
  },
  {
    id: 'mangosteen',
    local_name: 'Manggis',
    english_name: 'Mangosteen',
    botanical_name: 'Garcinia mangostana',
    description: 'Sweet, tangy purple fruit often called the queen of fruits.',
    nutrition: ['Antioxidant xanthones', 'Vitamin C', 'Low calorie'],
    season: 'May – September',
    emoji: '🟣',
  },
  {
    id: 'rambutan',
    local_name: 'Rambutan',
    english_name: 'Rambutan',
    botanical_name: 'Nephelium lappaceum',
    description: 'Hairy red shell with juicy, translucent white flesh similar to lychee.',
    nutrition: ['Vitamin C', 'Iron', 'Fiber'],
    season: 'June – October',
    emoji: '🔴',
  },
]

export const FruitEncyclopediaGrid = () => {
  const [query, setQuery] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)

  const filtered = useMemo(() => {
    const q = query.toLowerCase().trim()
    if (!q) return MOCK_FRUITS
    return MOCK_FRUITS.filter(
      (f) =>
        f.local_name.toLowerCase().includes(q) ||
        f.english_name.toLowerCase().includes(q) ||
        f.botanical_name.toLowerCase().includes(q)
    )
  }, [query])

  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-background p-4">
      <div className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground">Fruit Encyclopedia</h1>
          <p className="mt-2 text-muted-foreground">
            {MOCK_FRUITS.length} Malaysian local fruit classes (preview)
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
                  <div className="flex aspect-[4/3] items-center justify-center bg-muted text-6xl">
                    {fruit.emoji}
                  </div>
                  <div className="p-5">
                    <h2 className="text-xl font-semibold text-foreground">{fruit.local_name}</h2>
                    <p className="text-sm text-muted-foreground">{fruit.english_name}</p>
                    <p className="mt-1 text-xs italic text-muted-foreground">
                      {fruit.botanical_name}
                    </p>
                    <button
                      type="button"
                      onClick={() => setExpanded(open ? null : fruit.id)}
                      className="mt-4 text-sm font-medium text-primary transition hover:underline focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded"
                      aria-expanded={open}
                    >
                      {open ? 'Show less' : 'Show details'}
                    </button>
                    {open && (
                      <div className="mt-4 space-y-3 border-t border-border pt-4 text-sm">
                        <p className="text-muted-foreground">{fruit.description}</p>
                        <div>
                          <p className="font-medium text-foreground">Nutrition</p>
                          <ul className="mt-1 list-inside list-disc text-muted-foreground">
                            {fruit.nutrition.map((n) => (
                              <li key={n}>{n}</li>
                            ))}
                          </ul>
                        </div>
                        <p>
                          <span className="font-medium text-foreground">Season: </span>
                          <span className="text-muted-foreground">{fruit.season}</span>
                        </p>
                      </div>
                    )}
                  </div>
                </article>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
