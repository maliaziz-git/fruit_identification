import { Link } from 'react-router-dom'

const features = [
  {
    title: 'Smart cropping',
    description: 'Focus on one fruit before inference for higher accuracy on cluttered photos.',
    icon: '✂️',
  },
  {
    title: 'Top-3 predictions',
    description: 'See confidence scores for the best matches with clear status messaging.',
    icon: '📊',
  },
  {
    title: 'Fruit encyclopedia',
    description: 'Browse all 28 classes with descriptions, nutrition facts, and seasons.',
    icon: '📖',
  },
  {
    title: 'Shareable posters',
    description: 'Download a branded PNG poster with your result and fruit details.',
    icon: '🖼️',
  },
]

export const SaaSLanding = () => {
  return (
    <div className="w-full">
      <section className="relative overflow-hidden px-4 py-16 sm:px-6 sm:py-24">
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-secondary via-background to-accent/40" />
        <div className="relative mx-auto max-w-4xl text-center">
          <p className="mb-4 inline-flex rounded-full border border-border bg-card px-4 py-1 text-sm font-medium text-secondary-foreground">
            MobileNetV3 · 28 Malaysian fruit classes
          </p>
          <h1 className="text-4xl font-bold tracking-tight text-foreground sm:text-5xl lg:text-6xl">
            Identify local fruits{' '}
            <span className="text-primary">in seconds</span>
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground">
            Upload a photo, crop to focus on one fruit, and get instant CNN-powered
            predictions with nutrition info and downloadable posters.
          </p>
          <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Link
              to="/identify"
              className="inline-flex h-12 min-w-[180px] items-center justify-center rounded-xl bg-primary px-8 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/25 transition hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
            >
              Start identifying
            </Link>
            <Link
              to="/encyclopedia"
              className="inline-flex h-12 min-w-[180px] items-center justify-center rounded-xl border border-border bg-card px-8 text-base font-semibold text-foreground transition hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
            >
              Browse encyclopedia
            </Link>
          </div>
        </div>
      </section>

      <section className="border-t border-border bg-card px-4 py-16 sm:px-6">
        <div className="mx-auto max-w-6xl">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-bold text-foreground">Everything you need</h2>
            <p className="mt-3 text-muted-foreground">
              A complete SaaS-style workflow for fruit identification and learning.
            </p>
          </div>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {features.map((f) => (
              <div
                key={f.title}
                className="rounded-2xl border border-border bg-background p-6 transition hover:border-primary/30 hover:shadow-md"
              >
                <span className="text-3xl" aria-hidden>
                  {f.icon}
                </span>
                <h3 className="mt-4 text-lg font-semibold text-foreground">{f.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {f.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
