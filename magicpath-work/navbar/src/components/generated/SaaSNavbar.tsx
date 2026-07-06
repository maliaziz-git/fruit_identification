import { useState } from 'react';
const links = [{
  id: 'home',
  label: 'Home',
  href: '/'
}, {
  id: 'identify',
  label: 'Identify',
  href: '/identify'
}, {
  id: 'encyclopedia',
  label: 'Encyclopedia',
  href: '/encyclopedia'
}] as const;
export const SaaSNavbar = () => {
  const [active, setActive] = useState<string>('home');
  const [menuOpen, setMenuOpen] = useState(false);
  return <div className="flex min-h-screen w-full items-start justify-center bg-background p-4">
      <header className="sticky top-0 z-50 w-full max-w-6xl rounded-xl border border-border bg-card/80 shadow-sm backdrop-blur-md">
        <div className="flex items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <button type="button" onClick={() => setActive('home')} className="flex items-center gap-2 font-semibold text-foreground transition hover:opacity-80" aria-label="MyFruitID home">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
            </span>
            <span>MyFruitID</span>
          </button>

          <nav className="hidden items-center gap-1 sm:flex" aria-label="Main">
            {links.map(({
            id,
            label
          }) => {
            const isActive = active === id;
            return <button key={id} type="button" onClick={() => setActive(id)} className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${isActive ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground hover:bg-muted hover:text-foreground'}`} aria-current={isActive ? 'page' : undefined}>
                  
                  {label}
                </button>;
          })}
          </nav>

          <button type="button" onClick={() => setMenuOpen(o => !o)} className="rounded-lg p-2 text-muted-foreground hover:bg-muted sm:hidden" aria-expanded={menuOpen} aria-label="Toggle menu">
            
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              {menuOpen ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /> : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />}
            </svg>
          </button>
        </div>

        {menuOpen && <nav className="border-t border-border px-4 py-3 sm:hidden" aria-label="Mobile">
            <div className="flex flex-col gap-1">
              {links.map(({
            id,
            label
          }) => {
            const isActive = active === id;
            return <button key={id} type="button" onClick={() => {
              setActive(id);
              setMenuOpen(false);
            }} className={`rounded-lg px-3 py-2 text-left text-sm font-medium transition-colors ${isActive ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-muted hover:text-foreground'}`}>
                  
                    {label}
                  </button>;
          })}
            </div>
          </nav>}
      </header>
    </div>;
};