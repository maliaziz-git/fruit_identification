import { Outlet } from 'react-router-dom'
import { SaaSNavbar } from './magicpath/saa-s-navbar/SaaSNavbar'

export function AppLayout() {
  return (
    <div className="flex min-h-screen flex-col">
      <SaaSNavbar />
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-border bg-card px-4 py-8 text-center text-sm text-muted-foreground">
        Local Fruit Identification System · MobileNetV3 transfer learning
      </footer>
    </div>
  )
}
