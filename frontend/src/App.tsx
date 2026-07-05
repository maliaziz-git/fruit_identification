import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import { EncyclopediaPage } from './pages/EncyclopediaPage'
import { IdentifyPage } from './pages/IdentifyPage'
import { LandingPage } from './pages/LandingPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/identify" element={<IdentifyPage />} />
          <Route path="/encyclopedia" element={<EncyclopediaPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
