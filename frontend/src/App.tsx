import { useCallback, useEffect, useState, type ComponentType } from 'react'
import { IconHand, IconHome, IconLearn, IconScreen, IconSliders, IconStudio, Logo } from './components/Icons'
import { ConversationProvider } from './lib/conversation'
import { applyPrefs, usePrefs } from './lib/prefs'
import { Home } from './pages/Home'
import { LearnHub } from './pages/LearnHub'
import { Live } from './pages/Live'
import { Partner } from './pages/Partner'
import { Settings } from './pages/Settings'
import { Studio } from './pages/Studio'

type Route = 'home' | 'live' | 'partner' | 'partner-window' | 'learn' | 'studio' | 'settings'
const NAV: Array<[Route, string, ComponentType<{ size?: number }>]> = [
  ['home', 'Главная', IconHome], ['live', 'Разговор', IconHand], ['partner', 'Собеседник', IconScreen],
  ['learn', 'Обучение', IconLearn], ['studio', 'Студия', IconStudio], ['settings', 'Настройки', IconSliders],
]
const ROUTES = new Set<string>([...NAV.map(([id]) => id), 'partner-window'])
const TITLES: Record<Route, string> = { home: 'Главная', live: 'Разговор', partner: 'Собеседник', 'partner-window': 'Собеседник', learn: 'Обучение', studio: 'Студия жестов', settings: 'Настройки' }

const readRoute = (): Route => { const hash = location.hash.slice(1); return (ROUTES.has(hash) ? hash : 'home') as Route }

export default function App() {
  const prefs = usePrefs()
  const [route, setRoute] = useState<Route>(readRoute)
  useEffect(() => { applyPrefs(prefs) }, [prefs])
  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)')
    const sync = () => applyPrefs(prefs)
    media.addEventListener('change', sync)
    return () => media.removeEventListener('change', sync)
  }, [prefs])
  useEffect(() => {
    const onHash = () => setRoute(readRoute())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  useEffect(() => { document.title = `${TITLES[route]} · SignBridge` }, [route])
  const navigate = useCallback((next: string) => { location.hash = next }, [])

  if (route === 'partner-window') return <ConversationProvider><Partner navigate={navigate} solo /></ConversationProvider>

  return <ConversationProvider>
    <div className="app">
      <nav className="rail" aria-label="Разделы приложения">
        <span className="rail__logo"><Logo /></span>
        {NAV.map(([id, label, Icon]) => <a key={id} href={`#${id}`} aria-current={route === id ? 'page' : undefined}><Icon size={26} />{label}</a>)}
        <span className="rail__spacer" />
        <span className="rail__offline"><span className="dot" />ОФЛАЙН</span>
      </nav>
      <main className="view">
        {route === 'home' && <Home navigate={navigate} />}
        {route === 'live' && <Live navigate={navigate} />}
        {route === 'partner' && <Partner navigate={navigate} />}
        {route === 'learn' && <LearnHub />}
        {route === 'studio' && <Studio />}
        {route === 'settings' && <Settings />}
      </main>
    </div>
  </ConversationProvider>
}
