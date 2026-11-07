import { useEffect, useState } from 'react'
import { IconArrow, IconLock, IconSpeaker } from '../components/Icons'
import { useConversation } from '../lib/conversation'
import { usePrefs } from '../lib/prefs'
import { speak } from '../lib/speech'
import { v1, type Health, type Progress } from '../lib/v1'

const QUICK = ['Я не слышу. Смотрите на экран — я покажу текст.', 'Мне нужен врач.', 'Напишите, пожалуйста, на листке.']

function greeting() {
  const h = new Date().getHours()
  return h < 6 ? 'Доброй ночи.' : h < 12 ? 'Доброе утро.' : h < 18 ? 'Добрый день.' : 'Добрый вечер.'
}

export function Home({ navigate }: { navigate: (route: string) => void }) {
  const prefs = usePrefs()
  const convo = useConversation()
  const [health, setHealth] = useState<Health | null>(null)
  const [backendDown, setBackendDown] = useState(false)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [camera, setCamera] = useState<'granted' | 'denied' | 'prompt' | 'unknown'>('unknown')

  useEffect(() => {
    const load = () => v1.health().then(h => { setHealth(h); setBackendDown(false) }).catch(() => setBackendDown(true))
    load()
    const timer = window.setInterval(load, 4000)
    v1.progress().then(setProgress).catch(() => {})
    navigator.permissions?.query({ name: 'camera' as PermissionName }).then(s => setCamera(s.state as typeof camera)).catch(() => {})
    return () => window.clearInterval(timer)
  }, [])

  const start = () => { sessionStorage.setItem('sb-autostart', '1'); navigate('live') }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.code === 'Space' && (e.target as HTMLElement).tagName === 'BODY') { e.preventDefault(); start() } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const modelReady = health?.model_status === 'ready'
  const modelText = backendDown ? 'СЕРВЕР НЕ ЗАПУЩЕН' : !health ? 'ПРОВЕРЯЕМ МОДЕЛЬ' : modelReady ? 'МОДЕЛЬ РЖЯ ЗАГРУЖЕНА' : health.model_status === 'loading' ? 'МОДЕЛЬ ЗАГРУЖАЕТСЯ' : 'МОДЕЛЬ НЕДОСТУПНА'
  const cameraText = camera === 'granted' ? 'КАМЕРА ГОТОВА' : camera === 'denied' ? 'КАМЕРА ЗАПРЕЩЕНА' : 'КАМЕРА · СПРОСИМ ДОСТУП'

  return <div className="screen">
    <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 20, flexWrap: 'wrap' }}>
      <h1 className="h1">{greeting()}</h1>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <span className="pill"><span className={`dot ${camera === 'denied' ? 'dot--bad' : camera === 'granted' ? '' : 'dot--warn'}`} />{cameraText}</span>
        <span className="pill"><span className={`dot ${modelReady ? '' : backendDown ? 'dot--bad' : 'dot--warn'}`} />{modelText}</span>
        <span className="pill"><IconLock size={14} />Без интернета</span>
      </div>
    </header>
    {backendDown && <div className="banner banner--bad" role="alert">Локальный сервер не отвечает. Запустите <code>./start.sh</code> в папке проекта.</div>}

    <div className="home-grid">
      <a href="#live" className="hero-card dots-bg" onClick={e => { e.preventDefault(); start() }}>
        <svg aria-hidden="true" viewBox="0 0 600 460" style={{ position: 'absolute', right: -20, top: -10, width: '62%', height: '110%' }}>
          <defs><filter id="hmG" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="9" /></filter></defs>
          <path d="M266 132 C 220 40, 90 40, 70 140 S 20 300, -20 260" fill="none" stroke="#FF6B35" strokeOpacity=".7" strokeWidth="22" strokeLinecap="round" filter="url(#hmG)" className="trail" />
          <path d="M266 132 C 220 40, 90 40, 70 140 S 20 300, -20 260" fill="none" stroke="#FF6B35" strokeWidth="7" strokeLinecap="round" className="trail" />
          <g transform="translate(130,-10)" stroke="currentColor" strokeOpacity=".55" strokeWidth="4" fill="none" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="200,470 150,440 122,400 130,360 158,340" /><polyline points="200,470 160,310 150,240 146,195 142,152" /><polyline points="205,305 215,255 200,272 190,300" /><polyline points="245,315 252,268 238,285 228,310" /><polyline points="200,470 280,335 288,295 276,308 266,328" /><polyline points="160,310 205,305 245,315 280,335" />
          </g>
          <g transform="translate(130,-10)" fill="currentColor">
            {[[150,440],[122,400],[130,360],[158,340],[160,310],[150,240],[146,195],[205,305],[215,255],[200,272],[190,300],[245,315],[252,268],[238,285],[228,310],[280,335],[288,295],[276,308],[266,328]].map(([x, y]) => <circle key={`${x}-${y}`} cx={x} cy={y} r="6" />)}
            <circle cx="142" cy="152" r="13" fill="#FF6B35" className="breathe" />
          </g>
        </svg>
        <span className="mono" style={{ position: 'relative', fontSize: 13, opacity: .85 }}>Разговор</span>
        <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: 22 }}>
          <span className="display" style={{ fontWeight: 900, fontSize: 'clamp(48px,5.2vw,80px)', lineHeight: .94, maxWidth: '8ch' }}>Начать разговор</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
            <span className="btn btn--ember btn--lg">Включить камеру<IconArrow size={22} /></span>
            <span style={{ fontSize: 16, opacity: .85 }}>или нажмите <span className="kbd">ПРОБЕЛ</span></span>
          </span>
        </div>
      </a>

      <div style={{ flex: '1 1 340px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 22 }}>
        <a href="#partner" className="card-link" style={{ flex: 1 }} onClick={e => { e.preventDefault(); navigate('partner') }}>
          <span className="mono" style={{ fontSize: 13, color: 'var(--ink2)' }}>Собеседник</span>
          <span className="display" style={{ fontWeight: 800, fontSize: 56, lineHeight: .95 }}>Аа</span>
          <span style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'end', gap: 12 }}><span style={{ fontSize: 20, fontWeight: 600, lineHeight: 1.25 }}>Крупный текст и кнопки-ответы для слышащего</span><IconArrow size={28} style={{ flex: 'none' }} /></span>
        </a>
        <a href="#learn" className="card-link" style={{ flex: 1 }} onClick={e => { e.preventDefault(); navigate('learn') }}>
          <span className="mono" style={{ fontSize: 13, color: 'var(--ink2)', display: 'flex', justifyContent: 'space-between' }}><span>Обучение</span><span>{progress ? `${progress.attempts} попыток` : ''}</span></span>
          <span className="display" style={{ fontWeight: 800, fontSize: 34, lineHeight: 1 }}>{progress?.reviews_due ? `${progress.reviews_due} на повторение` : 'Алфавит и практика'}</span>
          <span className="note">{progress?.success_rate != null ? `Успешных попыток: ${Math.round(progress.success_rate * 100)}%` : 'Дактильная азбука, практика своих жестов, прогресс'}</span>
        </a>
      </div>
    </div>

    <section aria-labelledby="hm-q" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <h2 id="hm-q" className="h2">Быстрые фразы</h2>
      <div className="grid-auto">
        {QUICK.map(text => <button key={text} type="button" className="quick" onClick={() => { speak(text, { voice: prefs.voice, rate: prefs.rate }); convo.setSpoken(text) }}>
          <span style={{ flex: 1 }}>{text}</span><span className="quick__icon" aria-hidden="true"><IconSpeaker size={20} /></span>
        </button>)}
      </div>
      <p className="note">Фраза прозвучит голосом этого Mac и появится на экране собеседника.</p>
    </section>
  </div>
}
