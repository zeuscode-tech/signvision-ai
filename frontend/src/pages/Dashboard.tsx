import { useEffect, useState } from 'react'
import { ArrowRight, BrainCircuit, Camera, Flame, ShieldCheck } from 'lucide-react'
import { api } from '../api'
import type { NavKey } from '../types'

export function Dashboard({ navigate }: { navigate: (page: NavKey) => void }) {
  const [progress, setProgress] = useState<Record<string, unknown>>({})
  const [ready, setReady] = useState(false); const [bukvaReady,setBukvaReady]=useState(false)
  useEffect(() => { api.progress().then(setProgress).catch(()=>{}); api.health().then(x=>{setReady(x.model_ready);setBukvaReady(x.bukva_ready)}).catch(()=>{}) }, [])
  const attempts = Number(progress.attempts || 0), successes = Number(progress.successes || 0)
  return <div className="page dashboard">
    <section className="hero">
      <div><div className="eyebrow"><span className="pulse"/> Всё работает локально</div><h1>Научитесь видеть<br/><em>движение точнее.</em></h1><p>Персональный AI-тренажёр с готовой моделью алфавита и возможностью обучения на ваших примерах.</p><div className="hero__actions"><button className="button button--primary button--large" onClick={()=>navigate(bukvaReady?'alphabet':ready?'practice':'studio')}>{bukvaReady?'Открыть алфавит':ready?'Начать практику':'Обучить модель'} <ArrowRight/></button><button className="button button--ghost button--large" onClick={()=>navigate('learn')}>Посмотреть уроки</button></div></div>
      <div className="hero__orb"><div className="hand-glyph">🤟</div><span className="orbit orbit--one"/><span className="orbit orbit--two"/><div className="privacy-pill"><ShieldCheck/> Видео не покидает устройство</div></div>
    </section>
    <section className="stats-grid"><div className="stat card"><Flame/><span>Попытки</span><strong>{attempts}</strong><small>только реальные результаты</small></div><div className="stat card"><BrainCircuit/><span>Точность практики</span><strong>{attempts ? Math.round(successes/attempts*100)+'%' : '—'}</strong><small>{attempts?'по сохранённым попыткам':'пока нет данных'}</small></div><div className="stat card"><Camera/><span>Модели</span><strong>{bukvaReady?'Алфавит готов':ready?'Своя готова':'Нет'}</strong><small>{bukvaReady?'Bukva · 33 буквы':ready?'работает без облака':'проверьте setup'}</small></div></section>
    <section className="section-head"><div><span className="eyebrow">Ваш маршрут</span><h2>Продолжите с нужного шага</h2></div></section>
    <div className="route-grid"><button className="route card" onClick={()=>navigate('learn')}><span>01</span><h3>Освойте основу</h3><p>Настройте камеру и поймите, что именно видит трекер.</p><ArrowRight/></button><button className="route card" onClick={()=>navigate('studio')}><span>02</span><h3>Создайте словарь</h3><p>Запишите независимые сессии и обучите свою модель.</p><ArrowRight/></button><button className="route card" onClick={()=>navigate('practice')}><span>03</span><h3>Закрепите навык</h3><p>Практикуйте классы и возвращайтесь к сложным.</p><ArrowRight/></button></div>
  </div>
}

