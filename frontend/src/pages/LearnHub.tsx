import { useEffect, useState } from 'react'
import { v1, type Lesson, type Progress as ProgressData, type SrsItem } from '../lib/v1'
import { Alphabet } from './Alphabet'
import { Practice } from './Practice'
import { Progress } from './Progress'

type Tab = 'route' | 'alphabet' | 'practice' | 'quiz' | 'progress'
const TABS: Array<[Tab, string]> = [['route', 'Маршрут'], ['alphabet', 'Алфавит'], ['practice', 'Практика'], ['quiz', 'Квиз'], ['progress', 'Прогресс']]
const STATUS: Record<string, [string, string]> = {
  ready: ['готово', 'tag--ok'],
  ready_after_training: ['после обучения в студии', ''],
  user_trained: ['ваши жесты', ''],
  requires_verified_rsl_material: ['ждёт проверенный материал', 'tag--warn'],
  requires_expert_review: ['нужна проверка эксперта', 'tag--warn'],
}

export function LearnHub() {
  const [tab, setTab] = useState<Tab>('route')
  return <div className="screen">
    <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 20, flexWrap: 'wrap' }}>
      <h1 className="h1">Обучение</h1>
      <div className="tabs" role="tablist" aria-label="Разделы обучения">
        {TABS.map(([id, label]) => <button key={id} type="button" role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>{label}</button>)}
      </div>
    </header>
    {tab === 'route' && <Route go={setTab} />}
    {tab === 'alphabet' && <Alphabet />}
    {tab === 'practice' && <Practice />}
    {tab === 'quiz' && <Practice quiz />}
    {tab === 'progress' && <Progress />}
  </div>
}

function Route({ go }: { go: (tab: Tab) => void }) {
  const [lessons, setLessons] = useState<Lesson[]>([])
  const [progress, setProgress] = useState<ProgressData | null>(null)
  const [due, setDue] = useState<SrsItem[]>([])
  useEffect(() => {
    v1.lessons().then(setLessons).catch(() => {})
    v1.progress().then(setProgress).catch(() => {})
    v1.dueReviews().then(setDue).catch(() => {})
  }, [])
  return <div className="two-col">
    <section style={{ display: 'flex', flexDirection: 'column', gap: 14, flex: '1.4 1 520px' }}>
      <h2 className="h2">Маршрут</h2>
      <div className="route-list">
        {lessons.map(lesson => {
          const [label, cls] = STATUS[lesson.material_status] ?? [lesson.material_status, '']
          return <article key={lesson.id} className="route-item">
            <span className="route-item__n">{String(lesson.level).padStart(2, '0')}</span>
            <div><h3>{lesson.title}</h3><p>{lesson.description}</p></div>
            <span className={`tag ${cls}`}>{label}</span>
          </article>
        })}
      </div>
      <p className="note">Уроки по РЖЯ появляются только с проверенным лицензированным материалом. Пока он не добавлен, раздел честно отмечен как ожидающий.</p>
    </section>
    <aside style={{ display: 'flex', flexDirection: 'column', gap: 14, flex: '1 1 340px' }}>
      <div className="stats">
        <div className="stat-box"><strong>{progress?.attempts ?? 0}</strong><span>попыток</span></div>
        <div className="stat-box"><strong>{progress?.success_rate != null ? `${Math.round(progress.success_rate * 100)}%` : '—'}</strong><span>успешно</span></div>
        <div className="stat-box"><strong>{Math.round((progress?.practice_seconds ?? 0) / 60)}</strong><span>минут практики</span></div>
      </div>
      <section className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <h2 className="h2" style={{ fontSize: 20 }}>Повторить сегодня</h2>
        {due.length ? <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>{due.map(item => <span key={item.gesture_id} className="choice choice--warn">{item.gesture_id}</span>)}</div>
          : <p className="note" style={{ margin: 0 }}>На сегодня повторений нет.</p>}
        <button type="button" className="btn btn--ink" onClick={() => go('practice')}>К практике</button>
      </section>
      <section className="panel panel--ink" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <span className="mono" style={{ fontSize: 13, color: 'var(--ink2)' }}>Готово без обучения</span>
        <strong className="display" style={{ fontSize: 30 }}>Дактильная азбука</strong>
        <span className="note">33 буквы РЖЯ, модель Bukva, работает сразу.</span>
        <button type="button" className="btn btn--ember" onClick={() => go('alphabet')}>Открыть алфавит</button>
      </section>
    </aside>
  </div>
}
