import { useCallback, useEffect, useRef, useState } from 'react'
import { LiveCamera, type HandsInfo } from '../components/LiveCamera'
import { IconTrash } from '../components/Icons'
import { v1, type Prediction, type TrainingClass, type TrainingJob } from '../lib/v1'

const NEGATIVE = 'фон'
const TARGET_REPEATS = 10
const MIN_REPEATS = 2
const SAMPLES_PER_REPEAT = 4

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

export function Studio() {
  const [classes, setClasses] = useState<TrainingClass[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [name, setName] = useState('')
  const [cameraOn, setCameraOn] = useState(false)
  const [recording, setRecording] = useState<'gesture' | 'background' | null>(null)
  const [message, setMessage] = useState('')
  const [job, setJob] = useState<TrainingJob | null>(null)
  const [check, setCheck] = useState(false)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const hands = useRef<HandsInfo | null>(null)
  const lastPredict = useRef(0)

  const refresh = useCallback(() => v1.classes().then(rows => {
    setClasses(rows)
    setSelected(current => current ?? rows.find(r => !r.is_negative)?.id ?? null)
  }).catch(() => setMessage('Локальный сервер недоступен')), [])
  useEffect(() => { void refresh() }, [refresh])

  const gestures = classes.filter(c => !c.is_negative)
  const background = classes.find(c => c.is_negative)
  const current = classes.find(c => c.id === selected) ?? null

  const create = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!name.trim()) return
    try { const row = await v1.createClass(name.trim()); setName(''); setSelected(row.id); await refresh(); setCameraOn(true) }
    catch (e) { setMessage(e instanceof Error ? e.message : 'Не удалось создать жест') }
  }

  // One repeat = one independent capture session with a few frames while the hand is visible.
  const recordRepeat = async (kind: 'gesture' | 'background') => {
    let target = kind === 'gesture' ? current : background
    if (kind === 'background' && !target) target = await v1.createClass(NEGATIVE, true)
    if (!target) return
    setRecording(kind); setMessage('')
    try {
      await sleep(600)
      const session = await v1.createSession(target.id)
      let saved = 0, attempts = 0
      while (saved < SAMPLES_PER_REPEAT && attempts < 30) {
        attempts += 1
        const landmarks = hands.current?.landmarks
        if (landmarks) { await v1.addSample(session.id, landmarks).then(() => { saved += 1 }).catch(() => {}) }
        await sleep(220)
      }
      setMessage(saved ? `Повтор записан: ${saved} кадра. Немного меняйте угол и свет.` : 'Руки не видны — повтор не записан.')
    } finally { setRecording(null); await refresh() }
  }

  const train = async () => {
    setMessage(''); setPrediction(null)
    try {
      let state = await v1.startJob()
      setJob(state)
      while (state.status === 'queued' || state.status === 'running') { await sleep(800); state = await v1.job(state.id); setJob(state) }
      if (state.status === 'completed') { setMessage('Модель обучена и включена. Проверьте жест.'); setCheck(true); setCameraOn(true) }
      else setMessage(state.error_message || 'Обучение не удалось')
    } catch (e) { setMessage(e instanceof Error ? e.message : 'Обучение не удалось'); setJob(null) }
  }

  const onHands = useCallback((info: HandsInfo) => {
    hands.current = info
    if (!check || !info.landmarks || performance.now() - lastPredict.current < 300) return
    lastPredict.current = performance.now()
    v1.predict(info.landmarks).then(setPrediction).catch(() => {})
  }, [check])

  const repeats = current?.session_count ?? 0
  const metrics = job?.metrics as { accuracy?: number; macro_f1?: number; test_sessions?: number } | null

  return <div className="screen">
    <header style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <h1 className="h1">Студия жестов</h1>
      <p className="note" style={{ fontSize: 19, maxWidth: 760, margin: 0 }}>Научите SignBridge жестам, которых нет в словаре: жест-имя, домашние слова, местные варианты.</p>
    </header>
    <div className="two-col">
      <div style={{ flex: '1.5 1 520px', display: 'flex', flexDirection: 'column', gap: 16 }}>
        <LiveCamera active={cameraOn} onActiveChange={setCameraOn} onHands={onHands} label="Камера студии">
          {cameraOn && <div className="cam__top"><div>
            <span className="pill pill--dark">{recording ? <><span className="dot dot--ember pulse" />Запись · {recording === 'gesture' ? current?.name : 'фон'}</> : check ? 'Проверка' : current ? `Жест «${current.name}»` : 'Выберите жест'}</span>
          </div></div>}
          {check && prediction && <div className="cam__bottom"><div className="cam__hint" style={{ fontSize: 24 }}>
            {prediction.label ? <>Узнаю: <strong className="display">{prediction.label}</strong> · {Math.round(prediction.confidence * 100)}%</> : prediction.state === 'unknown' ? 'Это не ваш жест (фон)' : `Не уверен · ${Math.round(prediction.confidence * 100)}%`}
          </div></div>}
        </LiveCamera>
        {message && <div className="banner" role="status">{message}</div>}
      </div>

      <aside style={{ flex: '1 1 380px', display: 'flex', flexDirection: 'column', gap: 16 }}>
        <section className="panel panel--ink" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <span className="mono" style={{ fontSize: 13, color: 'var(--ink2)' }}>Новый жест</span>
          <div className="steps">
            <div className={`step ${current ? 'step--done' : 'step--now'}`}><span className="step__n">1</span>
              <form onSubmit={create} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <strong>Название жеста</strong>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <input className="input" style={{ flex: '1 1 160px', minHeight: 52 }} value={name} onChange={e => setName(e.target.value)} placeholder="Например: бабушка" aria-label="Название жеста" />
                  <button type="submit" className="btn btn--ink" disabled={!name.trim()}>Создать</button>
                </div>
              </form></div>
            <div className={`step ${repeats >= TARGET_REPEATS ? 'step--done' : current ? 'step--now' : ''}`}><span className="step__n">2</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <strong>Повторите жест {TARGET_REPEATS} раз</strong>
                <span className="note">записано {repeats} · минимум {MIN_REPEATS}, лучше {TARGET_REPEATS}; немного меняйте угол и свет</span>
                <div className="meter meter--small">{Array.from({ length: TARGET_REPEATS }, (_, i) => <i key={i} className={i < repeats ? 'on' : ''} />)}</div>
                <button type="button" className="btn btn--ember" disabled={!current || !cameraOn || !!recording} onClick={() => recordRepeat('gesture')}>{recording === 'gesture' ? 'Записываю…' : 'Записать повтор'}</button>
                <button type="button" className="btn btn--soft" disabled={!cameraOn || !!recording} onClick={() => recordRepeat('background')}>{recording === 'background' ? 'Записываю фон…' : `Записать фон · ${background?.session_count ?? 0}`}</button>
                <span className="note">Фон — руки в кадре, но без жеста. Нужен, чтобы система умела говорить «это не ваш жест».</span>
              </div></div>
            <div className={`step ${check ? 'step--now' : ''}`}><span className="step__n">3</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <strong>Проверка</strong>
                <span className="note">Обучение займёт несколько секунд. Потом покажите жест — проверим, узнаёт ли система.</span>
                <button type="button" className="btn btn--ink" disabled={!gestures.length || (job?.status === 'running' || job?.status === 'queued')} onClick={train}>
                  {job?.status === 'running' || job?.status === 'queued' ? 'Обучаю…' : 'Обучить и проверить'}</button>
                {metrics?.accuracy !== undefined && <span className="note">Точность на отложенных повторах: {Math.round(metrics.accuracy * 100)}% · F1 {Math.round((metrics.macro_f1 ?? 0) * 100)}% · проверено на {metrics.test_sessions} независимых повторах</span>}
              </div></div>
          </div>
        </section>

        <section className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span className="mono" style={{ fontSize: 13, color: 'var(--ink2)', marginBottom: 6 }}>Мои жесты</span>
          {!gestures.length && <p className="note" style={{ margin: 0 }}>Пока пусто. Создайте первый жест.</p>}
          {gestures.map(g => {
            const ready = g.session_count >= MIN_REPEATS && g.sample_count >= 6
            return <div key={g.id} className="gesture-row" style={{ cursor: 'pointer' }}>
              <button type="button" style={{ border: 0, background: 'none', textAlign: 'left', cursor: 'pointer', flex: 1, padding: 0 }} aria-pressed={g.id === selected} onClick={() => setSelected(g.id)}>
                <strong style={{ textDecoration: g.id === selected ? 'underline' : 'none', textUnderlineOffset: 5 }}>{g.name}</strong>
                <div className="note">{g.session_count} повторов · {g.sample_count} кадров</div>
              </button>
              <span className={`tag ${ready ? 'tag--ok' : 'tag--warn'}`}>{ready ? 'можно обучать' : `ещё ${Math.max(1, MIN_REPEATS - g.session_count)} повт.`}</span>
              <button type="button" className="btn--round" style={{ width: 44, height: 44 }} aria-label={`Удалить жест ${g.name}`} onClick={async () => { await v1.deleteClass(g.id); if (selected === g.id) setSelected(null); void refresh() }}><IconTrash size={20} /></button>
            </div>
          })}
          <p className="note" style={{ marginBottom: 0 }}>Жесты хранятся только на этом ноутбуке. Сохраняются точки рук, а не видео.</p>
        </section>
      </aside>
    </div>
  </div>
}
