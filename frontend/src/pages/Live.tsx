import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { LiveCamera, type HandsInfo } from '../components/LiveCamera'
import { IconBackspace, IconClose, IconPause, IconPlay, IconPoints, IconRepeat, IconSpeaker, IconSwap, IconUpRight } from '../components/Icons'
import { useConversation } from '../lib/conversation'
import { usePrefs } from '../lib/prefs'
import { phraseToText, speak } from '../lib/speech'
import { useRecognition, type Candidate, type WsPrediction } from '../lib/useRecognition'
import { v1, type DomainPack } from '../lib/v1'

const BUKVA = 'bukva-mobilenetv2-tsm'
const TICKS = 21
type Pending = { label: string; confidence: number; top3: Candidate[]; eventId: number | null; at: number }

function Meter({ value, warn = false }: { value: number; warn?: boolean }) {
  const on = Math.round(Math.max(0, Math.min(1, value)) * TICKS)
  return <div className={`meter ${warn ? 'meter--warn' : ''}`} role="img" aria-label={`Уверенность ${Math.round(value * 100)} процентов`}>
    {Array.from({ length: TICKS }, (_, i) => <i key={i} className={i < on ? 'on' : ''} />)}
  </div>
}

export function Live({ navigate }: { navigate: (route: string) => void }) {
  const prefs = usePrefs()
  const convo = useConversation()
  const [cameraOn, setCameraOn] = useState(() => sessionStorage.getItem('sb-autostart') === '1')
  const [video, setVideo] = useState<HTMLVideoElement | null>(null)
  const [paused, setPaused] = useState(false)
  const [showPoints, setShowPoints] = useState(true)
  const [mode, setMode] = useState<'words' | 'spell'>('words')
  const [packs, setPacks] = useState<DomainPack[]>([])
  const [packId, setPackId] = useState<string | null | undefined>(undefined)
  const [hands, setHands] = useState<HandsInfo>({ count: 0, partlyOut: null, landmarks: null, fps: 0 })
  const [pending, setPending] = useState<Pending | null>(null)
  const [decisive, setDecisive] = useState<WsPrediction | null>(null)
  const [letters, setLetters] = useState<string[]>([])
  const [notice, setNotice] = useState('')
  const [replySeenAt, setReplySeenAt] = useState(() => convo.replyAt)
  const pendingRef = useRef(pending)
  pendingRef.current = pending

  useEffect(() => { sessionStorage.removeItem('sb-autostart') }, [])
  useEffect(() => {
    Promise.all([v1.packs(), v1.settings()]).then(([list, settings]) => {
      const usable = list.filter(p => p.status !== 'invalid' && p.compatibility?.compatible !== false)
      setPacks(usable)
      const preferred = settings.active_domain_pack_id ?? (usable.some(p => p.id === 'medical') ? 'medical' : null)
      setPackId(preferred)
    }).catch(() => setPackId(null))
  }, [])

  const commit = useCallback((label: string) => {
    convo.addToken(label)
    setPending(null)
  }, [convo])

  const onPrediction = useCallback((p: WsPrediction) => {
    if (p.inference_ms !== undefined || p.state === 'no_hands') setDecisive(p.state === 'no_hands' ? null : p)
    if (!p.accepted || !p.label) return
    if (mode === 'spell') { setLetters(prev => [...prev, p.label!]); return }
    const previous = pendingRef.current
    if (previous) convo.addToken(previous.label) // a new word arrived: keep the earlier one
    setPending({ label: p.label, confidence: p.confidence ?? 0, top3: p.top3, eventId: p.event_id ?? null, at: Date.now() })
  }, [convo, mode])

  const onNotice = useCallback((code: string, message: string) => {
    if (code === 'RATE_LIMIT_EXCEEDED') return
    setNotice(message)
    window.setTimeout(() => setNotice(''), 3500)
  }, [])

  const enabled = cameraOn && !!video && !paused && packId !== undefined
  const socket = useRecognition({
    video, enabled, modelId: mode === 'spell' ? BUKVA : null, domainPack: mode === 'spell' ? null : packId ?? null,
    onPrediction, onNotice,
  })

  // auto-add a confident word after the delay chosen in settings
  useEffect(() => {
    if (!pending || prefs.autoAddMs <= 0) return
    const timer = window.setTimeout(() => commit(pending.label), prefs.autoAddMs)
    return () => window.clearTimeout(timer)
  }, [pending, prefs.autoAddMs, commit])

  const replace = (label: string) => {
    if (pending?.eventId != null) socket.send({ type: 'correct', event_id: pending.eventId, label })
    commit(label)
  }
  const dismiss = () => setPending(null)
  const sayPhrase = () => {
    const text = phraseToText(convo.tokens)
    if (!text) return
    speak(text, { voice: prefs.voice, rate: prefs.rate })
    convo.setSpoken(text)
  }
  const finishSpelling = () => {
    if (letters.length) convo.addToken(letters.join('').toLowerCase())
    setLetters([]); setMode('words'); setDecisive(null)
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const tag = (event.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      if (event.code === 'Space') {
        event.preventDefault()
        if (pendingRef.current) commit(pendingRef.current.label)
        else if (!cameraOn) setCameraOn(true)
      } else if (event.code === 'Escape' && pendingRef.current) {
        event.preventDefault(); dismiss()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cameraOn, commit])

  const newReply = convo.reply && convo.replyAt > replySeenAt
  const handsLabel = hands.count >= 2 ? 'ОБЕ РУКИ В КАДРЕ' : hands.count === 1 ? 'ОДНА РУКА В КАДРЕ' : 'РУК НЕ ВИДНО'
  const outLabel = hands.partlyOut === 'right' ? 'ПРАВАЯ РУКА ЧАСТИЧНО ВНЕ КАДРА' : hands.partlyOut === 'left' ? 'ЛЕВАЯ РУКА ЧАСТИЧНО ВНЕ КАДРА' : ''
  const packName = useMemo(() => packs.find(p => p.id === packId)?.name, [packs, packId])

  return <div className="live">
    {newReply && <div className={`banner ${prefs.flashOnReply ? 'flash' : ''}`} role="alert" style={{ background: 'var(--ember)', color: '#17140F' }}>
      <span className="mono" style={{ fontSize: 13 }}>Собеседник ответил</span>
      <strong className="display" style={{ fontSize: 34, flex: 1 }}>{convo.reply}</strong>
      <button type="button" className="btn btn--ink" onClick={() => setReplySeenAt(convo.replyAt)}>Понятно</button>
    </div>}

    <div className="live__stage">
      <LiveCamera active={cameraOn} onActiveChange={setCameraOn} onVideo={setVideo} onHands={setHands} showPoints={showPoints} showTrail={showPoints && !prefs.reduceMotion}>
        {cameraOn && <>
          <div className="cam__top">
            <div>
              <span className="pill pill--dark"><span className="dot" style={{ background: '#5BD6B0' }} />КАМЕРА · ОФЛАЙН</span>
              {outLabel ? <span className="pill pill--warn">{outLabel}</span> : <span className="pill pill--dark">{handsLabel}</span>}
            </div>
            <div>
              <button type="button" className="cam__icon" aria-label={showPoints ? 'Скрыть точки рук' : 'Показать точки рук'} aria-pressed={!showPoints} onClick={() => setShowPoints(!showPoints)}><IconPoints size={22} /></button>
              <button type="button" className="cam__icon" aria-label={paused ? 'Продолжить распознавание' : 'Пауза распознавания'} onClick={() => setPaused(!paused)}>{paused ? <IconPlay size={22} /> : <IconPause size={22} />}</button>
              <button type="button" className="cam__icon" aria-label="Выключить камеру" onClick={() => { setCameraOn(false); setPending(null) }}><IconClose size={22} /></button>
            </div>
          </div>
          <div className="cam__bottom">
            {outLabel ? <div className="cam__hint"><IconUpRight size={24} style={{ color: '#FFB547' }} />Сдвиньте руку ближе к центру кадра</div> : <span />}
            <span className="cam__stats">{hands.fps} FPS{socket.ready ? ` · ${socket.ready.model_id}` : ''}{socket.last?.inference_ms ? ` · ${Math.round(socket.last.inference_ms)} мс` : ''}</span>
          </div>
        </>}
      </LiveCamera>

      {renderResult()}
    </div>

    <section aria-label="Фраза" className="phrase">
      <span className="mono" style={{ fontSize: 13, color: 'var(--ink2)', marginRight: 6 }}>Фраза</span>
      <div className="phrase__words" aria-live="polite">
        {convo.tokens.map((token, i) => <span key={`${i}-${token}`} className={`chip-word ${i === convo.tokens.length - 1 && Date.now() - convo.lastAddedAt < 1500 ? 'chip-word--new' : ''}`}>{token}</span>)}
        {!convo.tokens.length && <span className="note">Подтверждённые слова появятся здесь. Это глоссы, а не перевод.</span>}
        <span className="phrase__caret" aria-hidden="true" />
      </div>
      <div className="phrase__tools">
        <button type="button" className="btn--round" aria-label="Убрать последнее слово" onClick={convo.undo} disabled={!convo.tokens.length}><IconBackspace /></button>
        <button type="button" className="btn--round" aria-label="Очистить фразу" onClick={convo.clear} disabled={!convo.tokens.length}><IconClose /></button>
        <button type="button" className="btn" onClick={() => navigate('partner')}><IconSwap size={22} />Показать собеседнику</button>
        <button type="button" className="btn btn--ember btn--lg" onClick={sayPhrase} disabled={!convo.tokens.length}><IconSpeaker size={26} />Озвучить</button>
      </div>
    </section>
  </div>

  function renderResult() {
    if (mode === 'spell') return <section className="result result--word" aria-live="polite">
      <div className="result__head"><span>По буквам · дактиль</span><span>{socket.status === 'live' ? 'модель Bukva' : 'подключение…'}</span></div>
      <p className="result__sub">Покажите буквы по одной. Каждая распознанная буква добавится ниже.</p>
      <div className="spell" aria-label={`Набрано: ${letters.join('')}`}>{letters.length ? letters.map((l, i) => <span key={i}>{l}</span>) : <span style={{ background: 'transparent', color: 'var(--ink2)' }}>…</span>}</div>
      {decisive?.state === 'uncertain' && <p className="note">Не уверен в букве ({Math.round((decisive.confidence ?? 0) * 100)}%) — покажите ещё раз.</p>}
      <div className="result__grow" />
      <div className="result__actions">
        <button type="button" className="btn btn--ember" onClick={finishSpelling} disabled={!letters.length}>Слово во фразу</button>
        <button type="button" className="btn" onClick={() => setLetters(l => l.slice(0, -1))} disabled={!letters.length}><IconBackspace size={22} />Буква</button>
        <button type="button" className="btn btn--soft" onClick={() => { setLetters([]); setMode('words'); setDecisive(null) }}>Отмена</button>
      </div>
    </section>

    if (pending) return <section className="result result--word" aria-labelledby="lv-word" aria-live="polite">
      <div className="result__head"><span>Распознано</span><span>{packName ? `словарь · ${packName}` : 'все слова'}</span></div>
      <h1 id="lv-word" className="result__word">{pending.label}</h1>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <Meter value={pending.confidence} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <span style={{ fontWeight: 600, color: pending.confidence >= 0.8 ? 'var(--ok)' : 'var(--ink2)' }}>{pending.confidence >= 0.8 ? 'Уверенно' : 'Выше порога, проверьте слово'}</span>
          <span style={{ fontFamily: 'var(--mono)', fontWeight: 700, fontSize: 26 }}>{Math.round(pending.confidence * 100)}%</span>
        </div>
      </div>
      {pending.top3.length > 1 && <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <span className="note">Похожие жесты — нажмите, чтобы заменить</span>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {pending.top3.slice(1).map(c => <button key={c.label} type="button" className="choice" onClick={() => replace(c.label)}>{c.label}<small>{Math.max(1, Math.round(c.confidence * 100))}%</small></button>)}
        </div>
      </div>}
      <div className="result__grow" />
      {prefs.autoAddMs > 0 && <div className="deplete" role="img" aria-label="Слово добавится автоматически"><div key={pending.at} style={{ animationDuration: `${prefs.autoAddMs}ms` }} /></div>}
      <div className="result__actions">
        <button type="button" className="btn btn--ember" onClick={() => commit(pending.label)}>Во фразу <span className="kbd">ПРОБЕЛ</span></button>
        <button type="button" className="btn" style={{ flex: '0 1 auto' }} onClick={dismiss}>Не то <span className="kbd" style={{ color: 'var(--ink2)' }}>ESC</span></button>
      </div>
    </section>

    if (!cameraOn) return <section className="result" aria-live="polite">
      <div className="result__head"><span>Разговор</span><span>офлайн</span></div>
      <h1 className="result__word" style={{ fontSize: 'clamp(44px,4.6vw,72px)' }}>Включите камеру</h1>
      <p className="result__sub" style={{ color: 'var(--ink2)', fontWeight: 500 }}>Показывайте жесты по одному, с короткой паузой. Слово появится здесь — подтвердите его, и оно встанет во фразу.</p>
      <div className="result__grow" />
      {packPicker()}
      <button type="button" className="btn btn--ember btn--lg" onClick={() => setCameraOn(true)}>Включить камеру <span className="kbd">ПРОБЕЛ</span></button>
    </section>

    if (socket.status === 'unavailable') return <section className="result result--warn" aria-live="assertive">
      <div className="result__head" style={{ color: 'var(--warn)' }}><span>Модель недоступна</span></div>
      <h1 className="result__word" style={{ fontSize: 'clamp(40px,4vw,64px)' }}>Модель не загружена</h1>
      <p className="result__sub" style={{ fontWeight: 500 }}>{socket.unavailableReason || 'Запустите setup.sh, чтобы скачать модель, или выберите другую в настройках.'}</p>
      <div className="result__grow" />
      <button type="button" className="btn" onClick={() => navigate('settings')}>Открыть настройки</button>
    </section>

    if (socket.status !== 'live') return <section className="result" aria-live="polite">
      <div className="result__head"><span>{socket.status === 'reconnecting' ? 'Переподключение' : 'Подключение'}</span></div>
      <h1 className="result__word result__word--muted pulse" style={{ fontSize: 'clamp(40px,4vw,64px)' }}>Готовим модель…</h1>
      <p className="note">Локальный сервер распознавания: 127.0.0.1</p>
    </section>

    if (paused) return <section className="result"><div className="result__head"><span>Пауза</span></div>
      <h1 className="result__word result__word--muted" style={{ fontSize: 'clamp(44px,4.6vw,72px)' }}>На паузе</h1>
      <button type="button" className="btn btn--ember" onClick={() => setPaused(false)}><IconPlay size={22} />Продолжить</button></section>

    const d = decisive
    if (d?.state === 'uncertain') {
      const confidence = d.confidence ?? 0
      return <section className="result result--warn" aria-labelledby="un-h" aria-live="assertive">
        <div className="result__head" style={{ color: 'var(--warn)', fontWeight: 700 }}><span>Не уверен</span><span>ниже порога {Math.round((socket.ready?.thresholds.confidence ?? 0.65) * 100)}%</span></div>
        <div className="shake" key={d.timestamp.slice(0, 19)}>
          <h1 id="un-h" className="result__word">Не понял.</h1>
          <p className="result__sub">Повторите жест чуть медленнее.</p>
        </div>
        <Meter value={confidence} warn />
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ fontWeight: 600, color: 'var(--warn)' }}>{hands.partlyOut ? 'Рука частично вне кадра' : hands.count === 0 ? 'Мало точек в кадре' : 'Жест не похож на слова словаря'}</span>
          <span style={{ fontFamily: 'var(--mono)', fontWeight: 700, fontSize: 26, color: 'var(--warn)' }}>{Math.round(confidence * 100)}%</span>
        </div>
        <div className="result__grow" />
        <div className="result__actions">
          <button type="button" className="btn btn--ink" onClick={() => { socket.send({ type: 'reset' }); setDecisive(null) }}><IconRepeat size={22} />Повторю жест</button>
          <button type="button" className="btn" style={{ flex: '0 1 auto' }} onClick={() => { setMode('spell'); setDecisive(null) }}>По буквам</button>
        </div>
      </section>
    }

    const candidate = d && (d.state === 'recognized' || d.state === 'cooldown') && d.label
    const status = socket.last
    const [title, hint] = status?.state === 'no_hands' || hands.count === 0 ? ['Руки не видны', 'Поднимите руки в рамку камеры']
      : status?.state === 'collecting' ? ['Собираю движение', `${status.buffered ?? 0} / ${status.required ?? 0} кадров`]
      : d?.reason === 'no_event' ? ['Жду жест', 'Покажите жест из словаря'] : ['Покажите жест', 'По одному, с короткой паузой']
    return <section className="result" aria-live="polite">
      <div className="result__head"><span>{candidate ? 'Узнаю…' : 'Слушаю'}</span><span>{packName ? `словарь · ${packName}` : 'все слова'}</span></div>
      {candidate ? <>
        <h1 className="result__word result__word--muted">{d!.label}</h1>
        <p className="note">{d!.state === 'cooldown' ? 'Короткая пауза между словами' : d!.reason === 'duplicate_suppressed' ? 'Это слово уже добавлено — опустите руки, чтобы повторить' : 'Удерживайте жест — проверяю стабильность'}</p>
        <Meter value={d!.confidence ?? 0} />
      </> : <>
        <h1 className="result__word result__word--muted" style={{ fontSize: 'clamp(44px,4.6vw,72px)', overflowWrap: 'normal' }}>{title}</h1>
        <p className="result__sub" style={{ fontWeight: 500, color: 'var(--ink2)' }}>{hint}</p>
      </>}
      <div className="result__grow" />
      {notice && <div className="banner banner--warn" role="status">{notice}</div>}
      {packPicker()}
      <button type="button" className="btn" onClick={() => setMode('spell')}>По буквам — для имён</button>
    </section>
  }

  function packPicker() {
    return <label className="field" style={{ fontSize: 15 }}>
      <span className="mono" style={{ fontSize: 12, color: 'var(--ink2)' }}>Словарь</span>
      <select className="input" value={packId ?? ''} onChange={e => { setPackId(e.target.value || null); setDecisive(null) }}>
        {packs.map(p => <option key={p.id} value={p.id}>{p.name} · {p.enabled_labels.length} слов</option>)}
        <option value="">Все слова модели</option>
      </select>
    </label>
  }
}
