import { useEffect, useState } from 'react'
import { IconBack, IconExternal, IconHand, IconRepeat } from '../components/Icons'
import { useConversation } from '../lib/conversation'
import { phraseToText } from '../lib/speech'
import { v1 } from '../lib/v1'

const GENERAL = ['Да', 'Нет', 'Подождите, пожалуйста', 'Повторите ещё раз', 'Напишите, пожалуйста', 'Спасибо']

export function Partner({ navigate, solo = false }: { navigate: (route: string) => void; solo?: boolean }) {
  const convo = useConversation()
  const [sets, setSets] = useState<Array<{ id: string; name: string; replies: string[] }>>([{ id: 'general', name: 'Общие', replies: GENERAL }])
  const [setId, setSetId] = useState('general')
  const [flipped, setFlipped] = useState(false)
  const [scale, setScale] = useState(1)
  const [own, setOwn] = useState('')
  const [shown, setShown] = useState<string | null>(null)

  useEffect(() => {
    v1.packs().then(list => {
      const packSets = list.filter(p => p.status !== 'invalid' && p.quick_responses?.length)
        .map(p => ({ id: p.id, name: p.name, replies: ['Да', 'Нет', ...p.quick_responses] }))
      setSets([...packSets, { id: 'general', name: 'Общие', replies: GENERAL }])
      if (packSets.length) setSetId(packSets[0].id)
    }).catch(() => {})
  }, [])

  const text = convo.tokens.length ? phraseToText(convo.tokens) : convo.spoken
  const spokenAloud = !!text && convo.spoken === text
  const reply = (value: string) => {
    const clean = value.trim()
    if (!clean) return
    convo.sendReply(clean)
    setShown(clean)
    setOwn('')
  }
  const active = sets.find(s => s.id === setId) ?? sets[0]

  return <div className={`partner ${flipped ? 'partner--flipped' : ''}`} style={{ ['--partner-scale' as string]: scale }}>
    <header className="partner__head">
      {solo ? <span className="mono" style={{ fontSize: 14 }}>SignBridge</span>
        : <button type="button" className="btn btn--soft" onClick={() => navigate('live')}><IconBack size={22} />К камере</button>}
      <span className="mono" style={{ fontSize: 14, color: 'var(--ink2)' }}>Экран для собеседника</span>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {!solo && <button type="button" className="btn" onClick={() => window.open('/#partner-window', 'signbridge-partner', 'width=1280,height=800')}><IconExternal size={20} />На второй экран</button>}
        <button type="button" className="btn" onClick={() => setFlipped(!flipped)} aria-pressed={flipped}><IconRepeat size={22} />Перевернуть экран</button>
        <button type="button" className="btn" style={{ width: 56, padding: 0, fontFamily: 'var(--display)', fontWeight: 800 }} aria-label="Увеличить текст" onClick={() => setScale(s => (s >= 1.4 ? 0.8 : s + 0.2))}>А+</button>
      </div>
    </header>

    <section className="partner__said" aria-live="polite">
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10, padding: '10px 18px', borderRadius: 999, background: 'var(--ink)', color: 'var(--bg)', fontWeight: 600 }}><IconHand size={22} />Вам говорят жестами</span>
        {spokenAloud && <><span className="wave" aria-hidden="true"><i /><i /><i /><i /></span><span className="note">Фраза прозвучала вслух</span></>}
      </div>
      {text ? <p key={text} className="partner__text" style={{ ['--len-scale' as string]: text.length <= 32 ? 1 : text.length <= 64 ? 0.7 : 0.5 }}>{text}</p>
        : <p className="partner__text" style={{ color: 'var(--ink2)', opacity: .5, ['--len-scale' as string]: 0.55 }}>Здесь появится фраза собеседника.</p>}
      {text && <p className="note">Распознанные жесты — отдельные слова, а не дословный перевод.</p>}
    </section>

    <section className="partner__replies" aria-labelledby="pr-replies">
      <div style={{ maxWidth: 1440, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
          <h2 id="pr-replies" className="h2" style={{ fontSize: 26 }}>Ответьте кнопкой — человек увидит ответ крупно</h2>
          <div className="seg" role="tablist" aria-label="Набор ответов">
            {sets.map(s => <button key={s.id} type="button" role="tab" aria-selected={s.id === active.id} onClick={() => setSetId(s.id)}>{s.name}</button>)}
          </div>
        </div>
        <div className="grid-auto" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(min(200px,100%),1fr))' }}>
          {active.replies.map((r, i) => <button key={r} type="button" className={`reply-btn ${r.length <= 4 ? 'reply-btn--big' : ''} ${i === 0 ? 'reply-btn--ember' : ''}`} onClick={() => reply(r)}>{r}</button>)}
        </div>
        <form style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }} onSubmit={e => { e.preventDefault(); reply(own) }}>
          <label htmlFor="pr-own" className="sr-only">Свой ответ</label>
          <input id="pr-own" className="input" style={{ flex: '1 1 360px', minHeight: 64, fontSize: 20 }} value={own} onChange={e => setOwn(e.target.value)} placeholder="Напишите свой ответ…" />
          <button type="submit" className="btn btn--ink" style={{ minHeight: 64 }} disabled={!own.trim()}>Показать ответ</button>
        </form>
      </div>
    </section>

    {shown && <div className="reply-overlay" role="dialog" aria-modal="true" aria-label="Ответ собеседника">
      <span className="mono" style={{ fontSize: 16 }}>Ответ для вас</span>
      <p>{shown}</p>
      <div><button type="button" className="btn btn--ink btn--lg" onClick={() => setShown(null)} autoFocus>Готово</button></div>
    </div>}
  </div>
}
