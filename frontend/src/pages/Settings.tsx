import { useEffect, useRef, useState } from 'react'
import { setPrefs, usePrefs } from '../lib/prefs'
import { russianVoices, speak } from '../lib/speech'
import { v1, type AppSettings, type DomainPack, type ModelEntry, type SystemInfo } from '../lib/v1'

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return <button type="button" role="switch" aria-checked={checked} aria-label={label} className="toggle" onClick={() => onChange(!checked)} />
}

export function Settings() {
  const prefs = usePrefs()
  const [server, setServer] = useState<AppSettings | null>(null)
  const [packs, setPacks] = useState<DomainPack[]>([])
  const [models, setModels] = useState<ModelEntry[]>([])
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [voices, setVoices] = useState(russianVoices())
  const [status, setStatus] = useState('')
  const [confirmText, setConfirmText] = useState('')
  const [busy, setBusy] = useState(false)
  const file = useRef<HTMLInputElement>(null)

  const load = () => {
    v1.settings().then(setServer).catch(() => {})
    v1.packs().then(list => setPacks(list.filter(p => p.status !== 'invalid'))).catch(() => {})
    v1.models().then(setModels).catch(() => {})
    v1.system().then(setInfo).catch(() => {})
  }
  useEffect(() => {
    load()
    const update = () => setVoices(russianVoices())
    speechSynthesis?.addEventListener?.('voiceschanged', update)
    return () => speechSynthesis?.removeEventListener?.('voiceschanged', update)
  }, [])

  const save = async (patch: Partial<AppSettings>) => {
    try { setServer(await v1.updateSettings(patch)); setStatus('Сохранено') }
    catch (e) { setStatus(e instanceof Error ? e.message : 'Не удалось сохранить') }
  }
  const threshold = Math.round((server?.confidence_threshold ?? 0.65) * 100)

  return <div className="screen" style={{ maxWidth: 1100 }}>
    <h1 className="h1">Настройки</h1>
    {status && <div className="banner" role="status">{status}</div>}

    <section className="panel settings-group">
      <span className="mono">Вид</span>
      <div className="setting-row"><div><strong>Тема</strong></div>
        <div className="seg">{([['light', 'Светлая'], ['dark', 'Тёмная'], ['system', 'Как в системе']] as const).map(([v, l]) => <button key={v} type="button" aria-pressed={prefs.theme === v} onClick={() => setPrefs({ theme: v })}>{l}</button>)}</div></div>
      <div className="setting-row"><div><strong>Размер текста</strong></div>
        <div className="seg">{([[0.9, 'Аа', 15], [1, 'Аа', 19], [1.15, 'Аа', 24]] as const).map(([v, l, size]) => <button key={v} type="button" aria-pressed={prefs.scale === v} aria-label={`Масштаб ${Math.round(v * 100)}%`} style={{ fontSize: size }} onClick={() => setPrefs({ scale: v })}>{l}</button>)}</div></div>
      <div className="setting-row"><div><strong>Повышенный контраст</strong><small>Толще линии, без полупрозрачных цветов</small></div><Toggle label="Повышенный контраст" checked={prefs.contrast} onChange={v => setPrefs({ contrast: v })} /></div>
      <div className="setting-row"><div><strong>Меньше движения</strong><small>Без следа жеста и прилёта слов</small></div><Toggle label="Меньше движения" checked={prefs.reduceMotion} onChange={v => setPrefs({ reduceMotion: v })} /></div>
    </section>

    <section className="panel settings-group">
      <span className="mono">Распознавание</span>
      <div className="setting-row"><div><strong>Порог уверенности · {threshold}%</strong><small>Ниже — «не понял, повторите». Выше — слово предлагается.</small></div>
        <input type="range" min={50} max={90} step={5} value={threshold} aria-label="Порог уверенности" onChange={e => setServer(s => s && { ...s, confidence_threshold: Number(e.target.value) / 100 })} onPointerUp={e => save({ confidence_threshold: Number((e.target as HTMLInputElement).value) / 100 })} onKeyUp={e => save({ confidence_threshold: Number((e.target as HTMLInputElement).value) / 100 })} /></div>
      <div className="setting-row"><div><strong>Добавлять слово само</strong><small>Если уверенность выше порога</small></div>
        <div className="seg">{([[0, 'Нет'], [1500, '1,5 с'], [3000, '3 с']] as const).map(([v, l]) => <button key={v} type="button" aria-pressed={prefs.autoAddMs === v} onClick={() => setPrefs({ autoAddMs: v })}>{l}</button>)}</div></div>
      <div className="setting-row"><div><strong>Словарь по умолчанию</strong><small>Ограничивает распознавание словами сценария — так точнее</small></div>
        <select className="input" aria-label="Словарь по умолчанию" value={server?.active_domain_pack_id ?? ''} onChange={e => save({ active_domain_pack_id: e.target.value })}>
          <option value="">Все слова модели</option>
          {packs.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select></div>
      <div className="setting-row"><div><strong>Модель РЖЯ</strong><small>{info ? `${info.active_model ?? '—'} · ${info.model_status}` : ''}</small></div>
        <select className="input" aria-label="Модель распознавания" value={models.find(m => m.is_active)?.id ?? ''} disabled={busy} onChange={async e => {
          setBusy(true); setStatus('Проверяем и загружаем модель…')
          try { await v1.activateModel(e.target.value); setStatus('Модель переключена') } catch (err) { setStatus(err instanceof Error ? err.message : 'Ошибка') }
          setBusy(false); load()
        }}>
          {models.map(m => <option key={m.id} value={m.id}>{m.name} · {m.labels_count || '?'} классов</option>)}
        </select></div>
    </section>

    <section className="panel settings-group">
      <span className="mono">Голос</span>
      <div className="setting-row"><div><strong>Голос озвучки</strong><small>Голоса macOS, работают без интернета</small></div>
        <select className="input" aria-label="Голос озвучки" value={prefs.voice ?? ''} onChange={e => setPrefs({ voice: e.target.value || null })}>
          <option value="">По умолчанию</option>
          {voices.map(v => <option key={v.name} value={v.name}>{v.name}</option>)}
        </select></div>
      <div className="setting-row"><div><strong>Скорость речи · {prefs.rate.toFixed(1)}×</strong></div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <input type="range" min={0.6} max={1.6} step={0.1} value={prefs.rate} aria-label="Скорость речи" onChange={e => setPrefs({ rate: Number(e.target.value) })} />
          <button type="button" className="btn btn--soft" onClick={() => speak('Здравствуйте, мне нужен врач.', { voice: prefs.voice, rate: prefs.rate })}>Проверить</button>
        </div></div>
    </section>

    <section className="panel settings-group">
      <span className="mono">Сигналы без звука</span>
      <div className="setting-row"><div><strong>Вспышка рамки</strong><small>Когда собеседник ответил кнопкой</small></div><Toggle label="Вспышка рамки" checked={prefs.flashOnReply} onChange={v => setPrefs({ flashOnReply: v })} /></div>
    </section>

    <section className="panel settings-group">
      <span className="mono">Приватность и данные</span>
      <div className="setting-row"><div><strong>Видео не сохраняется и не отправляется</strong><small>Всё работает на этом ноутбуке. Отключить нельзя — так задумано.</small></div><span className="tag tag--ok">всегда</span></div>
      <div className="setting-row"><div><strong>Экспорт данных</strong><small>ZIP: настройки, прогресс, ваши жесты (точки рук, без видео)</small></div><a className="btn" href={v1.exportUrl} download>Скачать архив</a></div>
      <div className="setting-row"><div><strong>Импорт</strong><small>Заменит текущие данные содержимым архива</small></div>
        <button type="button" className="btn" onClick={() => file.current?.click()}>Выбрать архив</button>
        <input ref={file} type="file" accept=".zip,application/zip" hidden onChange={async e => {
          const f = e.target.files?.[0]; if (!f) return
          try { const r = await v1.importData(f); setStatus(`Импортировано: ${r.imported_samples} образцов, ${r.imported_attempts} попыток`) } catch (err) { setStatus(err instanceof Error ? err.message : 'Ошибка импорта') }
          e.target.value = ''; load()
        }} /></div>
      <div className="setting-row"><div><strong style={{ color: 'var(--danger)' }}>Удалить все мои данные</strong><small>Прогресс, сессии, ваши жесты и модели. Встроенные модели останутся. Введите «УДАЛИТЬ».</small></div>
        <form style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }} onSubmit={async e => {
          e.preventDefault()
          try { await v1.deleteAll(); setStatus('Данные удалены'); setConfirmText('') } catch (err) { setStatus(err instanceof Error ? err.message : 'Ошибка') }
        }}>
          <input className="input" style={{ width: 180 }} aria-label="Подтверждение удаления" value={confirmText} onChange={e => setConfirmText(e.target.value)} placeholder="УДАЛИТЬ" />
          <button type="submit" className="btn" style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }} disabled={confirmText.trim().toUpperCase() !== 'УДАЛИТЬ'}>Удалить</button>
        </form></div>
    </section>
    {info && <p className="note">Backend {info.backend_version} · ONNX Runtime {info.onnxruntime_version} · детектор рук: {info.hand_detector}</p>}
  </div>
}
